import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

from datasets import Dataset, Features, Image as HFImage, Sequence, Value

from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer

POINT_METRICS = ["PointInBox_ACC"]
QWEN3_DISPLAY_WIDTH = 1000
QWEN3_DISPLAY_HEIGHT = 1000
QWEN3_TOOL_PROMPT_TEMPLATE = (
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\\n* The screen\'s resolution is 1000x1000.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.\\n* you can only use the left_click and mouse_move action to interact with the computer. if you can\'t find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button with coordinate (x, y).\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.IGNORECASE | re.DOTALL)
_COORDINATE_RE = re.compile(r"coordinate\"?\s*[:=]\s*\[\s*([^\]]+)\]", flags=re.IGNORECASE)
SCREENSPOTPRO_REPO_ID = "likaixin/ScreenSpot-Pro"
SCREENSPOTPRO_DATASET_DIR = "datasets--likaixin--ScreenSpot-Pro"
_SCREENSPOTPRO_ROOT: Optional[str] = None


class PointJsonFilter(RegexFilter):
    def __init__(
        self,
        regex_pattern: str = r'(?s)<answer>.*?(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\]).*?</answer>|(?s)</think>.*?(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\])|(?s)(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\])',
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


def _normalize_hub_root(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    root = Path(os.path.expanduser(path))
    if root.name == "hub":
        return root
    return root / "hub"


def _pick_snapshot(snapshots_dir: Path) -> Optional[Path]:
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def _find_snapshot_in_root(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    if (root / "annotations").is_dir() and (root / "images").is_dir():
        return root
    if (root / "snapshots").is_dir():
        return _pick_snapshot(root / "snapshots")
    dataset_dir = root / SCREENSPOTPRO_DATASET_DIR
    if (dataset_dir / "annotations").is_dir() and (dataset_dir / "images").is_dir():
        return dataset_dir
    if (dataset_dir / "snapshots").is_dir():
        return _pick_snapshot(dataset_dir / "snapshots")
    return None


def _candidate_hub_roots() -> Iterable[Path]:
    env_hub = os.getenv("HF_HUB_CACHE")
    if env_hub:
        yield Path(os.path.expanduser(env_hub))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        maybe_hub = _normalize_hub_root(hf_home)
        if maybe_hub is not None:
            yield maybe_hub
    yield Path(os.path.expanduser("~/.cache/huggingface/hub"))


def _resolve_screenspotpro_root() -> str:
    global _SCREENSPOTPRO_ROOT
    if _SCREENSPOTPRO_ROOT:
        return _SCREENSPOTPRO_ROOT

    env_root = os.getenv("SCREENSPOTPRO_ROOT")
    if env_root:
        env_root = os.path.expanduser(env_root)
        if os.path.isdir(env_root):
            found = _find_snapshot_in_root(Path(env_root))
            if found is not None:
                _SCREENSPOTPRO_ROOT = str(found)
                return _SCREENSPOTPRO_ROOT

    searched: List[str] = []
    for hub_root in _candidate_hub_roots():
        searched.append(str(hub_root))
        found = _find_snapshot_in_root(hub_root)
        if found is not None:
            _SCREENSPOTPRO_ROOT = str(found)
            return _SCREENSPOTPRO_ROOT

    raise RuntimeError(
        "ScreenSpot-Pro dataset not found in the HF cache. "
        "Set SCREENSPOTPRO_ROOT to a snapshot directory or ensure HF_HOME/HF_HUB_CACHE points to the cache root. "
        "Searched: " + ", ".join(searched)
    )


def _build_screenspotpro_dataset() -> Dataset:
    root = Path(_resolve_screenspotpro_root())
    ann_dir = root / "annotations"
    image_dir = root / "images"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"ScreenSpot-Pro annotations not found at {ann_dir}")

    images: List[str] = []
    instructions: List[str] = []
    bboxes: List[List[float]] = []
    file_names: List[str] = []
    data_types: List[str] = []
    data_sources: List[str] = []

    for ann_path in sorted(ann_dir.glob("*.json")):
        with ann_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            img_filename = entry.get("img_filename") or ""
            if not img_filename:
                continue
            image_path = image_dir / img_filename
            if not image_path.is_file():
                continue

            instruction = entry.get("instruction") or entry.get("instruction_cn") or ""
            bbox = entry.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox) < 4:
                continue

            ui_type = str(entry.get("ui_type") or "").lower()
            data_type = "text" if ui_type == "text" else "icon"
            platform = str(entry.get("platform") or "").lower()
            application = str(entry.get("application") or "").lower()
            if platform and application:
                data_source = f"{platform}_{application}"
            else:
                data_source = platform or application

            images.append(str(image_path))
            instructions.append(str(instruction))
            bboxes.append([float(v) for v in bbox[:4]])
            file_names.append(str(entry.get("id") or img_filename))
            data_types.append(data_type)
            data_sources.append(data_source)

    features = Features(
        {
            "image": HFImage(),
            "instruction": Value("string"),
            "bbox": Sequence(Value("float32")),
            "file_name": Value("string"),
            "data_type": Value("string"),
            "data_source": Value("string"),
        }
    )
    return Dataset.from_dict(
        {
            "image": images,
            "instruction": instructions,
            "bbox": bboxes,
            "file_name": file_names,
            "data_type": data_types,
            "data_source": data_sources,
        },
        features=features,
    )


def screenspot_point_doc_to_visual(doc):
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def screenspot_point_doc_to_text_abs(doc, lmms_eval_specific_kwargs=None):
    instruction = doc.get("instruction") or doc.get("instruction_cn") or ""
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    instruction = str(instruction).strip()
    if instruction and not instruction.endswith((".", "?", "!")):
        instruction += "."
    if pre_prompt and not pre_prompt.endswith((" ", "\n", "\t")):
        pre_prompt = pre_prompt + " "
    if post_prompt and not post_prompt.startswith((" ", "\n", "\t")):
        post_prompt = " " + post_prompt
    return f"{pre_prompt}{instruction}{post_prompt}"


def screenspot_point_doc_to_text_rel1000(doc, lmms_eval_specific_kwargs=None):
    instruction = doc.get("instruction") or doc.get("instruction_cn") or ""
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    instruction = str(instruction).strip()
    if instruction and not instruction.endswith((".", "?", "!")):
        instruction += "."
    if pre_prompt and not pre_prompt.endswith((" ", "\n", "\t")):
        pre_prompt = pre_prompt + " "
    if post_prompt and not post_prompt.startswith((" ", "\n", "\t")):
        post_prompt = " " + post_prompt
    return f"{pre_prompt}{instruction}{post_prompt}"


def _get_image_size(doc):
    image = doc.get("image")
    if hasattr(image, "size"):
        return image.size
    return None, None


def _build_qwen3_tool_prompt(doc, lmms_eval_specific_kwargs=None):
    template = QWEN3_TOOL_PROMPT_TEMPLATE
    if lmms_eval_specific_kwargs:
        template = lmms_eval_specific_kwargs.get("tool_prompt_template", template)
    return (
        template.replace("<display_width_px>", str(QWEN3_DISPLAY_WIDTH))
        .replace("<display_height_px>", str(QWEN3_DISPLAY_HEIGHT))
        .replace("{screen_width}", str(QWEN3_DISPLAY_WIDTH))
        .replace("{screen_height}", str(QWEN3_DISPLAY_HEIGHT))
    )


def screenspot_point_doc_to_text_qwen3_tool(doc, lmms_eval_specific_kwargs=None):
    instruction = doc.get("instruction") or doc.get("instruction_cn") or ""
    instruction = str(instruction).strip()
    if instruction and not instruction.endswith((".", "?", "!")):
        instruction += "."
    prompt = _build_qwen3_tool_prompt(doc, lmms_eval_specific_kwargs)
    if prompt and not prompt.endswith((" ", "\n", "\t")):
        prompt = prompt + " "
    return f"{prompt}{instruction}"


def screenspot_point_doc_to_target(doc, model_specific_target_kwargs=None):
    bbox = doc.get("bbox")
    if bbox is None:
        return json.dumps([0, 0, 0, 0])
    return json.dumps([float(v) for v in bbox])


def _extract_point_from_obj(obj: Any) -> Optional[List[float]]:
    if isinstance(obj, dict):
        if "point_2d" in obj and isinstance(obj["point_2d"], (list, tuple)) and len(obj["point_2d"]) >= 2:
            try:
                return [float(v) for v in obj["point_2d"][:2]]
            except (TypeError, ValueError):
                return None
        for value in obj.values():
            candidate = _extract_point_from_obj(value)
            if candidate is not None:
                return candidate
    if isinstance(obj, list):
        for item in obj:
            candidate = _extract_point_from_obj(item)
            if candidate is not None:
                return candidate
    return None


def _parse_pred_point(text: Any) -> Optional[List[float]]:
    if not isinstance(text, str):
        return None

    point: Optional[List[float]] = None
    try:
        loaded = json.loads(text)
        point = _extract_point_from_obj(loaded)
    except Exception:
        point = None

    if point is None:
        json_blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
        for block in reversed(json_blocks):
            try:
                loaded = json.loads(block)
                point = _extract_point_from_obj(loaded)
            except Exception:
                point = None
            if point is not None:
                break

    if point is None:
        point_matches = re.findall(r"point_2d\"?\s*[:=]\s*\[\s*([^\]]+)\]", text, flags=re.IGNORECASE)
        if point_matches:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", point_matches[-1])
            if len(nums) >= 2:
                point = [float(nums[0]), float(nums[1])]

    if point is None:
        matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]", text)
        if matches:
            point = [float(matches[-1][0]), float(matches[-1][1])]

    if point is None:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) >= 2:
            point = [float(nums[-2]), float(nums[-1])]

    if point is None:
        return None

    try:
        return [float(v) for v in point[:2]]
    except (TypeError, ValueError):
        return None


def _strip_after_think(text: str) -> str:
    return extract_final_answer(text)


def _parse_tool_call_point(text: Any) -> Optional[List[float]]:
    if not isinstance(text, str):
        return None

    blocks = _TOOL_CALL_BLOCK_RE.findall(text)
    if not blocks:
        blocks = [text]

    for block in reversed(blocks):
        coord_matches = _COORDINATE_RE.findall(block)
        if not coord_matches:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coord_matches[-1])
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
    return None


def _point_rel1000_to_abs(point: Optional[List[float]], width: Optional[int], height: Optional[int]) -> Optional[List[float]]:
    if point is None or not width or not height:
        return None
    return [point[0] * width / 1000.0, point[1] * height / 1000.0]


def _coerce_response_text(value: Any) -> str:
    while isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _clamp_point(point: List[float], width: Optional[int], height: Optional[int], fmt: str) -> List[float]:
    x, y = point
    fmt = (fmt or "abs_pixel").lower()
    if fmt == "abs_pixel":
        if width:
            x = max(0.0, min(float(width), x))
        if height:
            y = max(0.0, min(float(height), y))
        return [x, y]
    if fmt == "relative_1000":
        x = max(0.0, min(1000.0, x))
        y = max(0.0, min(1000.0, y))
        return [x, y]
    if fmt == "normalized":
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return [x, y]
    return [x, y]


def _bbox_to_abs(bbox: List[float], width: Optional[int], height: Optional[int], fmt: str) -> List[float]:
    x1, y1, x2, y2 = bbox
    fmt = (fmt or "abs_pixel").lower()
    if fmt == "abs_pixel":
        return [x1, y1, x2, y2]
    if not width or not height:
        return [0.0, 0.0, 0.0, 0.0]
    if fmt == "relative_1000":
        return [x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0]
    if fmt == "normalized":
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    return [x1, y1, x2, y2]


def _abs_bbox_to_format(bbox: List[float], width: Optional[int], height: Optional[int], fmt: str) -> List[float]:
    x1, y1, x2, y2 = bbox
    fmt = (fmt or "abs_pixel").lower()
    if fmt == "abs_pixel":
        if width:
            x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        if height:
            y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return [x1, y1, x2, y2]
    if not width or not height:
        return [0.0, 0.0, 0.0, 0.0]
    if fmt == "relative_1000":
        x1, x2 = x1 / width * 1000.0, x2 / width * 1000.0
        y1, y2 = y1 / height * 1000.0, y2 / height * 1000.0
        x1, x2 = sorted([max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))])
        y1, y2 = sorted([max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))])
        return [x1, y1, x2, y2]
    if fmt == "normalized":
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
        x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
        y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
        return [x1, y1, x2, y2]
    return [x1, y1, x2, y2]


def _prepare_bbox_for_compare(bbox: List[float], width: Optional[int], height: Optional[int], gt_format: str, compare_format: str) -> List[float]:
    abs_bbox = _bbox_to_abs(bbox, width, height, gt_format)
    return _abs_bbox_to_format(abs_bbox, width, height, compare_format)


def _annotate_point_format(dataset: Dataset, pred_format: str, gt_format: str) -> Dataset:
    def _add(doc):
        if "pred_point_format" not in doc:
            doc["pred_point_format"] = pred_format
        if "gt_bbox_format" not in doc:
            doc["gt_bbox_format"] = gt_format
        return doc

    try:
        return dataset.map(_add)
    except Exception:
        return dataset


def screenspot_point_process_docs_abs(dataset: Dataset) -> Dataset:
    if "instruction" not in dataset.column_names or "bbox" not in dataset.column_names:
        dataset = _build_screenspotpro_dataset()
    return _annotate_point_format(dataset, "abs_pixel", "abs_pixel")


def screenspot_point_process_docs_rel1000(dataset: Dataset) -> Dataset:
    if "instruction" not in dataset.column_names or "bbox" not in dataset.column_names:
        dataset = _build_screenspotpro_dataset()
    return _annotate_point_format(dataset, "relative_1000", "abs_pixel")


def screenspot_point_process_result(doc, result):
    pred_text = result[0] if len(result) > 0 else ""
    pred_text = _coerce_response_text(pred_text)
    pred_text = extract_final_answer(pred_text)
    pred_point = _parse_pred_point(pred_text)
    width, height = _get_image_size(doc)
    pred_format = doc.get("pred_point_format", "abs_pixel")
    gt_format = doc.get("gt_bbox_format", "abs_pixel")
    if pred_point is None:
        pred = [0.0, 0.0]
    else:
        pred = _clamp_point(pred_point, width, height, pred_format)
    gt_bbox = doc.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    gt_bbox = _prepare_bbox_for_compare([float(v) for v in gt_bbox], width, height, gt_format, pred_format)
    ann_id = doc["file_name"]
    correct = compute_point_in_box(gt_bbox, pred)
    data_dict = {
        "instruction": doc.get("instruction") or doc.get("instruction_cn") or "",
        "pred": pred,
        "ann_id": ann_id,
        "bbox": gt_bbox,
        "data_type": doc["data_type"],
        "data_source": doc["data_source"],
        "correct": correct,
    }
    return {f"screenspot_{metric}": data_dict for metric in POINT_METRICS}


def _screenspot_point_process_result_qwen3(doc, result, strip_think: bool):
    pred_text = result[0] if len(result) > 0 else ""
    pred_text = _coerce_response_text(pred_text)
    pred_text = extract_final_answer(pred_text)
    if strip_think:
        pred_text = _strip_after_think(pred_text)
    pred_point = _parse_tool_call_point(pred_text)
    pred_source = "tool_call"
    if pred_point is None:
        pred_point = _parse_pred_point(pred_text)
        pred_source = "json_fallback" if pred_point is not None else "none"
    width, height = _get_image_size(doc)
    pred_format = doc.get("pred_point_format", "abs_pixel")
    gt_format = doc.get("gt_bbox_format", "abs_pixel")
    pred_point_rel1000 = pred_point if (pred_point is not None and pred_format == "relative_1000") else None
    pred_point_abs = _point_rel1000_to_abs(pred_point_rel1000, width, height)
    if pred_point is None:
        pred = [0.0, 0.0]
    else:
        pred = _clamp_point(pred_point, width, height, pred_format)
    gt_bbox = doc.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    gt_bbox = _prepare_bbox_for_compare([float(v) for v in gt_bbox], width, height, gt_format, pred_format)
    ann_id = doc["file_name"]
    correct = compute_point_in_box(gt_bbox, pred)
    data_dict = {
        "instruction": doc.get("instruction") or doc.get("instruction_cn") or "",
        "pred": pred,
        "pred_raw_point": pred_point,
        "pred_point_rel1000": pred_point_rel1000,
        "pred_point_abs": pred_point_abs,
        "pred_point_source": pred_source,
        "prompt_display_size": [QWEN3_DISPLAY_WIDTH, QWEN3_DISPLAY_HEIGHT],
        "ann_id": ann_id,
        "bbox": gt_bbox,
        "data_type": doc["data_type"],
        "data_source": doc["data_source"],
        "correct": correct,
    }
    return {f"screenspot_{metric}": data_dict for metric in POINT_METRICS}


def screenspot_point_process_result_qwen3(doc, result):
    return _screenspot_point_process_result_qwen3(doc, result, strip_think=False)


def screenspot_point_process_result_qwen3_thinking(doc, result):
    return _screenspot_point_process_result_qwen3(doc, result, strip_think=True)


def compute_point_in_box(box, point):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def screenspot_point_aggregation_result(results, metric):
    scorers = {
        "PointInBox_ACC": compute_point_in_box,
    }
    results_dict = {
        metric: [],
        metric + "-mobile_text": [],
        metric + "-mobile_icon": [],
        metric + "-web_text": [],
        metric + "-web_icon": [],
        metric + "-desktop_text": [],
        metric + "-desktop_icon": [],
    }

    for result in results:
        gt = result["bbox"]
        pred = result["pred"]
        score = scorers[metric](gt, pred)

        results_dict[metric].append(score)
        if result["data_type"] == "text":
            if "ios" in result["data_source"] or "android" in result["data_source"]:
                results_dict[metric + "-mobile_text"].append(score)
            elif "macos" in result["data_source"] or "windows" in result["data_source"]:
                results_dict[metric + "-desktop_text"].append(score)
            else:
                results_dict[metric + "-web_text"].append(score)
        else:
            if "ios" in result["data_source"] or "android" in result["data_source"]:
                results_dict[metric + "-mobile_icon"].append(score)
            elif "macos" in result["data_source"] or "windows" in result["data_source"]:
                results_dict[metric + "-desktop_icon"].append(score)
            else:
                results_dict[metric + "-web_icon"].append(score)

    for key in results_dict:
        if len(results_dict[key]) == 0:
            results_dict[key] = 0
        else:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

    return results_dict[metric]


def screenspot_point_in_box_acc(results):
    return screenspot_point_aggregation_result(results, "PointInBox_ACC")
