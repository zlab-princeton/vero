import json
import re
from typing import Any, List, Optional

from datasets import Dataset

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


class PointJsonFilter(RegexFilter):
    def __init__(
        self,
        regex_pattern: str = r'(?s)<answer>.*?(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\]).*?</answer>|(?s)</think>.*?(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\])|(?s)(\[\s*\{[\s\S]*?"point_2d"[\s\S]*?\}\s*\])',
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


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
    # Placeholder for few-shot/demo contexts; evaluation ignores doc_to_target.
    return json.dumps([{"point_2d": [0, 0]}])


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


def _normalize_point(point: List[float], width: Optional[int], height: Optional[int], fmt: str) -> List[float]:
    x, y = point
    fmt = (fmt or "normalized").lower()
    if fmt == "abs_pixel":
        if not width or not height:
            return [0.0, 0.0]
        x = x / width
        y = y / height
    elif fmt == "relative_1000":
        x = x / 1000.0
        y = y / 1000.0
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return [x, y]


def _normalize_bbox(bbox: List[float], width: Optional[int], height: Optional[int], fmt: str) -> List[float]:
    x1, y1, x2, y2 = bbox
    fmt = (fmt or "normalized").lower()
    if fmt == "abs_pixel":
        if not width or not height:
            return [0.0, 0.0, 0.0, 0.0]
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    elif fmt == "relative_1000":
        x1, x2 = x1 / 1000.0, x2 / 1000.0
        y1, y2 = y1 / 1000.0, y2 / 1000.0

    x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
    y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
    return [x1, y1, x2, y2]


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
    return _annotate_point_format(dataset, "abs_pixel", "normalized")


def screenspot_point_process_docs_rel1000(dataset: Dataset) -> Dataset:
    return _annotate_point_format(dataset, "relative_1000", "normalized")


def screenspot_point_process_result(doc, result):
    pred_text = result[0] if len(result) > 0 else ""
    pred_text = _coerce_response_text(pred_text)
    pred_text = extract_final_answer(pred_text)
    pred_point = _parse_pred_point(pred_text)
    width, height = _get_image_size(doc)
    pred_format = doc.get("pred_point_format", "normalized")
    gt_format = doc.get("gt_bbox_format", "normalized")
    if pred_point is None:
        pred = [0.0, 0.0]
    else:
        pred = _normalize_point(pred_point, width, height, pred_format)
    gt_bbox = doc.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    gt_bbox = _normalize_bbox([float(v) for v in gt_bbox], width, height, gt_format)
    ann_id = doc["file_name"]
    correct = compute_point_in_box(gt_bbox, pred)
    data_dict = {
        "instruction": doc["instruction"],
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
    pred_format = doc.get("pred_point_format", "normalized")
    gt_format = doc.get("gt_bbox_format", "normalized")
    pred_point_rel1000 = pred_point if (pred_point is not None and pred_format == "relative_1000") else None
    pred_point_abs = _point_rel1000_to_abs(pred_point_rel1000, width, height)
    if pred_point is None:
        pred = [0.0, 0.0]
    else:
        pred = _normalize_point(pred_point, width, height, pred_format)
    gt_bbox = doc.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    gt_bbox = _normalize_bbox([float(v) for v in gt_bbox], width, height, gt_format)
    ann_id = doc["file_name"]
    correct = compute_point_in_box(gt_bbox, pred)
    data_dict = {
        "instruction": doc["instruction"],
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
