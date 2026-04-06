import json
import re
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, IterableDataset
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
AERIALVG_REPO = "IPEC-COMMUNITY/AerialVG"


def _get_image_path(doc: Dict[str, Any]) -> str:
    image_path = doc.get("image")
    if isinstance(image_path, str) and image_path:
        return image_path
    filename = doc.get("filename")
    if not filename:
        raise ValueError("AerialVG doc is missing filename.")
    image_path = hf_hub_download(AERIALVG_REPO, filename=f"images/{filename}", repo_type="dataset")
    doc["image"] = image_path
    return image_path


class BboxJsonFilter(RegexFilter):
    def __init__(
        self,
        regex_pattern: str = r'(?s)<answer>.*?(\[\s*\{[\s\S]*?"bbox_2d"[\s\S]*?\}\s*\]).*?</answer>|</think>.*?(\[\s*\{[\s\S]*?"bbox_2d"[\s\S]*?\}\s*\])|(\[\s*\{[\s\S]*?"bbox_2d"[\s\S]*?\}\s*\])',
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


def _normalize_bbox(bbox: Iterable[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    if not box_a or not box_b:
        return 0.0
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _clip_and_sort_bbox(bbox: List[float], width: Optional[float] = None, height: Optional[float] = None) -> List[float]:
    x1, y1, x2, y2 = bbox
    if width is not None and height is not None:
        x1, x2 = sorted([max(0.0, min(width, x1)), max(0.0, min(width, x2))])
        y1, y2 = sorted([max(0.0, min(height, y1)), max(0.0, min(height, y2))])
    else:
        x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
        y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
    return [x1, y1, x2, y2]


def _extract_bbox_from_obj(obj: Any) -> Optional[List[float]]:
    if isinstance(obj, dict):
        if "bbox_2d" in obj and isinstance(obj["bbox_2d"], (list, tuple)) and len(obj["bbox_2d"]) >= 4:
            try:
                return [float(v) for v in obj["bbox_2d"][:4]]
            except (TypeError, ValueError):
                return None
        for value in obj.values():
            candidate = _extract_bbox_from_obj(value)
            if candidate is not None:
                return candidate
    if isinstance(obj, list):
        for item in obj:
            candidate = _extract_bbox_from_obj(item)
            if candidate is not None:
                return candidate
    return None


def _parse_pred_bbox(text: str) -> Optional[List[float]]:
    """Parse bbox_2d from JSON output; fall back to last 4 numbers."""
    if not isinstance(text, str):
        return None

    coords: Optional[List[float]] = None
    try:
        loaded = json.loads(text)
        coords = _extract_bbox_from_obj(loaded)
    except Exception:
        coords = None

    if coords is None:
        json_blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
        for block in reversed(json_blocks):
            try:
                loaded = json.loads(block)
                coords = _extract_bbox_from_obj(loaded)
            except Exception:
                coords = None
            if coords is not None:
                break

    if coords is None:
        bbox_matches = re.findall(r"bbox_2d\"?\s*[:=]\s*\[\s*([^\]]+)\]", text, flags=re.IGNORECASE)
        if bbox_matches:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", bbox_matches[-1])
            if len(nums) >= 4:
                coords = [float(v) for v in nums[:4]]

    if coords is None:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) >= 4:
            coords = [float(v) for v in nums[-4:]]

    if coords is None:
        return None

    try:
        return [float(v) for v in coords[:4]]
    except (TypeError, ValueError):
        return None


def _coerce_response_text(value: Any) -> str:
    while isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _scale_pred_bbox(bbox: List[float], width: float, height: float, pred_format: str) -> Optional[List[float]]:
    pred_format = (pred_format or "abs_pixel").lower()
    if pred_format == "abs_pixel":
        return _clip_and_sort_bbox(bbox, width, height)
    if pred_format == "relative_1000":
        scaled = [bbox[0] * width / 1000.0, bbox[1] * height / 1000.0, bbox[2] * width / 1000.0, bbox[3] * height / 1000.0]
        return _clip_and_sort_bbox(scaled, width, height)
    if pred_format == "relative_1":
        scaled = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        return _clip_and_sort_bbox(scaled, width, height)
    return _clip_and_sort_bbox(bbox, width, height)


def _aerialvg_process_docs(dataset, pred_bbox_format: str = "abs_pixel"):
    """Use only the first annotated region per image."""
    rows: List[Dict[str, Any]] = []
    for sample in dataset:
        caption = sample.get("grounding", {}).get("caption", "")
        regions = sample.get("grounding", {}).get("regions", [])
        if not regions:
            continue
        region = regions[0]
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        rows.append(
            {
                "filename": sample["filename"],
                "width": sample["width"],
                "height": sample["height"],
                "phrase": region.get("phrase", ""),
                "relation": region.get("realation", None),
                "caption": caption,
                "bbox": bbox,
                "region_idx": 0,
                "pred_bbox_format": pred_bbox_format,
            }
        )

    if isinstance(dataset, IterableDataset):
        return Dataset.from_list(rows)
    return Dataset.from_list(rows)


def aerialvg_process_docs(dataset):
    return _aerialvg_process_docs(dataset, pred_bbox_format="abs_pixel")


def aerialvg_process_docs_rel1000(dataset):
    return _aerialvg_process_docs(dataset, pred_bbox_format="relative_1000")


def aerialvg_doc_to_visual(doc: Dict[str, Any]):
    image_path = _get_image_path(doc)
    with PILImage.open(image_path) as img:
        return [img.convert("RGB")]


def aerialvg_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    caption = doc.get("caption", "").strip()
    phrase = doc.get("phrase", "").strip()
    relation = doc.get("relation")
    relation_text = f" Relation: {relation}." if relation else ""
    question_phrase = re.sub(r'^(?:a|an|the)\s+', '', phrase, flags=re.IGNORECASE)
    if pre and not pre.endswith((" ", "\n", "\t")):
        pre = pre + " "
    if post and not post.startswith((" ", "\n", "\t")):
        post = " " + post
    return f"{pre}{caption} Where is the {question_phrase} located?{relation_text}{post}"


def aerialvg_doc_to_text_rel1000(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    caption = doc.get("caption", "").strip()
    phrase = doc.get("phrase", "").strip()
    relation = doc.get("relation")
    relation_text = f" Relation: {relation}." if relation else ""
    question_phrase = re.sub(r'^(?:a|an|the)\s+', '', phrase, flags=re.IGNORECASE)
    if pre and not pre.endswith((" ", "\n", "\t")):
        pre = pre + " "
    if post and not post.startswith((" ", "\n", "\t")):
        post = " " + post
    return f"{pre}{caption} Where is the {question_phrase} located?{relation_text}{post}"


def aerialvg_doc_to_target(doc: Dict[str, Any]) -> str:
    bbox = _clip_and_sort_bbox(doc["bbox"], doc["width"], doc["height"])
    label = doc.get("phrase") or "object"
    return json.dumps([{"bbox_2d": bbox, "label": label}])


def aerialvg_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    _get_image_path(doc)
    pred_text = results[0] if isinstance(results, (list, tuple)) else results
    pred_text = _coerce_response_text(pred_text)
    pred_text = extract_final_answer(pred_text)
    gt_bbox = _clip_and_sort_bbox(doc["bbox"], doc["width"], doc["height"])
    pred_bbox = _parse_pred_bbox(pred_text)
    pred_format = doc.get("pred_bbox_format", "abs_pixel")
    pred_bbox = _scale_pred_bbox(pred_bbox, doc["width"], doc["height"], pred_format) if pred_bbox else None

    iou = _compute_iou(pred_bbox, gt_bbox) if pred_bbox else 0.0
    acc05 = 1.0 if iou >= 0.5 else 0.0
    acc075 = 1.0 if iou >= 0.75 else 0.0
    return {"aerialvg_iou": iou, "aerialvg_acc@0.5": acc05, "aerialvg_acc@0.75": acc075}


def _point_in_bbox(px: float, py: float, bbox: List[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def aerialvg_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
