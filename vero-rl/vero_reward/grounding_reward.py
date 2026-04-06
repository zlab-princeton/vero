#!/usr/bin/env python3
"""Grounding reward (bbox IoU/F1 with penalties), Perception-R1 style."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import string_match

__all__ = [
    "GROUNDING_COMPONENT_WEIGHTS",
    "GROUNDING_IOU_THRESHOLD",
    "parse_json_from_text",
    "parse_prediction_items",
    "parse_ground_truth_items",
    "evaluate_detections_hungarian",
    "grounding_component_score",
    "compute_score_accuracy",
    "compute_score",
    "compute_score_from_data_source",
]

GROUNDING_COMPONENT_WEIGHTS = {
    "location": 1.0,
    "recall": 0.75,
    "penalty": 0.5,
}
GROUNDING_IOU_THRESHOLD = 0.5

# Default assumes absolute pixel bboxes; enable env var to treat 0-1000 as normalized (Qwen style).
ASSUME_QWEN_NORMALIZED_BBOX = os.getenv("CGS_QWEN_BBOX_NORMALIZED", "0").lower() not in {"0", "false", "no"}


def _resolve_assume_qwen_normalized(assume_qwen_normalized: Optional[bool]) -> bool:
    return ASSUME_QWEN_NORMALIZED_BBOX if assume_qwen_normalized is None else assume_qwen_normalized


def parse_json_from_text(text: str) -> Optional[List]:
    """Extract JSON array from text, handling markdown code blocks."""
    code_block_pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    json_pattern = r"\[\s*\{.*?\}\s*\]"
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    return None


def _normalize_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()


def convert_qwen_point_to_standard(point_qwen: List[float], img_width: int, img_height: int) -> np.ndarray:
    """Convert Qwen2.5-VL point format (0-1000 normalized) to absolute pixels."""
    x_norm, y_norm = point_qwen
    x_abs = (x_norm / 1000.0) * img_width
    y_abs = (y_norm / 1000.0) * img_height
    return np.array([x_abs, y_abs])


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def _coerce_image_size(image_size: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(image_size, dict):
        width = image_size.get("width")
        height = image_size.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return int(width), int(height)
        return None, None
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        width, height = image_size
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return int(width), int(height)
    return None, None


def _convert_bbox_coords(
    bbox_coords: List[float],
    img_width: Optional[int],
    img_height: Optional[int],
    assume_qwen_normalized: Optional[bool] = None,
) -> Optional[np.ndarray]:
    if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
        return None
    try:
        coords = [float(x) for x in bbox_coords]
    except (TypeError, ValueError):
        return None

    x1, y1, x2, y2 = coords
    # if x1 > x2:
    #     x1, x2 = x2, x1
    # if y1 > y2:
    #     y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)


def _convert_bbox_coords_to_qwen(
    bbox_coords: List[float],
    img_width: Optional[int],
    img_height: Optional[int],
) -> Optional[np.ndarray]:
    if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
        return None
    try:
        coords = [float(x) for x in bbox_coords]
    except (TypeError, ValueError):
        return None

    if not img_width or not img_height:
        return None
    x_min, y_min, x_max, y_max = coords
    coords = [
        max(0.0, min(float(round((x_min / img_width) * 1000.0)), 1000.0)),
        max(0.0, min(float(round((y_min / img_height) * 1000.0)), 1000.0)),
        max(0.0, min(float(round((x_max / img_width) * 1000.0)), 1000.0)),
        max(0.0, min(float(round((y_max / img_height) * 1000.0)), 1000.0)),
    ]


    # delete this 
    x1, y1, x2, y2 = coords
    # if x1 > x2:
    #     x1, x2 = x2, x1
    # if y1 > y2:
    #     y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)


def parse_prediction_items(
    text: str,
    image_path: Optional[str],
    image_size: Optional[Tuple[int, int]] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    parsed = parse_json_from_text(text)
    if not parsed:
        return []

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    img_width, img_height = _coerce_image_size(image_size)
    if img_width is None or img_height is None:
        if image_path:
            try:
                img_width, img_height = get_image_dimensions(image_path)
            except Exception:
                img_width = img_height = None

    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bbox = _convert_bbox_coords(
            item.get("bbox_2d"),
            None if use_qwen_normalized else img_width,
            None if use_qwen_normalized else img_height,
            assume_qwen_normalized=use_qwen_normalized,
        )
        if bbox is None:
            continue
        label = _normalize_label(item.get("label", ""))
        items.append({"bbox": bbox, "label": label})
    return items


def parse_ground_truth_items(
    gt_answer: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    parsed = parse_json_from_text(gt_answer)
    if not parsed:
        return []

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if use_qwen_normalized:
            bbox = _convert_bbox_coords_to_qwen(item.get("bbox_2d"), img_width, img_height)
        else:
            bbox = _convert_bbox_coords(
                item.get("bbox_2d"),
                img_width,
                img_height,
                assume_qwen_normalized=use_qwen_normalized,
            )
        if bbox is None:
            continue
        label = _normalize_label(item.get("label", ""))
        items.append({"bbox": bbox, "label": label})
    return items


def iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    pred_boxes = np.asarray(pred_boxes, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    N, M = pred_boxes.shape[0], gt_boxes.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)

    pa = (pred_boxes[:, 2] - pred_boxes[:, 0]).clip(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clip(min=0)
    ga = (gt_boxes[:, 2] - gt_boxes[:, 0]).clip(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clip(min=0)

    ix1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    iy1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    ix2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    iy2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])
    iw = (ix2 - ix1).clip(min=0)
    ih = (iy2 - iy1).clip(min=0)
    inter = iw * ih

    union = pa[:, None] + ga[None, :] - inter
    union = np.maximum(union, 1e-12)
    return inter / union


def evaluate_detections_hungarian(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
    return_matches: bool = False,
) -> Dict[str, Any]:
    ious = iou_matrix(pred_boxes, gt_boxes)
    N, M = ious.shape
    matches: List[Tuple[int, int, float]] = []

    if N == 0 or M == 0:
        tp = 0
        fp = N
        fn = M
        precision = 0.0 if N > 0 else (1.0 if M == 0 else 0.0)
        recall = 0.0 if M > 0 else (1.0 if N == 0 else 0.0)
        f1 = 0.0
        mean_iou_tp = None
        result = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou_tp": mean_iou_tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if return_matches:
            result["matches"] = matches
        return result

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - explicit failure path
        raise ImportError(
            "scipy is required for Hungarian matching in grounding_reward.py; please install scipy>=1.10."
        ) from exc

    cost = 1.0 - ious
    cost[ious < iou_thr] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)

    used_pred = set()
    used_gt = set()
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_thr and r not in used_pred and c not in used_gt:
            matches.append((r, c, float(ious[r, c])))
            used_pred.add(r)
            used_gt.add(c)
    tp = len(matches)
    fp = N - tp
    fn = M - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou_tp = float(np.mean([m[2] for m in matches])) if tp > 0 else None

    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_iou_tp": mean_iou_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if return_matches:
        result["matches"] = matches
    return result


def grounding_component_score(
    pred_items: List[Dict[str, Any]],
    gt_items: List[Dict[str, Any]],
    weights: Dict[str, float],
    iou_threshold: float,
) -> float:
    if not gt_items:
        return 0.0

    pred_boxes = np.array([item["bbox"] for item in pred_items], dtype=float) if pred_items else np.zeros((0, 4))
    gt_boxes = np.array([item["bbox"] for item in gt_items], dtype=float)

    results = evaluate_detections_hungarian(pred_boxes, gt_boxes, iou_thr=iou_threshold, return_matches=True)
    mean_iou = results["mean_iou_tp"] or 0.0
    f1 = results["f1"]
    num_pred = len(pred_items)
    num_gt = len(gt_items)
    tp = results["tp"]
    fp_rate = (num_pred - tp) / num_pred if num_pred > 0 else 0.0
    fn_rate = (num_gt - tp) / num_gt if num_gt > 0 else 0.0
    miss_penalty = 0.5 * (fp_rate + fn_rate)

    score = (
        weights["location"] * mean_iou
        + weights["recall"] * f1
        - weights["penalty"] * miss_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_score(
    predict_str: str,
    ground_truth: str,
    image_path: Optional[str],
    format_score: float = 0.5,
    component_weights: Dict[str, float] | None = None,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
    image_size: Optional[Tuple[int, int]] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> float:
    """Format reward + grounding component reward."""
    if component_weights is None:
        component_weights = GROUNDING_COMPONENT_WEIGHTS

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    format_r = string_match.format_reward(predict_str)
    img_width, img_height = _coerce_image_size(image_size)
    if img_width is None or img_height is None:
        if image_path is not None:
            try:
                img_width, img_height = get_image_dimensions(image_path)
            except Exception:
                img_width = img_height = None

    if image_path is None and (img_width is None or img_height is None):
        return format_score * format_r

    resolved_image_size = (
        (img_width, img_height) if img_width is not None and img_height is not None else None
    )
    pred_items = parse_prediction_items(
        predict_str,
        image_path,
        image_size=resolved_image_size,
        assume_qwen_normalized=use_qwen_normalized,
    )
    gt_items = parse_ground_truth_items(
        ground_truth,
        img_width=img_width,
        img_height=img_height,
        assume_qwen_normalized=use_qwen_normalized,
    )
    component_score = grounding_component_score(pred_items, gt_items, component_weights, iou_threshold)
    total = format_score * format_r + (1.0 - format_score) * component_score
    return float(np.clip(total, 0.0, 1.0))


def compute_score_accuracy(
    predict_str: str,
    ground_truth: str,
    image_path: Optional[str],
    component_weights: Dict[str, float] | None = None,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
    image_size: Optional[Tuple[int, int]] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> float:
    """Return only the grounding component score (no format weighting)."""
    if component_weights is None:
        component_weights = GROUNDING_COMPONENT_WEIGHTS

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    img_width, img_height = _coerce_image_size(image_size)
    if img_width is None or img_height is None:
        if image_path is not None:
            try:
                img_width, img_height = get_image_dimensions(image_path)
            except Exception:
                img_width = img_height = None

    if image_path is None and (img_width is None or img_height is None):
        return 0.0

    resolved_image_size = (
        (img_width, img_height) if img_width is not None and img_height is not None else None
    )
    pred_items = parse_prediction_items(
        predict_str,
        image_path,
        image_size=resolved_image_size,
        assume_qwen_normalized=use_qwen_normalized,
    )
    gt_items = parse_ground_truth_items(
        ground_truth,
        img_width=img_width,
        img_height=img_height,
        assume_qwen_normalized=use_qwen_normalized,
    )
    component_score = grounding_component_score(pred_items, gt_items, component_weights, iou_threshold)
    return float(np.clip(component_score, 0.0, 1.0))


def compute_score_from_data_source(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    format_score: float = 0.5,
    component_weights: Dict[str, float] | None = None,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
    **kwargs: Dict[str, Any],
) -> float:
    image_path = None
    image_size = None
    assume_qwen_normalized = None
    if extra_info is not None:
        images = extra_info.get("images", [])
        if images:
            image_path = images[0]
        image_sizes = extra_info.get("image_sizes") or extra_info.get("image_size")
        if isinstance(image_sizes, list) and image_sizes:
            image_size = image_sizes[0]
        elif image_sizes is not None:
            image_size = image_sizes
        if "normalize_bbox_to_1000" in extra_info:
            assume_qwen_normalized = bool(extra_info.get("normalize_bbox_to_1000"))
    return compute_score(
        predict_str=solution_str,
        ground_truth=ground_truth,
        image_path=image_path,
        format_score=format_score,
        component_weights=component_weights,
        iou_threshold=iou_threshold,
        image_size=image_size,
        assume_qwen_normalized=assume_qwen_normalized,
    )
