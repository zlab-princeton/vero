#!/usr/bin/env python3
"""Clicking reward: does the predicted point fall inside the GT bbox?"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import grounding_reward
from . import string_match

__all__ = [
    "compute_score_accuracy",
    "compute_score",
    "compute_score_from_data_source",
]


def _resolve_assume_qwen_normalized(assume_qwen_normalized: Optional[bool]) -> bool:
    if assume_qwen_normalized is not None:
        return assume_qwen_normalized
    point_env = os.getenv("CGS_QWEN_POINT_NORMALIZED", "0").lower() not in {"0", "false", "no"}
    return point_env or grounding_reward.ASSUME_QWEN_NORMALIZED_BBOX


def _parse_point_prediction(
    text: str,
    img_width: int,
    img_height: int,
    assume_qwen_normalized: Optional[bool] = None,
) -> Optional[np.ndarray]:
    parsed = grounding_reward.parse_json_from_text(text)
    if not parsed or len(parsed) == 0:
        return None
    item = parsed[0]
    if not isinstance(item, dict):
        return None
    point_coords = item.get("point_2d")
    if not point_coords or not isinstance(point_coords, list) or len(point_coords) != 2:
        return None
    try:
        x_raw, y_raw = float(point_coords[0]), float(point_coords[1])
    except (ValueError, TypeError):
        return None

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    if use_qwen_normalized:
        return np.array([x_raw, y_raw], dtype=float)

    # Heuristic: accept absolute pixels if they already fit the image.
    if 0 <= x_raw <= img_width and 0 <= y_raw <= img_height:
        return np.array([x_raw, y_raw], dtype=float)

    return None


def _parse_ground_truth_bboxes(
    gt_answer: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> List[np.ndarray]:
    return [
        item["bbox"]
        for item in grounding_reward.parse_ground_truth_items(
            gt_answer,
            img_width=img_width,
            img_height=img_height,
            assume_qwen_normalized=assume_qwen_normalized,
        )
    ]


def _point_in_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_score_accuracy(
    predict_str: str,
    ground_truth: str,
    image_path: Optional[str],
    image_size: Optional[tuple[int, int]] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> float:
    """Return only the clicking accuracy (no format weighting)."""
    img_width, img_height = grounding_reward._coerce_image_size(image_size)
    if img_width is None or img_height is None:
        if image_path is None:
            return 0.0
        try:
            img_width, img_height = grounding_reward.get_image_dimensions(image_path)
        except Exception:
            return 0.0

    use_qwen_normalized = _resolve_assume_qwen_normalized(assume_qwen_normalized)
    pred_point = _parse_point_prediction(
        predict_str,
        img_width,
        img_height,
        assume_qwen_normalized=use_qwen_normalized,
    )
    if pred_point is None:
        return 0.0

    if use_qwen_normalized:
        gt_bboxes = _parse_ground_truth_bboxes(
            ground_truth,
            img_width=img_width,
            img_height=img_height,
            assume_qwen_normalized=use_qwen_normalized,
        )
    else:
        gt_bboxes = _parse_ground_truth_bboxes(
            ground_truth,
            assume_qwen_normalized=use_qwen_normalized,
        )
    if not gt_bboxes:
        return 0.0

    return float(1.0 if _point_in_bbox(pred_point, gt_bboxes[0]) else 0.0)


def compute_score(
    predict_str: str,
    ground_truth: str,
    image_path: Optional[str],
    format_score: float = 0.5,
    image_size: Optional[tuple[int, int]] = None,
    assume_qwen_normalized: Optional[bool] = None,
) -> float:
    format_r = string_match.format_reward(predict_str)
    acc_r = compute_score_accuracy(
        predict_str,
        ground_truth,
        image_path,
        image_size=image_size,
        assume_qwen_normalized=assume_qwen_normalized,
    )
    total = (1.0 - format_score) * acc_r + format_score * format_r
    return float(total)


def compute_score_from_data_source(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    format_score: float = 0.5,
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
        image_size=image_size,
        assume_qwen_normalized=assume_qwen_normalized,
    )
