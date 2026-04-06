"""Shared evaluation utilities for bbox parsing and normalization.

Adapted from XiaomiMiMo/lmms-eval (mimo_vl_eval branch).
"""

import re

from lmms_eval.filters.extraction import ExtendedRegexFilter


def extract_final_boxed_content(text, strict=False):
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_matches:
        return boxed_matches[-1]
    return "" if strict else text


def extract_after_think_content(text, strict=False):
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        return text[last_think_end + len("</think>") :].strip()
    return "" if strict else text


def parse_bbox(input_str):
    """Extract four floats [x1, y1, x2, y2] from a string in various formats."""
    input_str = input_str.lower()

    # Pattern 1: Four floats within square brackets
    pattern1 = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(pattern1, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # Pattern 2: Four floats within parentheses
    pattern2 = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern2, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # Pattern 3: Four floats with labels
    top_left_x = re.search(r"top-left x:?\s*(-?\d+(?:\.\d+)?)", input_str)
    top_left_y = re.search(r"top-left y:?\s*(-?\d+(?:\.\d+)?)", input_str)
    bottom_right_x = re.search(r"bottom-right x:?\s*(-?\d+(?:\.\d+)?)", input_str)
    bottom_right_y = re.search(r"bottom-right y:?\s*(-?\d+(?:\.\d+)?)", input_str)

    if top_left_x and top_left_y and bottom_right_x and bottom_right_y:
        return [float(top_left_x.group(1)), float(top_left_y.group(1)), float(bottom_right_x.group(1)), float(bottom_right_y.group(1))]

    return [0, 0, 0, 0]


def parse_bbox_from_point(input_str):
    """Extract two floats [x, y] and return as a degenerate bbox [x, y, x, y]."""
    input_str = input_str.lower()

    pattern1 = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(pattern1, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 3)] * 2

    pattern2 = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern2, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 3)] * 2

    return [0, 0, 0, 0]


def normalize_bbox(bbox, width, height, resize_max_pixels=0):
    """Normalize bbox to [0,1] range. If values are >1, treat as absolute pixels."""
    if any(x > 1 for x in bbox):
        if resize_max_pixels > 0:
            try:
                from lmms_eval.models.model_utils.qwen.vision_process import smart_resize

                height, width = smart_resize(height=height, width=width, max_pixels=resize_max_pixels)
            except ImportError:
                pass
        bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
    return bbox


class BoxedFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = [[extract_final_boxed_content(r)][0] for resp in resps for r in resp]
        return filtered_resps


class StrictBoxedFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = [[extract_final_boxed_content(r, strict=True)][0] for resp in resps for r in resp]
        return filtered_resps


class AfterThinkFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = []
        for resp in resps:
            for r in resp:
                filtered_resps.append(extract_after_think_content(r))
        return filtered_resps
