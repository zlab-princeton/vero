import re
from typing import Any, Dict, Optional

from PIL import Image as PILImage
from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer


def _get_first_image(doc: Dict[str, Any]):
    # Common HF Image feature key is 'image'
    if "image" in doc:
        img = doc["image"]
        try:
            return img.convert("RGB")
        except Exception:
            return img
    # Some datasets use 'images' as a list
    if "images" in doc and isinstance(doc["images"], (list, tuple)) and len(doc["images"]) > 0:
        img = doc["images"][0]
        try:
            return img.convert("RGB")
        except Exception:
            return img
    # If an image path is provided
    if "image_path" in doc and isinstance(doc["image_path"], str):
        try:
            return PILImage.open(doc["image_path"]).convert("RGB")
        except Exception:
            return PILImage.open(doc["image_path"])  # best effort
    # If nothing is found, return None to let caller decide
    return None


def countbenchqa_doc_to_visual(doc: Dict[str, Any]):
    img = _get_first_image(doc)
    return [img] if img is not None else []


_QUESTION_KEYS = [
    "question",
    "prompt",
    "query",
    "instruction",
    "text",
    "caption",
]


def countbenchqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
        pre = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        post = lmms_eval_specific_kwargs["post_prompt"]

    question = None
    for k in _QUESTION_KEYS:
        if k in doc and isinstance(doc[k], str) and doc[k].strip():
            question = doc[k].strip()
            break

    if question is None:
        # Fallback generic prompt if schema differs; still returns a valid string
        question = "How many objects are requested?"

    return f"{pre}{question}{post}"


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _extract_int_from_text(s: str) -> Optional[int]:
    if not isinstance(s, str):
        try:
            return int(s)
        except Exception:
            return None
    t = s.strip().lower()
    # First try to find an integer in the string
    m = re.search(r"[-+]?\d+", t)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # Try number words (simple 0-20)
    for w, v in _NUM_WORDS.items():
        if w in t:
            return v
    return None


def _extract_last_int_from_text(s: str) -> Optional[int]:
    if not isinstance(s, str):
        try:
            return int(s)
        except Exception:
            return None
    t = s.strip().lower()
    matches = re.findall(r"[-+]?\d+", t)
    if matches:
        try:
            return int(matches[-1])
        except Exception:
            pass
    # Try number words (take the last occurrence in text order)
    found = None
    for w, v in _NUM_WORDS.items():
        if w in t:
            found = v
    return found


def _get_ground_truth_count(doc: Dict[str, Any]) -> Optional[int]:
    # Common field names for answers (CountBenchQA uses 'number')
    for k in ["number", "answer", "count", "label", "gt", "target"]:
        if k in doc:
            return _extract_int_from_text(doc[k])
    return None


def countbenchqa_process_results(doc: Dict[str, Any], results):
    # results is a list with a single generated string (may be wrapped in a list by filters)
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred = extract_final_answer(pred)
    pred_int = _extract_int_from_text(pred)
    gt_int = _get_ground_truth_count(doc)

    score = 0.0
    if (pred_int is not None) and (gt_int is not None) and (pred_int == gt_int):
        score = 1.0

    return {"exact_match": score}


def countbenchqa_cot_process_results(doc: Dict[str, Any], results):
    # When using CoT, prefer the last number in the response to avoid capturing intermediate counts
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred = extract_final_answer(pred)
    pred_int = _extract_last_int_from_text(pred)
    gt_int = _get_ground_truth_count(doc)

    score = 0.0
    if (pred_int is not None) and (gt_int is not None) and (pred_int == gt_int):
        score = 1.0
    return {"exact_match": score}


def countbenchqa_cot_process_results_json(doc: Dict[str, Any], results):
    import json

    raw_prediction = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(raw_prediction, (list, tuple)):
        raw_prediction = raw_prediction[-1] if len(raw_prediction) > 0 else ""
    parsed = None
    if isinstance(raw_prediction, dict):
        parsed = raw_prediction
    elif isinstance(raw_prediction, str):
        try:
            parsed = json.loads(raw_prediction)
        except Exception:
            parsed = None
    json_answer = parsed.get("answer") if isinstance(parsed, dict) else raw_prediction
    json_answer = extract_final_answer(json_answer)
    pred_int = _extract_last_int_from_text(json_answer)
    gt_int = _get_ground_truth_count(doc)

    score = 0.0
    if (pred_int is not None) and (gt_int is not None) and (pred_int == gt_int):
        score = 1.0
    return {"exact_match": score}


class DigitsOnlyFilter(RegexFilter):
    """Filter that extracts the first integer (optionally signed) from a response.

    If no integer is found, returns the fallback token (default: "[invalid]").
    This helps enforce numbers-only outputs when paired with strict prompts.
    """

    def __init__(self, group_select: int = 0, fallback: str = "[invalid]") -> None:
        super().__init__(regex_pattern=r"(-?\d+)", group_select=group_select, fallback=fallback)


class FinalAnswerRegexFilter(RegexFilter):
    """Filter to extract the final numeric answer when formatted as `Answer: <number>`.

    The default regex is case-insensitive and supports fullwidth colon as well.
    """

    def __init__(self, regex_pattern: str = r"(?i)answer[\s]*[:：][\s]*(-?\d+)", group_select: int = 0, fallback: str = "[invalid]") -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


class FinalAnswerNumberFilter(RegexFilter):
    """Extract the final integer, preferring content after </think>."""

    def __init__(
        self,
        regex_pattern: str = r"(?s)<answer>.*?(-?\d+).*?</answer>|(?s)</think>.*?(-?\d+)\s*$|(-?\d+)\s*$",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)
