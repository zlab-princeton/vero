from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from math_verify import parse
import re


_RATIO_RE = re.compile(r"^\s*([^:]+?)\s*:\s*([^:]+?)\s*$")


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_cot_process_results(doc, results):
    pred = results[0]
    pred = extract_answer_from_response(pred)
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def chartqa_process_results(doc, results):
    pred = results[0]
    pred = extract_final_answer(pred)
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _coerce_str(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _to_float(text):
        def _parse_scalar(value):
            try:
                parsed_value = parse(value)
            except Exception:
                return None
            if not parsed_value:
                return None
            try:
                return float(parsed_value[0])
            except (TypeError, ValueError):
                return None

        def _parse_candidates(value):
            candidates = []
            raw = value.strip()
            if raw:
                candidates.append(raw)
                # Backward-compatible percent parsing:
                # treat both "23%" and "23\\%" like "23" for numeric matching.
                if raw.endswith("\\%"):
                    stripped = raw[:-2].strip()
                    if stripped:
                        candidates.append(stripped)
                if raw.endswith("%"):
                    stripped = raw[:-1].strip()
                    if stripped:
                        candidates.append(stripped)
            for candidate in candidates:
                parsed = _parse_scalar(candidate)
                if parsed is not None:
                    return parsed
            return None

        raw_text = text if isinstance(text, str) else str(text)
        parsed_value = _parse_candidates(raw_text)
        if parsed_value is not None:
            return parsed_value

        # Safe ratio improvement: only parse simple standalone ratios (a:b).
        ratio_match = _RATIO_RE.fullmatch(raw_text)
        if ratio_match:
            left = _parse_candidates(ratio_match.group(1).strip())
            right = _parse_candidates(ratio_match.group(2).strip())
            if left is not None and right not in (None, 0.0):
                return left / right
        return None

    def _as_iterable(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _within_tolerance(value, reference):
        if reference == 0.0:
            return abs(value) <= max_relative_change
        return abs(value - reference) / abs(reference) <= max_relative_change

    predictions = [_coerce_str(v) for v in _as_iterable(prediction)]
    targets = [_coerce_str(v) for v in _as_iterable(target)]

    for pred_text in filter(None, predictions):
        if pred_text.endswith("."):
            pred_text = pred_text[:-1]
        pred_text = pred_text.strip()
        pred_has_percent = pred_text.endswith("%")
        pred_float = _to_float(pred_text)
        for target_text in filter(None, targets):
            target_has_percent = target_text.endswith("%")
            target_float = _to_float(target_text)
            if pred_float is not None and target_float is not None:
                if _within_tolerance(pred_float, target_float):
                    return True
                if (
                    pred_has_percent
                    and not target_has_percent
                    and _within_tolerance(pred_float * 100.0, target_float)
                ):
                    return True
                if (
                    pred_has_percent
                    and not target_has_percent
                    and _within_tolerance(pred_float / 100.0, target_float)
                ):
                    return True
                if (
                    target_has_percent
                    and not pred_has_percent
                    and _within_tolerance(pred_float, target_float * 100.0)
                ):
                    return True
                if (
                    target_has_percent
                    and not pred_has_percent
                    and _within_tolerance(pred_float, target_float / 100.0)
                ):
                    return True
                if (
                    not pred_has_percent
                    and not target_has_percent
                    and target_float != 0.0
                    and 0 < pred_float < 1
                    and _within_tolerance(pred_float * 100.0, target_float)
                ):
                    return True
                if (
                    not pred_has_percent
                    and not target_has_percent
                    and pred_float != 0.0
                    and 0 < target_float < 1
                    and _within_tolerance(pred_float, target_float * 100.0)
                ):
                    return True
            if pred_text.lower() == target_text.lower():
                return True

    return False
