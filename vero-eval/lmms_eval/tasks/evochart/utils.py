import re
from typing import Any, Dict, Iterable, List

from math_verify import parse
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\d*\.\d+")


def evochart_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    image = doc.get("image")
    if image is None:
        return []
    try:
        return [image.convert("RGB")]
    except AttributeError:
        return [image]


def evochart_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    question = str(doc.get("question", "")).strip()
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def _extract_final_answer(raw_response: Any) -> str:
    return extract_final_answer(raw_response)


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _parse_math_verify_number(text: str):
    try:
        parsed = parse(text)
    except Exception:
        return None
    if not parsed:
        return None
    try:
        return float(parsed[0])
    except (ValueError, TypeError):
        return None


def _to_float(text: str):
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None

    # math_verify already normalizes percent strings to decimals when possible
    parsed_value = _parse_math_verify_number(t)
    if parsed_value is not None:
        return parsed_value

    try:
        if t.endswith("%"):
            return float(t.rstrip("%")) / 100.0
        return float(t)
    except ValueError:
        return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_numeric_value(value: Any):
    if isinstance(value, (int, float)):
        return float(value)
    text = _coerce_text(value)
    if not text:
        return None

    if text.endswith("."):
        text = text[:-1].strip()

    has_percent = text.endswith("%") and not text.endswith("\\%")

    parsed_value = _parse_math_verify_number(text)
    if parsed_value is not None:
        return parsed_value * 100.0 if has_percent else parsed_value

    match = _NUMBER_PATTERN.search(text)
    if not match:
        return None
    candidate = match.group(0).replace(",", "")
    if candidate in {"", "+", "-"}:
        return None
    try:
        return float(candidate)
    except ValueError:
        return None


def _within_tolerance(value: float, reference: float, max_relative_change: float) -> bool:
    if reference == 0.0:
        return abs(value) <= max_relative_change
    return abs(value - reference) / abs(reference) <= max_relative_change


def _apply_relaxed_tolerance(prediction: Any, target: Any, max_relative_change: float = 0.05) -> bool:
    pred_text = _coerce_text(prediction)
    target_text = _coerce_text(target)
    if not pred_text or not target_text:
        return False

    if pred_text.endswith("."):
        pred_text = pred_text[:-1].strip()
    if target_text.endswith("."):
        target_text = target_text[:-1].strip()

    pred_has_percent = pred_text.endswith("%")
    target_has_percent = target_text.endswith("%")

    pred_float = _to_float(pred_text)
    target_float = _to_float(target_text)

    if pred_float is not None and target_float is not None:
        if _within_tolerance(pred_float, target_float, max_relative_change):
            return True
        if (
            pred_has_percent
            and not target_has_percent
            and _within_tolerance(pred_float * 100.0, target_float, max_relative_change)
        ):
            return True
        if (
            target_has_percent
            and not pred_has_percent
            and _within_tolerance(pred_float, target_float * 100.0, max_relative_change)
        ):
            return True
        if (
            not pred_has_percent
            and not target_has_percent
            and target_float != 0.0
            and 0 < pred_float < 1
            and _within_tolerance(pred_float * 100.0, target_float, max_relative_change)
        ):
            return True
        if (
            not pred_has_percent
            and not target_has_percent
            and pred_float != 0.0
            and 0 < target_float < 1
            and _within_tolerance(pred_float, target_float * 100.0, max_relative_change)
        ):
            return True

    return pred_text.lower() == target_text.lower()


def _compare_numeric_with_tolerance(
    prediction: Any,
    target: Any,
    max_relative_change: float = 0.05,
) -> bool:
    target_number = _extract_numeric_value(target)
    if target_number is None:
        return False
    pred_number = _extract_numeric_value(prediction)
    if pred_number is None:
        return False
    return _within_tolerance(pred_number, target_number, max_relative_change)


def evochart_process_results(doc: Dict[str, Any], results: Iterable[str]) -> Dict[str, float]:
    raw_prediction = results[0] if isinstance(results, (list, tuple)) else results
    prediction = _extract_final_answer(raw_prediction)
    answer = doc.get("answer", "")
    is_clear = doc.get("is_clear", True)

    if bool(is_clear):
        pred_norm = _normalize_text(prediction)
        answer_norm = _normalize_text(answer)
        if pred_norm == answer_norm:
            correct = 1.0
        else:
            answer_number = _extract_numeric_value(answer)
            pred_number = _extract_numeric_value(prediction) if answer_number is not None else None
            correct = float(
                answer_number is not None
                and pred_number is not None
                and abs(pred_number - answer_number) <= 1e-6
            )
    else:
        correct = float(
            _apply_relaxed_tolerance(prediction, answer)
            or _compare_numeric_with_tolerance(prediction, answer)
        )

    return {"accuracy": correct}
