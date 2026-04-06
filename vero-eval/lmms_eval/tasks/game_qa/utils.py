import logging
import re
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig

eval_logger = logging.getLogger("lmms-eval")

_CHOICE_STRINGS = tuple(list("ABCDEFGHIJKLM") + [str(i) for i in range(1, 10)])


def _is_simple_choice_answer(answer: Any) -> bool:
    if isinstance(answer, (list, tuple, dict)):
        return False
    if isinstance(answer, bool):
        return False
    if isinstance(answer, int):
        return 1 <= answer <= 9
    if isinstance(answer, float):
        return False
    if not isinstance(answer, str):
        return False
    text = answer.strip()
    if not text:
        return False
    if text.lower() in {"yes", "no"}:
        return False
    cleaned = re.sub(r"^[\s\(\[\{]+", "", text)
    cleaned = re.sub(r"[\s\)\]\}\.\,\:\;]+$", "", cleaned)
    if re.fullmatch(r"[A-Ma-m]", cleaned):
        return True
    if re.fullmatch(r"[+-]?\d+", cleaned):
        try:
            number = int(cleaned)
        except ValueError:
            return False
        return 1 <= number <= 9
    if re.fullmatch(r"[+-]?\d*\.\d+", cleaned):
        return False
    return False


def _extract_choice_token(text: str, allow_choice_parsing: bool = True) -> str:
    """
    Extract an option token from model output that may reference choices as letters (A, B, ...)
    or numbers (1, 2, ...). Returns empty string if no clean option token is found.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"^(answer|option|choice)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)

    if allow_choice_parsing and math_verify_parse and StringExtractionConfig:
        try:
            parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=_CHOICE_STRINGS)])
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice:
                parsed_choice = parsed_choice.upper()
                if parsed_choice in _CHOICE_STRINGS:
                    return parsed_choice

    letter_match = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()

    number_match = re.search(r"\b([0-9]+)\b", text)
    if number_match:
        return number_match.group(1)

    return ""


def game_qa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = f"{pre_prompt}{doc['question']}"
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt.strip()


def game_qa_doc_to_visual(doc: Dict[str, Any]) -> List:
    return [doc["image"].convert("RGB")]


def _strip_option_prefix(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"^\s*[\(\[]?(?:[A-Z][\)\].:\-]?\s+|\d+[\)\].:\-]\s*)", "", text, flags=re.IGNORECASE)


def _build_option_labels(choices: Optional[List[Any]]) -> List[str]:
    if not isinstance(choices, (list, tuple)) or not choices:
        return []
    labels = []
    for i, item in enumerate(choices):
        label = None
        if isinstance(item, str):
            match = re.match(r"\s*[\(\[]?([A-Z])[\)\].:\s]", item, flags=re.IGNORECASE)
            if match:
                label = match.group(1).upper()
        labels.append(label or chr(65 + i))
    return labels


def _strip_answer_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    matches = list(re.finditer(r"\banswer\b", text, flags=re.IGNORECASE))
    if not matches:
        return text.strip()
    tail = text[matches[-1].end() :].lstrip(" :-\t\r\n")
    return tail if tail else text.strip()


def _normalize_choice_token(token: str, option_labels: List[str]) -> str:
    if not token:
        return ""
    if option_labels and token.isdigit():
        number = int(token)
        if 1 <= number <= len(option_labels):
            return option_labels[number - 1]
    return token


def _choice_to_index(token: str, option_labels: List[str]) -> Optional[int]:
    if not token or not option_labels:
        return None
    if token.isdigit():
        number = int(token)
        if 1 <= number <= len(option_labels):
            return number
        return None
    if len(token) == 1 and token.isalpha():
        token = token.upper()
        try:
            return option_labels.index(token) + 1
        except ValueError:
            return None
    return None


def _build_extraction_question(doc: Dict[str, Any]) -> str:
    question = doc.get("question", "")
    if not isinstance(question, str):
        question = str(question)
    choices = doc.get("choices") or doc.get("options")
    if isinstance(choices, (list, tuple)) and choices:
        labels = _build_option_labels(list(choices))
        formatted = []
        for label, item in zip(labels, choices):
            text = _strip_option_prefix(item)
            formatted.append(f"{label}) {text}")
        options_string = "\n".join(formatted)
        question = f"{question}\n{options_string}" if question else options_string
    return question


def game_qa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    key_name = "game_qa_acc"
    grounded_output_raw = doc["answer"]
    grounded_output_text = str(grounded_output_raw)
    response = results[0]

    response = extract_final_answer(response)

    response = _strip_answer_prefix(response)

    option_labels = _build_option_labels(doc.get("choices") or doc.get("options"))

    allow_choice_parsing = _is_simple_choice_answer(grounded_output_raw)
    pred_choice = _extract_choice_token(response, allow_choice_parsing=allow_choice_parsing)
    pred_parsed = _normalize_choice_token(pred_choice, option_labels)
    pred_normalized = pred_parsed if pred_parsed else response.strip()

    gt_choice = _extract_choice_token(grounded_output_text, allow_choice_parsing=allow_choice_parsing)
    gt_parsed = _normalize_choice_token(gt_choice, option_labels)
    gt_normalized = gt_parsed if gt_parsed else grounded_output_text.strip()

    pred_index = _choice_to_index(pred_choice, option_labels)
    gt_index = _choice_to_index(gt_choice, option_labels)
    if pred_index is not None and gt_index is not None:
        is_correct = pred_index == gt_index
    else:
        is_correct = pred_normalized == gt_normalized
    doc_id = doc.get("state") or doc.get("game_name") or ""
    submission = {
        "id": doc_id,
        "gt_content": gt_normalized,
        "pred_parsed": pred_normalized,
        "pred": response,
        "game_name": doc.get("game_name", ""),
        "state": doc.get("state", ""),
        "is_correct": is_correct,
    }
    return {key_name: submission}


def game_qa_process_results_extract(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    key_name = "game_qa_acc"
    grounded_output_raw = doc["answer"]
    grounded_output_text = str(grounded_output_raw)
    response = results[0]

    response = extract_final_answer(response)

    response = _strip_answer_prefix(response)

    question = _build_extraction_question(doc)
    extracted = extract_answer_from_response(response, question=question)
    if extracted:
        response = extracted

    option_labels = _build_option_labels(doc.get("choices") or doc.get("options"))

    allow_choice_parsing = _is_simple_choice_answer(grounded_output_raw)
    pred_choice = _extract_choice_token(response, allow_choice_parsing=allow_choice_parsing)
    pred_parsed = _normalize_choice_token(pred_choice, option_labels)
    pred_normalized = pred_parsed if pred_parsed else response.strip()

    gt_choice = _extract_choice_token(grounded_output_text, allow_choice_parsing=allow_choice_parsing)
    gt_parsed = _normalize_choice_token(gt_choice, option_labels)
    gt_normalized = gt_parsed if gt_parsed else grounded_output_text.strip()

    pred_index = _choice_to_index(pred_choice, option_labels)
    gt_index = _choice_to_index(gt_choice, option_labels)
    if pred_index is not None and gt_index is not None:
        is_correct = pred_index == gt_index
    else:
        is_correct = pred_normalized == gt_normalized
    doc_id = doc.get("state") or doc.get("game_name") or ""
    submission = {
        "id": doc_id,
        "gt_content": gt_normalized,
        "pred_parsed": pred_normalized,
        "pred": response,
        "game_name": doc.get("game_name", ""),
        "state": doc.get("state", ""),
        "is_correct": is_correct,
    }
    return {key_name: submission}


def game_qa_aggregate_results(results: List[Dict]) -> float:
    total_samples = len(results)
    if total_samples == 0:
        return 0.0
    total_correct = sum(1 for sample in results if sample["is_correct"])
    return total_correct / total_samples
