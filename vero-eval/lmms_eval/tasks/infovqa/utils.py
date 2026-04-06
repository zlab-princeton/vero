import json
import re

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response
from lmms_eval.api.metrics import anls as compute_anls


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def infovqa_test_process_results(doc, results):
    pred = results[0]
    pred = extract_final_answer(pred)
    questionId = doc["questionId"]
    return {"submission": {"questionId": int(questionId), "answer": pred}}


def infovqa_test_aggregate_results(results, args):
    # save results as json
    file = generate_submission_file("infovqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {file}")


def _strip_answer_tags(response: str) -> str:
    return extract_final_answer(response)


def _strip_trailing_percent(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.endswith("%"):
        return value[:-1].strip()
    return value


def _normalize_question_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _coerce_text(value) -> str:
    return "" if value is None else str(value)


def _normalize_format_only(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = text.replace("$", "").replace("€", "").replace("£", "").replace("%", "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace(" & ", ", ").replace(" and ", ", ")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"(?<!\S)\+(\d)", r"\1", text)

    def _canon_number(match):
        token = match.group(0)
        number = float(token)
        return str(int(number)) if number.is_integer() else f"{number:.15g}"

    text = re.sub(r"(?<!\S)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\S)", _canon_number, text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_anls_with_format_override(references, parsed) -> float:
    score = compute_anls(references, [parsed])["anls"]
    if score == 1.0:
        return score

    normalized_pred = _normalize_format_only(parsed)
    if any(_normalize_format_only(reference) == normalized_pred for reference in references):
        return 1.0
    return score


def _compute_legacy_anls_score(references, raw_pred) -> float:
    # Match old evaluation behavior: score directly on raw model output.
    refs = [_coerce_text(ans) for ans in references]
    pred = _coerce_text(raw_pred)
    return compute_anls(refs, [pred])["anls"]


def _compute_enhanced_anls_score(references, parsed_pred) -> float:
    # Formatting-aware path: parse/normalize the extracted answer before ANLS.
    refs = [_strip_trailing_percent(_coerce_text(ans)) for ans in references]
    parsed = _strip_trailing_percent(_coerce_text(parsed_pred))
    return _compute_anls_with_format_override(refs, parsed)


def _compute_infovqa_score(references, raw_pred, parsed_pred) -> float:
    legacy_score = _compute_legacy_anls_score(references, raw_pred)
    stripped_score = _compute_enhanced_anls_score(references, parsed_pred)
    # Never regress old runs; keep only improvements from better parsing.
    return max(legacy_score, stripped_score)


def infovqa_cot_process_results(doc, results):
    raw_pred = results[0]
    parsed = extract_answer_from_response(raw_pred)
    if not parsed:
        parsed = _strip_answer_tags(raw_pred)
    parsed = _strip_trailing_percent(_coerce_text(parsed))

    references = doc.get("answers", [])
    score = _compute_infovqa_score(references, raw_pred, parsed)

    question_id = _normalize_question_id(doc.get("questionId"))
    output = {"anls": score}

    if question_id is not None:
        output["submission"] = {"questionId": question_id, "answer": parsed}
    return output

def infovqa_zs_process_results(doc, results):
    raw_pred = results[0]
    parsed = _strip_answer_tags(raw_pred)
    parsed = _strip_trailing_percent(_coerce_text(parsed))

    references = doc.get("answers", [])
    score = _compute_infovqa_score(references, raw_pred, parsed)

    question_id = _normalize_question_id(doc.get("questionId"))
    output = {"anls": score}
    if question_id is not None:
        output["submission"] = {"questionId": question_id, "answer": parsed}
    return output

def infovqa_zs_process_results_base(doc, results):
    raw_pred = results[0]
    parsed = _strip_answer_tags(raw_pred)
    parsed = _strip_trailing_percent(_coerce_text(parsed))

    references = doc.get("answers", [])
    score = _compute_infovqa_score(references, raw_pred, parsed)

    question_id = _normalize_question_id(doc.get("questionId"))
    output = {"anls": score}
    return output
    
def infovqa_cot_process_results_base(doc, results):
    raw_pred = results[0]
    parsed = extract_answer_from_response(raw_pred)
    if not parsed:
        parsed = _strip_answer_tags(raw_pred)
    parsed = _strip_trailing_percent(_coerce_text(parsed))

    references = doc.get("answers", [])
    score = _compute_infovqa_score(references, raw_pred, parsed)

    output = {"anls": score}

    return output
