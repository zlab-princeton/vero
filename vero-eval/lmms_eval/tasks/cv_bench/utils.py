import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig


def _parse_multi_choice_response(
    response: str,
    all_choices: List[str],
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response.upper():
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response.upper():
                candidates.append(choice)

    if len(candidates) == 0 and index2ans and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return random.choice(all_choices) if all_choices else ""
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[max(range(len(start_indexes)), key=start_indexes.__getitem__)]
    else:
        pred_index = candidates[0]
    return pred_index


def _extract_answer_letter(
    text: str,
    option_labels: Optional[List[str]] = None,
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    if option_labels and math_verify_parse and StringExtractionConfig:
        try:
            parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=tuple(option_labels))])
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice:
                parsed_choice = parsed_choice.upper()
                if parsed_choice in option_labels:
                    return parsed_choice
    if option_labels:
        return _parse_multi_choice_response(text, option_labels, index2ans=index2ans)
    letter_match = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return ""


def cv_bench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options_labels = ["A", "B", "C", "D", "E"]
    num_options = len(doc["choices"])
    options_current_task = ", ".join(options_labels[:num_options])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(options_current_task) + "\n" + doc["prompt"] + "\n" + lmms_eval_specific_kwargs.get("post_prompt", "")
    return prompt.strip()


def cv_bench_doc_to_visual(doc: dict) -> list:
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


def _build_index2ans(option_labels: List[str], choices: Optional[List[Any]]) -> Dict[str, str]:
    if not option_labels or not isinstance(choices, (list, tuple)):
        return {}
    index2ans = {}
    for label, item in zip(option_labels, choices):
        index2ans[label] = _strip_option_prefix(item)
    return index2ans


def _build_extraction_question(doc: Dict[str, Any]) -> str:
    question = doc.get("prompt", "")
    if not isinstance(question, str):
        question = str(question)
    choices = doc.get("choices")
    if isinstance(choices, (list, tuple)) and choices:
        labels = _build_option_labels(list(choices))
        formatted = []
        for label, item in zip(labels, choices):
            text = _strip_option_prefix(item)
            formatted.append(f"{label}) {text}")
        options_string = "\n".join(formatted)
        question = f"{question}\n{options_string}" if question else options_string
    return question


def cv_bench_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "cv_bench_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    response = extract_final_answer(response)

    # extract predicted answer
    choices = doc.get("choices")
    option_labels = _build_option_labels(choices)
    index2ans = _build_index2ans(option_labels, choices)
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    cv_bench_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "type": doc["type"], "task": doc["task"], "source": doc["source"], "is_correct": flag}
    return {key_name: cv_bench_submission}


def cv_bench_process_results_extract(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "cv_bench_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    response = extract_final_answer(response)

    question = _build_extraction_question(doc)
    extracted = extract_answer_from_response(response, question=question)
    if extracted:
        response = extracted

    # extract predicted answer
    choices = doc.get("choices")
    option_labels = _build_option_labels(choices)
    index2ans = _build_index2ans(option_labels, choices)
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    cv_bench_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "type": doc["type"], "task": doc["task"], "source": doc["source"], "is_correct": flag}
    return {key_name: cv_bench_submission}


def cv_bench_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def cv_bench_default_aggregate_results(results: List[Dict]):
    source_samples = defaultdict(list)
    for elem in results:
        source = elem["source"]
        source_samples[source].append(elem["is_correct"])
    source_accuracies = {source: sum(scores) / len(scores) for source, scores in source_samples.items()}
    ade20k_2d = source_accuracies["ADE20K"]
    coco_2d = source_accuracies["COCO"]
    omni_3d = source_accuracies["Omni3D"]

    # original formula
    cv_bench_accuracy = 1 / 2 * ((ade20k_2d + coco_2d) / 2 + omni_3d)
    return cv_bench_accuracy
