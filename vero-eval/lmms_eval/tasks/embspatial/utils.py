import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig

eval_logger = logging.getLogger("lmms-eval")

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


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
        parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=tuple(option_labels))])
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice:
                parsed_choice = parsed_choice.upper()
                return parsed_choice
    if option_labels:
        return _parse_multi_choice_response(text, option_labels, index2ans=index2ans)
    letter_match = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return ""


def embspatial_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options = doc["answer_options"]
    formatted_lines = ["Options:"]
    for i, item in enumerate(options):
        letter = chr(65 + i)
        formatted_lines.append(f"{letter}. {item}")
    options_string = "\n".join(formatted_lines)
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") + doc["question"] + "\n" + options_string + "\n" + lmms_eval_specific_kwargs.get("post_prompt", "")
    return prompt.strip()


def embspatial_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def _strip_option_prefix(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"^\s*[\(\[]?(?:[A-Z][\)\].:\-]?\s+|\d+[\)\].:\-]\s*)", "", text, flags=re.IGNORECASE)


def _build_option_labels(options: Optional[List[Any]]) -> List[str]:
    if not isinstance(options, (list, tuple)) or not options:
        return []
    labels = []
    for i, item in enumerate(options):
        label = None
        if isinstance(item, str):
            match = re.match(r"\s*[\(\[]?([A-Z])[\)\].:\s]", item, flags=re.IGNORECASE)
            if match:
                label = match.group(1).upper()
        labels.append(label or chr(65 + i))
    return labels


def _build_index2ans(option_labels: List[str], options: Optional[List[Any]]) -> Dict[str, str]:
    if not option_labels or not isinstance(options, (list, tuple)):
        return {}
    index2ans = {}
    for label, item in zip(option_labels, options):
        index2ans[label] = _strip_option_prefix(item)
    return index2ans


def _build_extraction_question(doc: Dict[str, Any]) -> str:
    question = doc.get("question", "")
    if not isinstance(question, str):
        question = str(question)
    options = doc.get("answer_options")
    if isinstance(options, (list, tuple)) and options:
        labels = _build_option_labels(list(options))
        formatted = []
        for label, item in zip(labels, options):
            text = _strip_option_prefix(item)
            formatted.append(f"{label}) {text}")
        options_string = "\n".join(formatted)
        question = f"{question}\n{options_string}" if question else options_string
    return question


def embspatial_process_results(doc, results):
    choices = ["A", "B", "C", "D"]
    key_name = "embspatial_acc"
    # extract grounded answer
    grounded_output = choices[doc["answer"]]
    response = results[0]

    response = extract_final_answer(response)

    # extract predicted answer
    options = doc.get("answer_options")
    option_labels = choices
    index2ans = _build_index2ans(option_labels, options)
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["relation"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def embspatial_process_results_extract(doc, results):
    choices = ["A", "B", "C", "D"]
    key_name = "embspatial_acc"
    # extract grounded answer
    grounded_output = choices[doc["answer"]]
    response = results[0]

    response = extract_final_answer(response)

    question = _build_extraction_question(doc)
    extracted = extract_answer_from_response(response, question=question)
    if extracted:
        response = extracted

    # extract predicted answer
    options = doc.get("answer_options")
    option_labels = choices
    index2ans = _build_index2ans(option_labels, options)
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["relation"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def embspatial_aggregate_results(results: List[Dict]):
    sub_task_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        sub_task = sample["sub_task"]
        is_correct = sample["is_correct"]

        if is_correct:
            total_correct += 1
            sub_task_to_eval_samples[sub_task].append(1)
        else:
            sub_task_to_eval_samples[sub_task].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    sub_task_accuracies = {sub_task: sum(scores) / len(scores) for sub_task, scores in sub_task_to_eval_samples.items()}

    eval_logger.info("%-40s", "EmbSpatial Per-Sub-Task Accuracy")
    eval_logger.info("-" * 40)

    for sub_task, acc in sub_task_accuracies.items():
        eval_logger.info("%-20s: %.4f", sub_task, acc)

    eval_logger.info("=" * 40)
    return accuracy
