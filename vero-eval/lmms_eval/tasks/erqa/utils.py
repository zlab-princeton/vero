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

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."
_MCQ_OPTION_PATTERN = re.compile(r"(?:^|\s)[\(\[]?([A-D])[\)\]]?\s*[\.:\)]\s*", re.IGNORECASE)

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


def erqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = f"{pre_prompt}{doc['question']}"
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt.strip()


def _extract_mcq_from_question(question: str):
    if not isinstance(question, str):
        return None, None
    matches = list(_MCQ_OPTION_PATTERN.finditer(question))
    if not matches:
        return None, None
    start_index = next((i for i, match in enumerate(matches) if match.group(1).upper() == "A"), None)
    if start_index is None:
        return None, None
    matches = matches[start_index:]
    options = []
    for i, match in enumerate(matches):
        letter = match.group(1).upper()
        if letter not in "ABCD":
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        option_text = question[start:end].strip()
        if option_text:
            options.append((letter, option_text))
    if len(options) < 2:
        return None, None
    question_text = question[: matches[0].start()].strip()
    question_text = re.sub(r"\bChoices:?\s*$", "", question_text, flags=re.IGNORECASE).strip()
    return question_text, options


def _is_mcq_question(question: str) -> bool:
    if not isinstance(question, str):
        return False
    letters = {match.group(1).upper() for match in _MCQ_OPTION_PATTERN.finditer(question)}
    return len(letters) >= 2


def _format_erqa_mcq_question(question: str) -> str:
    question_text, options = _extract_mcq_from_question(question)
    if not options:
        return question
    formatted_options = "\n".join([f"{letter}. {text}" for letter, text in options])
    if question_text:
        return f"{question_text}\nChoices:\n{formatted_options}"
    return f"Choices:\n{formatted_options}"


def erqa_doc_to_text_removepost(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip().replace(REPLACE_PROMPT, "").strip()
    question = _format_erqa_mcq_question(question)
    prompt = f"{pre_prompt}{question}"
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt.strip()


def erqa_doc_to_visual(doc: dict) -> list:
    image_list = []
    for image in doc["images"]:
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


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
    else:
        question = _format_erqa_mcq_question(question)
    return question


def erqa_process_results(doc, results):
    key_name = "erqa_acc"
    # extract grounded answer
    grounded_output = doc["answer"]
    response = results[0]

    response = extract_final_answer(response)

    # extract predicted answer
    if _is_mcq_question(doc.get("question", "")):
        option_labels = ["A", "B", "C", "D"]
        index2ans = None
    else:
        option_labels = None
        index2ans = None
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["question_type"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def erqa_process_results_extract(doc, results):
    key_name = "erqa_acc"
    # extract grounded answer
    grounded_output = doc["answer"]
    response = results[0]

    response = extract_final_answer(response)

    question = _build_extraction_question(doc)
    extracted = extract_answer_from_response(response, question=question)
    if extracted:
        response = extracted

    # extract predicted answer
    if _is_mcq_question(doc.get("question", "")):
        option_labels = ["A", "B", "C", "D"]
        index2ans = None
    else:
        option_labels = None
        index2ans = None
    pred_letter = _extract_answer_letter(
        response,
        option_labels=option_labels or None,
        index2ans=index2ans or None,
    )
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["question_type"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def erqa_aggregate_results(results):
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

    eval_logger.info("%-40s", "ERQA per-sub-task accuracy")
    eval_logger.info("-" * 40)
    for sub_task, acc in sub_task_accuracies.items():
        eval_logger.info("%-20s: %.4f", sub_task, acc)
    eval_logger.info("=" * 40)

    return accuracy
