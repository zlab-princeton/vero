import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

try:
    from math_verify import parse as math_verify_parse
    from math_verify.parser import StringExtractionConfig
except Exception:
    math_verify_parse = None
    StringExtractionConfig = None

TASKS = [
    "Reasoning",
    "Perception",
]

SUBTASKS = [
    "Monitoring",
    "Autonomous_Driving",
    "OCR with Complex Context",
    "Diagram and Table",
    "Remote Sensing",
]


def mme_realworld_doc_to_visual(doc):
    img = decode_base64_to_image(doc["bytes"])
    return [img.convert("RGB")]


import base64
import io

from PIL import Image


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def mme_realworld_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre}{question} {option_prompt}{post}"


def mme_realworld_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre}{question} {option_prompt}{post}"


def mme_realworld_doc_to_text_exact_match(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    question += " Please respond to the question with a single word or phrase.\nThe best answer is: "
    return question


def mme_realworld_cn_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "选项如下所示:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    question += " " + option_prompt + "根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n最佳答案为： "
    return question


def mme_realworld_cn_doc_to_text_exact_match(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    question += "请用一个单词或短语回答这个问题。\n最佳答案为： "
    return question


# [Image] [Question] The choices are listed below:
# (A) [Choice A]
# (B) [Choice B]
# (C) [Choice C]
# (D) [Choice D]
# (E) [Choice E]
# Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.
# The best answer is:


def _strip_reasoning_tags(text: str) -> str:
    return extract_final_answer(text)


def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)"]):
    if type(s) is dict:
        s = ""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = _strip_reasoning_tags(s).strip()
    s = s.strip()
    if math_verify_parse and StringExtractionConfig and choices:
        letters = []
        for choice in choices:
            match = re.search(r"[A-Z]", choice, re.IGNORECASE)
            if match:
                letters.append(match.group(0).upper())
        if letters:
            try:
                parsed = math_verify_parse(s, extraction_config=[StringExtractionConfig(strings=tuple(letters))])
            except Exception:
                parsed = None
            if parsed:
                parsed_choice = str(parsed[0]).strip()
                if parsed_choice:
                    parsed_choice = parsed_choice.upper()
                    if parsed_choice in letters:
                        return parsed_choice
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search(r"\b([A-E])\b|\(([A-E])\)", s, flags=re.IGNORECASE):
        return ""

    answer_patterns = [
        r"(?i)the best answer is\s*[:：]?\s*([A-E])",
        r"(?i)the correct answer is\s*[:：]?\s*([A-E])",
        r"(?i)the answer is\s*[:：]?\s*([A-E])",
        r"(?i)answer\s*[:：]?\s*([A-E])",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, s)
        if matches:
            return matches[-1].upper()

    matches = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"\(([A-E])\)", s, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"[ABCDE]", s)
    if matches:
        return matches[-1].upper()

    for choice in choices:
        if s.lower() in choice.lower():
            return choice[1]
    return ""


class FinalAnswerLetterFilter(RegexFilter):
    """Filter to extract the final answer letter when formatted as `The best answer is: <LETTER>`."""

    def __init__(
        self,
        regex_pattern: str = r"(?s)<answer>.*?([A-E]).*?</answer>|(?s)</think>.*?([A-E])\s*$|(?i)the best answer is[\s]*[:：][\s]*([A-E])",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


def mme_realworld_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme_realworld score), value: metric value
    """
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred_ans = extract_characters_regex(pred)

    category = "Perception" if "perception" in doc["category"].lower() else "Reasoning"
    sub_category = doc["category"].split("/")[-1]
    task_category = doc["l2-category"]
    pred_ans_norm = str(pred_ans).strip().lower()
    answer_norm = str(doc["answer"]).strip().lower()
    correct = bool(pred_ans_norm) and (pred_ans_norm == answer_norm or answer_norm in pred_ans_norm)
    data_dict = {
        "question_id": doc["index"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "correct": correct,
    }

    return {f"mme_realworld_score": data_dict}


def get_correct_answer(sample):
    sample["multi-choice options"] = [option.replace("（", "(").replace("）", ")") for option in sample["multi-choice options"]]

    correct_answer = next(option.split(") ")[1] for option in sample["multi-choice options"] if option.startswith(f"({sample['answer']})"))
    return correct_answer


def mme_realworld_exact_match(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme_realworld score), value: metric value
    """
    pred_ans = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred_ans, (list, tuple)):
        pred_ans = pred_ans[-1] if len(pred_ans) > 0 else ""
    answer = get_correct_answer(doc)

    category = "Perception" if "perception" in doc["category"].lower() else "Reasoning"
    sub_category = doc["category"].split("/")[-1]
    task_category = doc["l2-category"]
    data_dict = {"question_id": doc["index"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": answer}

    return {f"mme_realworld_exact_match": data_dict}


def mme_realworld_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    metrics = {}
    for task in TASKS:
        metrics[f"{task}"] = {}
        for subtask in SUBTASKS:
            metrics[f"{task}"][f"{subtask}"] = {}

    for i in range(len(results)):
        result = results[i]
        Task = result["category"]
        Subtask = result["sub_category"]
        Category = result["task_category"].lower()
        if "attribute" in Category.lower():
            Category = Category.split("/")[0] + "/attribute"
        cnt = result["pred_answer"].lower() == result["answer"].lower() or result["answer"].lower() in result["pred_answer"].lower()
        if Category not in metrics[Task][Subtask].keys():
            metrics[Task][Subtask][f"{Category}"] = {"true": cnt, "false": 1 - cnt, "is_E": result["pred_answer"] == "E"}
        else:
            metrics[Task][Subtask][f"{Category}"]["true"] += cnt
            metrics[Task][Subtask][f"{Category}"]["false"] += 1 - cnt
            metrics[Task][Subtask][f"{Category}"]["is_E"] += result["pred_answer"] == "E"

    sum_all, succ_all = 0, 0
    for task, tasks_values in metrics.items():
        eval_logger.info(f"*" * 32 + f"{task} (Task Start)")
        cnt_task, cnt_E, sum_task = 0, 0, 0
        for substask, subtask_value in tasks_values.items():
            eval_logger.info(f"+" * 16 + f"{substask} (Subtask Start)")
            cnt_subtask, sum_subtask, e_subtask = 0, 0, 0
            for category, category_dict in subtask_value.items():
                cnt_subtask += category_dict["true"]
                sum_subtask += category_dict["false"] + category_dict["true"]
                e_subtask += category_dict["is_E"]
                acc = category_dict["true"] / (category_dict["false"] + category_dict["true"])
                eval_logger.info(f"-" * 4 + f"\t" + "Acc " + "{:.4f}".format(acc) + f"\t{category.capitalize()} ({category_dict['false'] + category_dict['true']} items)")

            if sum_subtask == 0:
                acc_subtasks = 0
                e_subtask = 0
            else:
                acc_subtasks = cnt_subtask / sum_subtask
            eval_logger.info(f"+" * 16 + f"\t Acc " + "{:.4f}".format(acc_subtasks) + f"\t E choice {e_subtask} \t{substask} ({sum_subtask} items)")
            cnt_task += cnt_subtask
            sum_task += sum_subtask
            cnt_E += e_subtask

        if sum_task == 0:
            acc_task = 0
        else:
            acc_task = cnt_task / sum_task
        succ_all += cnt_task
        sum_all += sum_task
        eval_logger.info(f"*" * 32 + f"Acc " + "{:.4f}".format(acc_task) + f"\t E choice {cnt_E} \t{task} ({sum_task} items)\n")
    eval_logger.info(f"*" * 32 + f"Overall Acc " + "{:.4f}".format(succ_all / sum_all))
    return succ_all / sum_all
