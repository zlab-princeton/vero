import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.utils import JudgePromptBuilder, ResponseParser
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import get_judge_engine

# try:
from lmms_eval.tasks.mathvision.eval_utils import (
    find_math_answer,
    is_equal,
    is_number,
)
# except ImportError as e:
#     eval_logger.warning(f"Error importing eval_utils from lmms_eval.tasks.mathvision.eval_utils: {e}")
#     pass

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig

NUM_SECONDS_TO_SLEEP = 5

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "vllm").lower()
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

USE_LOCAL_JUDGE = API_TYPE not in ("openai", "azure")
if USE_LOCAL_JUDGE and os.getenv("VLLM_JUDGE_SYSTEM_PROMPT") is None:
    os.environ["VLLM_JUDGE_SYSTEM_PROMPT"] = ""

server = None
_local_engine = None

def _get_server():
    server_config = ServerConfig(
        model_name=GPT_MODEL,
    )
    global server
    if server is None and not USE_LOCAL_JUDGE:
        server = get_server(server_name=API_TYPE, config=server_config)
    return server


def _get_local_engine():
    global _local_engine
    if _local_engine is None and USE_LOCAL_JUDGE:
        _local_engine = get_judge_engine(os.getenv("JUDGE_MODEL_PATH") or GPT_MODEL)
    return _local_engine


def mathvision_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    return query_prompt

def mathvision_doc_to_text_custom(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        if "mc_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["mc_prompt"]:
            mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    query_prompt = '' #'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    
    # force images at the beginning - image tags here are kept in vllm so we need to remove
    for image_tag in ["<image1>", "<image2>", "<image3>", "<image4>", "<image5>", "<image6>", "<image7>", "<image8>", "<image>"]:
        query_prompt = query_prompt.replace(image_tag, "")
    return query_prompt.strip()


def mathvision_gpt_eval_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = extract_final_answer(pred)
        model_answer = truncate_response_tail_tiktoken(model_answer)
        gt_answer = str(doc["answer"])
        question = doc["question"]
        options = doc.get("options", [])

        # Include options in the question so the judge can map values to letters
        full_question = question
        if options:
            choices_str = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
            full_question = f"{question}\nChoices:\n{choices_str}"

        try:
            if USE_LOCAL_JUDGE:
                prompt = JudgePromptBuilder.build_binary_prompt(question=full_question, answer=gt_answer, prediction=model_answer, output_format="0/1")
                engine = _get_local_engine()
                raw = engine.generate_json(prompt, max_tokens=512)
                judge_response = ResponseParser.parse_binary_response(raw, "0/1")
                correct_list.append(judge_response)
            else:
                # Use the llm_judge API for binary evaluation
                result = _get_server().evaluate_binary(question=full_question, answer=gt_answer, prediction=model_answer, output_format="0/1")

                # Parse the result
                if result["success"]:
                    judge_response = result["result"]
                    correct_list.append(judge_response)
                else:
                    eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                    correct_list.append(False)

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct_list.append(False)

    # Calculate the average score for this document
    avg_score = sum(1 if score else 0 for score in correct_list) / len(correct_list) if correct_list else 0
    return {"llm_as_judge_eval": avg_score}


def mathvision_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = extract_final_answer(pred)
        model_answer = truncate_response_tail_tiktoken(model_answer)

        gt_answer = str(doc["answer"])
        if len(doc["options"]) > 0:
            gt_answer_value = doc["options"][ord(gt_answer) - ord("A")]
        else:
            gt_answer_value = ""

        if len(doc["options"]) > 0 and math_verify_parse and StringExtractionConfig:
            all_choices = [chr(ord("A") + i) for i in range(len(doc["options"]))]
            try:
                parsed = math_verify_parse(model_answer, extraction_config=[StringExtractionConfig(strings=tuple(all_choices))])
            except Exception:
                parsed = None
            if parsed:
                parsed_choice = str(parsed[0]).strip()
                if parsed_choice:
                    parsed_choice = parsed_choice.upper()
                    if parsed_choice in all_choices:
                        model_answer = parsed_choice

        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "oxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    return {
        "mathvision_standard_eval": {
            # "question": doc["question"],
            # "answer": doc["answer"],
            "response": results,
            # "subject": doc["subject"],
            # "level": doc["level"],
            "scores": correct_list,
            "resp_key": str(doc.get("id") or doc.get("question_id") or doc.get("image_id") or doc.get("question") or ""),
        },
    }


def mathvision_aggregate_results_eval(results):
    individual_scores = {}
    per_sample_scores = []

    for idx, result in enumerate(results):
        scores = result.get("scores", [])
        score = sum(1 if s else 0 for s in scores) / len(scores) if scores else 0
        per_sample_scores.append(score)

        resp_key = str(result.get("resp_key") or idx)
        individual_scores[resp_key] = {
            "score": score,
            "response": result.get("response"),
        }

    total = len(per_sample_scores) if per_sample_scores else 1
    accuracy = round(sum(per_sample_scores) / total, 4)
    mathvision_aggregate_results_eval.individual_scores = individual_scores
    return accuracy
