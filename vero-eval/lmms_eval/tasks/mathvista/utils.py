import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathvista.mathvista_evals import (
    MathVistaEvaluator,
    normalize_extracted_answer_with_math_verify,
    safe_equal_with_math_verify,
)

with open(Path(__file__).parent / "mathvista.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


@lru_cache(maxsize=1)
def get_mathvista_evaluator() -> MathVistaEvaluator:
    # Lazily load the judge so we only spin up the local vLLM when needed.
    return MathVistaEvaluator()


def mathvista_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    evaluator = get_mathvista_evaluator()
    query_prompt = evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    return query_prompt


def mathvista_process_results(doc, results):
    prediction = extract_final_answer(results[0].strip())
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    evaluator = get_mathvista_evaluator()
    # extraction = evaluator.extract_answer(prediction, problem, config["metadata"]["quick_extract"])

    # prediction = evaluator.normalize_extracted_answer(extraction, problem["choices"], problem["question_type"], problem["answer_type"], problem["precision"])
    prediction = normalize_extracted_answer_with_math_verify(
        prediction,
        problem["choices"],
        problem["question_type"],
        problem["answer_type"],
        problem["precision"],
    )
    # set test set answer to None
    # true_false = evaluator.safe_equal(prediction, problem["answer"]) if problem["answer"] is not None else False
    true_false = (
        safe_equal_with_math_verify(
            prediction,
            problem["answer"],
            answer_type=problem["answer_type"],
            precision=problem["precision"],
        )
        if problem["answer"] is not None
        else False
    )

    result = {
        "question_id": doc["pid"],
        "resp_key": str(doc.get("pid") or doc.get("question_id") or doc.get("question") or ""),
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        # "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "precision": doc["precision"] if "precision" in doc else 0,
        "metadata": doc["metadata"],
    }

    return {
        "llm_as_judge_eval": result,
        "submission": result,
    }


def mathvista_process_results_judge(doc, results):
    """Like mathvista_process_results but uses LLM judge for answer extraction."""
    prediction = extract_final_answer(results[0].strip())
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    evaluator = get_mathvista_evaluator()
    extraction = evaluator.extract_answer(prediction, problem, quick_extract=False)
    prediction = evaluator.normalize_extracted_answer(extraction, problem["choices"], problem["question_type"], problem["answer_type"], problem["precision"])
    true_false = evaluator.safe_equal(prediction, problem["answer"]) if problem["answer"] is not None else False

    result = {
        "question_id": doc["pid"],
        "resp_key": str(doc.get("pid") or doc.get("question_id") or doc.get("question") or ""),
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "precision": doc["precision"] if "precision" in doc else 0,
        "metadata": doc["metadata"],
    }

    return {
        "llm_as_judge_eval": result,
        "submission": result,
    }


def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total, 4)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    individual_scores = {}
    for result in results:
        resp_key = str(result.get("resp_key") or result.get("question_id") or "")
        if resp_key:
            individual_scores[resp_key] = {
                "score": 1.0 if result.get("true_false") else 0.0,
                "prediction": result.get("prediction"),
                "answer": result.get("answer"),
            }

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    target_keys = ["question_type", "answer_type", "language", "source", "category", "task", "context", "grade", "skills"]

    for key in target_keys:
        values = df[key].explode().unique() if key == "skills" else df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = get_mathvista_evaluator().get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                    scores[key][sub_key]["acc_gain"] = gain

    mathvista_aggregate_results.individual_scores = individual_scores
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
