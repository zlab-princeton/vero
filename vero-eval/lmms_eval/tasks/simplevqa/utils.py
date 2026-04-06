import json
import logging
import os
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken

eval_logger = logging.getLogger("lmms-eval")

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "simplevqa_judge_prompt.txt"
_MODEL_KEYS = ["model_response"]
_LABEL_CORRECT = "\u6b63\u786e"
_LABEL_INCORRECT = "\u9519\u8bef"
_LABEL_NOT_ATTEMPTED = "\u672a\u5c1d\u8bd5"
_LABEL_MAPPER = {
    _LABEL_CORRECT: "is_correct",
    _LABEL_INCORRECT: "is_incorrect",
    _LABEL_NOT_ATTEMPTED: "is_not_attempted",
}
_CANDIDATE_PREFIX = "\u9884\u6d4b\u7b54\u6848"


def _strip_think_and_answer(text: str) -> str:
    return extract_final_answer(text)


def _build_resp_key(doc: Dict[str, Any]) -> str:
    # Prefer dataset-provided unique identifiers when available.
    for key in ("id", "question_id", "sample_id", "uid", "qid"):
        value = doc.get(key)
        if value not in (None, ""):
            return str(value)

    # Fallback to a deterministic hash of stable fields to avoid collisions
    # when `source` repeats across many samples.
    payload = {
        "question": str(doc.get("question", "")).strip(),
        "answer": str(doc.get("answer", "")).strip(),
        "source": str(doc.get("source", "")).strip(),
        "original_category": str(doc.get("original_category", "")).strip(),
        "vqa_category": str(doc.get("vqa_category", "")).strip(),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"simplevqa_{digest}"


@lru_cache(maxsize=1)
def _load_judge_prompt() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _format_candidates(predictions: Sequence[str]) -> str:
    if not predictions:
        return ""
    lines = [f"[{_CANDIDATE_PREFIX}{idx}]\uff1a{pred}" for idx, pred in enumerate(predictions)]
    return "\n" + "\n".join(lines)


def _format_judge_prompt(question: str, answer: str, candidates: str) -> str:
    template = _load_judge_prompt()
    prompt = template.replace("{question}", question or "")
    prompt = prompt.replace("{answer}", answer or "")
    prompt = prompt.replace("{candidates}", candidates or "")
    return prompt


def _clean_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    cleaned = (
        cleaned.replace("```json", "")
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )
    if cleaned and not cleaned.endswith("}"):
        cleaned = f"{cleaned}}}"
    parsed = extract_json_candidate(cleaned)
    if isinstance(parsed, dict):
        return parsed
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_judge_res(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(parsed, dict) or not parsed:
        return {_MODEL_KEYS[0]: {"conclusion": "\u7b54\u6848\u89e3\u6790\u5931\u8d25"}}
    new_res: Dict[str, Any] = {}
    for i, key in enumerate(parsed.keys()):
        if i >= len(_MODEL_KEYS):
            break
        new_res[_MODEL_KEYS[i]] = parsed[key]
    if not new_res:
        new_res[_MODEL_KEYS[0]] = {"conclusion": "\u7b54\u6848\u89e3\u6790\u5931\u8d25"}
    return new_res


def _normalize_judge_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("conclusion", "")
    text = str(value).replace("**", "")
    if text not in _LABEL_MAPPER:
        if len(text) > 3:
            return _LABEL_NOT_ATTEMPTED
    return text


def _simplevqa_eval(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    amount = len(data)
    result_metrics: Dict[str, Dict[str, int]] = {}
    model_keys = list(_MODEL_KEYS)

    for line in data:
        judge_res = dict(line.get("judge_res") or {})
        for key, value in judge_res.items():
            judge_res[key] = _normalize_judge_value(value)

            if key not in result_metrics:
                result_metrics[key] = {
                    "is_correct": 0,
                    "is_incorrect": 0,
                    "is_not_attempted": 0,
                }

            try:
                result_metrics[key][_LABEL_MAPPER[judge_res[key]]] += 1
            except Exception:
                eval_logger.warning("Error in mapper key: %s", judge_res.get(key))

            if key not in model_keys:
                model_keys.append(key)

    eval_logger.info("ALL data count: %d", amount)
    aggregate_metrics: Dict[str, Any] = {}
    for model_key in model_keys:
        metrics = result_metrics.get(
            model_key,
            {"is_correct": 0, "is_incorrect": 0, "is_not_attempted": 0},
        )
        aggregate_metrics = {
            "LVLM_name": model_key,
            "is_correct": round(metrics["is_correct"] / amount, 4) if amount else 0.0,
            "is_incorrect": round(metrics["is_incorrect"] / amount, 4) if amount else 0.0,
            "is_not_attempted": round(metrics["is_not_attempted"] / amount, 4) if amount else 0.0,
        }
        aggregate_metrics["is_given_attempted"] = round(
            aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"], 4
        )
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"] / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )
        aggregate_metrics["f1"] = (
            2
            * aggregate_metrics["accuracy_given_attempted"]
            * aggregate_metrics["is_correct"]
            / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
            if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
            else 0
        )
        eval_logger.info("%s: AGGREGATE METRICS", model_key)
        eval_logger.info("%s", aggregate_metrics)

    return aggregate_metrics


def simplevqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = f"{pre_prompt}{doc['question']}"
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt


def simplevqa_doc_to_visual(doc: Dict[str, Any]) -> List:
    return [doc["image"].convert("RGB")]


def simplevqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    key_name = "simplevqa_acc"
    raw_gt = str(doc.get("answer", "")).strip()
    raw_predictions = list(results) if isinstance(results, (list, tuple)) else [results]
    predictions = [truncate_response_tail_tiktoken(_strip_think_and_answer(pred)) for pred in raw_predictions]
    prediction = predictions[0] if predictions else ""
    resp_key = _build_resp_key(doc)

    submission = {
        "resp_key": str(resp_key),
        "question": str(doc.get("question", "")).strip(),
        "answer": raw_gt,
        "prediction": prediction,
        "model_response": prediction,
        "reference": raw_gt,
        "predictions": predictions,
        "raw_predictions": raw_predictions,
        "original_category": doc.get("original_category", ""),
        "source": doc.get("source", ""),
        "vqa_category": doc.get("vqa_category", ""),
    }
    return {key_name: submission}


def simplevqa_aggregate_results(results: List[Dict]) -> float:
    total_samples = len(results)
    if total_samples == 0:
        simplevqa_aggregate_results.details = []
        simplevqa_aggregate_results.individual_scores = {}
        return 0.0

    prompts = []
    metadata = []
    for item in results:
        prediction = item.get("prediction") or ""
        candidates = _format_candidates([prediction])
        prompts.append(
            _format_judge_prompt(
                str(item.get("question", "")),
                str(item.get("answer", "")),
                candidates,
            )
        )
        metadata.append(item)

    engine = get_judge_engine(MODEL_HINT)
    judge_outputs = engine.generate_json_batch(prompts, use_tqdm=True)

    eval_data: List[Dict[str, Any]] = []
    details = {}
    for idx, (meta, raw) in enumerate(zip(metadata, judge_outputs)):
        parsed = _clean_judge_response(raw)
        judge_res = _build_judge_res(parsed)
        prediction = meta.get("prediction") or ""
        label = _normalize_judge_value(judge_res.get(_MODEL_KEYS[0], ""))
        score = 1.0 if label == _LABEL_CORRECT else 0.0
        entry = {
            "question": meta.get("question"),
            "answer": meta.get("answer"),
            "model_response": prediction,
            "judge_res": judge_res,
        }
        eval_data.append(entry)

        resp_key = meta.get("resp_key") or f"sample_{idx}"
        details[str(resp_key)] = {
            "question": meta.get("question"),
            "answer": meta.get("answer"),
            "prediction": prediction,
            "extracted_answer": prediction,
            "raw_predictions": meta.get("raw_predictions"),
            "judge_output": raw,
            "judge_res": judge_res,
            "score": score,
            "original_category": meta.get("original_category", ""),
            "source": meta.get("source", ""),
            "vqa_category": meta.get("vqa_category", ""),
        }

    aggregate_metrics = _simplevqa_eval(eval_data)
    simplevqa_aggregate_results.details = list(details.values())
    simplevqa_aggregate_results.individual_scores = details
    return aggregate_metrics.get("is_correct", 0.0)
