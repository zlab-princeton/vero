import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.vllm_judge import (
    extract_json_candidate,
    get_judge_engine,
)
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
REASONING_TYPES = ("visual", "visual/text", "synthesis", "text")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "compare_answer_prompt.txt"


def chartmuseum_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """Return the RGB image for a ChartMuseum sample."""

    image = doc.get("image")
    return [image.convert("RGB")]


def chartmuseum_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    """Compose the text prompt by surrounding the question with pre/post prompts."""

    question = str(doc.get("question", "")).strip()
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def chartmuseum_process_results(doc: Dict[str, Any], results: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Prepare judge inputs for a single sample."""

    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    prediction = extract_final_answer(raw_prediction)
    prediction = truncate_response_tail_tiktoken(prediction)
    resp_key = doc.get("id") or doc.get("question")
    if resp_key is None:
        resp_key = ""
    resp_key = str(resp_key)
    reference = str(doc.get("answer")).strip()
    question = str(doc.get("question")).strip()

    payload = {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "reasoning_type": doc.get("reasoning_type"),
        "source": doc.get("source"),
        "resp_key": resp_key,
    }
    return {"judge_accuracy": payload}


def chartmuseum_process_results_json(doc: Dict[str, Any], results: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Prepare judge inputs for JSON-formatted predictions."""

    import json

    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    parsed = None
    if isinstance(raw_prediction, dict):
        parsed = raw_prediction
    elif isinstance(raw_prediction, str):
        try:
            parsed = json.loads(raw_prediction)
        except Exception:
            parsed = None
    json_answer = parsed.get("answer") if isinstance(parsed, dict) else raw_prediction
    prediction = extract_final_answer(json_answer)
    prediction = truncate_response_tail_tiktoken(prediction)
    resp_key = doc.get("id") or doc.get("question")
    if resp_key is None:
        resp_key = ""
    resp_key = str(resp_key)
    reference = str(doc.get("answer")).strip()
    question = str(doc.get("question")).strip()

    payload = {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "reasoning_type": doc.get("reasoning_type"),
        "source": doc.get("source"),
        "resp_key": resp_key,
    }
    return {"judge_accuracy": payload}


@lru_cache(maxsize=1)
def _load_compare_prompt() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _format_compare_prompt(question: str, reference: str, prediction: str) -> str:
    template = _load_compare_prompt()
    prompt = template.replace("[QUESTION]", question)
    prompt = prompt.replace("[ANSWER1]", reference if reference is not None else "")
    prompt = prompt.replace("[ANSWER2]", prediction if prediction is not None else "")
    return prompt


def _parse_judge_output(raw: str) -> bool:
    """Interpret judge output as a boolean equivalence decision."""

    if not isinstance(raw, str):
        return False
    raw_text = raw.strip()
    parsed = extract_json_candidate(raw_text)
    if isinstance(parsed, dict):
        for key in ("equivalent", "is_equivalent", "match", "verdict", "result", "answer", "decision"):
            if key not in parsed:
                continue
            value = parsed[key]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower.startswith("y"):
                    return True
                if lower.startswith("n"):
                    return False
        # Fall back to checking a stringified dict
        raw_text = str(parsed)
    lowered = raw_text.lower()
    if lowered in {"yes", '"yes"', "'yes'"}:
        return True
    if lowered in {"no", '"no"', "'no'"}:
        return False
    if lowered.startswith("yes"):
        return True
    if lowered.startswith("no"):
        return False
    if "yes" in lowered and "no" not in lowered:
        return True
    if "no" in lowered and "yes" not in lowered:
        return False
    return False


def chartmuseum_aggregate_judge_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate judge-labeled equivalence decisions into an accuracy score."""

    if not results:
        chartmuseum_aggregate_judge_results.details = []
        chartmuseum_aggregate_judge_results.individual_scores = {}
        chartmuseum_aggregate_judge_results.reasoning_type_breakdown = {}
        return 0.0

    prompts = []
    metadata = []
    for item in results:
        question = item.get("question")
        reference = item.get("reference")
        prediction = item.get("prediction")
        prompts.append(_format_compare_prompt(question, reference, prediction))
        metadata.append(item)

    engine = get_judge_engine(MODEL_HINT)
    judge_outputs = engine.generate_json_batch(prompts, use_tqdm=True)

    total = len(judge_outputs)
    correct = 0
    details = {}
    reasoning_totals: Dict[str, Dict[str, int]] = {
        reasoning_type: {"correct": 0, "total": 0} for reasoning_type in REASONING_TYPES
    }
    reasoning_totals["unknown"] = {"correct": 0, "total": 0}
    for idx, (meta, raw) in enumerate(zip(metadata, judge_outputs)):
        is_equivalent = _parse_judge_output(raw)
        if is_equivalent:
            correct += 1
        reasoning_type = meta.get("reasoning_type") or "unknown"
        if reasoning_type not in reasoning_totals:
            reasoning_totals[reasoning_type] = {"correct": 0, "total": 0}
        reasoning_totals[reasoning_type]["total"] += 1
        if is_equivalent:
            reasoning_totals[reasoning_type]["correct"] += 1
        resp_key = meta.get("resp_key") or f"sample_{idx}"
        resp_key = str(resp_key)
        details[resp_key] = {
            "question": meta.get("question"),
            "reference": meta.get("reference"),
            "prediction": meta.get("prediction"),
            "raw_prediction": meta.get("raw_prediction"),
            "reasoning_type": reasoning_type,
            "judge_output": raw,
            "extracted_answer": meta.get("prediction", ""),
            "score": 1.0 if is_equivalent else 0.0,
        }

    breakdown: Dict[str, float] = {}
    for reasoning_type, counts in reasoning_totals.items():
        total_count = counts["total"]
        if total_count == 0:
            continue
        accuracy = counts["correct"] / total_count
        breakdown[reasoning_type] = accuracy
        print(f"chartmuseum_accuracy_{reasoning_type}: {accuracy:0.4f}")

    chartmuseum_aggregate_judge_results.details = list(details.values())
    chartmuseum_aggregate_judge_results.individual_scores = details
    chartmuseum_aggregate_judge_results.reasoning_type_breakdown = breakdown
    return correct / total if total else 0.0
