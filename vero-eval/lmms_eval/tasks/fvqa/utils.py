import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image as PILImage

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt_text_only.txt"


def _extract_question(doc: Dict[str, Any]) -> str:
    prompt = doc.get("prompt")
    question = ""
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            question = str(first.get("content", "") or "")
        else:
            question = str(first or "")
    question = question.replace("<image>", "").strip()
    return question


def _extract_ground_truth(doc: Dict[str, Any]) -> str:
    reward_model = doc.get("reward_model")
    if not isinstance(reward_model, dict):
        return ""
    return str(reward_model.get("ground_truth", "") or "").strip()


def _extract_candidate_answers(doc: Dict[str, Any]) -> List[Any]:
    reward_model = doc.get("reward_model")
    if not isinstance(reward_model, dict):
        return []
    candidates = reward_model.get("candidate_answers", [])
    if candidates is None:
        return []
    if isinstance(candidates, (list, tuple)):
        return list(candidates)
    return [candidates]


def _extract_resp_key(doc: Dict[str, Any], question: str) -> str:
    data_id = doc.get("data_id")
    if data_id is None:
        return question
    return str(data_id)


def fvqa_doc_to_visual(doc: Dict[str, Any]) -> List[PILImage.Image]:
    images = doc.get("images")
    if not isinstance(images, list) or not images:
        return []

    image_item = images[0]
    try:
        if isinstance(image_item, PILImage.Image):
            return [image_item.convert("RGB")]

        image_bytes: Optional[bytes] = None
        if isinstance(image_item, dict):
            raw_bytes = image_item.get("bytes")
            if isinstance(raw_bytes, bytes):
                image_bytes = raw_bytes
            elif isinstance(raw_bytes, bytearray):
                image_bytes = bytes(raw_bytes)
            elif raw_bytes is not None:
                image_bytes = bytes(raw_bytes)
        elif isinstance(image_item, bytes):
            image_bytes = image_item
        elif isinstance(image_item, bytearray):
            image_bytes = bytes(image_item)

        if not image_bytes:
            return []

        with PILImage.open(io.BytesIO(image_bytes)) as image:
            return [image.convert("RGB")]
    except Exception:
        return []


def fvqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]) -> str:
    question = _extract_question(doc)
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def fvqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return _extract_ground_truth(doc)


def fvqa_process_results(doc: Dict[str, Any], results: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    prediction = extract_final_answer(raw_prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)

    question = _extract_question(doc)
    reference = _extract_ground_truth(doc)
    candidate_answers = _extract_candidate_answers(doc)
    resp_key = _extract_resp_key(doc, question)

    payload = {
        "question": question,
        "reference": reference,
        "candidate_answers": candidate_answers,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
        "source": doc.get("source"),
    }
    return {"judge_accuracy": payload}


def fvqa_process_results_json(doc: Dict[str, Any], results: Iterable[str]) -> Dict[str, Dict[str, Any]]:
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
    prediction = extract_final_answer(json_answer, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)

    question = _extract_question(doc)
    reference = _extract_ground_truth(doc)
    candidate_answers = _extract_candidate_answers(doc)
    resp_key = _extract_resp_key(doc, question)

    payload = {
        "question": question,
        "reference": reference,
        "candidate_answers": candidate_answers,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
        "source": doc.get("source"),
    }
    return {"judge_accuracy": payload}


@lru_cache(maxsize=1)
def _load_judge_prompt() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _serialize_candidate_answers(candidate_answers: List[Any]) -> str:
    try:
        return json.dumps(candidate_answers, ensure_ascii=False)
    except Exception:
        return str(candidate_answers)


def _format_judge_prompt(question: str, reference: str, candidate_answers: List[Any], prediction: str) -> str:
    template = _load_judge_prompt()
    prompt = template.replace("{question}", question or "")
    prompt = prompt.replace("{ground truth answer}", reference or "")
    prompt = prompt.replace("{candidate answers}", _serialize_candidate_answers(candidate_answers))
    prompt = prompt.replace("{model response}", prediction or "")
    return prompt


def _parse_judge_output(raw: Any) -> bool:
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    raw_text = raw.strip()

    parsed = extract_json_candidate(raw_text)
    if isinstance(parsed, dict):
        for key in ("is_equivalent", "equivalent", "result", "answer", "decision"):
            if key not in parsed:
                continue
            value = parsed[key]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered.startswith(("y", "t")):
                    return True
                if lowered.startswith(("n", "f")):
                    return False
        raw_text = str(parsed)

    lowered = raw_text.lower()
    if lowered in {"yes", '"yes"', "'yes'", "true"}:
        return True
    if lowered in {"no", '"no"', "'no'", "false"}:
        return False
    if lowered.startswith(("yes", "true")):
        return True
    if lowered.startswith(("no", "false")):
        return False
    if "yes" in lowered and "no" not in lowered:
        return True
    if "no" in lowered and "yes" not in lowered:
        return False
    return False


def fvqa_aggregate_judge_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        fvqa_aggregate_judge_results.details = []
        fvqa_aggregate_judge_results.individual_scores = {}
        return 0.0

    prompts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for item in results:
        prompts.append(
            _format_judge_prompt(
                item.get("question", ""),
                item.get("reference", ""),
                item.get("candidate_answers", []),
                item.get("prediction", ""),
            )
        )
        metadata.append(item)

    engine = get_judge_engine(MODEL_HINT)
    judge_outputs = engine.generate_json_batch(prompts, use_tqdm=True)

    correct = 0
    details: Dict[str, Dict[str, Any]] = {}
    for idx, (meta, raw) in enumerate(zip(metadata, judge_outputs)):
        is_equivalent = _parse_judge_output(raw)
        if is_equivalent:
            correct += 1

        resp_key = str(meta.get("resp_key") or f"sample_{idx}")
        details[resp_key] = {
            "question": meta.get("question"),
            "reference": meta.get("reference"),
            "candidate_answers": meta.get("candidate_answers", []),
            "prediction": meta.get("prediction"),
            "raw_prediction": meta.get("raw_prediction"),
            "source": meta.get("source"),
            "judge_output": raw,
            "extracted_answer": meta.get("prediction", ""),
            "score": 1.0 if is_equivalent else 0.0,
        }

    fvqa_aggregate_judge_results.details = list(details.values())
    fvqa_aggregate_judge_results.individual_scores = details
    total = len(judge_outputs)
    return correct / total if total else 0.0
