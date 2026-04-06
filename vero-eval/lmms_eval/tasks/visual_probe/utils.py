import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"
_PREFIXES = ("VisualProbe_Easy/", "VisualProbe_Medium/", "VisualProbe_Hard/", "VisualProbe_train/")
_REPO_HINTS = {
    "easy": "Mini-o3/VisualProbe_Easy",
    "medium": "Mini-o3/VisualProbe_Medium",
    "hard": "Mini-o3/VisualProbe_Hard",
}


def _normalize_image_path(raw_path: str) -> str:
    path = raw_path.replace("\\", "/").lstrip("./")
    for prefix in _PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _infer_repo_id(doc: Dict[str, Any], path: Optional[str]) -> Optional[str]:
    text = f"{doc.get('data_source', '')} {path or ''}".lower()
    if "visual_probe_easy" in text or "visualprobe_easy" in text:
        return _REPO_HINTS["easy"]
    if "visual_probe_medium" in text or "visualprobe_medium" in text:
        return _REPO_HINTS["medium"]
    if "visual_probe_hard" in text or "visualprobe_hard" in text:
        return _REPO_HINTS["hard"]
    for key, repo in _REPO_HINTS.items():
        if key in text:
            return repo
    return None


def _open_image(path: Path) -> Optional[PILImage.Image]:
    if not path.exists():
        return None
    try:
        with PILImage.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _download_image(repo_id: str, filename: str) -> Optional[PILImage.Image]:
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    except Exception:
        return None
    return _open_image(Path(local_path))


def _coerce_image(img: Any, doc: Dict[str, Any]) -> Optional[PILImage.Image]:
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        normalized = _normalize_image_path(img)
        repo_id = _infer_repo_id(doc, img)
        for candidate in (Path(img), Path(normalized)):
            loaded = _open_image(candidate)
            if loaded is not None:
                return loaded
        if repo_id and normalized:
            return _download_image(repo_id, normalized)
    return None


def visual_probe_doc_to_visual(doc: Dict[str, Any]) -> List[PILImage.Image]:
    visuals: List[PILImage.Image] = []
    for img in doc.get("images", []):
        coerced = _coerce_image(img, doc)
        if coerced is not None:
            visuals.append(coerced)
    return visuals


def _strip_image_token(text: str) -> str:
    return text.replace("<image>", "").strip()


def visual_probe_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = (
        doc.get("problem")
        or doc.get("extra_info", {}).get("question")
        or doc.get("prompt")
        or doc.get("instruction")
        or ""
    )
    cleaned_question = _strip_image_token(str(question))
    return f"{pre}{cleaned_question}{post}"


def visual_probe_doc_to_target(doc: Dict[str, Any]) -> str:
    if "solution" in doc:
        return str(doc["solution"]).strip()
    if "answer" in doc:
        return str(doc["answer"]).strip()
    extra = doc.get("extra_info", {})
    if "answer" in extra:
        return str(extra["answer"]).strip()
    reward = doc.get("reward_model", {})
    if isinstance(reward, dict) and "ground_truth" in reward:
        return str(reward["ground_truth"]).strip()
    return ""


def _extract_final_answer(raw_response: Any) -> str:
    return extract_final_answer(raw_response)


class AnswerTagTextFilter(RegexFilter):
    """Extract answer content from <answer> or after </think>."""

    def __init__(
        self,
        regex_pattern: str = r"(?s)<answer>(.*?)</answer>|(?s)</think>\\s*(.*)$|(?s)(.*)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


def visual_probe_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    prediction = _extract_final_answer(raw_prediction)
    prediction = truncate_response_tail_tiktoken(prediction)
    question = (
        doc.get("problem")
        or doc.get("extra_info", {}).get("question")
        or doc.get("prompt")
        or doc.get("instruction")
        or ""
    )
    question = _strip_image_token(str(question))
    reference = str(doc.get("solution") or doc.get("answer") or "").strip()
    resp_key = str(doc.get("doc_id") or doc.get("id") or question)
    payload = {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
        "data_source": doc.get("data_source"),
    }
    return {"judge_accuracy": payload}


def visual_probe_process_results_json(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
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
    prediction = _extract_final_answer(json_answer)
    prediction = truncate_response_tail_tiktoken(prediction)
    question = (
        doc.get("problem")
        or doc.get("extra_info", {}).get("question")
        or doc.get("prompt")
        or doc.get("instruction")
        or ""
    )
    question = _strip_image_token(str(question))
    reference = str(doc.get("solution") or doc.get("answer") or "").strip()
    resp_key = str(doc.get("doc_id") or doc.get("id") or question)
    payload = {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
        "data_source": doc.get("data_source"),
    }
    return {"judge_accuracy": payload}


@lru_cache(maxsize=1)
def _load_judge_prompt() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _format_judge_prompt(question: str, reference: str, prediction: str) -> str:
    template = _load_judge_prompt()
    prompt = template.replace("[QUESTION]", question or "")
    prompt = prompt.replace("[REFERENCE]", reference or "")
    prompt = prompt.replace("[PREDICTION]", prediction or "")
    return prompt


def _parse_judge_output(raw: Any) -> bool:
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    raw_text = raw.strip()
    parsed = extract_json_candidate(raw_text)
    if isinstance(parsed, dict):
        for key in ("is_equivalent", "equivalent", "match", "result", "answer", "decision"):
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


def visual_probe_aggregate_judge_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        visual_probe_aggregate_judge_results.details = []
        visual_probe_aggregate_judge_results.individual_scores = {}
        return 0.0

    prompts = []
    metadata = []
    for item in results:
        prompts.append(_format_judge_prompt(item.get("question"), item.get("reference"), item.get("prediction")))
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
            "prediction": meta.get("prediction"),
            "raw_prediction": meta.get("raw_prediction"),
            "data_source": meta.get("data_source"),
            "judge_output": raw,
            "extracted_answer": meta.get("prediction", ""),
            "score": 1.0 if is_equivalent else 0.0,
        }

    visual_probe_aggregate_judge_results.details = list(details.values())
    visual_probe_aggregate_judge_results.individual_scores = details
    total = len(judge_outputs)
    return correct / total if total else 0.0
