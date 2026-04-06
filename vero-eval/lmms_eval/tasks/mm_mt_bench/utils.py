import base64
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, Features, Value
from loguru import logger as eval_logger
from PIL import Image as PILImage
import requests
import yaml

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import (
    get_judge_engine,
    judge_supports_multimodal,
    resolve_judge_mode,
)


SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
    "to the most recent question given the previous conversation as context. Your evaluation should consider "
    "correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your "
    "evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. "
    "Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 "
    "by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"
)
BRACKET_SCORE_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")


# ============================
# Dataset Processing Functions
# ============================

with open(Path(__file__).parent / "mm_mt_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


NUM_SECONDS_TO_SLEEP = 10
API_TYPE = os.getenv("API_TYPE", "openai")
API_TYPE_LOWER = API_TYPE.lower()
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-05-13")
JUDGE_MODEL_HINT = os.getenv("JUDGE_MODEL_PATH") or GPT_EVAL_MODEL_NAME
JUDGE_MODE = resolve_judge_mode(JUDGE_MODEL_HINT)
_JUDGE_MODE_OVERRIDDEN = os.getenv("LMMS_EVAL_JUDGE_MODE", "").strip().lower() in {"vlm", "llm"}
_JUDGE_TEXT_ONLY_FLAG = os.getenv("LMMS_EVAL_JUDGE_TEXT_ONLY", "").strip().lower() in {"1", "true", "yes"}
USE_LOCAL_JUDGE = API_TYPE_LOWER not in ("openai", "azure")
JUDGE_TEXT_ONLY = (JUDGE_MODE == "llm") if _JUDGE_MODE_OVERRIDDEN else (_JUDGE_TEXT_ONLY_FLAG or JUDGE_MODE == "llm")

if API_TYPE_LOWER == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE_LOWER == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {"api-key": API_KEY, "Content-Type": "application/json", "api-version": "2023-07-01-preview"}
else:
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

eval_logger.info(
    f"MM-MT judge routing: mode={JUDGE_MODE}, text_only={JUDGE_TEXT_ONLY}, use_local_judge={USE_LOCAL_JUDGE}"
)


_local_engine = None


def _get_local_engine():
    global _local_engine
    if _local_engine is None:
        _local_engine = get_judge_engine(JUDGE_MODEL_HINT)
    return _local_engine


def _prepend_system_prompt(prompt: str) -> str:
    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        return prompt
    if not prompt:
        return system_prompt
    return f"{system_prompt}\n{prompt}"


def _prompt_chunks_to_text(prompt: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for chunk in prompt or []:
        if not isinstance(chunk, dict):
            parts.append(str(chunk))
            continue
        chunk_type = chunk.get("type")
        if chunk_type == "text":
            parts.append(str(chunk.get("text", "")))
            continue
        if chunk_type in ("image", "image_url"):
            continue
        if "text" in chunk:
            parts.append(str(chunk.get("text", "")))
            continue
    return "".join(parts)


def _strip_image_chunks(prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped: List[Dict[str, Any]] = []
    for chunk in prompt or []:
        if isinstance(chunk, dict) and chunk.get("type") in ("image", "image_url"):
            continue
        stripped.append(chunk)
    return stripped


def _parse_conversation(raw_conversation: Any) -> List[Dict[str, Any]]:
    if raw_conversation is None:
        return []
    if isinstance(raw_conversation, list):
        return raw_conversation
    if isinstance(raw_conversation, str):
        try:
            parsed = json.loads(raw_conversation)
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            eval_logger.error(f"Failed to parse conversation JSON: {e}")
            return []
    return []


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text", ""))
    return str(content)


def _safe_get_image(doc: Dict[str, Any]):
    img = doc.get("image")
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if hasattr(img, "convert"):
        try:
            return img.convert("RGB")
        except Exception:
            return img
    if isinstance(img, str):
        try:
            return PILImage.open(img).convert("RGB")
        except Exception:
            return None
    return img


def mm_mt_bench_process_docs(dataset: Dataset) -> Dataset:
    conversations: List[str] = []
    images: List[Any] = []
    categories: List[Any] = []
    turns: List[int] = []
    message_indices: List[int] = []
    reference_answers: List[str] = []

    for example_idx, example in enumerate(dataset):
        raw_conversation = example.get("conversation")
        messages = _parse_conversation(raw_conversation)
        if not messages:
            continue
        conversation_json = json.dumps(messages, ensure_ascii=False)
        for msg_index in range(0, len(messages), 2):
            if msg_index + 1 >= len(messages):
                continue
            user_msg = messages[msg_index]
            assistant_msg = messages[msg_index + 1]
            if user_msg.get("role") != "user":
                continue
            ref_answer = _content_to_text(assistant_msg.get("content"))
            conversations.append(conversation_json)
            images.append(example.get("image"))
            categories.append(example.get("category"))
            turns.append(msg_index // 2)
            message_indices.append(msg_index)
            reference_answers.append(str(ref_answer).strip())

    if not conversations:
        return dataset

    features = Features(
        {
            "conversation": Value("string"),
            "image": dataset.features.get("image", Value("string")),
            "category": dataset.features.get("category", Value("string")),
            "turn": Value("int32"),
            "message_index": Value("int32"),
            "reference_answer": Value("string"),
        }
    )

    flat_dataset = Dataset.from_dict(
        {
            "conversation": conversations,
            "image": images,
            "category": categories,
            "turn": turns,
            "message_index": message_indices,
            "reference_answer": reference_answers,
        },
        features=features,
    )
    return flat_dataset


# ============================
# Input Formatting Functions
# ============================


def mm_mt_bench_doc_to_visual(doc: Dict[str, Any]):
    img = _safe_get_image(doc)
    return [img] if img is not None else []


def mm_mt_bench_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if lmms_eval_specific_kwargs.get("pre_prompt"):
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs.get("post_prompt"):
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    messages = _parse_conversation(doc.get("conversation"))
    message_index = doc.get("message_index")
    if isinstance(message_index, int) and messages:
        messages = messages[: message_index + 1]
    last_user = ""
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            last_user = _content_to_text(msg.get("content"))
            break
    return f"{pre_prompt}{last_user}{post_prompt}".strip()


def _normalize_messages_for_chat(messages: List[Dict[str, Any]], image: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        normalized_content: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in ("image", "image_url"):
                        normalized_content.append({"type": "image", "url": image})
                    elif item_type == "text":
                        text = item.get("text", "")
                        normalized_content.append({"type": "text", "text": str(text)})
                    elif "text" in item:
                        normalized_content.append({"type": "text", "text": str(item.get("text", ""))})
                elif isinstance(item, str):
                    normalized_content.append({"type": "text", "text": item})
        elif isinstance(content, str):
            normalized_content.append({"type": "text", "text": content})
        else:
            if content is not None:
                normalized_content.append({"type": "text", "text": str(content)})

        if not normalized_content:
            normalized_content.append({"type": "text", "text": ""})

        normalized.append({"role": role, "content": normalized_content})

    return normalized


def mm_mt_bench_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    messages = _parse_conversation(doc.get("conversation"))
    message_index = doc.get("message_index")
    if isinstance(message_index, int) and messages:
        messages = messages[: message_index + 1]
    image = _safe_get_image(doc)
    return _normalize_messages_for_chat(messages, image)


# ============================
# Judge Prompt Construction
# ============================


def _encode_image_base64(image: Any) -> Optional[str]:
    if image is None:
        return None
    if isinstance(image, str):
        if image.startswith("data:") or image.startswith("http"):
            return image
        try:
            image = PILImage.open(image).convert("RGB")
        except Exception:
            payload = image.strip()
            if payload:
                payload = "".join(payload.split())
                payload += "=" * (-len(payload) % 4)
                try:
                    base64.b64decode(payload, validate=False)
                    return "data:image/jpeg;base64," + payload
                except Exception:
                    return None
            return None
    elif isinstance(image, PILImage.Image):
        image = image.convert("RGB")
    elif hasattr(image, "convert"):
        try:
            image = image.convert("RGB")
        except Exception:
            return None
    else:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _append_text_chunk(prompt: List[Dict[str, Any]], text: str) -> None:
    if not text:
        return
    if prompt and prompt[-1].get("type") == "text":
        prompt[-1]["text"] += text
    else:
        prompt.append({"type": "text", "text": text})


def _append_image_chunk(prompt: List[Dict[str, Any]], image: Any) -> None:
    image_url = _encode_image_base64(image)
    if not image_url:
        return
    prompt.append({"type": "image_url", "image_url": {"url": image_url}})


def _add_or_append_chunk(prompt: List[Dict[str, Any]], chunk: Any, fallback_image: Any) -> None:
    if isinstance(chunk, dict):
        chunk_type = chunk.get("type")
        if chunk_type in ("image", "image_url"):
            image = chunk.get("url")
            if image is None and isinstance(chunk.get("image_url"), dict):
                image = chunk["image_url"].get("url")
            _append_image_chunk(prompt, image or fallback_image)
            return
        if chunk_type == "text":
            _append_text_chunk(prompt, str(chunk.get("text", "")))
            return
        if "text" in chunk:
            _append_text_chunk(prompt, str(chunk.get("text", "")))
            return
    if isinstance(chunk, str):
        _append_text_chunk(prompt, chunk)
        return
    if chunk is None:
        return
    _append_text_chunk(prompt, str(chunk))


def _replay_conversation(
    prompt: List[Dict[str, Any]],
    questions: List[Any],
    ref_answers: List[str],
    final_answer: str,
    image: Any,
) -> None:
    for question, ref_answer in zip(questions, ref_answers):
        _append_text_chunk(prompt, "### User:\n")
        if isinstance(question, list):
            for item in question:
                _add_or_append_chunk(prompt, item, image)
        else:
            _add_or_append_chunk(prompt, question, image)
        _append_text_chunk(prompt, f"\n\n### Reference answer:\n{ref_answer}\n\n")
    _append_text_chunk(prompt, f"\n\n### Assistant's answer:\n{final_answer}\n\n")


def _build_judge_prompt(
    questions: List[Any],
    ref_answers: List[str],
    final_answer: str,
    image: Any,
) -> List[Dict[str, Any]]:
    prompt: List[Dict[str, Any]] = [
        {"type": "text", "text": "<|The Start of Conversation with User|>\n\n"}
    ]
    _replay_conversation(prompt, questions, ref_answers, final_answer, image)
    _append_text_chunk(prompt, "<|The End of Conversation with User|>\n\n\n")
    return prompt


# ============================
# Judge API
# ============================


def get_eval(prompt: List[Dict[str, Any]], max_tokens: int, retries: int = 5):
    global headers

    if USE_LOCAL_JUDGE:
        model_used = JUDGE_MODEL_HINT
        for attempt in range(retries):
            try:
                engine = _get_local_engine()
                if JUDGE_TEXT_ONLY:
                    prompt_text = _prepend_system_prompt(_prompt_chunks_to_text(prompt))
                    content = engine.generate_json(prompt_text, max_tokens=max_tokens)
                else:
                    if not judge_supports_multimodal(engine):
                        raise RuntimeError(
                            "MM-MT VLM judge requires a multimodal backend. "
                            "Use JUDGE_BACKEND=engine with a multimodal local model, or set VLLM_SERVER_JUDGE=1."
                        )
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    content = engine.generate_json_messages(messages, max_tokens=max_tokens)
                return content.strip() if isinstance(content, str) else str(content), model_used
            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < retries:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:
                    eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                    return "", model_used
        return "", model_used

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data.get("model", GPT_EVAL_MODEL_NAME)
            break

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def _extract_score(judgement: str) -> float:
    if not isinstance(judgement, str):
        return -1.0
    match = re.search(BRACKET_SCORE_RE, judgement)
    if match:
        try:
            return float(match.groups()[0])
        except Exception:
            return -1.0
    return -1.0


# ============================
# Result Processing Functions
# ============================


def mm_mt_bench_process_results(doc: Dict[str, Any], results: List[str]):
    try:
        if not results:
            return {
                "micro_average_score": {"score": -1.0, "category": doc.get("category"), "turn": doc.get("turn")},
                "macro_average_score": {"score": -1.0, "category": doc.get("category"), "turn": doc.get("turn")},
            }
        response = results[0].strip() if results[0] else ""
        response = extract_final_answer(response, parse_boxed=False, strip_latex_wrappers=True)
    except Exception as e:
        eval_logger.error(f"Error extracting response: {e}")
        response = ""
    response = truncate_response_tail_tiktoken(response)

    raw_messages = _parse_conversation(doc.get("conversation"))
    message_index = doc.get("message_index")
    if isinstance(message_index, int) and raw_messages:
        raw_messages = raw_messages[: message_index + 1]

    questions = [m.get("content") for m in raw_messages if m.get("role") == "user"]
    ref_answers = [_content_to_text(m.get("content")) for m in raw_messages if m.get("role") == "assistant"]
    ref_answers.append(str(doc.get("reference_answer", "")).strip())

    image = _safe_get_image(doc)
    judge_prompt = _build_judge_prompt(questions, ref_answers, response, image)
    if JUDGE_TEXT_ONLY:
        judge_prompt = _strip_image_chunks(judge_prompt)

    try:
        judgement, _ = get_eval(judge_prompt, 4096)
    except Exception as e:
        eval_logger.error(f"Error getting evaluation from LLM: {e}")
        judgement = ""

    score = _extract_score(judgement)
    payload = {
        "score": score,
        "category": doc.get("category"),
        "turn": doc.get("turn"),
        "judgement": judgement,
    }
    return {
        "micro_average_score": payload,
        "macro_average_score": payload,
    }


def mm_mt_bench_process_results_json(doc: Dict[str, Any], results: List[str]):
    """Parse JSON-formatted predictions and delegate to the standard evaluator."""
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
    prediction = "" if json_answer is None else str(json_answer)
    return mm_mt_bench_process_results(doc, [prediction])


# ============================
# Aggregation Functions
# ============================


def mm_mt_bench_aggregate_micro(results: List[Dict[str, Any]]) -> float:
    scores = [result.get("score", -1.0) for result in results if isinstance(result, dict)]
    scores = [score for score in scores if isinstance(score, (int, float)) and score >= 0]
    return sum(scores) / len(scores) if scores else 0.0


def mm_mt_bench_aggregate_macro(results: List[Dict[str, Any]]) -> float:
    """Macro average across category AND turn groups (legacy behavior)."""
    category_scores: Dict[str, List[float]] = defaultdict(list)
    for result in results:
        if not isinstance(result, dict):
            continue
        score = result.get("score", -1.0)
        category = result.get("category")
        turn = result.get("turn")
        if isinstance(score, (int, float)):
            if category is not None:
                category_scores[str(category)].append(score)
            if turn is not None:
                category_scores[f"turn_{turn}"].append(score)

    if not category_scores:
        return 0.0

    category_averages = [sum(vals) / len(vals) for vals in category_scores.values() if vals]
    return sum(category_averages) / len(category_averages) if category_averages else 0.0


def mm_mt_bench_aggregate_macro_category(results: List[Dict[str, Any]]) -> float:
    """Macro average across categories only (excludes turn groups)."""
    category_scores: Dict[str, List[float]] = defaultdict(list)
    for result in results:
        if not isinstance(result, dict):
            continue
        score = result.get("score", -1.0)
        category = result.get("category")
        if isinstance(score, (int, float)) and score >= 0 and category is not None:
            category_scores[str(category)].append(score)

    if not category_scores:
        return 0.0

    category_averages = [sum(vals) / len(vals) for vals in category_scores.values() if vals]
    return sum(category_averages) / len(category_averages) if category_averages else 0.0
