"""Multiple-choice answer extraction helpers with optional judge fallback."""
from __future__ import annotations

import os
import random
import re
import string
import time
from typing import Any, Dict, Iterable, Optional

from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")

_EXTRACT_PROMPT_TEMPLATE = (
    "You are an AI assistant who will help me to match "
    "an answer with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, "
    "and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
    "Example 1: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: a cute teddy bear\nYour output: A\n"
    "Example 2: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: Spider\nYour output: Z\n"
    "Example 3: \n"
    "Question: {question}\nOptions: {options}\nAnswer: {prediction}\nYour output: "
)


def should_use_judge_extractor(lmms_eval_specific_kwargs: Optional[Dict[str, Any]]) -> bool:
    if not lmms_eval_specific_kwargs:
        return False
    return bool(lmms_eval_specific_kwargs.get("use_judge_extractor"))


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _count_choice(splits: Iterable[str], choices: Iterable[str], prefix: str = "", suffix: str = "") -> int:
    count = 0
    for choice in choices:
        if f"{prefix}{choice}{suffix}" in splits:
            count += 1
    return count


def can_infer_option(answer: str, choices: Iterable[str]) -> Optional[str]:
    if not isinstance(answer, str):
        return None

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    answer_mod = answer
    chars = ".()[],:;!*#{}"
    for char in chars:
        answer_mod = answer_mod.replace(char, " ")

    splits = [part.strip() for part in answer_mod.split() if part.strip()]
    count = _count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3:
                return None
            if ch in splits:
                return ch
    elif count == 0 and _count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return None


def can_infer_text(answer: str, choices: Dict[str, str]) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    answer_lower = answer.lower()
    choice_map = {k: str(v).lower() for k, v in choices.items()}
    candidates = [key for key, value in choice_map.items() if value and value in answer_lower]
    if len(candidates) == 1:
        return candidates[0]
    return None


def can_infer(answer: str, choices: Dict[str, str]) -> Optional[str]:
    option = can_infer_option(str(answer), choices.keys())
    return option if option else can_infer_text(str(answer), choices)


def build_option_str(choice_map: Dict[str, str]) -> str:
    lines = ["There are several options:"]
    for key, content in choice_map.items():
        if content is None:
            continue
        lines.append(f"{key}. {content}")
    return "\n".join(lines)


def build_prompt(question: str, choice_map: Dict[str, str], prediction: str) -> str:
    question_text = str(question).strip()
    if question_text and not question_text.endswith("?"):
        question_text = f"{question_text}?"
    escaped_question = _escape_braces(question_text)
    escaped_prediction = _escape_braces(str(prediction))
    options = _escape_braces(build_option_str(choice_map))
    return _EXTRACT_PROMPT_TEMPLATE.format(
        question=escaped_question,
        options=options,
        prediction=escaped_prediction,
    )


def build_choice_map_from_doc(
    doc: Dict[str, Any],
    *,
    choice_fields: Iterable[str],
    question_field: str,
) -> Dict[str, str]:
    for field in choice_fields:
        if field not in doc:
            continue
        choices = doc[field]
        if isinstance(choices, dict):
            mapped = {str(k): str(v) for k, v in choices.items() if v is not None}
            if mapped:
                return mapped
        if isinstance(choices, (list, tuple)):
            mapped = {}
            for idx, option in enumerate(choices):
                if option is None:
                    continue
                key = string.ascii_uppercase[idx]
                mapped[key] = str(option)
            if mapped:
                return mapped
    question = doc.get(question_field, "")
    return parse_choices_from_text(question)


def parse_choices_from_text(question: str) -> Dict[str, str]:
    if not isinstance(question, str) or not question:
        return {}
    matches = re.findall(r"\b([A-Z])\.\s+([^\n]+)", question)
    if not matches:
        return {}
    return {letter: text.strip() for letter, text in matches}


def extract_choice_with_judge(
    *,
    question: str,
    choice_map: Dict[str, str],
    prediction: str,
    model_hint: Optional[str] = None,
    max_retries: int = 5,
    wait: float = 5.0,
) -> Optional[str]:
    if not choice_map:
        return None

    rule_based = can_infer(prediction, choice_map)
    if rule_based and rule_based != "Z":
        return rule_based

    try:
        engine = get_judge_engine(model_hint or MODEL_HINT)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialize judge engine: {exc}")
        return None

    prompt = build_prompt(question, choice_map, prediction)
    last_raw = ""

    for _ in range(max_retries):
        try:
            raw = engine.generate_json(prompt, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            print(f"Judge extraction failed: {exc}")
            raw = ""
        last_raw = raw or last_raw

        parsed = extract_json_candidate(raw or "")
        if isinstance(parsed, dict):
            for key in ("option", "answer", "extracted_answer", "prediction", "result"):
                if key in parsed:
                    candidate = parsed[key]
                    inferred = can_infer(str(candidate), choice_map)
                    if inferred and inferred != "Z":
                        return inferred
        inferred = can_infer(str(raw), choice_map)
        if inferred and inferred != "Z":
            return inferred
        time.sleep(random.random() * wait * 2)

    return None
