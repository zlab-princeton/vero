import ast
import re
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

from math_verify import parse
from PIL import Image

from lmms_eval.api.metrics import levenshtein_distance
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response

PROMPTS_DIR = Path(__file__).resolve().parent / "chartqa_pro_prompts"

def _load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing ChartQA-Pro prompt file at {prompt_path}") from exc


FACTOID_PROMPT = _load_prompt("Factoid.txt")
MULTI_CHOICE_PROMPT = _load_prompt("Multi_Choice.txt")
HYPOTHETICAL_PROMPT = _load_prompt("Hypothetical.txt")
FACT_CHECKING_PROMPT = _load_prompt("Fact_Checking.txt")
CONVERSATIONAL_PROMPT = _load_prompt("Conversational.txt")


PROMPTS_DIR_ALT = Path(__file__).resolve().parent / "chartqa_pro_prompts_alt"

def _load_prompt_alt(filename: str) -> str:
    prompt_path = PROMPTS_DIR_ALT / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing ChartQA-Pro prompt file at {prompt_path}") from exc


FACTOID_PROMPT_ALT = _load_prompt_alt("Factoid.txt")
MULTI_CHOICE_PROMPT_ALT = _load_prompt_alt("Multi_Choice.txt")
HYPOTHETICAL_PROMPT_ALT = _load_prompt_alt("Hypothetical.txt")
FACT_CHECKING_PROMPT_ALT = _load_prompt_alt("Fact_Checking.txt")
CONVERSATIONAL_PROMPT_ALT = _load_prompt_alt("Conversational.txt")

_RATIO_RE = re.compile(r"^\s*([^:]+?)\s*:\s*([^:]+?)\s*$")

def chartqa_pro_doc_to_visual(doc):
    image = doc["image"]

    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    return [image.convert("RGB")]


def chartqa_pro_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["Question"]
    question_type = doc["Question Type"]
    if question_type == "Factoid":
        if isinstance(question, list):
            question = question[0]
        question = FACTOID_PROMPT.replace("<question>", question)
    elif question_type == "Multi Choice":
        if isinstance(question, list):
            question = question[0]
        question = MULTI_CHOICE_PROMPT.replace("<question>", question)
    elif question_type == "Hypothetical":
        if isinstance(question, list):
            question = question[0]
        question = HYPOTHETICAL_PROMPT.replace("<question>", question)
    elif question_type == "Fact Checking":
        if isinstance(question, list):
            question = question[0]
        question = FACT_CHECKING_PROMPT.replace("<question>", question)
    elif question_type == "Conversational":
        if isinstance(question, list):
            final_question = question[-1]
            history_questions = question[:-1]
        else:
            final_question = question
            history_questions = []
        answers = doc["Answer"]
        if isinstance(answers, list):
            history_answers = answers[:-1]
        else:
            history_answers = []
        conversation = ""
        for q, a, in zip(history_questions, history_answers):
            conversation += f"\n\nQuestion: {q}\n\nAnswer: {a}"
        question = CONVERSATIONAL_PROMPT.replace("<conversation>", conversation).replace("<question>", final_question)
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def chartqa_pro_doc_to_text_alt(doc, lmms_eval_specific_kwargs):
    question = doc["Question"]
    question_type = doc["Question Type"]
    if question_type == "Factoid":
        if isinstance(question, list):
            question = question[0]
        question = FACTOID_PROMPT_ALT.replace("<question>", question)
    elif question_type == "Multi Choice":
        if isinstance(question, list):
            question = question[0]
        question = MULTI_CHOICE_PROMPT_ALT.replace("<question>", question)
    elif question_type == "Hypothetical":
        if isinstance(question, list):
            question = question[0]
        question = HYPOTHETICAL_PROMPT_ALT.replace("<question>", question)
    elif question_type == "Fact Checking":
        if isinstance(question, list):
            question = question[0]
        question = FACT_CHECKING_PROMPT_ALT.replace("<question>", question)
    elif question_type == "Conversational":
        if isinstance(question, list):
            final_question = question[-1]
            history_questions = question[:-1]
        else:
            final_question = question
            history_questions = []
        answers = doc["Answer"]
        if isinstance(answers, list):
            history_answers = answers[:-1]
        else:
            history_answers = []
        conversation = ""
        for q, a, in zip(history_questions, history_answers):
            conversation += f"Question: {q}\n\nAnswer: {a}"
        question = CONVERSATIONAL_PROMPT_ALT.replace("<conversation>", conversation).replace("<question>", final_question)
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def chartqa_pro_cot_process_results(doc, results):
    pred = results[0]
    pred = extract_answer_from_response(pred)
    score = relaxed_correctness(pred, doc["Answer"], doc=doc)
    return_dict = {"relaxed_overall": score}
    return return_dict

def chartqa_pro_process_results(doc, results):
    pred = results[0]
    pred = extract_final_answer(pred)
    score = relaxed_correctness(pred, doc["Answer"], doc=doc)
    return_dict = {"relaxed_overall": score}
    return return_dict

def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.strip(".")
    text = text.strip("\n")
    return text.strip()


def fix_list_format(item: str) -> Any:
    """Attempt to coerce poorly formatted list strings into valid Python lists."""
    if not isinstance(item, str):
        return item
    stripped = item.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return item
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        pass
    content = stripped[1:-1].strip()
    if not content:
        return []
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in content:
        if char == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)
        if char == "[":
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
    if current:
        piece = "".join(current).strip()
        if piece:
            parts.append(piece)
    quoted = []
    for piece in parts:
        if (piece.startswith("'") and piece.endswith("'")) or (piece.startswith('"') and piece.endswith('"')):
            quoted.append(piece)
        else:
            quoted.append(f"'{piece}'")
    try:
        return ast.literal_eval(f"[{', '.join(quoted)}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    """Parses a string representation of a list into a clean list of strings."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [_normalize_text(entry) for entry in parsed]
    return None


def to_float(text: str) -> Optional[float]:
    """Convert a string into a float while handling percents and math expressions."""
    def _parse_scalar(value: str) -> Optional[float]:
        try:
            parsed_value = parse(value)
        except Exception:
            return None
        if not parsed_value:
            return None
        try:
            return float(parsed_value[0])
        except (TypeError, ValueError):
            return None

    def _parse_candidates(value: str) -> Optional[float]:
        candidates: List[str] = []
        raw = value.strip()
        if raw:
            candidates.append(raw)
            # Backward-compatible percent parsing:
            # treat both "23%" and "23\\%" like "23" for numeric matching.
            if raw.endswith("\\%"):
                stripped = raw[:-2].strip()
                if stripped:
                    candidates.append(stripped)
            if raw.endswith("%"):
                stripped = raw[:-1].strip()
                if stripped:
                    candidates.append(stripped)
        for candidate in candidates:
            parsed = _parse_scalar(candidate)
            if parsed is not None:
                return parsed
        return None

    if isinstance(text, (int, float)):
        return float(text)
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    if not t:
        return None

    parsed = _parse_candidates(t)
    if parsed is not None:
        return parsed

    # Safe ratio improvement: only parse simple standalone ratios (a:b).
    ratio_match = _RATIO_RE.fullmatch(t)
    if ratio_match:
        left = _parse_candidates(ratio_match.group(1).strip())
        right = _parse_candidates(ratio_match.group(2).strip())
        if left is not None and right not in (None, 0.0):
            return left / right

    return None


def anls_score(prediction: str, gold_labels: List[str], threshold: float = 0.5) -> float:
    """Compute ANLS-style score with a minimum threshold."""
    pred_norm = " ".join(_normalize_text(prediction).lower().split())
    best = 0.0
    for gold in gold_labels:
        gold_norm = " ".join(_normalize_text(gold).lower().split())
        max_len = max(len(gold_norm), len(pred_norm))
        if max_len == 0:
            score = 1.0
        else:
            dist = levenshtein_distance(gold_norm, pred_norm)
            score = 1.0 - (float(dist) / float(max_len))
        best = max(best, score)
    return best if best >= threshold else 0.0


def evaluate_single_answer(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    """Compare a single target/prediction pair using relaxed numeric tolerance."""
    t = _normalize_text(target)
    p = _normalize_text(prediction)
    t_float = to_float(t)
    p_float = to_float(p)
    if t_float is not None and p_float is not None:
        p_has_percent = p.endswith("%") or p.endswith("\\%")
        t_has_percent = t.endswith("%") or t.endswith("\\%")

        def _within_tolerance(value: float, reference: float) -> bool:
            if reference == 0.0:
                return value == 0.0
            return abs(value - reference) / abs(reference) <= max_relative_change

        if _within_tolerance(p_float, t_float):
            return 1.0
        if p_has_percent and not t_has_percent:
            if _within_tolerance(p_float * 100.0, t_float):
                return 1.0
            if _within_tolerance(p_float / 100.0, t_float):
                return 1.0
        if t_has_percent and not p_has_percent:
            if _within_tolerance(p_float, t_float * 100.0):
                return 1.0
            if _within_tolerance(p_float, t_float / 100.0):
                return 1.0
        if t_float == 0.0:
            return 1.0 if p_float == 0.0 else 0.0
        change = abs(p_float - t_float) / abs(t_float)
        return 1.0 if change <= max_relative_change else 0.0
    return anls_score(prediction=p, gold_labels=[t], threshold=0.5)


def _prepare_year_flags(doc: Optional[dict], target_len: int, question_type: Optional[str]) -> List[str]:
    """Expand year flags so they align with the number of target elements."""
    if doc is None:
        return ["NO"] * target_len
    raw_flags = doc.get("Year", [])
    if isinstance(raw_flags, str):
        flags: List[str] = [raw_flags]
    elif isinstance(raw_flags, list):
        flags = []
        for flag in raw_flags:
            if isinstance(flag, list):
                flags.extend(flag)
            else:
                flags.append(flag)
    else:
        flags = []
    normalized = [str(flag).strip().upper() for flag in flags if flag is not None]
    if question_type == "Conversational" and normalized:
        normalized = normalized[-1:]
    if target_len == 0:
        return normalized
    if not normalized:
        normalized = ["NO"] * target_len
    elif len(normalized) < target_len:
        repeats = (target_len + len(normalized) - 1) // len(normalized)
        normalized = (normalized * repeats)[:target_len]
    else:
        normalized = normalized[:target_len]
    return normalized


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_normalize_text(v) for v in value]
    if isinstance(value, str):
        fixed = fix_list_format(value)
        if isinstance(fixed, list):
            return [_normalize_text(v) for v in fixed]
        parsed = parse_to_list(str(fixed))
        if parsed is not None:
            return parsed
        return [_normalize_text(value)]
    return [_normalize_text(value)]


def relaxed_correctness(prediction, target, doc=None, max_relative_change: float = 0.05) -> float:
    """Calculates relaxed correctness with support for multi-answer targets and year flags."""
    answer_field = None
    if doc is not None and "Answer" in doc:
        answer_field = doc["Answer"]
    else:
        answer_field = target
    if isinstance(answer_field, list) and answer_field:
        target_value = answer_field[-1]
    else:
        target_value = answer_field

    question_type = doc.get("Question Type") if isinstance(doc, dict) else None
    always_use_exact_match = question_type in {"Fact Checking", "Multi Choice"}

    target_list = _as_list(target_value)
    prediction_list = _as_list(prediction)

    year_flags = _prepare_year_flags(doc if isinstance(doc, dict) else None, len(target_list), question_type)
    total_slots = max(len(target_list), len(prediction_list))
    scores: List[float] = []

    for idx in range(total_slots):
        if idx >= len(target_list) or idx >= len(prediction_list):
            scores.append(0.0)
            continue
        t_item = target_list[idx]
        p_item = prediction_list[idx]
        flag = year_flags[idx] if idx < len(year_flags) else "NO"
        is_year = isinstance(flag, str) and flag.upper() == "YES"
        if is_year or always_use_exact_match:
            scores.append(1.0 if t_item.lower() == p_item.lower() else 0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))

    return sum(scores) / len(scores) if scores else 0.0
