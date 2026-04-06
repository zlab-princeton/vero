"""Reward helpers that require explicit reward types and support numeric tolerance."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable

import sympy
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.grader import sympy_numeric_eq as mv_sympy_numeric_eq
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
from sympy import Basic
from sympy.matrices import MatrixBase
from . import click_reward, grounding_reward, instructions as instruction_lib
from .text_normalization import normalize_text_for_match

__all__ = [
    "format_reward",
    "acc_reward",
    "compute_score",
    "compute_score_from_data_source",
]


_MAX_TEXT_LEN = 200  # Bound extremely long answers/prompts to avoid expensive Sympy work
_PARSE_TIMEOUT = 3
_VERIFY_TIMEOUT = 3

_FORMAT_PATTERN = re.compile(
    r"<think>(?:(?!<think>|</think>).)*</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)
_THINK_PATTERN = re.compile(r"<think>(?P<think>(?:(?!<think>|</think>).)*)</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_REWARD_TYPES = {
    "string_match",
    "multiple_choice",
    "number_list",
    "web_action",
    "numeric",
    "list_string_match",
    "counting",
    "search",
}

_FORMAT_BOXED_WEIGHT = 0.5

_NUMERIC_EXTRACTION_TARGETS = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]
_MOLMO_COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
_MOLMO_FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
_MOLMO_POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _safe_truncate(text: Any, limit: int = 200) -> str:
    try:
        rendered = str(text)
    except Exception:
        rendered = "<unprintable>"
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _log_timeout(context: str, truth: Any, pred: Any) -> None:
    truth_preview = _safe_truncate(truth)
    pred_preview = _safe_truncate(pred)
    print(f"[math_verify timeout] {context} | truth={truth_preview!r} | pred={pred_preview!r}")


def _is_single_letter_choice(text: str) -> bool:
    trimmed = text.strip()
    return len(trimmed) == 1 and trimmed.isalpha()


def _has_single_tag_pair(text: str, open_tag: str, close_tag: str) -> bool:
    return text.count(open_tag) == 1 and text.count(close_tag) == 1


def _extract_answer(predict_str: str) -> str:
    answer_match = _ANSWER_PATTERN.search(predict_str)
    candidate = answer_match.group(1).strip() if answer_match else predict_str.strip()
    return candidate


def _extract_boxed_contents(text: str) -> list[str]:
    if not text:
        return []

    contents: list[str] = []
    idx = 0
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            idx = cursor
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            content = text[content_start : cursor - 1].strip()
            content = _strip_tex_text_wrapper(content)
            contents.append(content)
            idx = cursor
        else:
            break
    return contents


def _extract_last_boxed_bracket_content(text: str) -> str | None:
    """Return payload from the last malformed '\\boxed[...]' block, if present."""
    if not text:
        return None

    idx = 0
    last_content: str | None = None
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] != "[":
            idx = cursor + 1
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_content = text[content_start : cursor - 1].strip()
            idx = cursor
        else:
            break
    return last_content


def _strip_tex_text_wrapper(text: str) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith(r"\text"):
        return text
    idx = len(r"\text")
    while idx < len(stripped) and stripped[idx].isspace():
        idx += 1
    if idx >= len(stripped) or stripped[idx] != "{":
        return text
    idx += 1
    depth = 1
    content_start = idx
    while idx < len(stripped) and depth > 0:
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return text
    content = stripped[content_start : idx - 1].strip()
    trailing = stripped[idx:].strip()
    if trailing:
        return text
    return content


def _strict_json_array_text(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    return candidate


def _strict_json_object_text(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return candidate


def _extract_last_json_array_fragment(text: str) -> str | None:
    """Extract and validate the last JSON-array-looking fragment from free text."""
    if not text:
        return None
    candidates = re.findall(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    for candidate in reversed(candidates):
        strict = _strict_json_array_text(candidate)
        if strict:
            return strict
    return None


def _extract_model_family(extra_info: Any) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    model_family = extra_info.get("model_family")
    if not isinstance(model_family, str):
        return None
    model_family = model_family.strip().lower()
    return model_family or None


def _extract_molmo_grounding_answer(text: str) -> str | None:
    strict_object = _strict_json_object_text(text)
    if not strict_object:
        return None
    try:
        parsed = json.loads(strict_object)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    results = parsed.get("results")
    if not isinstance(results, list):
        return None

    normalized_items: list[dict[str, list[int]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            x_min = int(round(float(item["x_min"])))
            y_min = int(round(float(item["y_min"])))
            x_max = int(round(float(item["x_max"])))
            y_max = int(round(float(item["y_max"])))
        except (KeyError, TypeError, ValueError):
            continue
        x_min, x_max = sorted((x_min, x_max))
        y_min, y_max = sorted((y_min, y_max))
        x_min = max(0, min(x_min, 1000))
        y_min = max(0, min(y_min, 1000))
        x_max = max(0, min(x_max, 1000))
        y_max = max(0, min(y_max, 1000))
        normalized_items.append({"bbox_2d": [x_min, y_min, x_max, y_max]})

    if not normalized_items and results:
        return None
    return json.dumps(normalized_items, ensure_ascii=True)


def _extract_molmo_clicking_answer(text: str) -> str | None:
    if not text:
        return None
    points: list[tuple[int, int]] = []
    for coord_match in _MOLMO_COORD_REGEX.finditer(text):
        coord_text = coord_match.group(1)
        for frame_match in _MOLMO_FRAME_REGEX.finditer(coord_text):
            point_payload = frame_match.group(2)
            for point_match in _MOLMO_POINTS_REGEX.finditer(point_payload):
                x_val = int(point_match.group(2))
                y_val = int(point_match.group(3))
                x_val = max(0, min(x_val, 1000))
                y_val = max(0, min(y_val, 1000))
                points.append((x_val, y_val))
    if not points:
        return None
    x_coord, y_coord = points[0]
    return json.dumps([{"point_2d": [x_coord, y_coord]}], ensure_ascii=True)


def _extract_grounding_clicking_answer(
    answer_text: str, reward_type: str | None = None, extra_info: Any | None = None
) -> str | None:
    # Grounding/clicking extraction order:
    # - if one-or-more valid \boxed{} blocks exist, use the last boxed payload
    # - otherwise, use raw answer text
    # No malformed-boxed recovery here; malformed boxed should fail strict JSON parsing.
    candidate = answer_text.strip()
    if not candidate:
        return None

    boxed_values = _extract_boxed_contents(candidate)
    source = boxed_values[-1] if boxed_values and boxed_values[-1] else candidate

    model_family = _extract_model_family(extra_info)
    reward_type = reward_type.strip().lower() if isinstance(reward_type, str) else None
    if model_family == "molmo2" and reward_type == "grounding":
        molmo_payload = _extract_molmo_grounding_answer(source)
        if molmo_payload:
            return molmo_payload
        return _strict_json_array_text(source)
    if model_family == "molmo2" and reward_type == "clicking":
        molmo_payload = _extract_molmo_clicking_answer(source)
        if molmo_payload:
            return molmo_payload
        return _strict_json_array_text(source)

    return _strict_json_array_text(source)


def _extract_tolerance(extra_info: Any) -> float | None:
    if not isinstance(extra_info, dict):
        return None
    tolerance = extra_info.get("tolerance")
    if tolerance is None:
        return None
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        return None
    if tolerance_value < 0:
        return None
    return tolerance_value


def _extract_reward_type(extra_info: Any) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    reward_type = extra_info.get("reward_type")
    if not isinstance(reward_type, str):
        return None
    reward_type = reward_type.strip().lower()
    allowed = {
        "string_match",
        "multiple_choice",
        "number_list",
        "web_action",
        "numeric",
        "list_string_match",
        "counting",
        "search",
        "grounding",
        "clicking",
        "instruction_following_llm_judge",
        "instruction_following",
        "llm_judge",
    }
    if reward_type in allowed:
        return reward_type
    return None


def _extract_image_path(extra_info: Any) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    images = extra_info.get("images")
    if isinstance(images, (list, tuple)) and images:
        first = images[0]
        if isinstance(first, str):
            return first
    image_path = extra_info.get("image_path") or extra_info.get("image")
    if isinstance(image_path, str):
        return image_path
    return None


def _extract_image_size(extra_info: Any) -> tuple[int, int] | None:
    if not isinstance(extra_info, dict):
        return None
    image_sizes = extra_info.get("image_sizes") or extra_info.get("image_size")
    if isinstance(image_sizes, list) and image_sizes:
        image_size = image_sizes[0]
    else:
        image_size = image_sizes
    if isinstance(image_size, dict):
        width = image_size.get("width")
        height = image_size.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return int(width), int(height)
        return None
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        width, height = image_size
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return int(width), int(height)
    return None


def _extract_normalize_bbox_to_1000(extra_info: Any) -> bool | None:
    """Prefer extra_info["normalize_bbox_to_1000"]; CGS_QWEN_BBOX_NORMALIZED/CGS_QWEN_POINT_NORMALIZED are deprecated."""
    if not isinstance(extra_info, dict):
        return None
    if "normalize_bbox_to_1000" in extra_info:
        return bool(extra_info.get("normalize_bbox_to_1000"))
    return None


def _string_match_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    normalized_pred = _normalize_text_for_match(stripped_predicted)
    normalized_truth = _normalize_text_for_match(truth)
    if not normalized_pred or not normalized_truth:
        return 0.0
    return 1.0 if normalized_pred == normalized_truth else 0.0


# def _multiple_choice_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
#     try:
#         gold_parsed = parse(
#             truth, extraction_config=[StringExtractionConfig()], parsing_timeout=_PARSE_TIMEOUT
#         )
#         pred_parsed = parse(
#             stripped_predicted,
#             extraction_config=[StringExtractionConfig()],
#             parsing_timeout=_PARSE_TIMEOUT,
#         )
#         gold_target = gold_parsed if gold_parsed else truth
#         pred_target = pred_parsed if pred_parsed else stripped_predicted
#         return (
#             1.0
#             if verify(gold_target, pred_target, timeout_seconds=_VERIFY_TIMEOUT)
#             else 0.0
#         )
#     except TimeoutException:
#         _log_timeout("multiple_choice", truth, stripped_predicted)
#         return 0.0
#     except Exception:
#         return 1.0 if stripped_predicted == truth else 0.0


def _multiple_choice_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    try:
        parsed = parse(
            truth,
            extraction_config=[
                StringExtractionConfig(
                    strings=(
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "K",
                        "L",
                        "M",
                        "N",
                        "O",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "U",
                        "V",
                        "W",
                        "X",
                        "Y",
                        "Z",
                    )
                )
            ],
            parsing_timeout=_PARSE_TIMEOUT,
        )
    except TimeoutException:
        _log_timeout("multiple_choice", truth, stripped_predicted)
        parsed = None
    except Exception:
        parsed = None

    if parsed is None:
        parsed_values = []
    elif isinstance(parsed, (list, tuple, set)):
        parsed_values = list(parsed)
    else:
        parsed_values = [parsed]

    normalized_truth = str(parsed_values[0]) if parsed_values else truth
    return _string_match_reward(stripped_predicted, normalized_truth, **_)


_NUMBER_LIST_DISCOUNT = 0.2  # Discount for partially correct permutations
_NUMBER_PATTERN = re.compile(r"-?\d+")


def _parse_number_list(text: str) -> list[int]:
    return [int(val) for val in _NUMBER_PATTERN.findall(text)]


def _number_list_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    gold_list = _parse_number_list(truth)
    pred_list = _parse_number_list(stripped_predicted)
    if not gold_list or not pred_list:
        return 0.0

    k = len(gold_list)
    if len(pred_list) != k:
        return 0.0
    if len(set(pred_list)) != k:
        return 0.0
    if sorted(pred_list) != sorted(gold_list):
        return 0.0
    if pred_list == gold_list:
        return 1.0

    correct_positions = sum(1 for gold, pred in zip(gold_list, pred_list) if gold == pred)
    return _NUMBER_LIST_DISCOUNT * (correct_positions / k)


def _normalize_web_action_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {str(key).upper(): value for key, value in payload.items()}


def _load_web_action_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return _normalize_web_action_payload(payload)
    if not isinstance(payload, str):
        return {}

    payload = payload.strip()
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            parsed = parser(payload)
            return _normalize_web_action_payload(parsed)
        except Exception:
            continue
    return {}


def _load_instruction_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return {}

    payload = payload.strip()
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            parsed = parser(payload)
        except Exception:
            continue

        if isinstance(parsed, str):
            inner = parsed.strip()
            if inner and inner != payload:
                for inner_parser in parsers:
                    try:
                        parsed = inner_parser(inner)
                        break
                    except Exception:
                        continue

        if isinstance(parsed, dict):
            return parsed
    return {}


def _web_action_values_match(gold_value: Any, pred_value: Any) -> bool:
    if isinstance(gold_value, (int, float)) and isinstance(pred_value, str):
        try:
            pred_as_number = float(pred_value)
            return float(gold_value) == pred_as_number
        except Exception:
            pass
    if isinstance(gold_value, str) and isinstance(pred_value, (int, float)):
        try:
            gold_as_number = float(gold_value)
            return gold_as_number == float(pred_value)
        except Exception:
            pass
    return str(gold_value).strip() == str(pred_value).strip()


def _web_action_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    gold_payload = _load_web_action_payload(truth)
    pred_payload = _load_web_action_payload(stripped_predicted)
    if not gold_payload or not pred_payload:
        return 0.0

    considered_keys: list[str] = []
    if "ACTION" in gold_payload:
        considered_keys.append("ACTION")

    mark_value = gold_payload.get("MARK")
    ignore_mark = mark_value == -1 or (isinstance(mark_value, str) and mark_value.strip() == "-1")
    if "MARK" in gold_payload and not ignore_mark:
        considered_keys.append("MARK")

    value_value = gold_payload.get("VALUE")
    ignore_value = value_value == "None" or value_value is None
    if isinstance(value_value, str) and value_value.strip().lower() == "none":
        ignore_value = True
    if "VALUE" in gold_payload and not ignore_value:
        considered_keys.append("VALUE")

    if not considered_keys:
        return 0.0

    weight = 1.0 / len(considered_keys)
    reward = 0.0
    for key in considered_keys:
        if key in pred_payload and _web_action_values_match(gold_payload.get(key), pred_payload.get(key)):
            reward += weight
    return reward


def _default_acc_reward(
    stripped_predicted: str, truth: str, use_boxed: bool = False, extra_info=None  # noqa: ARG001
) -> float:

    _VERIFY_TIMEOUT=1
    _PARSE_TIMEOUT=1
    
    if stripped_predicted.strip().lower() == truth.strip().lower():
        return 1.0

    truth_for_parse = truth

    try:
        gold_parsed = parse(truth_for_parse, parsing_timeout=_PARSE_TIMEOUT)
        mcq_mode = False
        if not gold_parsed and _is_single_letter_choice(truth):
            gold_parsed = parse(
                truth,
                extraction_config=[StringExtractionConfig()],
                parsing_timeout=_PARSE_TIMEOUT,
            )
            mcq_mode = bool(gold_parsed)

        pred_parsed = parse(stripped_predicted, parsing_timeout=_PARSE_TIMEOUT)
        if not pred_parsed and mcq_mode:
            pred_parsed = parse(
                stripped_predicted,
                extraction_config=[StringExtractionConfig()],
                parsing_timeout=_PARSE_TIMEOUT,
            )

        gold_target = gold_parsed if gold_parsed else truth
        pred_target = pred_parsed if pred_parsed else stripped_predicted
    except Exception:
        gold_parsed = None
        pred_parsed = None
        gold_target = truth
        pred_target = stripped_predicted

    try:
        verified = verify(
            gold_target, 
            pred_target, 
            float_rounding=6,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=_VERIFY_TIMEOUT
            )
    except TimeoutException:
        _log_timeout("default_verify", truth, stripped_predicted)
        verified = False
    except Exception:
        verified = False

    return 1.0 if verified else 0.0


def _sympy_numeric_eq_with_tolerance(
    a: Basic | MatrixBase | str | float,
    b: Basic | MatrixBase | str | float,
    float_rounding: int,
    numeric_precision: int,
    tol: float,
) -> bool:
    """Leverage math_verify comparison but allow relative tolerance when numbers parse."""
    try:
        a_f = float(a)
        b_f = float(b)
        diff = abs(a_f - b_f)
        if abs(a_f) <= 1e-12:
            if diff <= tol:
                return True
        elif diff <= tol * abs(a_f):
            return True
    except Exception:
        pass

    return mv_sympy_numeric_eq(a, b, float_rounding, numeric_precision)


def _coerce_to_sympy(expr: Basic | MatrixBase | str | float) -> Basic | MatrixBase | None:
    if isinstance(expr, (Basic, MatrixBase)):
        return expr
    try:
        return sympy.sympify(expr)
    except Exception:
        return None


def _compare_within_tolerance(
    gold_values: Iterable[Basic | MatrixBase],
    pred_values: Iterable[Basic | MatrixBase | str],
    tol: float,
    float_rounding: int = 6,
    numeric_precision: int = 15,
) -> bool:
    for gold in gold_values:
        gold_sympy = _coerce_to_sympy(gold)
        if gold_sympy is None:
            continue
        for pred in pred_values:
            pred_sympy = _coerce_to_sympy(pred)
            if pred_sympy is None:
                continue
            if _sympy_numeric_eq_with_tolerance(
                gold_sympy, pred_sympy, float_rounding, numeric_precision, tol
            ):
                return True
    return False


def _normalize_text_for_match(text: Any) -> str:
    return normalize_text_for_match(text)


def _coerce_truth_to_list(truth: Any) -> list[str]:
    if isinstance(truth, (list, tuple, set)):
        return [str(item) for item in truth]
    if isinstance(truth, str):
        candidate = truth.strip()
        if not candidate:
            return []
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [candidate]
    return [str(truth)]


def _list_string_match_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    normalized_pred = _normalize_text_for_match(stripped_predicted)
    if not normalized_pred:
        return 0.0

    truth_items = _coerce_truth_to_list(truth)
    if not truth_items:
        return 0.0

    for truth_item in truth_items:
        if normalized_pred == _normalize_text_for_match(truth_item):
            return 1.0
    return 0.0


def _instruction_following_reward(
    stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any  # noqa: ARG001
) -> float:
    payload = _load_instruction_payload(truth)
    instructions_list = payload.get("instructions") if isinstance(payload, dict) else None
    if not isinstance(instructions_list, (list, tuple)) or not instructions_list:
        return 0.0

    satisfied = 0
    total = 0
    for item in instructions_list:
        total += 1
        if not isinstance(item, dict):
            continue
        class_name = item.get("class")
        if not isinstance(class_name, str) or not class_name:
            continue
        instruction_cls = getattr(instruction_lib, class_name, None)
        if instruction_cls is None:
            continue
        try:
            instruction = instruction_cls(class_name)
        except Exception:
            try:
                instruction = instruction_cls()
            except Exception:
                continue

        args = item.get("args")
        if not isinstance(args, dict):
            args = {}
        cleaned_args = {key: value for key, value in args.items() if value is not None}
        built = False
        try:
            instruction.build_description(**cleaned_args)
            built = True
        except TypeError:
            try:
                allowed_keys = instruction.get_instruction_args_keys()
            except Exception:
                allowed_keys = None
            if allowed_keys:
                filtered_args = {key: value for key, value in cleaned_args.items() if key in set(allowed_keys)}
            else:
                filtered_args = {}
            try:
                instruction.build_description(**filtered_args)
                built = True
            except Exception:
                built = False
        except Exception:
            built = False

        if not built:
            continue

        try:
            if stripped_predicted.strip() and instruction.check_following(stripped_predicted):
                satisfied += 1
        except Exception:
            continue

    if total == 0:
        return 0.0
    return satisfied / total


def _llm_judge_dummy_reward(
    stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any  # noqa: ARG001
) -> float:
    return 0.0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        pass
    try:
        expr = sympy.sympify(value)
        if expr.is_real or expr.is_number:
            evaluated = expr.evalf()
            return float(evaluated)
    except Exception:
        pass
    return None


def _numeric_candidates(text: str) -> list[float]:
    candidates: list[float] = []

    direct = _coerce_float(text)
    if direct is not None:
        candidates.append(direct)

    try:
        parsed = parse(text, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT)
    except TimeoutException:
        _log_timeout("numeric_parse", text, text)
        parsed = None
    except Exception:
        parsed = None

    parsed_values: list[Any]
    if parsed is None:
        parsed_values = []
    elif isinstance(parsed, (list, tuple, set)):
        parsed_values = list(parsed)
    else:
        parsed_values = [parsed]

    for value in parsed_values:
        coerced = _coerce_float(value)
        if coerced is not None:
            candidates.append(coerced)

    unique_candidates: list[float] = []
    for val in candidates:
        if not any(abs(val - existing) <= 1e-12 for existing in unique_candidates):
            unique_candidates.append(val)
    return unique_candidates


def grade_numeric(gold: str, pred: str, precision: int = 6, tolerance: float | None = None) -> int:
    """
    Returns:
        1 if pred matches gold numerically,
        0 if it does not,
       -1 if parsing/comparison failed.
    """

    if tolerance is not None:
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            return -1
        if tol < 0:
            return -1

        gold_candidates = _numeric_candidates(gold)
        pred_candidates = _numeric_candidates(pred)
        if not gold_candidates or not pred_candidates:
            return -1

        for gold_value in gold_candidates:
            for pred_value in pred_candidates:
                diff = abs(gold_value - pred_value)
                if abs(gold_value) <= 1e-12:
                    if diff <= tol:
                        return 1
                elif diff <= tol * abs(gold_value):
                    return 1
        return 0

    try:
        gold_parsed = parse(
            gold, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT
        )
        pred_parsed = parse(
            pred, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT
        )
    except TimeoutException:
        _log_timeout("numeric_parse_verify", gold, pred)
        return -1
    except Exception:
        return -1

    if gold_parsed is None or pred_parsed is None:
        return -1

    try:
        verified = verify(
            gold_parsed,
            pred_parsed,
            float_rounding=precision,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=_VERIFY_TIMEOUT,
        )
    except TimeoutException:
        _log_timeout("numeric_verify", gold, pred)
        return -1
    except Exception:
        return -1

    return 1 if verified else 0


def _numeric_reward(stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any) -> float:
    tolerance = _extract_tolerance(extra_info)
    result = grade_numeric(truth, stripped_predicted, precision=6, tolerance=tolerance)
    return 1.0 if result == 1 else 0.0


def _grounding_reward(stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any) -> float:
    image_path = _extract_image_path(extra_info)
    image_size = _extract_image_size(extra_info)
    assume_qwen_normalized = _extract_normalize_bbox_to_1000(extra_info)
    component_weights = dict(grounding_reward.GROUNDING_COMPONENT_WEIGHTS)
    iou_threshold = grounding_reward.GROUNDING_IOU_THRESHOLD

    if isinstance(extra_info, dict):
        weights_override = extra_info.get("grounding_component_weights") or extra_info.get("component_weights")
        if isinstance(weights_override, dict):
            numeric_overrides = {k: float(v) for k, v in weights_override.items() if isinstance(v, (int, float))}
            if numeric_overrides:
                component_weights.update(numeric_overrides)
        iou_override = extra_info.get("iou_threshold")
        try:
            if iou_override is not None:
                iou_threshold = float(iou_override)
        except (TypeError, ValueError):
            pass

    try:
        return float(
            grounding_reward.compute_score_accuracy(
                predict_str=stripped_predicted,
                ground_truth=truth,
                image_path=image_path,
                component_weights=component_weights,
                iou_threshold=iou_threshold,
                image_size=image_size,
                assume_qwen_normalized=assume_qwen_normalized,
            )
        )
    except Exception:
        return 0.0


def _clicking_reward(stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any) -> float:
    image_path = _extract_image_path(extra_info)
    image_size = _extract_image_size(extra_info)
    assume_qwen_normalized = _extract_normalize_bbox_to_1000(extra_info)
    try:
        return float(
            click_reward.compute_score_accuracy(
                predict_str=stripped_predicted,
                ground_truth=truth,
                image_path=image_path,
                image_size=image_size,
                assume_qwen_normalized=assume_qwen_normalized,
            )
        )
    except Exception:
        return 0.0


def format_reward(predict_str: str, extra_info: Any | None = None) -> float:
    if not predict_str:
        return 0.0

    if not _has_single_tag_pair(predict_str, "<think>", "</think>"):
        return 0.0
    if not _has_single_tag_pair(predict_str, "<answer>", "</answer>"):
        return 0.0

    if not re.fullmatch(_FORMAT_PATTERN, predict_str):
        return 0.0

    # Require non-whitespace reasoning content in the think block.
    think_match = _THINK_PATTERN.search(predict_str)
    if not think_match or not think_match.group("think").strip():
        return 0.0

    answer_text = _extract_answer(predict_str).strip()
    if not answer_text:
        return 0.0

    reward_type = _extract_reward_type(extra_info)
    if reward_type in _BOXED_REWARD_TYPES:
        boxed_values = _extract_boxed_contents(answer_text)
        if len(boxed_values) == 1 and boxed_values[0]:
            return 1.0
        return 1.0 - _FORMAT_BOXED_WEIGHT
    if reward_type in {"grounding", "clicking"}:
        # For grounding/clicking, only penalize repeated boxed usage.
        # A single malformed \boxed (e.g. \boxed[...]) keeps full format score;
        # accuracy path already returns 0 when strict JSON parsing fails.
        if answer_text.count(r"\boxed") > 1:
            return 1.0 - _FORMAT_BOXED_WEIGHT

    return 1.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = False, extra_info=None) -> float:  # noqa: ARG001
    raw_predicted = _extract_answer(predict_str)
    stripped_predicted = raw_predicted.strip()
    reward_type = _extract_reward_type(extra_info)
    if not reward_type:
        print(f"[acc_reward] Missing or invalid reward_type in extra_info: {extra_info!r}")
    if reward_type in _BOXED_REWARD_TYPES:
        boxed_values = _extract_boxed_contents(stripped_predicted)
        if boxed_values and boxed_values[-1]:
            stripped_predicted = boxed_values[-1].strip()
        elif r"\boxed" in stripped_predicted:
            malformed_bracket = _extract_last_boxed_bracket_content(stripped_predicted)
            if malformed_bracket:
                stripped_predicted = malformed_bracket
    elif reward_type in {"grounding", "clicking"}:
        strict_payload = _extract_grounding_clicking_answer(
            stripped_predicted,
            reward_type=reward_type,
            extra_info=extra_info,
        )
        if not strict_payload:
            return 0.0
        stripped_predicted = strict_payload
    if not stripped_predicted:
        return 0.0

    truth = ground_truth.strip()
    reward_handlers = {
        "string_match": _string_match_reward,
        "multiple_choice": _multiple_choice_reward,
        "number_list": _number_list_reward,
        "web_action": _web_action_reward,
        "numeric": _numeric_reward,
        "list_string_match": _list_string_match_reward,
        "counting": _string_match_reward,
        "search": _string_match_reward,
        "grounding": _grounding_reward,
        "clicking": _clicking_reward,
        "instruction_following_llm_judge": _instruction_following_reward,
        "instruction_following": _instruction_following_reward,
        "llm_judge": _llm_judge_dummy_reward,
    }
    handler = reward_handlers.get(reward_type)
    if not handler:
        return _default_acc_reward(stripped_predicted, truth, use_boxed=use_boxed, extra_info=extra_info)

    try:
        return handler(stripped_predicted, truth, extra_info=extra_info)
    except Exception:
        return 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,
    format_score: float = 0.5,
    extra_info=None,
) -> dict[str, float]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)
    accuracy = acc_reward(predict_str, ground_truth, use_boxed=use_boxed, extra_info=extra_info)
    formatting = format_reward(predict_str, extra_info=extra_info)
    if formatting == 0.0:
        accuracy = 0.0
    score = (1.0 - format_score) * accuracy + format_score * formatting
    return {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
    }


def compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    **_: dict,
) -> dict[str, float]:
    """Adapter for custom_reward_function configs that expect default signature."""

    return compute_score(solution_str, ground_truth, use_boxed=use_boxed, format_score=format_score, extra_info=extra_info)
