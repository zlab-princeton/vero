# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""String match reward that enforces <think>/<answer> formatting."""

from __future__ import annotations

import re

__all__ = [
    "format_reward",
    "acc_reward",
    "compute_score",
    "compute_score_from_data_source",
]


_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>\s*\Z", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CHOICE_PREFIX_PATTERN = re.compile(r"^[A-Za-z][\.\)]\s*")


def _strip_string(text: str | None) -> str:
    if text is None:
        return ""

    # Standardize whitespace and symbols before comparison.
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("−", "-")  # normalize minus sign
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove currency, percentage, and formatting markers.
    for token in ("$", "£", "€", "¥", "%", "‰", ",", "\\", "°"):
        text = text.replace(token, "")

    text = text.strip()
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    if text.lower().startswith("final answer:"):
        text = text[13:].strip()

    # Drop outer quotes and trailing punctuation that commonly appears in generations.
    text = text.strip("'\"")
    if text.endswith("."):
        text = text[:-1]
    text = text.strip()

    # Collapse internal whitespace and lowercase for stable comparisons.
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    return text


def _extract_answer(predict_str: str) -> str:
    answer_match = _ANSWER_PATTERN.search(predict_str)
    candidate = answer_match.group(1).strip() if answer_match else predict_str.strip()
    candidate = _CHOICE_PREFIX_PATTERN.sub("", candidate)
    return candidate


def format_reward(predict_str: str) -> float:
    if not predict_str:
        return 0.0

    if not re.fullmatch(_FORMAT_PATTERN, predict_str):
        return 0.0

    answer_text = _strip_string(_extract_answer(predict_str))
    return 1.0 if answer_text else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:  # noqa: ARG001
    predicted = _strip_string(_extract_answer(predict_str))
    truth = _strip_string(ground_truth)

    # Fallback to numeric token comparison when both normalize to empty (e.g., stripped units).
    if not predicted and not truth:
        return 0.0

    # Compare numeric forms if one side is numeric-like to tolerate units.
    if predicted != truth:
        number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
        pred_numbers = number_pattern.findall(predicted)
        truth_numbers = number_pattern.findall(truth)
        if pred_numbers and truth_numbers and pred_numbers == truth_numbers:
            return 1.0

    return 1.0 if predicted == truth else 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    format_score: float = 0.5,
) -> dict[str, float]:
    accuracy = acc_reward(predict_str, ground_truth, use_boxed=use_boxed)
    formatting = format_reward(predict_str)
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
    extra_info=None,  # noqa: ARG001
    use_boxed: bool = False,
    format_score: float = 0.5,
    **_: dict,
) -> dict[str, float]:
    """Adapter for custom_reward_function configs that expect default signature."""

    return compute_score(solution_str, ground_truth, use_boxed=use_boxed, format_score=format_score)
