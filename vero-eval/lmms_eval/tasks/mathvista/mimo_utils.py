"""MiMo-specific MathVista utilities.

Uses the 'reasoning' prompt pattern (boxed-format hints) from XiaomiMiMo/lmms-eval.
Delegates to the shared evaluator for answer extraction and normalization.
"""

import re
from functools import lru_cache

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from lmms_eval.tasks.mathvista.utils import (
    mathvista_aggregate_results,
    mathvista_doc_to_visual,
    mathvista_process_results,
)


def _build_reasoning_hint(question_type, answer_type, precision):
    """Build a boxed-format hint matching MiMo's 'reasoning' shot_type."""
    if question_type == "multi_choice":
        return "Answer the question with option letter from given choices. Put your final answer within \\boxed{}."

    if answer_type == "integer":
        return "Answer the question requiring an integer answer. Put your final answer within \\boxed{}."
    if answer_type == "float" and precision == 1:
        return "Answer the question requiring a floating-point number with one decimal place. Put your final answer within \\boxed{}."
    if answer_type == "float" and precision == 2:
        return "Answer the question requiring a floating-point number with two decimal places. Put your final answer within \\boxed{}."
    if answer_type == "list":
        return "Answer the question requiring a Python list. Put your final answer within \\boxed{}."
    return "Put your final answer within \\boxed{}."


def mimo_mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Generate MiMo reasoning-style prompt for MathVista.

    Prompt layout: Question -> Choices -> Boxed hint
    (no "First perform reasoning..." or other redundant instructions)
    """
    question = doc["question"]
    unit = doc["unit"] if "unit" in doc else ""
    choices = doc["choices"]
    question_type = doc["question_type"]
    answer_type = doc["answer_type"]
    precision = doc["precision"] if "precision" in doc else 0

    question_text = question
    if unit:
        question_text += f" (Unit: {unit})"

    if choices:
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A') + i)}) {choice}")
        choices_text = "\n".join(texts)
    else:
        choices_text = ""

    hint_text = _build_reasoning_hint(question_type, answer_type, precision)

    elements = [question_text, choices_text, hint_text]
    return "\n".join([e for e in elements if e]).strip()
