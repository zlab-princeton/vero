"""MiMo-specific V*Bench utilities.
Matches XiaomiMiMo/lmms-eval vstar_bench_boxed.yaml:
- BoxedFilter extracts last \\boxed{} content
- Simple regex letter extraction + accuracy scoring
"""

import re

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from lmms_eval.tasks.vstar_bench.utils import (
    vstar_doc_to_text,
    vstar_doc_to_visual,
)

# Patterns matching MiMo's official eval
_PRED_PATTERNS = [
    r"Answer:?\s*[\(]?([A-D])[\)]?",
    r"[\(]?([A-D])[\)\.]?",
    r"option\s*([A-D])",
    r"([A-D])\s*option",
    r"answer\s*is\s*([A-D])",
    r"([A-D])\s*is\s*the\s*answer",
    r"^([A-D])$",
]


def _extract_letter(text: str) -> str | None:
    for pattern in _PRED_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def vstar_bench_mimo_process_results(doc, results):
    """Simple letter-match accuracy after BoxedFilter, matching MiMo's official eval."""
    prediction = results[0] if results else ""
    answer = doc["label"]

    pred_letter = _extract_letter(prediction)
    ans_letter = _extract_letter(answer)

    score = 0
    if pred_letter and ans_letter:
        score = 1 if pred_letter == ans_letter else 0
    return {"accuracy": score}


def vstar_bench_mimo_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
