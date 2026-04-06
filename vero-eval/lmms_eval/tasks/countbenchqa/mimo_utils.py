"""MiMo-specific CountBenchQA utilities.
Matches XiaomiMiMo/lmms-eval countbench_boxed.yaml:
- BoxedFilter extracts last \boxed{} content
- Simple exact match scoring (no complex int extraction)
"""

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from lmms_eval.tasks.countbenchqa.utils import (
    countbenchqa_doc_to_text,
    countbenchqa_doc_to_visual,
)


def countbenchqa_mimo_process_results(doc, results):
    """Simple exact match after BoxedFilter extraction, matching MiMo's official eval."""
    prediction = results[0]
    answer = str(doc["number"])
    score = 0
    if prediction.strip().lower() == answer.strip().lower():
        score = 1
    return {"exact_match": score}


def countbenchqa_mimo_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
