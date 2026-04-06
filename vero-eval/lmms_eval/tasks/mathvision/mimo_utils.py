"""MiMo-specific MathVision utilities.

Uses boxed-format prompts and dual metrics (standard_eval + math_verify)
matching XiaomiMiMo/lmms-eval.
"""

import re

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks.mathvision.utils import (
    mathvision_aggregate_results_eval,
    mathvision_doc_to_visual,
    mathvision_process_results,
)

_math_verify_fn = MathVerifyFn()


def mimo_mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Generate MiMo-style prompt for MathVision.

    Clean format: question + choices + type-specific boxed instruction.
    No 'Please solve step by step' preamble.
    """
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    if choices_str:
        query_prompt = f"{question}\nChoices: {choices_str}"
    else:
        query_prompt = question

    if lmms_eval_specific_kwargs:
        if len_choices > 0:
            query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['mc_prompt']}"
        else:
            query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['short_answer_prompt']}"
    return query_prompt


def mimo_mathvision_boxed_process_results(doc, results):
    """Extract last \\boxed{} content, then run standard + math_verify evaluation."""
    assert len(results) == 1
    model_answer = results[0].strip()

    math_verify_score, math_verify_ext = _math_verify_fn(model_answer, doc["answer"])

    ext_model_answer = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", model_answer)
    if ext_model_answer:
        ext_model_answer = ext_model_answer[-1]
    else:
        ext_model_answer = ""

    res = mathvision_process_results(doc, [ext_model_answer])
    res["math_verify"] = {
        "score": math_verify_score,
        "extraction": math_verify_ext,
    }
    return res


def mimo_mathvision_math_verify_aggregate_results(results):
    total = len(results)
    score = sum(result["score"] for result in results)
    return score / total
