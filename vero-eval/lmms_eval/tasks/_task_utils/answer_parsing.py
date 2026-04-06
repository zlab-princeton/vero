"""Utilities for extracting final answers from CoT-style model responses."""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.vllm_judge import (
    extract_json_candidate,
    get_judge_engine,
)

_DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL_PATH", "")

EXTRACT_ANSWER_PROMPT = """**Task**: You will receive a *question* and a *model response*. Your job is to extract **only the final answer** from the response.

**Instructions**:

* Do **not** include explanations or reasoning.
* If there is **no model response**, leave the extracted answer empty.

**Output Format**:
Return a JSON object with this exact structure:

{
  "extracted_answer": "<string>"
}

**Guidelines for the extracted_answer**:

* **Numbers**: Use only digits and a decimal point if needed.

  * Remove percentage signs (`%`) and currency symbols (`$`).
  * Remove commas in large numbers.
* **Examples**:

  * Model response: `"The answer is 83%."` → `"83"`
  * Model response: `"Revenue per visit in Q4 '12 is 0.27$."` → `"0.27"`
  * Model response: `"The total was 3,280."` → `"3280"`
  * Model response: `"The answer is **22,500**."` → `"22500"`
  * Model response: `"The answer is 19,407.2 million U.S. dollars."` → `"19407.2"`

### Example 1 Starts ###
* Response: There is only one curve that intersects y=\\lambda exactly three times. The name of the curve is written as P55762.

{
    "extracted_answer": "P55762",
}
### Example 1 Ends ###


### Example 2 Starts ###
* Response: I see letter a, b, and c. a may be it but not close. b looks like the one corresponding to the subplot where all bars are above 35. c looks like the one corresponding to the subplot where all bars are below 35. The letter of the subplot where all bars are above 35 is b.

{
    "extracted_answer": "b",
}
### Example 2 Ends ###

### Your Turn ###
* Question: <|question|>
* Response: <|response|>
"""


def extract_answer_from_response(
    response: str,
    *,
    max_tokens: int = 512,
    prompt_template: Optional[str] = None,
    question: Optional[str] = None,
    model_hint: Optional[str] = None,
) -> str:
    """Parse the final answer from a free-form model response using a judge LLM."""

    if not isinstance(response, str):
        return ""

    template = prompt_template or EXTRACT_ANSWER_PROMPT
    prompt = (
        template.replace("<|question|>", question or "")
        .replace("<|response|>", response)
    )

    try:
        engine = get_judge_engine(model_hint or _DEFAULT_JUDGE_MODEL)
        raw_output = engine.generate_json(prompt, max_tokens=max_tokens)
    except Exception as exc:
        eval_logger.warning(f"Error during vLLM extraction: {exc}")
        return ""

    content = extract_json_candidate(raw_output)
    if content is None:
        eval_logger.warning(f"Failed to parse extraction output: {raw_output}")
        return ""

    extracted = content.get("extracted_answer", "")
    if extracted is None:
        return ""
    return extracted if isinstance(extracted, str) else str(extracted)
