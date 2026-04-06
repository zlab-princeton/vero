import os
from copy import deepcopy
from typing import List, Sequence, Tuple

from lmms_eval.tasks.charxiv.constant import (
    REASONING_GRADING_INST,
    REASONING_GRADING_PREFIX,
    REASONING_RESP_INST,
    REASONING_RESP_INST_NO_POST_PROMPT,
)
from lmms_eval.tasks._task_utils.vllm_judge import (
    extract_json_candidate,
    get_judge_engine,
)
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken


def _parse_reasoning_response(payload: str) -> Tuple[str, int] | None:
    content = extract_json_candidate(payload)
    if content is None:
        return None
    try:
        extracted = content["extracted_answer"]
        score = int(content["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return extracted, score


def get_reasoning_results_gpt_batch(
    prompts: Sequence[str],
    *,
    model: str = "qwen_vllm",
    max_retries: int = 10,
    use_tqdm: bool = False,
) -> List[Tuple[str, int]]:
    """Score multiple reasoning responses using a single vLLM batch call."""

    engine = get_judge_engine(model)
    total = len(prompts)
    pending = {idx: prompts[idx] for idx in range(total)}
    results: List[Tuple[str, int]] = [
        ("Failed to parse response", 0) for _ in range(total)
    ]
    retries = 0

    while pending and retries < max_retries:
        retries += 1
        batch_indices = list(pending.keys())
        batch_prompts = [pending[idx] for idx in batch_indices]
        try:
            responses = engine.generate_json_batch(
                batch_prompts,
                use_tqdm=use_tqdm and retries == 1,
            )
        except Exception as exc:
            print(f"Error during vLLM generation: {exc}")
            continue

        for idx, response in zip(batch_indices, responses):
            parsed = _parse_reasoning_response(response)
            if parsed is None:
                print(f"Failed to parse response: {response}")
                continue
            results[idx] = parsed
            pending.pop(idx, None)

    if pending:
        for idx in pending:
            print(f"Failed to get response for prompt: {prompts[idx]}")

    return results


def get_reasoning_result_gpt(
    client,
    prompt: str,
    model="qwen_vllm",
    max_retries: int = 10,
):
    """Backward-compatible single prompt wrapper around the batch scorer."""

    _ = client
    batched = get_reasoning_results_gpt_batch(
        [prompt],
        model=model,
        max_retries=max_retries,
    )
    return batched[0]


def get_number_instruction(answer):
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_grading_queries(input, resp):
    queries = {}
    for _, data in input.items():
        figure_id = str(data["figure_id"])
        # question without instruction, response
        query, response = resp[figure_id]["raw_question"], resp[figure_id]["response"]
        response = truncate_response_tail_tiktoken(response)
        # get query for answer type (inst_category), then
        # populate the query with the question, ground truth, and response
        grading_query = REASONING_GRADING_PREFIX + deepcopy(REASONING_GRADING_INST[data["inst_category"]]).replace("<|question|>", query).replace("<|ground_truth|>", data["answer"]).replace("<|response|>", response)
        query = {
            "figure_id": figure_id,
            "grading_query": grading_query,
        }
        queries[figure_id] = query
    return queries


def build_reasoning_queries(data, image_dir):
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        inst_category = d["inst_category"]
        # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d["query"])
        # 4: number-in-general -> need to specify the number of decimal places
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(d["query"], get_number_instruction(d["answer"]))
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")
        query = {
            "figure_id": d["figure_id"],  # figure_id
            "figure_path": figure_path,  # figure_path
            "inst_category": inst_category,  # instruction category
            "raw_question": d["query"],  # question @@@ without @@@ instruction
            "question": question,  # question with instruction
        }
        queries[d["figure_id"]] = query
    return queries
