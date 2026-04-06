import os

from tqdm import tqdm

from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST, REASONING_RESP_INST_NO_POST_PROMPT
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks.charxiv.reasoning_utils import (
    build_reasoning_grading_queries,
    get_number_instruction,
    get_reasoning_results_gpt_batch,
)

MODEL_VERSION = os.getenv("JUDGE_MODEL_PATH", "")

def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    question = question + lmms_eval_specific_kwargs["post_prompt"]
    return question

def charxiv_reasoning_doc_to_text_cot_no_post_prompt(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST_NO_POST_PROMPT[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST_NO_POST_PROMPT[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    question = question + lmms_eval_specific_kwargs["post_prompt"]
    return question


def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages

def charxiv_reasoning_process_results(doc, results):
    figure_id = doc["figure_path"]
    inst_category = doc["reasoning_q_source"]
    response = extract_final_answer(results[0].strip(), parse_boxed=False)
    answer = doc["reasoning_a"]
    data = {}
    data["inst_category"] = inst_category
    data["figure_id"] = figure_id
    data["answer"] = answer
    resp_value = {"raw_question": doc["reasoning_q"], "response": response}
    return {"reasoning_acc": {"resp_value": resp_value, "resp_key": figure_id, "data": data}}


def charxiv_reasoning_aggregate_results(results):
    data = {}
    resps = {}
    for i, result in enumerate(results):
        data[i] = result["data"]
        resps[result["resp_key"]] = result["resp_value"]
    queries = build_reasoning_grading_queries(data, resps)
    figure_ids = list(queries.keys())
    prompts = [queries[figure_id]["grading_query"] for figure_id in figure_ids]
    judge_outputs = get_reasoning_results_gpt_batch(
        prompts,
        model=MODEL_VERSION,
        use_tqdm=True,
    )
    for figure_id, (ext, scr) in zip(tqdm(figure_ids, desc="Scoring reasoning queries"), judge_outputs):
        queries[figure_id]["extracted_answer"] = ext
        queries[figure_id]["score"] = scr
        queries[figure_id].pop("grading_query")

    charxiv_reasoning_aggregate_results.individual_scores = queries

    # Return the average score
    scores = [query["score"] for query in queries.values()]
    return sum(scores) / len(scores)
