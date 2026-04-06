# flake8: noqa
import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import nltk
import requests
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import (
    get_judge_engine,
    judge_supports_multimodal,
    resolve_judge_mode,
)


# Ensure nltk can find punkt if available locally
for _path in [os.getenv("NLTK_DATA_PATH"), os.path.expanduser("~/nltk_data")]:
    if _path and _path not in nltk.data.path:
        nltk.data.path.append(_path)


# ============================
# Dataset Helpers
# ============================


def mmifeval_doc_to_visual(doc):
    image = doc.get("image")
    if image is None:
        raise ValueError("MMIFEval: missing image data in document")
    if isinstance(image, bytes):
        raw = image
    else:
        raw = base64.b64decode(image)
    try:
        return [Image.open(BytesIO(raw)).convert("RGB")]
    except Exception as exc:
        raise ValueError("MMIFEval: failed to decode image data") from exc


def mmifeval_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_text = doc.get("question") or doc.get("instruction") or ""

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    formatted_question = f"{pre_prompt}{question_text}{post_prompt}"

    return formatted_question


# ============================
# Rule-Based Constraint Checks
# ============================


# HumanCheck: True


def check_whether_response_paragraph_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    actual_count = len(paragraphs)
    # print(actual_count)

    return lower_bound <= actual_count <= upper_bound

# HumanCheck: True


def check_whether_response_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    # use nltk to split the response into sentences
    sentences = nltk.sent_tokenize(response)
    actual_count = len(sentences)
    # print(actual_count)

    return lower_bound <= actual_count <= upper_bound

# HumanCheck: True


def check_whether_each_paragraph_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        # use nltk to split the paragraph into sentences
        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)
        # print(f"paragraph {i}: {actual_count}")
        if actual_count < lower_bound or actual_count > upper_bound:
            return False

    return True

# HumanCheck: True


def check_whether_each_paragraph_sentence_number_in_range_list(
    response: str, ranges: List[List[int]]
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    if len(paragraphs) != len(ranges):
        return False

    for i, (paragraph, range_pair) in enumerate(zip(paragraphs, ranges)):
        lower_bound, upper_bound = range_pair
        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)
        # print(f"paragraph {i}: {actual_count}")
        if not (lower_bound <= actual_count <= upper_bound):
            return False

    return True

# HumanCheck: True


def check_whether_response_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    # this line is used to filter out all non-word characters
    response_clean = re.sub(r"[^\w\s.-]", "", response)
    word_list = response_clean.split()
    word_count = len(word_list)
    # print(word_count)
    return lower_bound <= word_count <= upper_bound

# HumanCheck: True


def check_whether_each_paragraph_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    # Check whether the number of words in each paragraph of the response is greater than or equal to lower_bound and less than or equal to upper_bound.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that the number of words in each paragraph
    # should be between 50 and 80, then lower_bound = 50 and upper_bound = 80.
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        paragraph_clean = re.sub(r"[^\w\s.-]", "", paragraph)
        word_count = len(paragraph_clean.split())
        # print(f"paragraph {i} word count: {word_count}")
        if not (lower_bound <= word_count <= upper_bound):
            return False

    return True

# HumanCheck: True


def check_whether_whole_response_not_contain_certain_substrings(
    response: str, substrings: List[str]
) -> bool:
    # Check whether the entire response does not contain any of the specified substrings.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that the response should not contain the
    # words "apple" and "banana", then substrings = ["apple", "banana"].
    return all(substring not in response for substring in substrings)

# HumanCheck: True


def check_whether_whole_response_not_contain_certain_substring(
    response: str, substring: str
) -> bool:
    return substring not in response

# HumanCheck: True


def check_whether_each_sentence_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    # Check whether each sentence in the response starts with the specified substring.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that each sentence should start with
    # exclamation point, then substring = "!".
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.startswith(substring) for sentence in sentences)

# HumanCheck: True


def check_whether_each_paragraph_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.startswith(substring) for paragraph in paragraphs)

# HumanCheck: True


def check_whether_each_paragraph_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.endswith(substring) for paragraph in paragraphs)

# HumanCheck: True


def check_whether_each_sentence_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.endswith(substring) for sentence in sentences)

# HumanCheck: True


def check_whether_whole_response_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().startswith(substring)

# HumanCheck: True


def check_whether_whole_response_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().endswith(substring)

# HumanCheck: True


def check_whether_each_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:
    # should notice case like "Reddit" is counted as "Redditor"
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    for keyword in keywords:
        # use \b to match the whole word
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        if len(matches) < lower_bound_times or len(
                matches) > upper_bound_times:
            return False

    return True

# HumanCheck: True


def check_whether_total_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:
    # should notice case like "Reddit" is counted as "Redditor"
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    count = 0
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        count += len(matches)

    return lower_bound_times <= count <= upper_bound_times

# HumanCheck: True


def check_percentage_number_precision_in_response(
        response: str, precision: int) -> bool:
    # All numeric values that appear before a percentage sign (%) must be
    # rounded and retained to two decimal places.
    pattern = r'(\d+\.\d+|\d+)\s*%'  # allow numbers and % to have spaces

    matches = re.findall(pattern, response)

    for num_str in matches:
        if '.' not in num_str:
            # no decimal point, not a float number
            return False
        decimal_part = num_str.split('.')[1]
        if len(decimal_part) != precision:
            return False

    return True

# HumanCheck: True


def check_number_precision_in_response(response: str, precision: int) -> bool:
    # Regex pattern to extract numbers, including scientific notation and
    # percentages
    number_pattern = r'''
        (?<!\w)                     # Not preceded by a word character
        [+-]?                      # Optional sign
        (?:                        # Number formats:
            \d{1,3}(?:,\d{3})*(?:\.\d+)?   # e.g., 1,234.56
            | \d+\.\d+             # e.g., 123.456
            | \.\d+                # e.g., .456
            | \d+                  # e.g., 123
        )
        (?:[eE][+-]?\d+)?          # Optional scientific notation
        %?                         # Optional percentage
        (?!\w)                     # Not followed by a word character
    '''

    matches = re.finditer(number_pattern, response, flags=re.VERBOSE)

    for match in matches:
        num_str = match.group()
        clean_num = num_str.replace(',', '').rstrip('%')

        # Split out mantissa if scientific notation
        if 'e' in clean_num.lower():
            mantissa = re.split('[eE]', clean_num)[0]
        else:
            mantissa = clean_num

        # Check digits after decimal in mantissa
        if '.' in mantissa:
            decimal_part = mantissa.split('.')[-1]
            if len(decimal_part) != precision:
                return False
        else:
            if precision != 0:
                return False

    return True

# HumanCheck: True


def check_whether_has_no_arabic_number_in_response(response: str) -> bool:
    number_pattern = r"""
        (?<![.\w])                            # Ensure no preceding . or word char
        (?:                                   # Start of number pattern
            \d{1,3}(?:,\d{3})+(?:\.\d+)?%?    |  # 1,000 or 1,000.00 or 1,000%
            \d+\.\d+%?                        |  # decimals: 3.14, 0.5%
            \d+%?                             |  # integers: 100, 100%
            \d+(?:\.\d+)?(?:[eE][+-]?\d+)        # scientific: 5e-10, 5.09e-10
        )
        (?![.\w])                             # Ensure no trailing . or word char
    """
    numbers = re.findall(
        number_pattern,
        response,
        flags=re.IGNORECASE | re.VERBOSE)
    # print(numbers)
    return len(numbers) == 0


# ============================
# Prompt Construction
# ============================


def generate_eval_pt_c_level(constraints, prediction):
    constraints_str = "\n".join(
        [f"Constraint_{i + 1}: {constraint['value']}" for i, constraint in enumerate(constraints)]
    )
    image_judging_requirement = (
        "2. Judge only from the text response and constraints. Ignore image content."
        if JUDGE_TEXT_ONLY
        else "2. You should refer to the content of image to make the judgment."
    )
    pt = f"""\
Your task is to evaluate whether the response from an AI assistant adheres to all of the given constraints. \
Please follow the requirements below to make the judgment:
1. Be strict and consistent in your assessment.
{image_judging_requirement}
3. For each constraint, if the response fails to fully meet the constraint, \
give it a score of 0. Otherwise, give it a score of 1.

<start of response>
{prediction}
<end of response>

<start of constraint list>
{constraints_str}
<end of constraint list>

You must evaluate and provide an explanation for each constraint listed, ensuring no constraint is omitted. \
At the end, summarize the scores for all constraints in one sentence.

Your output should strictly follow the format below:
Judgement: ...
Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, Score of constraint_3: x/1, ..., Score of \
constraint_n: x/1.
"""
    return pt


def generate_eval_pt_p_level(question, prediction, ground_truth):
    pt = f"""\
You are an expert evaluator. Your task is to extract the answer from the model output and \
compare it with the ground truth list \
to determine whether the model answer covers all the points in the ground truth list. \
The ground truth list is provided as a JSON array of strings, and the model answer is a text string. \
An answer is considered correct if every element from the ground truth list appears in the model \
answer (substring matching is acceptable). \
The order does not matter. \

Your response should only be 'right' if the model answer fully covers the ground truth, or 'wrong' if it does not. \
Do not provide any additional commentary.

Question: {question}
Response from the model: {prediction}
Ground Truth List: {ground_truth}
"""
    return pt


def generate_cmp_pt(constraint, pred_with_constraint, pred_without_constraint):
    pt = f"""\
You are an expert in judging whether the respone follow the given constraint. \
Your task is to assess whether the model's response satisfies \
the given constraint and return True or False. I will provide you \
with the constraint and the model's response under this constraint. \
To assist with your evaluation, I will also provide you with the model's response \
to the same question without the constraint.

<start of constraint>
{constraint}
<end of constraint>

<start of response under the constraint>
{pred_with_constraint}
<end of response under the constraint>

<start of response without the constraint>
{pred_without_constraint}
<end of response without the constraint>

**Please follow the steps below to evaluate**:
Step 1. Compare the model's response under the constraint with its response without the constraint. \
If you believe these two answers \
are very similar, it means the model has not fully considered the impact of the constraint on the answer. \
Please return False.
Step 2. Compare the model's response under the constraint with the content of the constraint. If you believe the model's response \
does not meet the requirements specified in the constraint, return False. Otherwise, \
if the response effectively satisfies the constraint, return True.

Start by briefly explaining your reasoning based on the above steps. At the end, provide a one-sentence \
summary of your evaluation.

Your output must strictly follow this format:
Reasoning: ...
Summary: "True" / "False".
"""
    return pt


# ============================
# Score Extraction
# ============================


# extract score from gpt_resp
# format: Score of instruction: x/1, Score of constraint_1: y/1, Score of constraint_2: z/1, ..., Score of constraint_n: w/1.
# return: score_dict {'instruction': x/1, 'constraint_1': y/1,
# 'constraint_2': z/1, ..., 'constraint_n': w/1}


def extract_score_from_direct_gpt_resp(raw_score):
    # Define regular expression patterns (updated to handle underscores in
    # constraint names)
    score_pattern = re.compile(r"Score\s+of\s+([a-zA-Z0-9_\-]+):\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)

    # Clean the raw score to remove unnecessary symbols (e.g., newlines,
    # multiple spaces)
    # Normalize whitespace
    cleaned_score = re.sub(r"\s+", " ", raw_score).strip()
    # delete all the '*'
    cleaned_score = re.sub(r"\*", "", cleaned_score)

    # Find all individual component scores
    score_matches = score_pattern.findall(cleaned_score)

    # If no valid score matches found, print and raise an exception
    if not score_matches:
        print(f"raw_score:\n{raw_score}")
        raise ValueError("raw_score format is incorrect, cannot parse scores")

    score_dict = {}

    # Parse each component score
    for match in score_matches:
        component_name = match[0].strip().lower()  # Component name, converted to lowercase
        component_name = component_name.replace(" ", "_")
        numerator = int(match[1])  # Numerator
        denominator = int(match[2])  # Denominator
        score = numerator / denominator  # Calculate the score
        score_dict[component_name] = score  # Store it in the dictionary

    return score_dict


# extract score from gpt_resp
# format: right or wrong
# return: score


def extract_score_from_p_level_gpt_resp(raw_score):
    if raw_score == "right":
        return 1
    elif raw_score == "wrong":
        return 0
    else:
        # try to find "right" or "wrong" in the raw_score
        if re.search(r"right", raw_score, re.IGNORECASE):
            return 1
        elif re.search(r"wrong", raw_score, re.IGNORECASE):
            return 0
        else:
            raise ValueError("raw_score format is incorrect, cannot parse scores")


# extract score from gpt_resp
# format: True or False
# return: score


def extract_score_from_cmp_gpt_resp(response_text):
    # Step 1: Find the last occurrence of 'summary:'
    summary_idx = response_text.lower().rfind("summary")
    if summary_idx == -1:
        raise ValueError("No 'summary' found in response.")

    # Step 2: Slice the string after 'summary:' and extract value
    after_summary = response_text[summary_idx + len("summary") :]

    # Match true/false ignoring markdown and formatting
    match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return 1 if value == "true" else 0

    raise ValueError("No valid 'True' or 'False' found after 'summary'.")


# ============================
# Judge API
# ============================


NUM_SECONDS_TO_SLEEP = 10
API_TYPE = os.getenv("API_TYPE", "openai")
API_TYPE_LOWER = API_TYPE.lower()
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
JUDGE_MODEL_HINT = os.getenv("JUDGE_MODEL_PATH") or GPT_EVAL_MODEL_NAME
JUDGE_MODE = resolve_judge_mode(JUDGE_MODEL_HINT)
_JUDGE_MODE_OVERRIDDEN = os.getenv("LMMS_EVAL_JUDGE_MODE", "").strip().lower() in {"vlm", "llm"}
_JUDGE_TEXT_ONLY_FLAG = os.getenv("LMMS_EVAL_JUDGE_TEXT_ONLY", "").strip().lower() in {"1", "true", "yes"}
USE_LOCAL_JUDGE = API_TYPE_LOWER not in ("openai", "azure")
JUDGE_TEXT_ONLY = (JUDGE_MODE == "llm") if _JUDGE_MODE_OVERRIDDEN else (_JUDGE_TEXT_ONLY_FLAG or JUDGE_MODE == "llm")

if API_TYPE_LOWER == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE_LOWER == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("GPT_API_KEY", os.getenv("AZURE_API_KEY", "YOUR_API_KEY"))
    headers = {"api-key": API_KEY, "Content-Type": "application/json", "api-version": "2023-07-01-preview"}
else:
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

eval_logger.info(
    f"MMIFEval judge routing: mode={JUDGE_MODE}, text_only={JUDGE_TEXT_ONLY}, use_local_judge={USE_LOCAL_JUDGE}"
)


def _read_mmifeval_batch_size() -> int:
    raw = os.getenv("MMIFEVAL_JUDGE_BATCH_SIZE", "8").strip()
    try:
        batch_size = int(raw)
    except ValueError:
        eval_logger.warning(f"Invalid MMIFEVAL_JUDGE_BATCH_SIZE='{raw}', defaulting to 8.")
        return 8
    if batch_size < 1:
        eval_logger.warning(f"MMIFEVAL_JUDGE_BATCH_SIZE={batch_size} is invalid; clamping to 1.")
        return 1
    return batch_size


MMIFEVAL_JUDGE_BATCH_SIZE = _read_mmifeval_batch_size()

_local_engine = None


def _get_local_engine():
    global _local_engine
    if _local_engine is None:
        _local_engine = get_judge_engine(JUDGE_MODEL_HINT)
    return _local_engine


def _coerce_image_url(image_data: Any) -> str:
    if not image_data:
        return ""
    if isinstance(image_data, bytes):
        return "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")
    if not isinstance(image_data, str):
        return ""
    if image_data.startswith("data:") or image_data.startswith("http"):
        return image_data
    return f"data:image/jpeg;base64,{image_data}"


def _run_local(prompt: str, max_tokens: int, retries: int) -> str:
    model_used = JUDGE_MODEL_HINT
    for attempt in range(retries):
        try:
            content = _get_local_engine().generate_json(prompt, max_tokens=max_tokens)
            return content.strip() if isinstance(content, str) else str(content)
        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return ""
    return ""


def _run_local_messages(messages: List[Dict[str, Any]], max_tokens: int, retries: int) -> str:
    model_used = JUDGE_MODEL_HINT
    for attempt in range(retries):
        try:
            engine = _get_local_engine()
            if not judge_supports_multimodal(engine):
                raise RuntimeError(
                    "MMIFEval VLM judge requires a multimodal backend. "
                    "Use JUDGE_BACKEND=engine with a multimodal local model, or set VLLM_SERVER_JUDGE=1."
                )
            content = engine.generate_json_messages(messages, max_tokens=max_tokens)
            return content.strip() if isinstance(content, str) else str(content)
        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return ""
    return ""


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _normalize_batch_outputs(outputs: List[Any], expected_size: int) -> List[str]:
    normalized = [content.strip() if isinstance(content, str) else str(content) for content in outputs]
    if len(normalized) != expected_size:
        raise RuntimeError(f"Batch output size mismatch: expected {expected_size}, got {len(normalized)}")
    return normalized


def _run_local_batch(prompts: List[str], max_tokens: int, retries: int) -> List[str]:
    if not prompts:
        return []
    for attempt in range(retries):
        try:
            outputs = _get_local_engine().generate_json_batch(prompts, max_tokens=max_tokens, use_tqdm=False)
            return _normalize_batch_outputs(outputs, len(prompts))
        except Exception as e:
            eval_logger.info(f"Batch attempt {attempt + 1} failed with error: {e}")
            if attempt + 1 < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} batch attempts failed. Last error message: {e}")
    eval_logger.warning("MMIFEval batch text judge failed; falling back to per-item judge calls.")
    return [_run_local(prompt, max_tokens=max_tokens, retries=retries) for prompt in prompts]


def _run_local_messages_batch_task_specific(engine: Any, messages_batch: List[List[Dict[str, Any]]], max_tokens: int) -> List[str]:
    llm = getattr(engine, "_llm", None)
    if llm is None or not hasattr(llm, "chat"):
        raise RuntimeError("MMIFEval task-specific multimodal batching requires a local vLLM judge engine.")
    if not hasattr(engine, "_resolve_sampling_params"):
        raise RuntimeError("MMIFEval task-specific multimodal batching could not resolve sampling params.")

    sampling_params = engine._resolve_sampling_params(max_tokens)
    system_prompt = getattr(engine, "_system_prompt", "")
    payload_batch: List[List[Dict[str, Any]]] = []
    for messages in messages_batch:
        payload_messages = list(messages or [])
        if system_prompt and not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in payload_messages):
            payload_messages = [{"role": "system", "content": system_prompt}] + payload_messages
        payload_batch.append(payload_messages)

    chat_kwargs: Dict[str, Any] = {}
    chat_template = getattr(engine, "_chat_template", None)
    if chat_template:
        chat_kwargs["chat_template"] = chat_template

    outputs = llm.chat(
        messages=payload_batch,
        sampling_params=sampling_params,
        use_tqdm=False,
        **chat_kwargs,
    )
    responses: List[str] = []
    for output in outputs:
        if not output.outputs:
            responses.append("")
            continue
        responses.append(output.outputs[0].text.strip())
    return _normalize_batch_outputs(responses, len(messages_batch))


def _run_local_messages_batch(messages_batch: List[List[Dict[str, Any]]], max_tokens: int, retries: int) -> List[str]:
    if not messages_batch:
        return []
    for attempt in range(retries):
        try:
            engine = _get_local_engine()
            if not judge_supports_multimodal(engine):
                raise RuntimeError(
                    "MMIFEval VLM judge requires a multimodal backend. "
                    "Use JUDGE_BACKEND=engine with a multimodal local model, or set VLLM_SERVER_JUDGE=1."
                )
            return _run_local_messages_batch_task_specific(engine, messages_batch, max_tokens=max_tokens)
        except Exception as e:
            eval_logger.info(f"Batch attempt {attempt + 1} failed with error: {e}")
            if attempt + 1 < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} batch attempts failed. Last error message: {e}")
    eval_logger.warning("MMIFEval batch multimodal judge failed; falling back to per-item judge calls.")
    return [_run_local_messages(messages, max_tokens=max_tokens, retries=retries) for messages in messages_batch]


def _run_without_image_batch(prompts: List[str], retry: int = 3) -> List[str]:
    if not prompts:
        return []
    if USE_LOCAL_JUDGE:
        responses: List[str] = []
        for prompt_batch in _chunk_list(prompts, MMIFEVAL_JUDGE_BATCH_SIZE):
            responses.extend(_run_local_batch(prompt_batch, max_tokens=4096, retries=retry))
        return responses
    return [run_once_without_image(prompt, retry=retry) for prompt in prompts]


def _run_with_messages_batch(messages_batch: List[List[Dict[str, Any]]], retry: int = 3) -> List[str]:
    if not messages_batch:
        return []
    if USE_LOCAL_JUDGE:
        responses: List[str] = []
        for message_chunk in _chunk_list(messages_batch, MMIFEVAL_JUDGE_BATCH_SIZE):
            responses.extend(_run_local_messages_batch(message_chunk, max_tokens=4096, retries=retry))
        return responses
    responses: List[str] = []
    for messages in messages_batch:
        responses.append(_post_chat(messages, max_tokens=4096, retries=retry))
    return responses


def _build_user_message_with_image(prompt: str, image_base64: str) -> List[Dict[str, Any]]:
    image_url = image_base64
    if image_url and not image_url.startswith("data:"):
        image_url = f"data:image/jpeg;base64,{image_url}"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
    ]
    return [{"role": "user", "content": content}]


def _build_user_message_text_only(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def _post_chat(messages: List[Dict[str, Any]], max_tokens: int, retries: int = 2) -> str:
    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content
            break

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return ""
    return ""


# <<< gpt >>>


judge_model = None


def run_once_with_image(pt, image, retry=3):
    if JUDGE_TEXT_ONLY:
        return run_once_without_image(pt, retry=retry)
    if not image:
        eval_logger.warning("MMIFEval: missing image for judge prompt, falling back to text-only")
        return run_once_without_image(pt, retry=retry)
    ans = ""
    if USE_LOCAL_JUDGE:
        image_url = _coerce_image_url(image)
        if not image_url:
            raise ValueError("MMIFEval: failed to build image URL for VLM judge.")
        messages = _build_user_message_with_image(pt, image_url)
        return _run_local_messages(messages, max_tokens=4096, retries=retry)
    messages = _build_user_message_with_image(pt, image)
    while retry:
        try:
            ans = _post_chat(messages, max_tokens=4096)
            return ans
        except Exception as e:
            eval_logger.info(f"Error in run_once_with_image: {e}")
            retry -= 1
    return ans


def run_once_without_image(pt, retry=3):
    ans = ""
    if USE_LOCAL_JUDGE:
        return _run_local(pt, max_tokens=4096, retries=retry)
    messages = _build_user_message_text_only(pt)
    while retry:
        try:
            ans = _post_chat(messages, max_tokens=4096)
            return ans
        except Exception as e:
            eval_logger.info(f"Error in run_once_without_image: {e}")
            retry -= 1
    return ans


# ============================
# Scoring
# ============================


def _safe_json_loads(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


aux_data_dict = {}
_IMAGE_CACHE: Dict[str, Any] = {}


def _cache_image(item_id: Any, image_value: Any) -> None:
    if item_id is None or image_value in (None, ""):
        return
    _IMAGE_CACHE[str(item_id)] = image_value


def _get_cached_image(item: Dict[str, Any]) -> Any:
    direct_image = item.get("image")
    if direct_image not in (None, ""):
        return direct_image
    item_id = item.get("id")
    if item_id is None:
        return None
    return _IMAGE_CACHE.get(str(item_id))


def judge_one_item(item):
    global aux_data_dict
    if isinstance(item, str):
        item = json.loads(item)

    if item.get("tag", None) == "P-Level":
        # in tsv file, answer is a string, need to be converted to list
        pt = generate_eval_pt_p_level(item.get("question"), item.get("prediction"), _safe_json_loads(item.get("answer"), []))
        gpt_resp = run_once_without_image(pt)
        try:
            score = extract_score_from_p_level_gpt_resp(gpt_resp)
            return (
                0,
                "success",
                {
                    "total_score": score,
                    "gpt_resp": gpt_resp,
                },
            )
        except Exception as e:
            eval_logger.error(f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}\n")
            return 1, "P-Level, fail in extract score", {}
    else:  # process C-Level data
        constraints_all = _safe_json_loads(item.get("constraints"), [])
        # split into direct_gpt and other
        # direct_gpt can be processed in batch
        # other needs to be processed one by one
        constraint_direct_gpt = []
        constraint_other = []
        for constraint in constraints_all:
            method = constraint["judge"]["method"]
            if method == "direct_gpt":
                constraint_direct_gpt.append(constraint)
            else:
                constraint_other.append(constraint)
        score_dict = {}
        # 1. process direct_gpt: if there is no direct_gpt, instruction is also
        # needed
        if len(constraint_direct_gpt) > 0:
            pt_direct_gpt = generate_eval_pt_c_level(constraint_direct_gpt, item.get("prediction"))
            gpt_resp = run_once_with_image(pt_direct_gpt, _get_cached_image(item))
            try:
                direct_gpt_score_dict = extract_score_from_direct_gpt_resp(gpt_resp)
                score_dict["gpt_resp_direct_gpt"] = gpt_resp
                for i, constraint in enumerate(constraint_direct_gpt):
                    score_dict[constraint["key"]] = direct_gpt_score_dict[f"constraint_{i + 1}"]
            except Exception as e:
                eval_logger.error(
                    f"\nError:\n{e}\nItem:\n{item}\npt_direct_gpt:\n{pt_direct_gpt}\ngpt_resp:\n{gpt_resp}\n"
                )
                return 1, "C-Level, direct_gpt, fail in extract score", {}
        # 2. process rule_based
        for constraint in constraint_other:
            if constraint["judge"]["method"] == "rule_based":
                # call function according to constraint["judge"]["verify_funcs"]
                # maybe a list of function names (str)
                # func in function_and_compare.py
                # example: {"method": "rule_based", "verify_funcs": [{"func":
                # "check_whether_response_paragraph_number_in_range", "params":
                # [3, 3]}]}}
                score = 1.0
                # breakpoint()
                for func_dict in constraint["judge"]["verify_funcs"]:
                    func = globals()[func_dict["func"]]
                    # use * to unpack the list, ** is used for dict
                    judge_result = func(item.get("prediction"), *func_dict["params"])
                    # breakpoint()
                    if not judge_result:  # False -> score = 0
                        score = 0.0
                        break
                # breakpoint()
                score_dict[constraint["key"]] = score
        # 3. process cmp_gpt
        for constraint in constraint_other:
            if constraint["judge"]["method"] == "cmp_gpt":
                del_cons_prediction = aux_data_dict[item["id"]][constraint["key"]]
                pt = generate_cmp_pt(constraint["value"], item.get("prediction"), del_cons_prediction)
                gpt_resp = run_once_without_image(pt)
                try:
                    score = extract_score_from_cmp_gpt_resp(gpt_resp)
                    score_dict[constraint["key"]] = score
                    score_dict[f"gpt_resp_cmp_gpt_{constraint['key']}"] = gpt_resp
                except Exception as e:
                    eval_logger.error(f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}")
                    return 1, "C-Level, cmp_gpt, fail in extract score", {}
        # add total_score
        total_score = 0.0
        cnt = 0
        for key, value in score_dict.items():
            if key.startswith("gpt_resp_"):
                continue
            total_score += value
            cnt += 1
        score_dict["total_score"] = total_score / cnt if cnt else 0.0
        eval_logger.info(f"score_dict:\n{score_dict}")
        return 0, "success", score_dict


# ============================
# Result Processing Functions
# ============================


_EVAL_CACHE: Dict[str, Any] = {}


def _extract_image_ref(doc: Dict[str, Any]) -> str:
    for key in ("image_path", "image_id", "image_name", "id"):
        value = doc.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int)):
            return str(value)
    return ""


def _make_resp_key(item_id: Any) -> str:
    if item_id is None:
        return ""
    return str(item_id)


def mmifeval_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    prediction = ""
    try:
        if results and len(results) > 0 and results[0] is not None:
            prediction = results[0].strip()
    except Exception:
        prediction = ""
    prediction = extract_final_answer(prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)
    item_id = doc.get("id")

    payload = {
        "id": item_id,
        "tag": doc.get("tag"),
        "infer_type": doc.get("infer_type"),
        "del_cons": doc.get("del_cons"),
        "question": doc.get("question"),
        "instruction": doc.get("instruction"),
        "answer": doc.get("answer"),
        "constraints": doc.get("constraints"),
        "image_ref": _extract_image_ref(doc),
        "resp_key": _make_resp_key(item_id),
        "prediction": prediction,
    }
    if not JUDGE_TEXT_ONLY:
        payload["image"] = doc.get("image")
    return {
        "mmifeval_overall_accuracy": payload,
        "mmifeval_p_level_accuracy": payload,
        "mmifeval_c_level_accuracy": payload,
    }


def mmifeval_process_results_json(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    parsed = None
    if isinstance(raw_prediction, dict):
        parsed = raw_prediction
    elif isinstance(raw_prediction, str):
        try:
            parsed = json.loads(raw_prediction)
        except Exception:
            parsed = None
    json_answer = parsed.get("answer") if isinstance(parsed, dict) else raw_prediction
    prediction = ""
    if json_answer is not None:
        prediction = str(json_answer).strip()
    prediction = extract_final_answer(prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)
    item_id = doc.get("id")

    payload = {
        "id": item_id,
        "tag": doc.get("tag"),
        "infer_type": doc.get("infer_type"),
        "del_cons": doc.get("del_cons"),
        "question": doc.get("question"),
        "instruction": doc.get("instruction"),
        "answer": doc.get("answer"),
        "constraints": doc.get("constraints"),
        "image_ref": _extract_image_ref(doc),
        "resp_key": _make_resp_key(item_id),
        "prediction": prediction,
    }
    if not JUDGE_TEXT_ONLY:
        payload["image"] = doc.get("image")
    return {
        "mmifeval_overall_accuracy": payload,
        "mmifeval_p_level_accuracy": payload,
        "mmifeval_c_level_accuracy": payload,
    }


def _mark_failed_state(state: Dict[str, Any], message: str) -> None:
    if state.get("ret_code", 0) != 0:
        return
    state["ret_code"] = 1
    state["msg"] = message
    state["score_dict"] = {}


def _score_direct_gpt_response(
    state: Dict[str, Any],
    item: Dict[str, Any],
    prompt: str,
    constraints: List[Dict[str, Any]],
    gpt_resp: str,
) -> None:
    try:
        direct_gpt_score_dict = extract_score_from_direct_gpt_resp(gpt_resp)
        score_dict = state["score_dict"]
        score_dict["gpt_resp_direct_gpt"] = gpt_resp
        for i, constraint in enumerate(constraints):
            score_dict[constraint["key"]] = direct_gpt_score_dict[f"constraint_{i + 1}"]
    except Exception as e:
        eval_logger.error(
            f"\nError:\n{e}\nItem:\n{item}\npt_direct_gpt:\n{prompt}\ngpt_resp:\n{gpt_resp}\n"
        )
        _mark_failed_state(state, "C-Level, direct_gpt, fail in extract score")


def _compute_mmifeval_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _EVAL_CACHE
    if _EVAL_CACHE:
        return _EVAL_CACHE

    data_all = [item for item in results if isinstance(item, dict)]

    main_data = []
    aux_data = []
    for line in data_all:
        if line.get("infer_type", None) == "main":
            main_data.append(line)
        else:
            aux_data.append(line)

    aux_data_local = {}
    for line in aux_data:
        if line.get("infer_type") != "aux_cmp_gpt":
            continue
        del_cons = line.get("del_cons")
        if not del_cons:
            continue
        if line.get("id") is None:
            continue
        if line["id"] not in aux_data_local:
            aux_data_local[line["id"]] = {}
        aux_data_local[line["id"]][del_cons] = line.get("prediction", "")

    global aux_data_dict
    aux_data_dict = aux_data_local

    eval_logger.info(
        f"MMIFEval judge batching: batch_size={MMIFEVAL_JUDGE_BATCH_SIZE}, "
        f"text_only={JUDGE_TEXT_ONLY}, use_local_judge={USE_LOCAL_JUDGE}"
    )

    states: List[Dict[str, Any]] = []
    p_level_jobs: List[Tuple[Dict[str, Any], str]] = []
    direct_text_jobs: List[Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]] = []
    direct_mm_jobs: List[Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]] = []

    for item in main_data:
        state = {
            "id": item.get("id"),
            "tag": item.get("tag"),
            "item": item,
            "ret_code": 0,
            "msg": "success",
            "score_dict": {},
            "constraint_other": [],
        }
        states.append(state)

        if item.get("tag") == "P-Level":
            pt = generate_eval_pt_p_level(item.get("question"), item.get("prediction"), _safe_json_loads(item.get("answer"), []))
            p_level_jobs.append((state, pt))
            continue

        constraints_all = _safe_json_loads(item.get("constraints"), [])
        constraint_direct_gpt = []
        constraint_other = []
        for constraint in constraints_all:
            if not isinstance(constraint, dict):
                continue
            judge_cfg = constraint.get("judge") or {}
            method = judge_cfg.get("method") if isinstance(judge_cfg, dict) else None
            if method == "direct_gpt":
                constraint_direct_gpt.append(constraint)
            else:
                constraint_other.append(constraint)
        state["constraint_other"] = constraint_other

        if constraint_direct_gpt:
            pt_direct_gpt = generate_eval_pt_c_level(constraint_direct_gpt, item.get("prediction"))
            if JUDGE_TEXT_ONLY:
                direct_text_jobs.append((state, item, pt_direct_gpt, constraint_direct_gpt))
            else:
                image = _get_cached_image(item)
                if not image:
                    eval_logger.warning("MMIFEval: missing image for judge prompt, falling back to text-only")
                    direct_text_jobs.append((state, item, pt_direct_gpt, constraint_direct_gpt))
                else:
                    if USE_LOCAL_JUDGE:
                        image_url = _coerce_image_url(image)
                        if not image_url:
                            eval_logger.error(
                                f"\nError:\nMMIFEval: failed to build image URL for VLM judge.\n"
                                f"Item:\n{item}\npt_direct_gpt:\n{pt_direct_gpt}\n"
                            )
                            _mark_failed_state(state, "C-Level, direct_gpt, fail in extract score")
                            continue
                        messages = _build_user_message_with_image(pt_direct_gpt, image_url)
                    else:
                        messages = _build_user_message_with_image(pt_direct_gpt, image)
                    direct_mm_jobs.append((state, item, pt_direct_gpt, constraint_direct_gpt, messages))

    if p_level_jobs:
        p_level_prompts = [prompt for _, prompt in p_level_jobs]
        p_level_responses = _run_without_image_batch(p_level_prompts, retry=3)
        for (state, prompt), gpt_resp in zip(p_level_jobs, p_level_responses):
            if state.get("ret_code", 0) != 0:
                continue
            try:
                score = extract_score_from_p_level_gpt_resp(gpt_resp)
                state["score_dict"] = {
                    "total_score": score,
                    "gpt_resp": gpt_resp,
                }
            except Exception as e:
                eval_logger.error(f"\nError:\n{e}\nItem:\n{state['item']}\ngpt_resp:\n{gpt_resp}\nPrompt:\n{prompt}\n")
                _mark_failed_state(state, "P-Level, fail in extract score")

    if direct_text_jobs:
        direct_text_prompts = [prompt for _, _, prompt, _ in direct_text_jobs]
        direct_text_responses = _run_without_image_batch(direct_text_prompts, retry=3)
        for (state, item, prompt, constraints), gpt_resp in zip(direct_text_jobs, direct_text_responses):
            if state.get("ret_code", 0) != 0:
                continue
            _score_direct_gpt_response(state, item, prompt, constraints, gpt_resp)

    if direct_mm_jobs:
        direct_mm_messages = [messages for _, _, _, _, messages in direct_mm_jobs]
        direct_mm_responses = _run_with_messages_batch(direct_mm_messages, retry=3)
        for (state, item, prompt, constraints, _), gpt_resp in zip(direct_mm_jobs, direct_mm_responses):
            if state.get("ret_code", 0) != 0:
                continue
            _score_direct_gpt_response(state, item, prompt, constraints, gpt_resp)

    cmp_jobs: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]] = []
    for state in states:
        if state.get("ret_code", 0) != 0 or state.get("tag") == "P-Level":
            continue
        item = state["item"]
        score_dict = state["score_dict"]
        for constraint in state.get("constraint_other", []):
            judge_cfg = constraint.get("judge") or {}
            method = judge_cfg.get("method") if isinstance(judge_cfg, dict) else None
            if method == "rule_based":
                score = 1.0
                for func_dict in judge_cfg.get("verify_funcs", []):
                    func = globals()[func_dict["func"]]
                    judge_result = func(item.get("prediction"), *func_dict["params"])
                    if not judge_result:
                        score = 0.0
                        break
                score_dict[constraint["key"]] = score
        for constraint in state.get("constraint_other", []):
            judge_cfg = constraint.get("judge") or {}
            method = judge_cfg.get("method") if isinstance(judge_cfg, dict) else None
            if method != "cmp_gpt":
                continue
            item_id = item.get("id")
            aux_prediction = aux_data_dict.get(item_id, {}).get(constraint["key"], None)
            if aux_prediction is None:
                eval_logger.error(
                    f"\nError:\nMissing aux_cmp_gpt prediction\nItem:\n{item}\nConstraint:\n{constraint}\n"
                )
                _mark_failed_state(state, "C-Level, cmp_gpt, fail in extract score")
                break
            prompt = generate_cmp_pt(constraint["value"], item.get("prediction"), aux_prediction)
            cmp_jobs.append((state, item, constraint, prompt))

    if cmp_jobs:
        cmp_prompts = [prompt for _, _, _, prompt in cmp_jobs]
        cmp_responses = _run_without_image_batch(cmp_prompts, retry=3)
        for (state, item, constraint, _), gpt_resp in zip(cmp_jobs, cmp_responses):
            if state.get("ret_code", 0) != 0:
                continue
            try:
                score = extract_score_from_cmp_gpt_resp(gpt_resp)
                state["score_dict"][constraint["key"]] = score
                state["score_dict"][f"gpt_resp_cmp_gpt_{constraint['key']}"] = gpt_resp
            except Exception as e:
                eval_logger.error(f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}")
                _mark_failed_state(state, "C-Level, cmp_gpt, fail in extract score")

    scored_items = []
    for state in states:
        if state.get("ret_code", 0) == 0 and state.get("tag") != "P-Level":
            score_dict = state["score_dict"]
            total_score = 0.0
            cnt = 0
            for key, value in score_dict.items():
                if key.startswith("gpt_resp_"):
                    continue
                total_score += value
                cnt += 1
            score_dict["total_score"] = total_score / cnt if cnt else 0.0
            eval_logger.info(f"score_dict:\n{score_dict}")
        scored_items.append(
            {
                "id": state.get("id"),
                "tag": state.get("tag"),
                "eval_ret_code": state.get("ret_code", 1),
                "eval_msg": state.get("msg", "failed"),
                "eval_score_dict": state.get("score_dict", {}),
            }
        )

    individual_scores = {}
    judge_failed_cnt = 0
    p_level_score_sum = 0.0
    c_level_score_sum = 0.0
    p_level_cnt = 0
    c_level_cnt = 0

    for line in scored_items:
        eval_score_dict = line.get("eval_score_dict", {})
        total_score = eval_score_dict.get("total_score", 0.0) if isinstance(eval_score_dict, dict) else 0.0
        if line.get("eval_ret_code", 1) != 0:
            judge_failed_cnt += 1

        resp_key = _make_resp_key(line.get("id"))
        if resp_key:
            constraint_scores = {}
            if isinstance(eval_score_dict, dict):
                for key, value in eval_score_dict.items():
                    if key.startswith("gpt_resp_"):
                        continue
                    constraint_scores[key] = value
            individual_scores[resp_key] = {
                "score": total_score,
                "tag": line.get("tag"),
                "eval_ret_code": line.get("eval_ret_code"),
                "eval_msg": line.get("eval_msg"),
                "constraint_scores": constraint_scores,
            }

        if line.get("tag") == "P-Level":
            p_level_score_sum += total_score
            p_level_cnt += 1
        elif line.get("tag") == "C-Level":
            c_level_score_sum += total_score
            c_level_cnt += 1

    p_level_accuracy = p_level_score_sum / p_level_cnt if p_level_cnt else 0.0
    c_level_accuracy = c_level_score_sum / c_level_cnt if c_level_cnt else 0.0
    overall_accuracy = (
        (p_level_accuracy * p_level_cnt + c_level_accuracy * c_level_cnt) / (p_level_cnt + c_level_cnt)
        if (p_level_cnt + c_level_cnt) > 0
        else 0.0
    )

    _EVAL_CACHE = {
        "p_level_accuracy": p_level_accuracy,
        "c_level_accuracy": c_level_accuracy,
        "overall_accuracy": overall_accuracy,
        "details": scored_items,
        "individual_scores": individual_scores,
        "judge_failed_cnt": judge_failed_cnt,
        "p_level_cnt": p_level_cnt,
        "c_level_cnt": c_level_cnt,
    }
    if judge_failed_cnt > 0:
        eval_logger.warning(f"MMIFEval judge failed on {judge_failed_cnt}/{len(scored_items)} items.")
    return _EVAL_CACHE


# ============================
# Aggregation Functions
# ============================


def mmifeval_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    scores = _compute_mmifeval_scores(results)
    mmifeval_aggregate_overall.details = scores.get("details", [])
    mmifeval_aggregate_overall.individual_scores = scores.get("individual_scores", {})
    return scores.get("overall_accuracy", 0.0)


def mmifeval_aggregate_p_level(results: List[Dict[str, Any]]) -> float:
    scores = _compute_mmifeval_scores(results)
    mmifeval_aggregate_p_level.details = scores.get("details", [])
    mmifeval_aggregate_p_level.individual_scores = scores.get("individual_scores", {})
    return scores.get("p_level_accuracy", 0.0)


def mmifeval_aggregate_c_level(results: List[Dict[str, Any]]) -> float:
    scores = _compute_mmifeval_scores(results)
    mmifeval_aggregate_c_level.details = scores.get("details", [])
    mmifeval_aggregate_c_level.individual_scores = scores.get("individual_scores", {})
    return scores.get("c_level_accuracy", 0.0)
