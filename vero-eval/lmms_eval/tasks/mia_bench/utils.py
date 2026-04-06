import os
import time
import io
import base64
from pathlib import Path

import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import get_judge_engine, judge_supports_multimodal, resolve_judge_mode

def mia_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mia_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_text = doc["instruction"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""

    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    formatted_question = f"{pre_prompt}{question_text}{post_prompt}"

    return formatted_question


# ============================
# Result Processing Functions
# ============================

with open(Path(__file__).parent / "mia_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

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

eval_logger.info(
    f"MIA judge routing: mode={JUDGE_MODE}, text_only={JUDGE_TEXT_ONLY}, use_local_judge={USE_LOCAL_JUDGE}"
)

if API_TYPE_LOWER == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE_LOWER == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {"api-key": API_KEY, "Content-Type": "application/json", "api-version": "2023-07-01-preview"}
else:
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("GPT_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


_local_engine = None


def _get_local_engine():
    global _local_engine
    if _local_engine is None:
        _local_engine = get_judge_engine(JUDGE_MODEL_HINT)
    return _local_engine


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


def _run_local_messages(messages, max_tokens: int, retries: int) -> str:
    model_used = JUDGE_MODEL_HINT
    for attempt in range(retries):
        try:
            engine = _get_local_engine()
            if not judge_supports_multimodal(engine):
                raise RuntimeError(
                    "MIA VLM judge requires a multimodal backend. "
                    "Use JUDGE_BACKEND=engine with a multimodal model, or set VLLM_SERVER_JUDGE=1."
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


def _encode_image_data_url(image):
    if image is None:
        return None
    if isinstance(image, str):
        if image.startswith("data:") or image.startswith("http"):
            return image
        if os.path.exists(image):
            try:
                from PIL import Image as PILImage
                image = PILImage.open(image).convert("RGB")
            except Exception:
                return None
        else:
            payload = image.strip()
            if payload:
                payload = "".join(payload.split())
                payload += "=" * (-len(payload) % 4)
                try:
                    base64.b64decode(payload, validate=False)
                    return f"data:image/jpeg;base64,{payload}"
                except Exception:
                    return None
            return None
    if isinstance(image, bytes):
        return "data:image/jpeg;base64," + base64.b64encode(image).decode("utf-8")
    if hasattr(image, "convert"):
        try:
            image = image.convert("RGB")
        except Exception:
            return None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None


def _build_messages(content: str, image=None):
    if JUDGE_TEXT_ONLY or image is None:
        return [{"role": "user", "content": content}]
    image_url = _encode_image_data_url(image)
    if not image_url:
        return [{"role": "user", "content": content}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def get_eval(content: str, max_tokens: int, retries: int = 5, image=None):
    global headers

    if USE_LOCAL_JUDGE:
        if JUDGE_TEXT_ONLY:
            return _run_local(content, max_tokens=max_tokens, retries=retries), JUDGE_MODEL_HINT
        messages = _build_messages(content, image=image)
        if image is not None and len(messages) > 0 and isinstance(messages[0].get("content"), list):
            return _run_local_messages(messages, max_tokens=max_tokens, retries=retries), JUDGE_MODEL_HINT
        return _run_local(content, max_tokens=max_tokens, retries=retries), JUDGE_MODEL_HINT

    messages = _build_messages(content, image=image)

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def generate_prompt(d, response):
    instruction = d["instruction"]
    weight = d["component_weight"] * 1
    d["num_of_component"] = len(d["components"])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d["num_of_component"] == 1:
        components = """The first component is:' """ + d["components"][0] + "'"
        score = """The first component is worth """ + weight[0] + " scores."
    elif d["num_of_component"] == 2:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + "'"
        score = """The first and second components are each worth """ + weight[0] + " and " + weight[1] + " scores."
    elif d["num_of_component"] == 3:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + """', and the third component is:' """ + d["components"][2] + "'"
        score = """The first, second, and third components are each worth """ + weight[0] + ", " + weight[1] + ", and " + weight[2] + " scores."
    elif d["num_of_component"] == 4:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + "'"
        )
        score = """The first, second, third, and fourth components are each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + ", and " + weight[3] + " scores."
    elif d["num_of_component"] == 5:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + """', and the fifth component is:' """
            + d["components"][4]
            + "'"
        )
        score = """The first, second, third, fourth, and fifth components are each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + ", " + weight[3] + ", and " + weight[4] + " scores."
    return (
        """Here is an instruction for a multimodal LLM: ' """
        + instruction
        + """ You need to grade if the response from the model follows each component of the instruction. """
        + components
        + """ The response is:' """
        + response
        + """' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. """
        + score
        + " List scores of each component, and the total score in one sentence in this EXACT format: "
        + ", ".join(f"score of component {i + 1}: ?/{weight[i]}" for i in range(d["num_of_component"]))
        + ", total score: ?/10. Replace each ? with the actual numeric score. Do not use markdown formatting or asterisks. Then explain your reasons."
    )


def process_rawscore(component_type, raw_score):
    import re

    from loguru import logger as eval_logger

    score_dict = {}

    try:
        # Validate inputs
        if not component_type or not isinstance(component_type, list):
            eval_logger.error(f"Invalid component_type: {component_type}")
            return {"total_score": 0}

        if not raw_score or not isinstance(raw_score, str):
            eval_logger.error(f"Invalid raw_score: {raw_score}")
            # Initialize with zeros for all components
            for component in component_type:
                score_dict[component] = 0
            score_dict["total_score"] = 0
            return score_dict

        # Extract only the score line (first sentence) to avoid matching
        # "component N:" mentions in the explanation text that follows.
        first_period = raw_score.find(".")
        score_line = raw_score[: first_period + 1] if first_period != -1 else raw_score

        # Support both "component N: a/b" and "constraint_N: a/b" judge styles.
        component_patterns = [
            r"(?:score\s+of\s+)?component\s+(\d+)\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
            r"(?:score\s+of\s+)?constraint[_\s]*(\d+)\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        ]

        # Pattern: "total score: a/b"
        total_pattern = r"total\s+score\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
        total_score_found = False
        parsed_component_scores = 0

        # Find component scores from score line only
        try:
            for component_pattern in component_patterns:
                component_matches = re.findall(component_pattern, score_line, re.IGNORECASE)
                for match in component_matches:
                    try:
                        component_num = int(match[0]) - 1  # Convert to 0-based index
                        if 0 <= component_num < len(component_type):
                            numerator = float(match[1].strip())
                            denominator = float(match[2].strip())
                            score = numerator / denominator if denominator != 0 else 0
                            # Clamp score between 0 and 1
                            score = max(0, min(1, score))
                            key = component_type[component_num]
                            if key not in score_dict:
                                parsed_component_scores += 1
                            score_dict[key] = score
                        else:
                            eval_logger.warning(f"Component number {component_num + 1} out of range for {len(component_type)} components")
                    except (ValueError, IndexError) as e:
                        eval_logger.warning(f"Error parsing component match {match}: {e}")
                        continue
        except Exception as e:
            eval_logger.error(f"Error in component pattern matching: {e}")

        # Find total score
        try:
            total_match = re.search(total_pattern, score_line, re.IGNORECASE)
            if total_match:
                total_numerator = float(total_match.group(1).strip())
                total_denominator = float(total_match.group(2).strip())
                total_score = total_numerator / total_denominator if total_denominator != 0 else 0
                # Clamp total score between 0 and 1
                total_score = max(0, min(1, total_score))
                score_dict["total_score"] = total_score
                total_score_found = True
            else:
                # Fallback: calculate total as average of component scores
                if score_dict:
                    total_score = sum(score_dict.values()) / len(score_dict)
                    score_dict["total_score"] = total_score
                else:
                    score_dict["total_score"] = 0
        except Exception as e:
            eval_logger.error(f"Error parsing total score: {e}")
            score_dict["total_score"] = 0

        # Ensure all components have scores.
        # Avoid warning spam when total score is already parsed and only breakdown is missing.
        missing_components = [component for component in component_type if component not in score_dict]
        if missing_components:
            if not total_score_found:
                if parsed_component_scores == 0:
                    eval_logger.warning("Failed to parse per-component scores from judge output; defaulting missing components to 0.")
                else:
                    eval_logger.warning("Partially parsed per-component scores; defaulting missing components to 0.")
            else:
                eval_logger.warning(
                    f"Total score parsed but {len(missing_components)} component scores missing; defaulting missing components to 0."
                )
            for component in missing_components:
                score_dict[component] = 0

        # Ensure total_score exists
        if "total_score" not in score_dict:
            if score_dict:
                score_dict["total_score"] = sum(v for k, v in score_dict.items() if k != "total_score") / len([k for k in score_dict.keys() if k != "total_score"])
            else:
                score_dict["total_score"] = 0

    except Exception as e:
        eval_logger.error(f"Unexpected error in process_rawscore: {e}")
        eval_logger.error(f"Raw score content: {raw_score[:500] if raw_score else 'None'}...")
        # Emergency fallback: return zeros
        score_dict = {}
        for component in component_type if component_type else []:
            score_dict[component] = 0
        score_dict["total_score"] = 0

    # Final validation
    try:
        # Ensure all values are numeric and within valid range
        for key, value in score_dict.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                eval_logger.warning(f"Invalid score for {key}: {value}, setting to 0")
                score_dict[key] = 0
    except Exception as e:
        eval_logger.error(f"Error in final validation: {e}")

    return score_dict


def mia_bench_process_results(doc, results):
    from loguru import logger as eval_logger

    try:
        # Validate inputs
        if not results or len(results) == 0:
            eval_logger.error("No results provided")
            return {"gpt_eval_score": {"total_score": 0}}

        if not doc or not isinstance(doc, dict):
            eval_logger.error(f"Invalid doc: {doc}")
            return {"gpt_eval_score": {"total_score": 0}}

        # Extract response safely
        try:
            response = results[0].strip() if results[0] else ""
            if not response:
                eval_logger.warning("Empty response from model")
                response = ""
        except (IndexError, AttributeError) as e:
            eval_logger.error(f"Error extracting response: {e}")
            response = ""
        response = extract_final_answer(response, parse_boxed=False, strip_latex_wrappers=True)
        response = truncate_response_tail_tiktoken(response)

        # Extract components safely
        try:
            components = doc.get("components", [])
            if not components or not isinstance(components, list):
                eval_logger.error(f"Invalid components in doc: {components}")
                return {"gpt_eval_score": {"total_score": 0}}
        except Exception as e:
            eval_logger.error(f"Error extracting components: {e}")
            return {"gpt_eval_score": {"total_score": 0}}

        # Generate evaluation prompt
        try:
            eval_prompt = generate_prompt(doc, response)
            if not eval_prompt:
                eval_logger.error("Failed to generate evaluation prompt")
                # Create fallback score_dict
                score_dict = {"total_score": 0}
                for component in components:
                    score_dict[component] = 0
                return {"gpt_eval_score": score_dict}
        except Exception as e:
            eval_logger.error(f"Error generating prompt: {e}")
            # Create fallback score_dict
            score_dict = {"total_score": 0}
            for component in components:
                score_dict[component] = 0
            return {"gpt_eval_score": score_dict}

        # Get evaluation from LLM
        try:
            eval_score, _ = get_eval(eval_prompt, 4096, image=doc.get("image"))
            if not eval_score or not isinstance(eval_score, str):
                eval_logger.warning(f"Invalid eval_score returned: {eval_score}")
                eval_score = ""
        except Exception as e:
            eval_logger.error(f"Error getting evaluation from LLM: {e}")
            eval_score = ""

        # Process the raw score
        try:
            score_dict = process_rawscore(components, eval_score)
            if not score_dict or not isinstance(score_dict, dict):
                eval_logger.error("process_rawscore returned invalid result")
                # Create fallback score_dict
                score_dict = {"total_score": 0}
                for component in components:
                    score_dict[component] = 0
        except Exception as e:
            eval_logger.error(f"Error processing raw score: {e}")
            # Create fallback score_dict
            score_dict = {"total_score": 0}
            for component in components:
                score_dict[component] = 0

        # Embed audit trail in score_dict (aggregation only reads total_score).
        weights = doc.get("component_weight", [])
        score_dict["_component_weights"] = dict(zip(components, weights)) if len(weights) == len(components) else weights
        score_dict["_judge_raw"] = eval_score

        return {"gpt_eval_score": score_dict}

    except Exception as e:
        eval_logger.error(f"Unexpected error in mia_bench_process_results: {e}")
        # Emergency fallback
        return {"gpt_eval_score": {"total_score": 0}}


def mia_bench_process_results_json(doc, results):
    import json

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
    prediction = "" if json_answer is None else str(json_answer)
    return mia_bench_process_results(doc, [prediction])


# ============================
# Aggregation Functions
# ============================


def mia_bench_aggregate_results(results):
    total_score = 0
    for result in results:
        # Overall accuracy
        total_score += result["total_score"]
    return total_score / len(results)
