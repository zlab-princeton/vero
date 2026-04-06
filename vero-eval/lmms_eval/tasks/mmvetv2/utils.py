import os
import re
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont

from lmms_eval.llm_judge import Request, ServerConfig, get_server
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine


@lru_cache(maxsize=1)
def _resolve_label_font_path():
    task_dir = Path(__file__).resolve().parent
    candidates = [
        task_dir / "arial.ttf",
        Path(ImageFont.__file__).resolve().parent / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=1)
def _log_font_fallback_once():
    eval_logger.warning(
        "No TrueType label font found for mmvetv2 image grouping. Falling back to PIL default font."
    )
    return True


def add_order_label(image, label, font_size=40):
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Prefer task-local arial.ttf, then bundled/system DejaVu, then PIL default.
    font_path = _resolve_label_font_path()
    if font_path is None:
        _log_font_fallback_once()
        font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except OSError:
            _log_font_fallback_once()
            font = ImageFont.load_default()

    # Calculate text size and position
    text_width = text_height = font_size
    label_background_margin = 10
    label_background_size = (
        text_width + 2 * label_background_margin,
        text_height + 2 * label_background_margin,
    )

    # Draw a solid white square for the label background
    label_background_position = (0, 0)  # Top-left corner
    draw.rectangle(
        [
            label_background_position,
            (
                label_background_position[0] + label_background_size[0],
                label_background_position[1] + label_background_size[1],
            ),
        ],
        fill="white",
    )

    # Add the label text in black over the white square
    label_position = (label_background_margin, label_background_margin)
    draw.text(label_position, label, font=font, fill="black")

    return image


def resize_image_height(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = int(width * fixed_size / height), fixed_size
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_horizontal(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    assert all(height == max_height for height in heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in image_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def resize_image_width(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = fixed_size, int(height * fixed_size / width)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_vertical(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_height = sum(heights)
    max_width = max(widths)
    assert all(width == max_width for width in widths)
    new_im = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def process_images_horizontal(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_height(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_horizontal(images)


def process_images_vertical(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_width(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_vertical(images)


def process_images(images, size=1008):
    concat_horizontal = process_images_horizontal(images, size)
    concat_vertical = process_images_vertical(images, size)

    hw, hh = concat_horizontal.size
    vw, vh = concat_vertical.size

    ha = hw / hh
    va = vh / vw

    if ha > va:
        return concat_vertical
    else:
        return concat_horizontal


def mmvet_group_img_doc_to_visual(doc):
    prompt = doc["question"]
    image_tokens = re.findall(r"<image_\d+>", prompt)
    visual = [doc[image_token.strip("<>")].convert("RGB") for image_token in image_tokens]
    visual = process_images(visual)
    return [visual]


def mmvet_doc_to_visual(doc):
    prompt = doc["question"]
    image_tokens = re.findall(r"<image_\d+>", prompt)
    visual = [doc[image_token.strip("<>")].convert("RGB") for image_token in image_tokens]
    return visual


def replace_images_tokens(input_string):
    if config["metadata"]["interleaved_format"]:
        for i in range(0, 18):
            question_text = f"<image_{i}>"
            query_text = "<image>"
            if question_text in input_string:
                input_string = input_string.replace(question_text, query_text)
    queries = input_string.split("<IMG>")
    return "".join(queries)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        return replace_images_tokens(doc["question"])
    question = replace_images_tokens(doc["question"])
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"


with open(Path(__file__).parent / "mmvetv2.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
JUDGE_MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
MMVETV2_JUDGE_BACKEND = os.getenv("MMVETV2_JUDGE_BACKEND", "vllm").strip().lower()
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_score_prompt.txt"


@lru_cache(maxsize=1)
def _get_legacy_server():
    server_config = ServerConfig(model_name=MODEL_VERSION, temperature=0.0, max_tokens=128)
    return get_server(server_name=API_TYPE, config=server_config)


@lru_cache(maxsize=1)
def _get_vllm_engine():
    return get_judge_engine(JUDGE_MODEL_HINT or MODEL_VERSION)


@lru_cache(maxsize=1)
def _load_judge_prompt():
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _format_judge_prompt(question, answer, prediction):
    template = _load_judge_prompt()
    prompt = template.replace("[QUESTION]", str(question or ""))
    prompt = prompt.replace("[GROUND_TRUTH]", str(answer or ""))
    prompt = prompt.replace("[PREDICTION]", str(prediction or ""))
    return prompt


def _extract_score(content):
    text = str(content or "").strip()
    if not text:
        return None

    parsed = extract_json_candidate(text)
    if isinstance(parsed, dict):
        for key in ("score", "correctness", "result", "value"):
            if key not in parsed:
                continue
            value = parsed.get(key)
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= score <= 1.0:
                return score

    first_token = text.split(" ")[0].strip()
    try:
        score = float(first_token)
        if 0.0 <= score <= 1.0:
            return score
    except ValueError:
        pass

    # Also accept wrappers like "Score: 0.7".
    matches = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    for match in matches:
        try:
            score = float(match)
        except ValueError:
            continue
        if 0.0 <= score <= 1.0:
            return score
    return None


def _normalize_prediction(raw_prediction):
    prediction = extract_final_answer(raw_prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)
    return prediction


def _get_chat_response_vllm(
    prompt,
    model=MODEL_VERSION,
    temperature=0.0,
    max_tokens=128,
    patience=3,
    sleep_time=5,
):
    del temperature  # governed by judge engine configuration
    model_name = JUDGE_MODEL_HINT or model
    try:
        engine = _get_vllm_engine()
    except Exception as e:
        eval_logger.error(f"Failed to initialize vLLM judge engine: {e}")
        return "", model_name

    while patience > 0:
        patience -= 1
        try:
            responses = engine.generate_json_batch([prompt], max_tokens=max_tokens, use_tqdm=False)
            content = responses[0].strip() if responses and responses[0] else ""
            if content:
                return content, model_name
        except Exception as e:
            eval_logger.error(f"vLLM judge error: {e}")
            if "rate limit" in str(e).lower():
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying vLLM judge...Patience left: {patience}")

    return "", model_name


def _get_chat_response_legacy(
    prompt,
    model=MODEL_VERSION,
    temperature=0.0,
    max_tokens=128,
    patience=3,
    sleep_time=5,
):
    custom_config = ServerConfig(model_name=model, temperature=temperature, max_tokens=max_tokens)
    server = _get_legacy_server()

    while patience > 0:
        patience -= 1
        try:
            request = Request(messages=[{"role": "user", "content": prompt}], config=custom_config)
            response = server.evaluate(request)
            content = response.content.strip() if response.content else ""
            if content:
                return content, response.model_used
        except Exception as e:
            eval_logger.error(f"Legacy judge error: {e}")
            if "rate limit" in str(e).lower():
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying legacy judge...Patience left: {patience}")

    return "", ""


def get_chat_response(
    prompt,
    model=MODEL_VERSION,
    temperature=0.0,
    max_tokens=128,
    patience=3,
    sleep_time=5,
):
    backend = MMVETV2_JUDGE_BACKEND
    if backend == "legacy":
        return _get_chat_response_legacy(prompt, model=model, temperature=temperature, max_tokens=max_tokens, patience=patience, sleep_time=sleep_time)

    if backend not in {"vllm", "auto"}:
        eval_logger.warning(f"Unknown MMVETV2_JUDGE_BACKEND={backend}, defaulting to vllm.")
        backend = "vllm"

    content, model_name = _get_chat_response_vllm(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        patience=patience,
        sleep_time=sleep_time,
    )
    if content:
        return content, model_name

    if backend == "vllm":
        raise RuntimeError(
            "MMVETV2_JUDGE_BACKEND is set to 'vllm', but vLLM judge returned no output. "
            "Fix vLLM judge configuration (e.g., JUDGE_MODEL_PATH, model availability, server/runtime) "
            "or switch MMVETV2_JUDGE_BACKEND to 'auto'/'legacy'."
        )

    eval_logger.info("vLLM judge returned empty output; falling back to legacy judge.")
    return _get_chat_response_legacy(prompt, model=model, temperature=temperature, max_tokens=max_tokens, patience=patience, sleep_time=sleep_time)


def mmvet_process_results(doc, results):
    # get pred and ground truth here
    raw_prediction = results[0] if isinstance(results, (list, tuple)) and results else results
    pred = _normalize_prediction(raw_prediction)

    # Phase 1: skip judge, store raw response for later Phase 2 judging
    if os.environ.get("SKIP_JUDGE_IN_PROCESS_RESULTS") == "1":
        return {
            "gpt_eval_score": {
                "question_id": doc["id"],
                "question": doc["question"],
                "gt_answer": doc["answer"],
                "capabilities": doc["capability"],
                "pred_answer": pred,
                "score": -1.0,
                "eval_model": "",
                "_skipped_judge": True,
                "_response": pred,
            }
        }

    question = doc["question"]
    answer = doc["answer"]
    gpt_query_prompt = _format_judge_prompt(
        question=question,
        answer=answer.replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
        prediction=pred,
    )
    grade_sample_run_complete = False
    temperature = 0.0
    model_name = ""

    while not grade_sample_run_complete:
        content, model_name = get_chat_response(
            gpt_query_prompt,
            temperature=temperature,
        )
        if content:
            score = _extract_score(content)
            if score is not None:
                grade_sample_run_complete = True
            else:
                time.sleep(5)
                temperature += 0.5
                eval_logger.info(f"Sleep 5 secs, {doc['question_id']} try again with increased temperature {temperature}.")
                content, model_name = get_chat_response(
                    gpt_query_prompt,
                    temperature=temperature,
                )
                score = _extract_score(content)
                if score is not None:
                    grade_sample_run_complete = True
                if temperature >= 2:  # Assuming a max temperature threshold
                    score = 0.0
                    grade_sample_run_complete = True
                    eval_logger.info(f"Reach to max trials, {doc['question_id']} failed to get a score.")
        else:
            score = 0.0
            grade_sample_run_complete = True
            eval_logger.info(f"{doc['question_id']} failed to get a score.")

    return {
        f"gpt_eval_score": {
            "question_id": doc["id"],
            "question": doc["question"],
            "gt_answer": doc["answer"],
            "capabilities": doc["capability"],
            "pred_answer": pred,
            "score": score,
            "eval_model": model_name,
        }
    }


cap_columns = pd.DataFrame(["rec", "ocr", "know", "gen", "spat", "math", "seq"])
cap_details_columns = pd.DataFrame(
    [
        "rec_know_gen",
        "rec",
        "rec_spat_ocr_gen",
        "rec_gen",
        "ocr",
        "rec_spat",
        "spat_ocr",
        "rec_spat_gen",
        "math_spat_ocr",
        "rec_seq_gen",
        "ocr_gen",
        "rec_know",
        "rec_know_ocr_gen",
        "rec_spat_ocr",
        "math_ocr",
        "rec_know_spat",
        "rec_spat_seq_gen",
        "rec_spat_seq",
        "rec_seq_ocr_gen",
        "rec_seq",
        "rec_ocr_gen",
        "rec_ocr",
        "rec_know_spat_ocr",
        "know_spat_ocr",
        "seq_ocr_gen_rec_spat",
        "rec_seq_spat_ocr",
        "rec_know_spat_gen",
        "spat_ocr_gen",
        "rec_know_math",
        "ocr_gen_rec_know_spat",
        "rec_know_seq_gen",
        "rec_math_spat_ocr",
        "seq_ocr_gen_rec_know",
        "math_spat_ocr_gen",
        "rec_math_ocr",
        "spat_seq_ocr_rec_math",
        "spat_seq_ocr_gen_math",
        "rec_know_seq",
        "rec_seq_ocr",
    ]
)


def mmvet_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # Calculate the overall score
    overall_score = sum([result["score"] for result in results]) / len(results)
    eval_logger.info(f"Overall Score: {overall_score:.4f}")

    # Initialize dictionaries to store scores for each capability and detail
    cap_scores = {cap: 0 for cap in cap_columns.squeeze().tolist()}
    cap_details_scores = {detail: 0 for detail in cap_details_columns.squeeze().tolist()}

    # Count the number of results for each capability and detail
    cap_counts = {cap: 0 for cap in cap_scores}
    cap_details_counts = {detail: 0 for detail in cap_details_scores}

    # Aggregate scores for each capability and detail
    for result in results:
        for cap in cap_scores:
            if cap in result["capabilities"]:
                cap_scores[cap] += result["score"]
                cap_counts[cap] += 1
        for detail in cap_details_scores:
            detail_set = set(detail.split("_"))
            result_detail_set = set(result["capabilities"])
            if detail_set == result_detail_set:
                cap_details_scores[detail] += result["score"]
                cap_details_counts[detail] += 1

    # Calculate the average score for each capability
    for cap in cap_scores:
        if cap_counts[cap] > 0:
            cap_scores[cap] = cap_scores[cap] / cap_counts[cap] * 100
        eval_logger.info(f"Score for {cap}: {cap_scores[cap]:.2f}")

    # Calculate the average score for each detailed capability
    for detail in cap_details_scores:
        if cap_details_counts[detail] > 0:
            cap_details_scores[detail] = cap_details_scores[detail] / cap_details_counts[detail] * 100
        eval_logger.info(f"Score for {detail}: {cap_details_scores[detail]:.2f}")

    return overall_score
