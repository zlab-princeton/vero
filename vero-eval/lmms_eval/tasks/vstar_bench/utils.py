import re
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.filters.extraction import RegexFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig


def vstar_doc_to_visual(doc):
    """Convert document to visual input."""
    return [doc["image"].convert("RGB")]


def vstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt with options."""
    question = doc["text"]

    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("strip_answer_instruction"):
        answer_instruction = "Answer with the option's letter from the given choices directly."
        question = question.replace(f"\n{answer_instruction}", "")
        question = question.replace(answer_instruction, "")
        question = question.strip()

    # Extract options from the question text
    # The format is typically: "Question text? (A) option1 (B) option2 (C) option3 (D) option4"
    options_match = re.findall(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)", question)

    if options_match:
        # Clean up the question text (remove the options part)
        question_text = question.split("(A)")[0].strip()

        # Format options
        option_letters = ["A", "B", "C", "D"]
        options_formatted = []
        for i, option in enumerate(options_match):
            if i < len(option_letters):
                options_formatted.append(f"{option_letters[i]}. {option.strip()}")

        # Reconstruct the question with properly formatted options
        question = f"{question_text}\n" + "\n".join(options_formatted)

    # Add pre-prompt and post-prompt if specified
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
            question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
            question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"

    return question


def _strip_reasoning_tags(text: str) -> str:
    return extract_final_answer(text)


def extract_answer_letter(response):
    """Extract the answer letter from model response."""
    if not isinstance(response, str):
        response = "" if response is None else str(response)
    response = _strip_reasoning_tags(response)
    response = response.strip().upper()

    if math_verify_parse and StringExtractionConfig:
        try:
            parsed = math_verify_parse(response, extraction_config=[StringExtractionConfig(strings=("A", "B", "C", "D"))])
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice:
                parsed_choice = parsed_choice.upper()
                if parsed_choice in ("A", "B", "C", "D"):
                    return parsed_choice

    # Try to find patterns like "A", "(A)", "A.", "A)", "Answer: A", etc.
    patterns = [
        r"^([A-D])\s*[\.)\]]*",  # Starts with letter
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",  # Answer: A or The answer is A format
        r"\(([A-D])\)",  # (A) format
        r"([A-D])\s*(?:\.|\)|])",  # A. or A) or A] format
        r"(?:^|\s)([A-D])(?:\s|$)",  # Standalone letter
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If no pattern matches, check if response contains only one letter A-D
    letters = re.findall(r"[A-D]", response)
    if len(letters) == 1:
        return letters[0]

    # Default to first character if it's A-D
    if response and response[0] in "ABCD":
        return response[0]

    return None


class FinalAnswerLetterFilter(RegexFilter):
    """Extract answer letter from <answer> tags or final line."""

    def __init__(
        self,
        regex_pattern: str = r"(?s)<answer>.*?([A-D]).*?</answer>|(?s)</think>.*?([A-D])\s*$|(?:^|\s)([A-D])(?:\s|$)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        super().__init__(regex_pattern=regex_pattern, group_select=group_select, fallback=fallback)


def vstar_process_results(doc, results):
    """
    Process the model results and compare with ground truth.

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0] if results else ""

    # Extract predicted answer letter
    pred_letter = extract_answer_letter(pred)

    # The label in the dataset should be the correct answer letter (A, B, C, or D)
    gt_letter = doc["label"].strip().upper()

    # Score 1 if correct, 0 otherwise
    score = 1.0 if pred_letter == gt_letter else 0.0

    # Get category for aggregation
    category = doc.get("category", "unknown")

    # Log for debugging
    if score == 0:
        eval_logger.debug(f"Question: {doc['text'][:100]}...")
        eval_logger.debug(f"Predicted: {pred_letter}, Ground Truth: {gt_letter}")
        eval_logger.debug(f"Raw prediction: {pred}")

    # Return metrics for different aggregations
    result = {"question_id": doc["question_id"], "category": category, "score": score, "prediction": pred_letter, "ground_truth": gt_letter}

    return {f"vstar_{category}_acc": result, "vstar_overall_acc": result}


def vstar_aggregate_results(results):
    """
    Aggregate results by category and overall.

    Args:
        results: a list of values returned by process_results
    Returns:
        Aggregated accuracy score
    """
    if not results:
        return 0.0

    # Group by category
    category_scores = defaultdict(list)
    all_scores = []

    for result in results:
        category = result["category"]
        score = result["score"]
        category_scores[category].append(score)
        all_scores.append(score)

    # Calculate accuracy for each category
    for category, scores in category_scores.items():
        if scores:
            acc = sum(scores) / len(scores)
            eval_logger.info(f"{category}: {acc:.4f} (n={len(scores)})")

    # Calculate overall accuracy
    if all_scores:
        overall_acc = sum(all_scores) / len(all_scores)
        eval_logger.info(f"Overall: {overall_acc:.4f} (n={len(all_scores)})")
        return overall_acc

    return 0.0
