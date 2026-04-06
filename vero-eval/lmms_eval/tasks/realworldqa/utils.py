import random
import re
from typing import Dict, List, Optional

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.answer_parsing import extract_answer_from_response

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."
_MCQ_OPTION_PATTERN = re.compile(r"(?:^|\s)[\(\[]?([A-D])[\)\]]?\s*[\.:\)]\s*", re.IGNORECASE)


def _extract_mcq_from_question(question: str):
    if not isinstance(question, str):
        return None, None
    matches = list(_MCQ_OPTION_PATTERN.finditer(question))
    if not matches:
        return None, None
    start_index = next((i for i, match in enumerate(matches) if match.group(1).upper() == "A"), None)
    if start_index is None:
        return None, None
    matches = matches[start_index:]
    options = []
    for i, match in enumerate(matches):
        letter = match.group(1).upper()
        if letter not in "ABCD":
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        option_text = question[start:end].strip()
        if option_text:
            options.append((letter, option_text))
    if len(options) < 2:
        return None, None
    question_text = question[: matches[0].start()].strip()
    return question_text, options


def _format_mcq_question(question: str) -> str:
    question_text, options = _extract_mcq_from_question(question)
    if not options:
        return question
    formatted_options = "\n".join([f"{letter}. {text}" for letter, text in options])
    if question_text:
        return f"{question_text}\nChoices:\n{formatted_options}"
    return f"Choices:\n{formatted_options}"


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = _format_mcq_question(question)
    return f"{pre_prompt}{question}\n{post_prompt}"


def realworldqa_doc_to_text_removepost(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "Please answer directly with a single word or number." in question:
        question = question.replace("Please answer directly with a single word or number.", "").strip()
    if "Please answer directly with only the letter of the correct option and nothing else." in question:
        question = question.replace("Please answer directly with only the letter of the correct option and nothing else.", "").strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = _format_mcq_question(question)
    return f"{pre_prompt}{question}\n{post_prompt}"
    


def _parse_multi_choice_response(
    response: str,
    all_choices: List[str],
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response.upper():
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response.upper():
                candidates.append(choice)

    if len(candidates) == 0 and index2ans and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return random.choice(all_choices) if all_choices else ""
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[max(range(len(start_indexes)), key=start_indexes.__getitem__)]
    else:
        pred_index = candidates[0]
    return pred_index


def _extract_answer_letter(
    text: str,
    option_labels: Optional[List[str]] = None,
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    if option_labels and math_verify_parse and StringExtractionConfig:
        try:
            parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=tuple(option_labels))])
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice:
                parsed_choice = parsed_choice.upper()
                if parsed_choice in option_labels:
                    return parsed_choice
    if option_labels:
        return _parse_multi_choice_response(text, option_labels, index2ans=index2ans)
    letter_match = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return ""


def _strip_option_prefix(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"^\s*[\(\[]?(?:[A-Z][\)\].:\-]?\s+|\d+[\)\].:\-]\s*)", "", text, flags=re.IGNORECASE)


def _build_option_labels(choices: Optional[List]) -> List[str]:
    if not isinstance(choices, (list, tuple)) or not choices:
        return []
    labels = []
    for i, item in enumerate(choices):
        label = None
        if isinstance(item, str):
            match = re.match(r"\s*[\(\[]?([A-Z])[\)\].:\s]", item, flags=re.IGNORECASE)
            if match:
                label = match.group(1).upper()
        labels.append(label or chr(65 + i))
    return labels


def _build_index2ans(option_labels: List[str], choices: Optional[List]) -> Dict[str, str]:
    if not option_labels or not isinstance(choices, (list, tuple)):
        return {}
    index2ans = {}
    for label, item in zip(option_labels, choices):
        index2ans[label] = _strip_option_prefix(item)
    return index2ans


def _build_extraction_question(doc) -> str:
    question = doc.get("question", "")
    if not isinstance(question, str):
        question = str(question)
    choices = doc.get("choices") or doc.get("options")
    if isinstance(choices, (list, tuple)) and choices:
        labels = _build_option_labels(list(choices))
        formatted = []
        for label, item in zip(labels, choices):
            text = _strip_option_prefix(item)
            formatted.append(f"{label}) {text}")
        options_string = "\n".join(formatted)
        question = f"{question}\n{options_string}" if question else options_string
    return question


def realworldqa_process_results(doc, results):
    response = results[0]
    gt_ans = doc["answer"].lower().strip()

    response = extract_final_answer(response)

    question_text, options = _extract_mcq_from_question(doc.get("question", ""))
    if options:
        option_labels = ["A", "B", "C", "D"]
        index2ans = {letter: text for letter, text in options}
        pred_letter = _extract_answer_letter(
            response,
            option_labels=option_labels,
            index2ans=index2ans or None,
        )
        if pred_letter:
            pred = pred_letter.lower().strip()
        else:
            pred = response.lower().strip().rstrip(".")
    else:
        pred = response.lower().strip().rstrip(".")

    score = 1.0 if pred == gt_ans else 0.0
    return {
        "exact_match": score,
    }


def realworldqa_process_results_extract(doc, results):
    response = results[0]
    gt_ans = doc["answer"].lower().strip()

    response = extract_final_answer(response)

    question = _build_extraction_question(doc)
    extracted = extract_answer_from_response(response, question=question)
    if extracted:
        response = extracted

    question_text, options = _extract_mcq_from_question(doc.get("question", ""))
    if options:
        option_labels = ["A", "B", "C", "D"]
        index2ans = {letter: text for letter, text in options}
        pred_letter = _extract_answer_letter(
            response,
            option_labels=option_labels,
            index2ans=index2ans or None,
        )
        if pred_letter:
            pred = pred_letter.lower().strip()
        else:
            pred = response.lower().strip().rstrip(".")
    else:
        pred = response.lower().strip().rstrip(".")

    score = 1.0 if pred == gt_ans else 0.0
    return {
        "exact_match": score,
    }


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Strip <think> blocks and \boxed{} wrappers before cleaning
                resp = extract_final_answer(resp)
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
