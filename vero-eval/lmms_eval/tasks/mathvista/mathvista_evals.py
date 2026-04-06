import math
import os
import re
import time
from pathlib import Path

import requests
import yaml
from Levenshtein import distance
from loguru import logger as eval_logger

from lmms_eval.llm_judge import Request, ServerConfig, get_server
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine

try:
    import sympy as _sympy
except Exception:
    _sympy = None

from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig

# pids: 799, 681, 615
shot_examples = [
    {
        "question": "How much money does Ruth need to buy a baking dish, a casserole dish, and an ice cream scoop? (Unit: $)",
        "caption": "The image shows a table with a variety of items on it, including a baking dish, ice cream scoop, casserole dish, and rolling pin. The text in the image says:\n\n```\nbaking dish\n$4.00\nice cream scoop\n$6.00\ncasserole dish\n$3.00\nrolling pin\n$4.00\n```",
        "ocr": "[([5, 3], 'baking dish'), ([177, 5], '$4.00'), ([7, 41], 'ice cream scoop'), ([177, 37], '$6.00'), ([9, 69], 'casserole dish'), ([177, 69], '$3.00'), ([5, 98], 'rolling pin'), ([177, 101], '$4.00')]",
        "solution": """
Find the total cost of a baking dish, a casserole dish, and an ice cream scoop.\n\n$4.00 + $3.00 + $6.00 = $13.00\n\nRuth needs $13.00.
""",
        "code": """
baking_dish_price = 4.00
casserole_dish_price = 3.00
ice_cream_scoop_price = 6.00

ans = baking_dish_price + casserole_dish_price + ice_cream_scoop_price
print(ans)
""",
    },
    {
        "question": "What is the largest city in the nation where this plane is headquartered?",
        "choices": ["hong kong", "osaka", "shanghai", "tokyo"],
        "caption": 'The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says "Japan. Endless Discovery".',
        "solution": """
The caption mentions that the text on the image says "Japan. Endless Discovery". This indicates that the plane is headquartered in Japan. 

Among the Japanese cities, Tokyo is the largest city.

Thus, the answer is D (tokyo).
""",
        "code": """
def largest_city(caption, choices):
    countries_largest_cities = {
        'Japan': 'tokyo',
        'China': 'shanghai'
    }

    if "Japan" in caption:
        country = 'Japan'
    elif "China" in caption:
        country = 'China'

    for choice in choices:
        if choice == countries_largest_cities[country]:
            return choice
    return ""

choices = ['hong kong', 'osaka', 'shanghai', 'tokyo']
caption = "The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says 'Japan. Endless Discovery'."

print(largest_city(caption, choices))
""",
    },
    {
        "question": "If two sides of a triangle measure 12 and 7, which of the following cannot be the perimeter of the triangle?",
        "choices": ["29", "34", "37", "38"],
        "caption": "The image shows a triangle with two sides labeled 7 and 12. The triangle is drawn on a white background. There is no text other than the labels.",
        "ocr": "[([70, 74], '7'), ([324, 74], '12')]",
        "solution": """
To determine which of the given perimeters cannot be possible for the triangle, we apply the triangle inequality theorem. The sum of any two sides of a triangle must be greater than the third side.

For the maximum possible value of the third side:
12 + 7 = 19

The minimum possible value for the third side:
12 - 7 = 5

The third side for each option:
(A) 29 - 12 - 7 = 10 (valid)
(B) 34 - 12 - 7 = 15 (valid)
(C) 37 - 12 - 7 = 18 (valid)
(D) 38 - 12 - 7 = 19 (invalid because it should be less than 19)

Thus, the answer is D.
""",
        "code": """
def is_valid_triangle(a, b, perimeter):
    # Given a and b, find the third side
    third_side = perimeter - a - b
    
    # Check triangle inequality
    if (a + b > third_side) and (a + third_side > b) and (b + third_side > a):
        return True
    return False

# Given sides
a = 12
b = 7

# Given perimeters
perimeters = [29, 34, 37, 38]

# Check which perimeter is not valid
for p in perimeters:
    if not is_valid_triangle(a, b, p):
        print(p)
""",
    },
]

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

with open(Path(__file__).parent / "mathvista.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


class MathVistaEvaluator:
    def __init__(self, quick_extract=False):
        self.quick_extract = quick_extract
        # Default to local vLLM judge when API_TYPE is unset.
        self.api_type = os.getenv("API_TYPE", "vllm").lower()
        self.gpt_model = os.getenv("MODEL_VERSION", os.getenv("JUDGE_MODEL_PATH", "gpt-4o-2024-11-20"))
        self.server = None
        self._local_engine = None
        self.use_local_judge = self.api_type not in ["openai", "azure"]
        if self.use_local_judge and os.getenv("VLLM_JUDGE_SYSTEM_PROMPT") is None:
            # Allow raw text outputs from the local judge.
            os.environ["VLLM_JUDGE_SYSTEM_PROMPT"] = ""

    def _get_server(self):
        if self.server is None and not self.use_local_judge:
            cfg = ServerConfig(model_name=self.gpt_model, temperature=0.0, max_tokens=256, timeout=60, num_retries=5, retry_delay=10)
            self.server = get_server(server_name=self.api_type, config=cfg)
        return self.server

    def _get_local_engine(self):
        if self._local_engine is None and self.use_local_judge:
            self._local_engine = get_judge_engine(os.getenv("JUDGE_MODEL_PATH") or self.gpt_model)
        return self._local_engine

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=5, sleep_time=0):
        if self.use_local_judge:
            engine = self._get_local_engine()
            return engine.generate_json(prompt, max_tokens=max_tokens).strip()

        # Create a custom server config for this specific request with different parameters
        request_config = ServerConfig(model_name=self.gpt_model, temperature=temperature, max_tokens=max_tokens, timeout=60, num_retries=patience, retry_delay=sleep_time)

        while patience > 0:
            patience -= 1
            try:
                # Use the core evaluate method with a Request object for direct text generation
                request = Request(messages=[{"role": "user", "content": prompt}], config=request_config)

                response = self._get_server().evaluate(request)

                if response.success:
                    prediction = response.content.strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    eval_logger.error(f"Server evaluation failed: {response.error}")

            except Exception as e:
                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""

    def verify_extraction(self, extraction):
        extraction = extraction.strip()
        if not extraction:
            return False
        return True

    def create_test_prompt(self, demo_prompt, query, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"{query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def extract_answer(self, response, problem, quick_extract=False):
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        choices = problem.get("choices", [])
        query = problem["query"]

        if not response:
            return ""

        def _maybe_from_json(text):
            parsed = extract_json_candidate(text)
            if parsed:
                for key in ("Extracted answer", "extracted_answer", "answer"):
                    if key in parsed and parsed[key] not in (None, {}):
                        try:
                            return str(parsed[key])
                        except Exception:
                            pass
            return text

        # Handle JSON-formatted outputs from the judge (e.g., {"Extracted answer": "C"}).
        response = _maybe_from_json(response)
        response = truncate_response_tail_tiktoken(response)

        if question_type == "multi_choice" and response in choices:
            return response

        if answer_type == "integer":
            try:
                extraction = int(response)
                return str(extraction)
            except ValueError:
                pass

        if answer_type == "float":
            try:
                extraction = str(float(response))
                return extraction
            except ValueError:
                pass

        # quick extraction
        if quick_extract:
            eval_logger.info("Quickly extracting answer...")
            # The answer is "text". -> "text"
            try:
                result = re.search(r'The answer is "(.*)"\.', response)
                if result:
                    extraction = result.group(1)
                    return _maybe_from_json(extraction)
            except re.error:
                pass

        # general extraction
        try:
            full_prompt = self.create_test_prompt(DEMO_PROMPT, query, response)
            extraction = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            return _maybe_from_json(extraction)
        except Exception as e:
            eval_logger.error(e)
            eval_logger.error(f"Error in extracting answer for problem")

        return response

    def get_most_similar(self, prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
        """
        distances = [distance(prediction, choice) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]

    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        """
        Normalize the extracted answer to match the answer type
        """
        if question_type == "multi_choice":
            # make sure the extraction is a string
            if isinstance(extraction, str):
                extraction = extraction.strip().upper()
            else:
                try:
                    extraction = str(extraction).upper()
                except:
                    extraction = ""

            # extract "A" from "(A) text"
            letter = re.findall(r"\(([a-zA-Z])\)", extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()

            options = [chr(ord("A") + i) for i in range(len(choices))]

            if extraction in options:
                # convert option letter to text, e.g. "A" -> "text"
                ind = options.index(extraction)
                extraction = choices[ind]
            else:
                # select the most similar option
                extraction = self.get_most_similar(extraction, choices)
            assert extraction in choices

        elif answer_type == "integer":
            try:
                extraction = str(int(float(extraction)))
            except:
                extraction = None

        elif answer_type == "float":
            try:
                # Coerce precision to an int (it can be float or string in the dataset).
                ndigits = int(float(precision)) if precision is not None else 0
            except Exception:
                ndigits = 0
            try:
                extraction = str(round(float(extraction), ndigits))
            except Exception:
                # If rounding fails, keep the raw extraction as a string rather than dropping to None.
                try:
                    extraction = str(extraction)
                except Exception:
                    extraction = None

        elif answer_type == "list":
            try:
                extraction = str(extraction)
            except:
                extraction = None

        return extraction

    def safe_equal(self, prediction, answer):
        """
        Check if the prediction is equal to the answer, even if they are of different types
        """
        try:
            if str(prediction).strip().lower() == str(answer).strip().lower():
                return True
            return False
        except Exception as e:
            eval_logger.info(e)
            return False

    def get_acc_with_contion(self, res_pd, key, value):
        """
        Calculate the accuracy of predictions with a specific condition
        """
        if key == "skills":
            total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
        else:
            total_pd = res_pd[res_pd[key] == value]

        correct_pd = total_pd[total_pd["true_false"] == True]
        acc = "{:.4f}".format(len(correct_pd) / len(total_pd)) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    def create_one_query(self, problem, shot_type, examples=shot_examples, shot_num=0, use_caption=False, use_ocr=False):
        ### [1] Demo prompt
        if shot_num == 0:
            demo_prompt = ""
        else:
            demos = []
            shot_num = min(shot_num, len(examples))
            for example in examples[:shot_num]:
                prompt = ""

                # question
                prompt += f"Question: {example['question']}"

                # choices
                if "choices" in example:
                    texts = ["Choices:"]
                    for i, choice in enumerate(example["choices"]):
                        texts.append(f"({chr(ord('A')+i)}) {choice}")
                    prompt += "\n" + "\n".join(texts)

                # caption
                if use_caption:
                    caption = example["caption"] if "caption" in example else ""
                    if caption != "":
                        prompt += "\n" + f"Image description: {caption}"

                # ocr
                if use_ocr:
                    ocr = example["ocr"] if "ocr" in example else ""
                    if ocr != "":
                        prompt += "\n" + f"Image detected text: {ocr}"

                # solution
                if shot_type == "solution":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"Solution: {solution}"

                # step-by-step
                if shot_type == "step-by-step":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # think-step-by-step
                if shot_type == "think-step-by-step":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # direct
                if shot_type == "direct":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # code
                if shot_type == "code":
                    code = example["code"].strip()
                    prompt += "\n" + f"Python code: {code}"

                demos.append(prompt)

            demo_prompt = "\n\n".join(demos)

        ### [2] Test query
        # problem info
        question = problem["question"]
        unit = problem["unit"]
        choices = problem["choices"]
        caption = problem["caption"]
        ocr = problem["ocr"]
        precision = problem["precision"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]

        # hint
        if shot_type == "solution":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        # step-by-step
        elif shot_type == "format-prompt":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Answer with the option's letter from the given choices directly."
            else:
                if answer_type == "integer":
                    hint_text = f"Answer the question using a single integer number."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Answer the question using a single floating-point number with one decimal place."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Answer the question using a single floating-point number with two decimal places."

                elif answer_type == "list":
                    hint_text = f"Answer the question using a Python list."
        # step-by-step
        elif shot_type == "step-by-step":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        # step-by-step
        elif shot_type == "reason-first":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"First perform reasoning, then finally answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end in the following format: Answer: xxx."

                elif answer_type == "float" and precision == 1:
                    hint_text = (
                        f"First perform reasoning, then finally answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end in the following format: Answer: xxx."
                    )

                elif answer_type == "float" and precision == 2:
                    hint_text = f"First perform reasoning, then finally answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end in the following format: Answer: xxx."

                elif answer_type == "list":
                    hint_text = f"First perform reasoning, then finally answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end in the following format: Answer: xxx."
        elif shot_type == "direct":
            hint_text = ""
        else:
            assert shot_type == "code"
            hint_text = "Hint: Please generate a python code to solve the problem"

        # question
        if shot_type == "format-prompt":
            question_text = f"{question}"
        else:
            question_text = f"{question}"
        if unit:
            question_text += f" (Unit: {unit})"

        # choices
        if choices:
            if shot_type == "format-prompt":
                texts = []
                for i, choice in enumerate(choices):
                    texts.append(f"{chr(ord('A')+i)}. {choice}")
                choices_text = "\n".join(texts)
            else:
                # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)
        else:
            choices_text = ""

        # caption
        caption_text = ""
        if use_caption and caption != "":
            caption_text = f"Image description: {caption}"

        # ocr
        ocr_text = ""
        if use_ocr and ocr != "":
            ocr_text = f"Image detected text: {ocr}"

        # prompt
        if shot_type == "solution":
            prompt = "Solution: "
        elif shot_type == "format-prompt":
            prompt = ""
        elif shot_type == "step-by-step":
            prompt = ""
        elif shot_type == "reason-first":
            prompt = ""
        elif shot_type == "direct":
            prompt = ""
        else:
            assert shot_type == "code"
            prompt = "Python code: "

        if shot_type == "reason-first":
            elements = [hint_text, question_text, choices_text, caption_text, ocr_text, prompt]
            test_query = "\n".join([e for e in elements if e != ""])
        else:
            elements = [question_text, choices_text, caption_text, ocr_text, hint_text, prompt]
            test_query = "\n".join([e for e in elements if e != ""])

        ### [3] Final query
        query = demo_prompt + "\n\n" + test_query
        query = query.strip()
        print(f"Final query:\n{query}")
        return query


_MATHVISTA_PARSE_TIMEOUT = 3
_MATHVISTA_VERIFY_TIMEOUT = 3
_MATHVISTA_NUMERIC_TARGETS = (
    [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    if LatexExtractionConfig and ExprExtractionConfig
    else []
)


def _mathvista_get_most_similar(prediction, choices):
    if not choices:
        return prediction
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def _mathvista_parse_choice_letter(text, options):
    if not (math_verify_parse and StringExtractionConfig):
        return ""
    if not text or not options:
        return ""
    try:
        parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=tuple(options))])
    except Exception:
        parsed = None
    if parsed:
        parsed_choice = str(parsed[0]).strip()
        if parsed_choice:
            parsed_choice = parsed_choice.upper()
            if parsed_choice in options:
                return parsed_choice
    return ""


def _mathvista_coerce_float(value):
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        pass
    else:
        return coerced if math.isfinite(coerced) else None
    if _sympy is not None:
        try:
            expr = _sympy.sympify(value)
            if expr.is_real or expr.is_number:
                coerced = float(expr.evalf())
                return coerced if math.isfinite(coerced) else None
        except Exception:
            pass
    return None


def _mathvista_numeric_candidates(text):
    candidates = []

    direct = _mathvista_coerce_float(text)
    if direct is not None:
        candidates.append(direct)

    if math_verify_parse and _MATHVISTA_NUMERIC_TARGETS:
        try:
            parsed = math_verify_parse(text, _MATHVISTA_NUMERIC_TARGETS, parsing_timeout=_MATHVISTA_PARSE_TIMEOUT)
        except TimeoutException:
            parsed = None
        except Exception:
            parsed = None

        if parsed is None:
            parsed_values = []
        elif isinstance(parsed, (list, tuple, set)):
            parsed_values = list(parsed)
        else:
            parsed_values = [parsed]

        for value in parsed_values:
            coerced = _mathvista_coerce_float(value)
            if coerced is not None:
                candidates.append(coerced)

    unique_candidates = []
    for val in candidates:
        if not any(abs(val - existing) <= 1e-12 for existing in unique_candidates):
            unique_candidates.append(val)
    return unique_candidates


def grade_numeric_with_math_verify(gold, pred, precision=6, tolerance=None):
    if not (math_verify_parse and math_verify_verify and _MATHVISTA_NUMERIC_TARGETS):
        try:
            return 1 if float(gold) == float(pred) else 0
        except Exception:
            return -1

    if tolerance is not None:
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            return -1
        if tol < 0:
            return -1

        gold_candidates = _mathvista_numeric_candidates(gold)
        pred_candidates = _mathvista_numeric_candidates(pred)
        if not gold_candidates or not pred_candidates:
            return -1

        for gold_value in gold_candidates:
            for pred_value in pred_candidates:
                diff = abs(gold_value - pred_value)
                if abs(gold_value) <= 1e-12:
                    if diff <= tol:
                        return 1
                elif diff <= tol * abs(gold_value):
                    return 1
        return 0

    try:
        gold_parsed = math_verify_parse(
            gold, _MATHVISTA_NUMERIC_TARGETS, parsing_timeout=_MATHVISTA_PARSE_TIMEOUT
        )
        pred_parsed = math_verify_parse(
            pred, _MATHVISTA_NUMERIC_TARGETS, parsing_timeout=_MATHVISTA_PARSE_TIMEOUT
        )
    except TimeoutException:
        return -1
    except Exception:
        return -1

    if gold_parsed is None or pred_parsed is None:
        return -1

    try:
        verified = math_verify_verify(
            gold_parsed,
            pred_parsed,
            float_rounding=precision,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=_MATHVISTA_VERIFY_TIMEOUT,
        )
    except TimeoutException:
        return -1
    except Exception:
        return -1

    return 1 if verified else 0


def _mathvista_normalize_numeric_extraction(extraction, answer_type, precision):
    if extraction is None:
        return None
    try:
        text = str(extraction).strip()
    except Exception:
        return None
    if not text:
        return None

    candidates = _mathvista_numeric_candidates(text)
    if candidates:
        value = float(candidates[0])
        if not math.isfinite(value):
            return text
        if answer_type == "integer":
            rounded = round(value)
            if abs(value - rounded) <= 1e-9:
                return str(int(rounded))
            return str(value)
        try:
            ndigits = int(float(precision)) if precision is not None else 0
        except Exception:
            ndigits = 0
        return str(round(value, ndigits))

    try:
        if answer_type == "integer":
            return str(int(float(text)))
        ndigits = int(float(precision)) if precision is not None else 0
        return str(round(float(text), ndigits))
    except Exception:
        return text


def normalize_extracted_answer_with_math_verify(extraction, choices, question_type, answer_type, precision):
    if question_type == "multi_choice":
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction).strip()
            except Exception:
                extraction = ""

        if not extraction:
            return ""

        options = [chr(ord("A") + i) for i in range(len(choices or []))]

        parsed_letter = _mathvista_parse_choice_letter(extraction, options)
        if not parsed_letter:
            letter = re.findall(r"\(([a-zA-Z])\)", extraction)
            if letter:
                parsed_letter = letter[0].upper()

        if parsed_letter in options:
            return choices[options.index(parsed_letter)]

        if extraction.upper() in options:
            return choices[options.index(extraction.upper())]

        return _mathvista_get_most_similar(extraction, choices)

    if answer_type in ("integer", "float"):
        try:
            return _mathvista_normalize_numeric_extraction(extraction, answer_type, precision)
        except Exception:
            return None

    if answer_type == "list":
        try:
            return str(extraction)
        except Exception:
            return None

    return extraction


def safe_equal_with_math_verify(prediction, answer, answer_type=None, precision=6):
    if answer is None:
        return False
    if answer_type in ("integer", "float"):
        try:
            ndigits = int(float(precision)) if precision is not None else 6
        except Exception:
            ndigits = 6
        result = grade_numeric_with_math_verify(str(answer), str(prediction), precision=ndigits)
        if result == 1:
            return True
        if result == 0:
            return False
    try:
        return str(prediction).strip().lower() == str(answer).strip().lower()
    except Exception:
        return False
