from __future__ import annotations

from typing import Any


def _extract_last_boxed_content(text: str) -> str | None:
    r"""Return content of the last balanced \boxed{...}, if present."""
    last_content: str | None = None
    start_idx = 0
    while True:
        idx = text.find("\\boxed{", start_idx)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            last_content = text[start : pos - 1]
            start_idx = pos
        else:
            break
    return last_content


def _strip_boxed(text: str) -> str:
    r"""Remove \boxed{...} wrappers, keeping inner content. Handles nested braces."""
    result = text
    while True:
        idx = result.find("\\boxed{")
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            inner = result[start : pos - 1]
            result = result[:idx] + inner + result[pos:]
        else:
            break  # Unbalanced braces — stop
    return result


def _strip_tex_text_wrapper(text: str) -> str:
    r"""If text is exactly \text{...}, unwrap it once (or repeatedly)."""
    result = text.strip()
    while result.startswith("\\text{") and result.endswith("}"):
        start = len("\\text{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0 or pos != len(result):
            break
        result = result[start : pos - 1].strip()
    return result


def extract_final_answer(
    raw_response: Any,
    parse_boxed: bool = True,
    strip_latex_wrappers: bool = False,
) -> str:
    """Extract the final answer, removing thinking reasoning.

    - If <answer>...</answer> exists, prefer its content.
    - Otherwise, drop content up to </think> if present.
    - Trim common answer prefixes.
    - *parse_boxed*: extract content of the last \\boxed{...} (for math tasks).
    - *strip_latex_wrappers*: remove \\boxed{...} and \\text{...} wrappers
      inline, keeping surrounding text intact (for instruction-following tasks
      where these are just noise from reasoning models).
    """
    if not isinstance(raw_response, str):
        return "" if raw_response is None else str(raw_response)

    text = raw_response.strip()
    if not text:
        return ""

    if "</think>" in text and "<think>" not in text:
        text = f"<think>\n{text}"
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # if "Answer:" in text:
    #     text = text.rsplit("Answer:", 1)[-1].strip()
    # elif "answer:" in text:
    #     text = text.rsplit("answer:", 1)[-1].strip()

    if parse_boxed:
        boxed_content = _extract_last_boxed_content(text)
        if boxed_content is not None:
            text = boxed_content.strip()
        else:
            stripped = _strip_boxed(text)
            if stripped != text:
                text = stripped.strip()

        text = _strip_tex_text_wrapper(text)
    elif strip_latex_wrappers:
        text = _strip_boxed(text)
        text = _strip_tex_text_wrapper(text)

    return text.strip()
