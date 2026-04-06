# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of instructions."""

from __future__ import annotations

import collections
import json
import os
import random
import re
import string
from collections.abc import Sequence

import langdetect
import nltk

try:
    from absl import logging as absl_logging

    logging = absl_logging
except Exception:  # pragma: no cover - fallback when absl is unavailable
    import logging as py_logging

    logging = py_logging

try:
    from open_instruct.IFEvalG import instructions_util as _instructions_util
except Exception:  # pragma: no cover - fallback when open_instruct is unavailable
    _instructions_util = None


_NLTK_DATA_PATH = os.environ.get("NLTK_DATA_PATH")
if _NLTK_DATA_PATH:
    nltk.data.path.append(_NLTK_DATA_PATH)


class _FallbackInstructionsUtil:
    nltk = nltk
    LANGUAGE_CODES = {
        "en": "English",
        "zh": "Chinese",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
    }
    _KEYWORD_POOL = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]

    @staticmethod
    def generate_keywords(num_keywords: int = 1) -> list[str]:
        if num_keywords <= 0:
            return []
        if num_keywords <= len(_FallbackInstructionsUtil._KEYWORD_POOL):
            return random.sample(_FallbackInstructionsUtil._KEYWORD_POOL, num_keywords)
        return random.choices(_FallbackInstructionsUtil._KEYWORD_POOL, k=num_keywords)

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        return nltk.sent_tokenize(text)

    @staticmethod
    def count_sentences(text: str) -> int:
        return len(_FallbackInstructionsUtil.split_into_sentences(text))

    @staticmethod
    def count_words(text: str) -> int:
        return len(nltk.word_tokenize(text))


instructions_util = _instructions_util or _FallbackInstructionsUtil

_InstructionArgsDtype = dict[str, int | str | Sequence[str]] | None

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

# The options of starter keywords.
_STARTER_OPTIONS = (
    "I would say",
    "My answer is",
    "I believe",
    "In my opinion",
    "I think",
    "I reckon",
    "I feel",
    "From my perspective",
    "As I see it",
    "According to me",
    "As far as I'm concerned",
    "To my understanding",
    "In my view",
    "My take on it is",
    "As per my perception",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# phrases
_PHRASES = [
    "Dance like nobody is watching you",
    "The early bird catches the worm",
    "Time flies when having fun",
    "Every cloud has a silver lining",
    "Actions speak louder than words",
    "Don't judge a book by cover",
    "Live each day to the fullest",
    "All that glitters is not gold",
    "Laughter is the best medicine",
    "The pen is mightier than sword",
]


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id=None):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self) -> _InstructionArgsDtype:
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self) -> list[str]:
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value: str) -> bool:
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        self._description_pattern = (
            "Your ENTIRE response should be in {language} language, no other " + "language is allowed."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Your response should contain {relation} {num_sentences} sentences."
        return self._description_pattern.format(
            relation=self._comparison_relation, num_sentences=self._num_sentences_threshold
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_sentences": self._num_sentences_threshold, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = instructions_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold  # pytype: disable=bad-return-type


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "Include at least {num_placeholders} bracketed placeholders like [address] or [name] in your response."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "Your answer must contain exactly {num_bullets} bullet points.\n"
            "Use Markdown bullets starting with either `*` or `-`, one bullet per line, for example:\n"
            "* This is point 1.\n"
            "- This is point 2."
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        options = "\n".join(f"- {opt}" for opt in self._constrained_responses)
        self._description_pattern = (
            "Reply with exactly one of the following options (copy it verbatim, including punctuation):\n"
            f"{options}"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        return any(value == constrained_response for constrained_response in self._constrained_responses)


class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.

        Args:
          starter: A string representing the keyward that the response should start
            with.

        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            "For every message you write in this conversation, start your response with the exact text: {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        return value.lstrip().startswith(self._starter)


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "Highlight at least {num_highlights} sections in your answer with "
            + "Markdown emphasis, for example *like this* or **like this**."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string repesenting the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = section_spliter.strip() if isinstance(section_spliter, str) else section_spliter
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            "Your response must have at least {num_sections} sections. Mark the beginning "
            + "of each section with {section_spliter} X, such as:\n"
            + "{section_spliter} 1\n"
            + "[content of section 1]\n"
            + "{section_spliter} 2\n"
            + "[content of section 2]"
        )

        return self._description_pattern.format(num_sections=self._num_sections, section_spliter=self._section_spliter)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"section_spliter": self._section_spliter, "num_sections": self._num_sections}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "Write exactly {num_paragraphs} paragraphs. Separate paragraphs using a line that contains only `***`."
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "At the end of your response, add a postscript that starts with {postscript} (for example: {postscript} ...)."
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)


class RephraseChecker(Instruction):
    """Checks the repharse."""

    _CHANGE_PATTERN = r"\*[^*\n]+\*"

    def build_description(self, *, original_message):
        """Build the instruction description.

        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.

        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(f"Message {original_message} does not contain changes in the form of *change me*.")

        self._reference_without_change = original_message
        self._description = (
            "Rephrase the message below, but ONLY change text that is inside single-asterisk spans like *change me*.\n"
            "Everything outside those *...* spans (including spacing and punctuation) must remain identical.\n"
            "Original message: {original_message}"
        )
        return self._description.format(original_message=original_message)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.

        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            return False

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return bool(re.search(self._CHANGE_PATTERN, response))

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(self._CHANGE_PATTERN, "", response)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = instructions_util.generate_keywords(num_keywords=_NUM_KEYWORDS)
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        keywords_text = ", ".join(self._keywords)
        self._description_pattern = (
            "Include all of the following keywords somewhere in your response (case-insensitive): {keywords_text}."
        )

        return self._description_pattern.format(keywords_text=keywords_text)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        return all(re.search(keyword, value, flags=re.IGNORECASE) for keyword in self._keywords)


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "In your response, the word {keyword} should appear {relation} " + "{frequency} times."
        )

        return self._description_pattern.format(
            keyword=self._keyword, relation=self._comparison_relation, frequency=self._frequency
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword, "frequency": self._frequency, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency  # pytype: disable=bad-return-type


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(_NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Answer with {relation} {num_words} words."

        return self._description_pattern.format(relation=self._comparison_relation, num_words=self._num_words)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = instructions_util.count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words  # pytype: disable=bad-return-type


class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "Output valid JSON only (no extra commentary).\n"
            "Optionally, wrap the JSON in a single Markdown code block like:\n"
            "```json\n"
            "{...}\n"
            "```"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(self, num_paragraphs=None, nth_paragraph=None, first_word=None):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if self._nth_paragraph is None or self._nth_paragraph <= 0 or self._nth_paragraph > self._num_paragraphs:
            self._nth_paragraph = random.randint(1, self._num_paragraphs)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "Write exactly {num_paragraphs} paragraphs.\n"
            "Separate paragraphs using exactly one blank line (i.e., two newline characters: \\n\\n). "
            "Do not add extra blank lines.\n"
            "Paragraph {nth_paragraph} (1-indexed) must start with the word {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs, nth_paragraph=self._nth_paragraph, first_word=self._first_word
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.

        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.

        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            self._key_sentences = ["For now, this is fine."]
        else:
            self._key_sentences = list(key_sentences)

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        key_sentences_text = "\n".join(f"- {s}" for s in self._key_sentences)
        self._description_pattern = (
            "Include exactly {num_sentences} of the following sentences verbatim as full sentences "
            "(same capitalization and punctuation):\n"
            "{key_sentences_text}"
        )

        return self._description_pattern.format(
            num_sentences=self._num_sentences, key_sentences_text=key_sentences_text
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_sentences": self._num_sentences, "key_sentences": list(self._key_sentences)}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = instructions_util.split_into_sentences(value)
        for sentence in self._key_sentences:
            if sentence in sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings respresenting words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = instructions_util.generate_keywords(num_keywords=_NUM_KEYWORDS)
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        forbidden_words_text = ", ".join(self._forbidden_words)
        self._description_pattern = (
            "Do not use any of these words in your response as standalone words (case-insensitive): {forbidden_words_text}."
        )

        return self._description_pattern.format(forbidden_words_text=forbidden_words_text)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        return all(not re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE) for word in self._forbidden_words)


class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.

        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have betweeb low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.

        Returns:
          A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            "Rephrase the following paragraph: "
            + "{original_paragraph}\nYour response should have "
            + "between {low} and {high} of the same words. "
            + "Words are the same if and only if all of the "
            + "letters, ignoring cases, are the same. For "
            + "example, 'run' is the same as 'Run' but different "
            + "to 'ran'."
        )

        return self._description.format(original_paragraph=original_paragraph, low=self._low, high=self._high)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"original_paragraph": self._original_paragraph, "low": self._low, "high": self._high}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Give exactly two different responses.\n"
            "Separate the two responses with exactly six asterisks: `******`.\n"
            "Do not include anything else besides the two responses and the separator."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "Start your output by repeating the following request verbatim (no extra characters before it):\n"
            "{prompt_to_repeat}\n"
            "Then, after the repeated request, provide your answer.\n"
            "The request to repeat is exactly the text shown above (it does not include these instructions)."
        )
        return self._description_pattern.format(prompt_to_repeat=self._prompt_to_repeat)

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return bool(value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()))


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            "Finish your response with this exact phrase {ender}. No other words should follow this phrase."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        return any(title.lstrip("<").rstrip(">").strip() for title in titles)


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not letter or len(letter) > 1 or ord(letter.lower()) < 97 or ord(letter.lower()) > 122:
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "In your response, the letter {letter} should appear {let_relation} {let_frequency} times."
        )

        return self._description_pattern.format(
            letter=self._letter, let_frequency=self._frequency, let_relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {"letter": self._letter, "let_frequency": self._frequency, "let_relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Your entire response should be in English, and in all capital letters."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "In your entire response, refrain from the use of any commas."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(self, capital_frequency=None, capital_relation=None):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            "In your response, words with all capital letters should appear {relation} {frequency} times."
        )

        return self._description_pattern.format(frequency=self._frequency, relation=self._comparison_relation)

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {"capital_frequency": self._frequency, "capital_relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = instructions_util.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Wrap your entire response with double quotation marks."
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class RepeatPhraseChecker(Instruction):
    """Repeat the phrase {phrase} exactly {small_n} times, transforming it slightly each time by replacing only one word in the center of the phrase."""

    def build_description(self, phrase=None, small_n=None):
        """Build the instruction description.

        Args:
          phrase: A string representing the phrase to be repeated.
          N: An integer representing the number of times to repeat the phrase.
          word_count: An integer representing the number of words in the phrase.

        Returns:
          A string representing the instruction description.
        """
        if not phrase:
            self._phrase = random.choice(_PHRASES)
        else:
            self._phrase = phrase.strip()
        if not small_n:
            self._small_n = random.randint(2, 3)
        else:
            self._small_n = small_n

        self._description_pattern = (
            "Write the phrase \"{phrase}\" exactly {small_n} times in your response.\n"
            "Exactly ONE of those occurrences must differ from the original by changing exactly one word, "
            "and all the other occurrences must match the original phrase exactly.\n"
            "Keep the first and last word the same. Do not add any extra words inside the phrase."
        )
        return self._description_pattern.format(phrase=self._phrase, small_n=self._small_n)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"phrase": self._phrase, "small_n": self._small_n}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["phrase", "small_n"]

    def check_following(self, value):
        """Checks if the response contains the expected number of phrases with the correct modifications."""
        ref_words = self._phrase.split()
        if not ref_words:
            return False
        first_word = ref_words[0]
        last_word = ref_words[-1]
        phrase_len = len(ref_words)

        words = re.findall(r"\b\w+\b", value)
        exact_count = 0
        one_diff_count = 0
        total_occurrences = 0

        for i in range(len(words) - phrase_len + 1):
            window = words[i : i + phrase_len]
            if window[0] != first_word or window[-1] != last_word:
                continue
            total_occurrences += 1
            if window == ref_words:
                exact_count += 1
                continue
            differences = sum(1 for a, b in zip(window, ref_words) if a != b)
            if differences == 1:
                one_diff_count += 1
            else:
                return False

        return total_occurrences == self._small_n and one_diff_count == 1 and exact_count == self._small_n - 1


class CopyChecker(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "Copy the following text verbatim and output only that text (no extra words): {prompt_to_repeat}"
        )
        return self._description_pattern.format(prompt_to_repeat=self._prompt_to_repeat)

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return value.strip().lower() == self._prompt_to_repeat.strip().lower()


class CopySpanIdxChecker(Instruction):
    """{prompt_to_repeat}. Copy the span of words that lies between (and including) index {n_start} and {n_end}, the indices are character indices!"""

    def build_description(self, prompt_to_repeat=None, n_start=None, n_end=None):
        """Build the instruction description.

        Args:
        n_start: An integer representing the start index of the span.
        n_end: An integer representing the end index of the span.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        if not n_start:
            self._n_start = random.randint(0, len(self._prompt_to_repeat) - 2)
        else:
            self._n_start = n_start
        if not n_end:
            self._n_end = random.randint(self._n_start + 1, len(self._prompt_to_repeat) - 1)
        else:
            self._n_end = n_end
        self._description_pattern = (
            "Given the text below, output ONLY the substring from character index {n_start} (inclusive) up to "
            "character index {n_end} (exclusive).\n"
            "Indices are 0-based character indices, counting every character including spaces.\n"
            "Text: {prompt_to_repeat}"
        )
        return self._description_pattern.format(
            n_start=self._n_start, n_end=self._n_end, prompt_to_repeat=self._prompt_to_repeat
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"n_start": self._n_start, "n_end": self._n_end, "prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["n_start", "n_end", "prompt_to_repeat"]

    def check_following(self, value):
        """Checks if the response contains the expected number of phrases with the correct modifications."""
        return value.strip().lower() == self._prompt_to_repeat[self._n_start : self._n_end].strip().lower()


class SentenceHyphenChecker(Instruction):
    """All sentences must be connected using hyphens, with no spaces between them."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Write multiple sentences and connect them using a single hyphen `-` between sentences, with no spaces around the hyphen.\n"
            "Do not use hyphens anywhere else in the response."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if all sentences are connected using hyphens, with no spaces between them."""
        if "-" not in value:
            return False
        sentences = value.split("-")
        if len(sentences) < 2:
            return False
        for sentence in sentences:
            if not sentence:
                return False
            if sentence.strip() != sentence:
                return False
            if len(instructions_util.split_into_sentences(sentence)) != 1:
                return False
        return True


class AdjacentLetterChecker(Instruction):
    """No two adjacent words can start with consecutive letters of the alphabet."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Ensure each word starts with a letter (A-Z or a-z). No two adjacent words may start with consecutive letters of the alphabet "
            "(for example, a word starting with 'a' cannot be immediately followed by a word starting with 'b')."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if no two adjacent words start with consecutive letters of the alphabet."""
        words = value.split()
        for i in range(len(words) - 1):
            first_letter = words[i][0].lower()
            second_letter = words[i + 1][0].lower()
            if len(first_letter) != 1 or len(second_letter) != 1:
                return False
            if ord(second_letter) - ord(first_letter) == 1:
                return False
        return True


class SquareBracketChecker(Instruction):
    """Enclose every word in your response within square brackets."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Enclose every whitespace-separated token in your response within square brackets, e.g. `[hello] [world]`."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if every word in the response is enclosed within square brackets."""
        words = value.split()
        return all(word.startswith("[") and word.endswith("]") for word in words)


class KeywordFrequencyOnceChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = 1

        self._description_pattern = "Include the keyword {keyword} in your response exactly once (case-insensitive)."

        return self._description_pattern.format(keyword=self._keyword, frequency=self._frequency)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        return actual_occurrences == 1


class KeywordFrequencyCheckerDifferent(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "In your response, the word {keyword} should appear {relation} {frequency} times."

        return self._description_pattern.format(
            keyword=self._keyword, relation=self._comparison_relation, frequency=self._frequency
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword, "frequency": self._frequency, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency  # pytype: disable=bad-return-type


class ExcludeWordHarderChecker(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, keyword=None, instruction=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings respresenting words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = random.choice(instruction.split())
        else:
            self._keyword = keyword.strip()

        self._description_pattern = "Do not include keyword {keyword} in the response."

        return self._description_pattern.format(keyword=self._keyword)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        pattern = r"\b" + re.escape(self._keyword) + r"\b"
        return not re.search(pattern, value, flags=re.IGNORECASE)


class ParagraphBasicChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = (
            "Write exactly 2 paragraphs. Separate them using a line that contains only `***`."
        )

        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == 2


class ParagraphBasicChecker2(Instruction):
    """Checks the paragraphs."""

    def build_description(self):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = (
            "Write exactly 2 paragraphs. Separate the two paragraphs using exactly one blank line (two newline characters: \\n\\n)."
        )

        return self._description_pattern.format()

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == 2


class FirstWordSentChecker(Instruction):
    """The first word of each sentence should be the word {first_word}."""

    def build_description(self, first_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the first word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not first_word:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(first_word, str):
                self._first_word = first_word[0].strip()
            else:
                self._first_word = first_word.strip()

        self._description_pattern = "The first word of each sentence should be the word {first_word}."

        return self._description_pattern.format(first_word=self._first_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["first_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        sentences = instructions_util.split_into_sentences(value)

        # Check if the first word of each sentence matches the expected word
        for sentence in sentences:
            if not sentence.strip():
                return False
            first_word = sentence.split()[0].strip()
            if first_word.lower() != self._first_word.lower():
                return False
        return True


class FirstWordAnswerChecker(Instruction):
    """The first word of each sentence should be the word {first_word}."""

    def build_description(self, first_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the first word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not first_word:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._first_word = first_word.strip()

        self._description_pattern = "The first word of your response should be the word {first_word}."

        return self._description_pattern.format(first_word=self._first_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["first_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        if not value.strip() or len(value.split()) == 0:
            return False
        first_word = value.split()[0].strip()
        return first_word.lower() == self._first_word.lower()


class LastWordSentChecker(Instruction):
    """The last word of each sentence should be the word {last_word}."""

    def build_description(self, last_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the last word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not last_word:
            self._last_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(last_word, str):
                self._last_word = last_word[0].strip()
            else:
                self._last_word = last_word.strip()

        self._description_pattern = (
            "The last word of each sentence, before punctuation, should be the word {last_word}."
        )

        return self._description_pattern.format(last_word=self._last_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["last_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        sentences = instructions_util.split_into_sentences(value)

        # Check if the first word of each sentence matches the expected word
        for sentence in sentences:
            if not sentence.strip():
                return False
            last_word = sentence.split()[-1].strip()
            # remove any punctuation from last_word
            last_word = re.sub(r"[^\w\s]", "", last_word)
            if last_word.lower() != self._last_word.lower():
                return False
        return True


class LastWordAnswerChecker(Instruction):
    """The last word of your response should be the word {last_word}."""

    def build_description(self, last_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the last word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not last_word:
            self._last_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._last_word = last_word.strip()

        self._description_pattern = "The last word of your response should be the word {last_word}."

        return self._description_pattern.format(last_word=self._last_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["last_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        last_word = value.split()[-1].strip()
        # remove any punctuation from last_word
        last_word = re.sub(r"[^\w\s]", "", last_word)
        return last_word.lower() == self._last_word.lower()


class BiGramWrappingChecker(Instruction):
    """Wrap every word bigram in double angular brackets, such as <<I am>> <<at home>> <<with my>> <<cute dog>>."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Write your response as an even number of words. Group words into consecutive non-overlapping pairs:\n"
            "- words 1–2, 3–4, 5–6, ...\n"
            "Wrap each 2-word pair in double angle brackets like: `<<I am>> <<at home>> <<with my>> <<cute dog>>`.\n"
            "Put `<<` immediately before the first word and `>>` immediately after the second word of each pair."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if every word bigram is enclosed within double angular brackets."""
        words = value.split()
        if len(words) == 0 or len(words) % 2 != 0:
            return False
        for i in range(0, len(words), 2):
            first = words[i]
            second = words[i + 1]
            if not first.startswith("<<"):
                return False
            if first.endswith(">>"):
                return False
            if not second.endswith(">>"):
                return False
            if second.startswith("<<"):
                return False
        return True


class CopyingSimpleChecker(Instruction):
    """Repeat the request without change (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"""

    def build_description(self, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
        prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "Repeat the following request exactly (verbatim) and output ONLY that request. Do not answer it.\n"
            "Request to repeat: {prompt_to_repeat}"
        )
        return self._description_pattern.format(prompt_to_repeat=self._prompt_to_repeat)

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return value.strip().lower() == self._prompt_to_repeat.strip().lower()


class CopyingMultipleChecker(Instruction):
    """Repeat the request without change {N} times, separated by 6 asterisk symbols (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"""

    def build_description(self, prompt_to_repeat=None, N=None):
        """Build the instruction description.

        Args:
        prompt_to_repeat: The prompt that is meant to be repeated.
        N: An integer representing the number of times to repeat the phrase.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        self._description_pattern = (
            "Repeat the following request exactly {N} times. Separate each repetition with exactly six asterisks: `******`.\n"
            "Output ONLY the repetitions and separators (do not answer the request, and do not add a leading/trailing separator).\n"
            "Request to repeat: {prompt_to_repeat}"
        )
        return self._description_pattern.format(N=self._N, prompt_to_repeat=self._prompt_to_repeat)

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat, "N": self._N}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat", "N"]

    def check_following(self, value):
        prompts = value.split("******")
        if len(prompts) != self._N:
            return False
        return all(prompt.strip().lower() == self._prompt_to_repeat.strip().lower() for prompt in prompts)


class PunctuationDotChecker(Instruction):
    """In your entire response, refrain from the use of . (i.e. dots) as punctuation and in general."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "In your entire response, refrain from the use of . (i.e. dots) as punctuation and in general."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain dots."""
        return not re.search(r"\.", value)


class PunctuationExclamationChecker(Instruction):
    """In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain exclamation marks."""
        return not re.search(r"\!", value)


class LowercaseCountingChecker(Instruction):
    """In your response, all lowercase words should appear at most {N} times."""

    def build_description(self, N=None):
        """Build the instruction description.

        Args:
        N: An integer representing the maximum number of lowercase words allowed.

        Returns:
        A string representing the instruction description.
        """
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        self._description_pattern = (
            "In your response, there may be at most {N} words that are made up of only lowercase letters a–z."
        )
        return self._description_pattern.format(N=self._N)

    def get_instruction_args(self):
        return {"N": self._N}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["N"]

    def check_following(self, value):
        """Checks that the response does not contain lowercase words more than N times."""
        lowercase_words = re.findall(r"\b[a-z]+\b", value)
        return len(lowercase_words) <= self._N


class LetterCountingChecker(Instruction):
    """Answer with {relation} {N} letters."""

    def build_description(self, N=None, relation=None):
        """Build the instruction description.

        Args:
        N: An integer representing the maximum number of letters allowed.

        Returns:
        A string representing the instruction description.
        """
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        if not relation:
            self._relation = random.choice(_COMPARISON_RELATION)
        else:
            self._relation = relation
        self._description_pattern = (
            "Your response must contain {relation} {N} alphabetic letters (A–Z or a–z) in total. "
            "Spaces, digits, and punctuation do not count toward the letter total."
        )
        return self._description_pattern.format(N=self._N, relation=self._relation)

    def get_instruction_args(self):
        return {"N": self._N, "relation": self._relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["N", "relation"]

    def check_following(self, value):
        """Checks that the response does not contain lowercase words more than N times."""
        letters = re.findall(r"[a-zA-Z]", value)
        if self._relation == "at least":
            return len(letters) >= self._N
        elif self._relation == "less than":
            return len(letters) < self._N


class CountingCompositionChecker(Instruction):
    """Write 3 paragraphs, delimited by the markdown divider: * * *, with exactly {n_sent} sentences each, with exactly {n_words} words in each sentence."""

    def build_description(self, n_sent=None, n_words=None):
        """Build the instruction description.

        Args:
        n_sent: An integer representing the number of sentences in each paragraph.
        n_words: An integer representing the number of words in each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not n_sent:
            self._n_sent = random.randint(2, 3)
        else:
            self._n_sent = n_sent
        if not n_words:
            self._n_words = random.randint(2, 3)
        else:
            self._n_words = n_words
        self._description_pattern = (
            "Write exactly 3 paragraphs, separated by a line that contains only `***`.\n"
            "Each paragraph must contain exactly {n_sent} sentences.\n"
            "Each sentence must contain exactly {n_words} words."
        )
        return self._description_pattern.format(n_sent=self._n_sent, n_words=self._n_words)

    def get_instruction_args(self):
        return {"n_sent": self._n_sent, "n_words": self._n_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["n_sent", "n_words"]

    def check_following(self, value):
        """Checks that the response contains the expected number of paragraphs, sentences, and words.

        Args:
          value: A string representing the response.

        Returns:
          True if the response meets the requirements; otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

            sentences = instructions_util.split_into_sentences(paragraph)
            num_sentences = len(sentences)

            if num_sentences != self._n_sent:
                return False

            for sentence in sentences:
                words = instructions_util.nltk.word_tokenize(sentence)
                num_words = len(words)

                if num_words != self._n_words:
                    return False

        return num_paragraphs == 3


class CountUniqueChecker(Instruction):
    """Only use unique words in your response, no word should be repeated!"""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Only use unique words in your response, no word should be repeated!"
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response contains unique words."""
        words = instructions_util.nltk.word_tokenize(value)
        unique_words = set(words)
        return len(words) == len(unique_words)


class CountIncrementWordChecker(Instruction):
    """Include keyword {keyword1} once in your response, keyword {keyword2} twice in your response."""

    def build_description(self, keyword1=None, keyword2=None):
        """Build the instruction description.

        Args:
        keyword1: A string representing a keyword that is expected in the response.
        keyword2: A string representing a keyword that is expected in the response.

        Returns:
        A string representing the instruction description.
        """
        if not keyword1:
            self._keyword1 = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword1 = keyword1.strip()
        if not keyword2:
            self._keyword2 = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword2 = keyword2.strip()

        self._description_pattern = (
            "Include the keyword \"{keyword1}\" exactly once in your response, and include the keyword \"{keyword2}\" exactly twice in your response."
        )

        return self._description_pattern.format(keyword1=self._keyword1, keyword2=self._keyword2)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword1": self._keyword1, "keyword2": self._keyword2}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword1", "keyword2"]

    def check_following(self, value):
        """Checks if the response contains the expected number of keywords.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains the expected number of keywords;
          otherwise, False.
        """
        actual_occurrences1 = len(re.findall(self._keyword1, value, flags=re.IGNORECASE))
        actual_occurrences2 = len(re.findall(self._keyword2, value, flags=re.IGNORECASE))

        return bool(actual_occurrences1 == 1 and actual_occurrences2 == 2)


class PalindromeBasicChecker(Instruction):
    """Include a palindrome in your response."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Include at least one palindrome word in your response (a single word that reads the same forward and backward), "
            "for example: level, racecar, or noon."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a palindrome.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains a palindrome; otherwise, False.
        """
        palindromes = [word for word in value.split() if word == word[::-1]]
        return len(palindromes) > 0


class KeywordSpecificPositionChecker(Instruction):
    """Include keyword {keyword1} in the {n}-th sentence, as the {m}-th word of that sentence."""

    def build_description(self, keyword=None, n=None, m=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          n: An integer representing the sentence number.
          m: An integer representing the word number.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(keyword, str):
                self._keyword = keyword[0].strip()
            else:
                self._keyword = keyword.strip()
        if not n:
            self._n = random.randint(1, 20)
        else:
            self._n = n
        if not m:
            self._m = random.randint(1, 30)
        else:
            self._m = m

        self._description_pattern = (
            "Include the keyword \"{keyword}\" (case-sensitive) in the {n}-th sentence (1-indexed), as the {m}-th word/token of that sentence (1-indexed).\n"
            "To avoid tokenization surprises, use the keyword as its own word with no punctuation attached."
        )

        return self._description_pattern.format(keyword=self._keyword, n=self._n, m=self._m)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword, "n": self._n, "m": self._m}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "n", "m"]

    def check_following(self, value):
        """Checks if the response contains the expected number of keywords.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains the expected number of keywords;
          otherwise, False.
        """
        sentences = instructions_util.split_into_sentences(value)
        if len(sentences) < self._n:
            return False
        words = instructions_util.nltk.word_tokenize(sentences[self._n - 1])
        if len(words) < self._m:
            return False
        return words[self._m - 1] == self._keyword


class StartEndChecker(Instruction):
    """Start and end your response with the same word (do not write anything after the last word, not even punctuation)."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Start and end your response with the same word (do not write anything after the last word, not even punctuation)."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response starts and ends with the same word.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts and ends with the same word;
          otherwise, False.
        """
        words = instructions_util.nltk.word_tokenize(value)
        if len(words) < 2:
            return False
        return words[0].lower() == words[-1].lower()


# Additional checkers derived from the first instruction set.


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


class CheckWhetherResponseParagraphNumberInRange(Instruction):
    """Check the paragraph count is within a range."""

    def build_description(self, *, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be set.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._description_pattern = (
            "The response must contain between {lower_bound} and {upper_bound} paragraphs (paragraphs are separated by blank lines)."
        )
        return self._description_pattern.format(lower_bound=self._lower_bound, upper_bound=self._upper_bound)

    def get_instruction_args(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}

    def get_instruction_args_keys(self):
        return ["lower_bound", "upper_bound"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)
        actual_count = len(paragraphs)
        return self._lower_bound <= actual_count <= self._upper_bound


class CheckWhetherResponseSentenceNumberInRange(Instruction):
    """Check the sentence count is within a range."""

    def build_description(self, *, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be set.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._description_pattern = (
            "The response should contain between {lower_bound} and {upper_bound} sentences."
        )
        return self._description_pattern.format(lower_bound=self._lower_bound, upper_bound=self._upper_bound)

    def get_instruction_args(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}

    def get_instruction_args_keys(self):
        return ["lower_bound", "upper_bound"]

    def check_following(self, value):
        response = _clean_text(value)
        sentences = nltk.sent_tokenize(response)
        actual_count = len(sentences)
        return self._lower_bound <= actual_count <= self._upper_bound


class CheckWhetherEachParagraphSentenceNumberInRange(Instruction):
    """Check each paragraph sentence count is within a range."""

    def build_description(self, *, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be set.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._description_pattern = (
            "Each paragraph should contain between {lower_bound} and {upper_bound} sentences."
        )
        return self._description_pattern.format(lower_bound=self._lower_bound, upper_bound=self._upper_bound)

    def get_instruction_args(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}

    def get_instruction_args_keys(self):
        return ["lower_bound", "upper_bound"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)

        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            actual_count = len(sentences)
            if actual_count < self._lower_bound or actual_count > self._upper_bound:
                return False

        return True


class CheckWhetherEachParagraphSentenceNumberInRangeList(Instruction):
    """Check each paragraph sentence count is within a provided range list."""

    def build_description(self, *, ranges=None):
        if not ranges:
            raise ValueError("ranges must be set.")
        self._ranges = ranges
        self._description_pattern = (
            "Your response must have exactly {num_paragraphs} paragraphs.\n"
            "For paragraph i (1-indexed), the number of sentences must fall within the corresponding inclusive range in: {ranges}."
        )
        return self._description_pattern.format(num_paragraphs=len(self._ranges or []), ranges=self._ranges)

    def get_instruction_args(self):
        return {"ranges": self._ranges}

    def get_instruction_args_keys(self):
        return ["ranges"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)

        if len(paragraphs) != len(self._ranges):
            return False

        for paragraph, range_pair in zip(paragraphs, self._ranges):
            lower_bound, upper_bound = range_pair
            sentences = nltk.sent_tokenize(paragraph)
            actual_count = len(sentences)
            if not (lower_bound <= actual_count <= upper_bound):
                return False

        return True


class CheckWhetherResponseWordCountInRange(Instruction):
    """Check the response word count is within a range."""

    def build_description(self, *, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be set.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._description_pattern = "The response should contain between {lower_bound} and {upper_bound} words."
        return self._description_pattern.format(lower_bound=self._lower_bound, upper_bound=self._upper_bound)

    def get_instruction_args(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}

    def get_instruction_args_keys(self):
        return ["lower_bound", "upper_bound"]

    def check_following(self, value):
        response_clean = re.sub(r"[^\w\s.-]", "", value)
        word_list = response_clean.split()
        word_count = len(word_list)
        return self._lower_bound <= word_count <= self._upper_bound


class CheckWhetherEachParagraphWordCountInRange(Instruction):
    """Check each paragraph word count is within a range."""

    def build_description(self, *, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be set.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._description_pattern = (
            "Each paragraph should contain between {lower_bound} and {upper_bound} words."
        )
        return self._description_pattern.format(lower_bound=self._lower_bound, upper_bound=self._upper_bound)

    def get_instruction_args(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}

    def get_instruction_args_keys(self):
        return ["lower_bound", "upper_bound"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)

        for paragraph in paragraphs:
            paragraph_clean = re.sub(r"[^\w\s.-]", "", paragraph)
            word_count = len(paragraph_clean.split())
            if not (self._lower_bound <= word_count <= self._upper_bound):
                return False

        return True


class CheckWhetherWholeResponseNotContainCertainSubstrings(Instruction):
    """Check the response does not contain any of the specified substrings."""

    def build_description(self, *, substrings=None):
        self._substrings = substrings or []
        self._description_pattern = (
            "The response must not contain any of these exact substrings (case-sensitive): {substrings}."
        )
        return self._description_pattern.format(substrings=self._substrings)

    def get_instruction_args(self):
        return {"substrings": self._substrings}

    def get_instruction_args_keys(self):
        return ["substrings"]

    def check_following(self, value):
        return all(substring not in value for substring in self._substrings)


class CheckWhetherWholeResponseNotContainCertainSubstring(Instruction):
    """Check the response does not contain a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "The response must not contain the exact substring \"{substring}\" (case-sensitive)."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        return self._substring not in value


class CheckWhetherEachSentenceBeginWithCertainSubstring(Instruction):
    """Check each sentence begins with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "Each sentence must begin with the exact substring \"{substring}\" (case-sensitive).\n"
            "Do not put any whitespace before this substring at the start of a sentence."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        response = _clean_text(value)
        sentences = nltk.sent_tokenize(response)
        return all(sentence.startswith(self._substring) for sentence in sentences)


class CheckWhetherEachParagraphBeginWithCertainSubstring(Instruction):
    """Check each paragraph begins with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "Each paragraph must begin with the exact substring \"{substring}\" (case-sensitive).\n"
            "Do not put any whitespace before this substring at the start of a paragraph."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)
        return all(paragraph.startswith(self._substring) for paragraph in paragraphs)


class CheckWhetherEachParagraphEndWithCertainSubstring(Instruction):
    """Check each paragraph ends with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "Each paragraph must end with the exact substring \"{substring}\" (case-sensitive).\n"
            "Do not add any characters after this substring within a paragraph."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        cleaned_response = _clean_text(value)
        paragraphs = _split_paragraphs(cleaned_response)
        return all(paragraph.endswith(self._substring) for paragraph in paragraphs)


class CheckWhetherEachSentenceEndWithCertainSubstring(Instruction):
    """Check each sentence ends with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "Each sentence must end with the exact substring \"{substring}\" (case-sensitive).\n"
            "Do not add any characters after this substring within a sentence."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        response = _clean_text(value)
        sentences = nltk.sent_tokenize(response)
        return all(sentence.endswith(self._substring) for sentence in sentences)


class CheckWhetherWholeResponseBeginWithCertainSubstring(Instruction):
    """Check the response begins with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "After trimming leading/trailing whitespace, the response must begin with the exact substring "
            "\"{substring}\" (case-sensitive)."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        return value.strip().startswith(self._substring)


class CheckWhetherWholeResponseEndWithCertainSubstring(Instruction):
    """Check the response ends with a specific substring."""

    def build_description(self, *, substring=None):
        if substring is None:
            raise ValueError("substring must be set.")
        self._substring = substring
        self._description_pattern = (
            "After trimming leading/trailing whitespace, the response must end with the exact substring "
            "\"{substring}\" (case-sensitive)."
        )
        return self._description_pattern.format(substring=self._substring)

    def get_instruction_args(self):
        return {"substring": self._substring}

    def get_instruction_args_keys(self):
        return ["substring"]

    def check_following(self, value):
        return value.strip().endswith(self._substring)


class CheckWhetherEachKeywordInListMetionedInRange(Instruction):
    """Check each keyword appears within a specified frequency range."""

    def build_description(self, *, keywords=None, lower_bound_times=None, upper_bound_times=None):
        if not keywords:
            raise ValueError("keywords must be set.")
        if lower_bound_times is None or upper_bound_times is None:
            raise ValueError("lower_bound_times and upper_bound_times must be set.")
        self._keywords = keywords or []
        self._lower_bound_times = lower_bound_times
        self._upper_bound_times = upper_bound_times
        self._description_pattern = (
            "Each of these keywords must appear as a standalone word between {lower_bound_times} and {upper_bound_times} times "
            "(case-insensitive): {keywords}."
        )
        return self._description_pattern.format(
            keywords=self._keywords,
            lower_bound_times=self._lower_bound_times,
            upper_bound_times=self._upper_bound_times,
        )

    def get_instruction_args(self):
        return {
            "keywords": self._keywords,
            "lower_bound_times": self._lower_bound_times,
            "upper_bound_times": self._upper_bound_times,
        }

    def get_instruction_args_keys(self):
        return ["keywords", "lower_bound_times", "upper_bound_times"]

    def check_following(self, value):
        response = _clean_text(value)
        response_lower = response.lower()

        for keyword in self._keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            matches = re.findall(pattern, response_lower)
            if len(matches) < self._lower_bound_times or len(matches) > self._upper_bound_times:
                return False

        return True


class CheckWhetherTotalKeywordInListMetionedInRange(Instruction):
    """Check total keyword occurrences across a list are within a range."""

    def build_description(self, *, keywords=None, lower_bound_times=None, upper_bound_times=None):
        if not keywords:
            raise ValueError("keywords must be set.")
        if lower_bound_times is None or upper_bound_times is None:
            raise ValueError("lower_bound_times and upper_bound_times must be set.")
        self._keywords = keywords or []
        self._lower_bound_times = lower_bound_times
        self._upper_bound_times = upper_bound_times
        self._description_pattern = (
            "Across these keywords {keywords}, the total number of standalone-word occurrences must be between "
            + "{lower_bound_times} and {upper_bound_times} (case-insensitive)."
        )
        return self._description_pattern.format(
            keywords=self._keywords,
            lower_bound_times=self._lower_bound_times,
            upper_bound_times=self._upper_bound_times,
        )

    def get_instruction_args(self):
        return {
            "keywords": self._keywords,
            "lower_bound_times": self._lower_bound_times,
            "upper_bound_times": self._upper_bound_times,
        }

    def get_instruction_args_keys(self):
        return ["keywords", "lower_bound_times", "upper_bound_times"]

    def check_following(self, value):
        response = _clean_text(value)
        response_lower = response.lower()

        count = 0
        for keyword in self._keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            matches = re.findall(pattern, response_lower)
            count += len(matches)

        return self._lower_bound_times <= count <= self._upper_bound_times


class CheckPercentageNumberPrecisionInResponse(Instruction):
    """Check precision of numbers preceding percentage signs."""

    def build_description(self, *, precision=None):
        if precision is None:
            raise ValueError("precision must be set.")
        self._precision = precision
        self._description_pattern = (
            "Write all percentages with a decimal point and exactly {precision} digits after the decimal, e.g. 12.34%."
        )
        return self._description_pattern.format(precision=self._precision)

    def get_instruction_args(self):
        return {"precision": self._precision}

    def get_instruction_args_keys(self):
        return ["precision"]

    def check_following(self, value):
        pattern = r"(\d+\.\d+|\d+)\s*%"

        matches = re.findall(pattern, value)

        for num_str in matches:
            if "." not in num_str:
                return False
            decimal_part = num_str.split(".")[1]
            if len(decimal_part) != self._precision:
                return False

        return True


class CheckNumberPrecisionInResponse(Instruction):
    """Check precision of all numbers in the response."""

    def build_description(self, *, precision=None):
        if precision is None:
            raise ValueError("precision must be set.")
        self._precision = precision
        self._description_pattern = "All numbers must have exactly {precision} decimal places (or none if 0)."
        return self._description_pattern.format(precision=self._precision)

    def get_instruction_args(self):
        return {"precision": self._precision}

    def get_instruction_args_keys(self):
        return ["precision"]

    def check_following(self, value):
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

        matches = re.finditer(number_pattern, value, flags=re.VERBOSE)

        for match in matches:
            num_str = match.group()
            clean_num = num_str.replace(",", "").rstrip("%")

            if "e" in clean_num.lower():
                mantissa = re.split("[eE]", clean_num)[0]
            else:
                mantissa = clean_num

            if "." in mantissa:
                decimal_part = mantissa.split(".")[-1]
                if len(decimal_part) != self._precision:
                    return False
            else:
                if self._precision != 0:
                    return False

        return True


class CheckWhetherHasNoArabicNumberInResponse(Instruction):
    """Check that no Arabic numerals appear in the response."""

    def build_description(self):
        self._description_pattern = "The response must not contain any digits 0–9 anywhere."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return re.search(r"\d", value) is None
