# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto

try:
    from verl.utils.reward_score import default_compute_score
    from verl.utils.reward_score.math_verify_reward_type_boxed import _extract_answer
except ImportError:
    from vero_reward import default_compute_score
    from vero_reward.math_verify_reward_type_boxed import _extract_answer

from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


_LLM_JUDGE_REWARD_TYPES = {"llm_judge", "instruction_following_llm_judge"}
_ONLINE_JUDGE_BACKENDS = {"openai", "openai_api", "openai_server", "vllm_server", "server"}
_LOCAL_JUDGE_BACKENDS = {"vllm_engine", "vllm", "local_vllm", "engine"}
_LOGGER = logging.getLogger(__name__)


@register("vero_vllm_judge")
class VeroVLLMJudgeRewardManager(AbstractRewardManager):
    """Vero reward manager with online or local vLLM-based LLM judge backends."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        llm_judge_kwargs: dict | None = None,
        format_score: float | None = None,
        instruction_following_reward_weight: float | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

        self.llm_judge_kwargs = llm_judge_kwargs or {}
        self._client = None
        self._async_client = None
        self._sampling_params: dict[str, Any] | None = None
        self._extra_body: dict[str, Any] = {}
        self._vllm_engine = None
        self._vllm_sampling_params = None
        self._vllm_tokenizer = None

        self._judge_model_name = self.llm_judge_kwargs.get("model_name", "qwen_vllm")
        self._judge_model_path = self.llm_judge_kwargs.get("model_path") or os.environ.get("VLLM_JUDGE_MODEL_PATH")
        if self._judge_model_path is None:
            # Backward compatibility: allow passing local model path in model_name.
            if "/" in self._judge_model_name or os.path.exists(self._judge_model_name):
                self._judge_model_path = self._judge_model_name

        raw_backend = self.llm_judge_kwargs.get("backend")
        if raw_backend is None and bool(self.llm_judge_kwargs.get("use_vllm_engine", False)):
            raw_backend = "vllm_engine"
        if raw_backend is None:
            raw_backend = "openai_server"
        self._judge_backend = str(raw_backend).strip().lower()
        if self._judge_backend in _ONLINE_JUDGE_BACKENDS:
            self._judge_backend = "openai_server"
        elif self._judge_backend in _LOCAL_JUDGE_BACKENDS:
            self._judge_backend = "vllm_engine"
        else:
            raise ValueError(
                "Unsupported llm_judge_kwargs.backend="
                f"{raw_backend!r}. Supported values: {sorted(_ONLINE_JUDGE_BACKENDS | _LOCAL_JUDGE_BACKENDS)}"
            )

        self._judge_api_key = self.llm_judge_kwargs.get("api_key") or os.environ.get("VLLM_API_KEY", "qwen")
        self._judge_base_url = (
            self.llm_judge_kwargs.get("base_url")
            or os.environ.get("VLLM_JUDGE_BASE_URL")
        )
        if self._judge_base_url is None:
            port = (
                self.llm_judge_kwargs.get("port")
                or os.environ.get("VLLM_JUDGE_PORT")
                or os.environ.get("VLLM_PORT")
                or "8000"
            )
            self._judge_base_url = f"http://localhost:{port}/v1"
        self._judge_server_root = self._judge_base_url
        if self._judge_server_root.endswith("/v1"):
            self._judge_server_root = self._judge_server_root[:-3]
        self._judge_timeout = float(self.llm_judge_kwargs.get("timeout", 120))

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self._judge_prompt_path = self.llm_judge_kwargs.get(
            "prompt_path",
            os.path.join(repo_root, "examples", "prompts", "llm_judge_reference.txt"),
        )
        self._judge_prompt_template = self.llm_judge_kwargs.get("prompt_template")
        if self._judge_prompt_template is None:
            if not os.path.exists(self._judge_prompt_path):
                raise FileNotFoundError(f"LLM judge prompt not found: {self._judge_prompt_path}")
            with open(self._judge_prompt_path, "r", encoding="utf-8") as handle:
                self._judge_prompt_template = handle.read()

        self._judge_score_scale = self.llm_judge_kwargs.get("score_scale", "minus1_to_1")
        self._judge_sleep_level = int(self.llm_judge_kwargs.get("sleep_level", 1))
        self._judge_enable_sleep = bool(self.llm_judge_kwargs.get("enable_sleep_mode", True))
        self._judge_wake_tags = self.llm_judge_kwargs.get("wake_tags", ["kv_cache", "weights"])
        self._judge_log_outputs = self.llm_judge_kwargs.get("log_outputs", True)
        self._debug_enabled = bool(self.llm_judge_kwargs.get("debug_enabled", True))
        self._debug_output_dir = self.llm_judge_kwargs.get(
            "debug_output_dir",
            os.path.join(repo_root, "outputs", "llm_judge_debug"),
        )
        self._debug_max_items = int(self.llm_judge_kwargs.get("debug_max_items", 16))
        self._debug_saved = 0
        self._debug_every = int(self.llm_judge_kwargs.get("debug_every", 1))
        self._debug_file_path = None
        if format_score is None:
            raise ValueError("format_score must be provided via reward_model.reward_kwargs.format_score")
        try:
            self._format_score = float(format_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid format_score={format_score!r}; must be a float.") from exc
        if not 0.0 <= self._format_score <= 1.0:
            raise ValueError(f"format_score must be in [0, 1], got {self._format_score}")
        self._format_score_checked = False
        if instruction_following_reward_weight is None:
            raise ValueError(
                "instruction_following_reward_weight must be provided via "
                "reward_model.reward_kwargs.instruction_following_reward_weight"
            )
        try:
            self._instruction_following_reward_weight = float(instruction_following_reward_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid instruction_following_reward_weight="
                f"{instruction_following_reward_weight!r}; must be a float."
            ) from exc
        if not 0.0 <= self._instruction_following_reward_weight <= 1.0:
            raise ValueError(
                "instruction_following_reward_weight must be in [0, 1], got "
                f"{self._instruction_following_reward_weight}"
            )
        if self._debug_enabled:
            os.makedirs(self._debug_output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._debug_file_path = os.path.join(
                self._debug_output_dir, f"llm_judge_debug_{timestamp}_{os.getpid()}.jsonl"
            )
            _LOGGER.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
            _LOGGER.info("LLM judge debug enabled, writing to %s", self._debug_file_path)
            print(f"[llm_judge] debug enabled, writing to {self._debug_file_path}")

        _LOGGER.info(
            "Initialized LLM judge backend=%s model_name=%s model_path=%s base_url=%s",
            self._judge_backend,
            self._judge_model_name,
            self._judge_model_path,
            self._judge_base_url,
        )
        self._debug(
            f"judge backend={self._judge_backend} model_name={self._judge_model_name} "
            f"model_path={self._judge_model_path} base_url={self._judge_base_url}"
        )

    def _debug(self, message: str) -> None:
        if not self._debug_enabled:
            return
        _LOGGER.info("%s", message)
        print(f"[llm_judge] {message}")

    def _ensure_client(self) -> None:
        if self._judge_backend != "openai_server":
            return
        if self._client is not None and self._async_client is not None:
            return
        try:
            from openai import AsyncOpenAI, OpenAI
        except Exception:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError("openai python client is required for LLM judge reward.") from exc
            AsyncOpenAI = None

        if not self._judge_base_url:
            raise ValueError("llm_judge_kwargs.base_url or VLLM_JUDGE_BASE_URL must be set for online judge.")
        _LOGGER.info("Initializing vLLM judge client base_url=%s model=%s", self._judge_base_url, self._judge_model_name)
        self._debug(f"init client base_url={self._judge_base_url} model={self._judge_model_name}")

        if self._client is None:
            self._client = OpenAI(
                api_key=self._judge_api_key, base_url=self._judge_base_url, timeout=self._judge_timeout
            )
        if self._async_client is None and AsyncOpenAI is not None:
            self._async_client = AsyncOpenAI(
                api_key=self._judge_api_key, base_url=self._judge_base_url, timeout=self._judge_timeout
            )
        self._sampling_params = {
            "temperature": self.llm_judge_kwargs.get("temperature", 0.7),
            "top_p": self.llm_judge_kwargs.get("top_p", 0.8),
            "max_tokens": self.llm_judge_kwargs.get("max_tokens", 1024),
            "presence_penalty": self.llm_judge_kwargs.get("presence_penalty", 1.5),
            "frequency_penalty": self.llm_judge_kwargs.get("frequency_penalty", 1.0),
        }
        self._extra_body = {}
        if self.llm_judge_kwargs.get("top_k") is not None:
            self._extra_body["top_k"] = self.llm_judge_kwargs.get("top_k")
        if self.llm_judge_kwargs.get("min_p") is not None:
            self._extra_body["min_p"] = self.llm_judge_kwargs.get("min_p")
        stop = self.llm_judge_kwargs.get("stop")
        if stop:
            self._sampling_params["stop"] = stop
        _LOGGER.info("Initialized vLLM judge sampling params=%s", self._sampling_params)
        self._debug(f"sampling params {self._sampling_params}")
        if self._extra_body:
            _LOGGER.info("Initialized vLLM judge extra_body=%s", self._extra_body)
            self._debug(f"extra_body {self._extra_body}")

    def _ensure_vllm_engine(self) -> None:
        if self._judge_backend != "vllm_engine":
            return
        if self._vllm_engine is not None and self._vllm_sampling_params is not None and self._vllm_tokenizer is not None:
            return

        if not self._judge_model_path:
            raise ValueError(
                "llm_judge_kwargs.model_path (or VLLM_JUDGE_MODEL_PATH) must be set when backend=vllm_engine."
            )

        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError("vllm python package is required when llm_judge backend is vllm_engine.") from exc

        tensor_parallel_size = int(
            self.llm_judge_kwargs.get(
                "tensor_parallel_size",
                os.environ.get("VLLM_JUDGE_TENSOR_PARALLEL_SIZE", os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            )
        )
        dtype = self.llm_judge_kwargs.get("dtype", os.environ.get("VLLM_JUDGE_DTYPE", "bfloat16"))
        max_model_len = self.llm_judge_kwargs.get("max_model_len")
        max_num_seqs = self.llm_judge_kwargs.get("max_num_seqs")
        gpu_memory_utilization = float(
            self.llm_judge_kwargs.get(
                "gpu_memory_utilization",
                os.environ.get("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", "0.9"),
            )
        )
        trust_remote_code = bool(self.llm_judge_kwargs.get("trust_remote_code", False))

        llm_kwargs = {
            "model": self._judge_model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if self._judge_enable_sleep:
            llm_kwargs["enable_sleep_mode"] = True
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)

        _LOGGER.info("Initializing local vLLM judge engine kwargs=%s", llm_kwargs)
        self._debug(f"init local vllm engine kwargs={llm_kwargs}")
        self._vllm_engine = LLM(**llm_kwargs)
        self._vllm_tokenizer = self._vllm_engine.get_tokenizer()

        # Optionally override chat template (e.g., Qwen3 non-thinking template).
        chat_template_path = self.llm_judge_kwargs.get("chat_template_path")
        if not chat_template_path and isinstance(self._judge_model_path, str) and "qwen3" in self._judge_model_path.lower():
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            default_template = os.path.join(repo_root, "examples", "prompts", "qwen3_nonthinking.jinja")
            if os.path.isfile(default_template):
                chat_template_path = default_template
        if chat_template_path:
            if not os.path.isfile(chat_template_path):
                raise FileNotFoundError(f"Chat template file not found: {chat_template_path}")
            with open(chat_template_path, "r", encoding="utf-8") as handle:
                chat_template = handle.read()
            if hasattr(self._vllm_tokenizer, "chat_template"):
                self._vllm_tokenizer.chat_template = chat_template
            self._debug(f"set local vllm chat_template={chat_template_path}")

        sampling_kwargs: dict[str, Any] = {
            "temperature": float(self.llm_judge_kwargs.get("temperature", 0.0)),
            "top_p": float(self.llm_judge_kwargs.get("top_p", 1.0)),
            "max_tokens": int(self.llm_judge_kwargs.get("max_tokens", 2048)),
            "presence_penalty": float(self.llm_judge_kwargs.get("presence_penalty", 0.0)),
            "frequency_penalty": float(self.llm_judge_kwargs.get("frequency_penalty", 0.0)),
        }
        top_k = self.llm_judge_kwargs.get("top_k")
        if top_k is not None:
            sampling_kwargs["top_k"] = int(top_k)
        min_p = self.llm_judge_kwargs.get("min_p")
        if min_p is not None:
            sampling_kwargs["min_p"] = float(min_p)
        stop = self.llm_judge_kwargs.get("stop")
        if stop:
            sampling_kwargs["stop"] = stop
        self._vllm_sampling_params = SamplingParams(**sampling_kwargs)
        self._debug(f"local vllm sampling params {sampling_kwargs}")

    def _build_local_chat_prompt(self, prompt: str) -> str:
        if self._vllm_tokenizer is None:
            raise RuntimeError("Local vLLM tokenizer is not initialized.")
        system_prompt = self.llm_judge_kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        return self._vllm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _wake_local_llm(self) -> None:
        if not self._judge_enable_sleep:
            return
        if self._vllm_engine is None:
            return
        try:
            self._vllm_engine.wake_up()
            _LOGGER.info("local vLLM engine wake_up")
            self._debug("local vllm wake_up")
        except Exception as exc:
            _LOGGER.warning("local vLLM engine wake_up failed: %s", exc)

    def _sleep_local_llm(self) -> None:
        if not self._judge_enable_sleep:
            return
        if self._vllm_engine is None:
            return
        try:
            self._vllm_engine.sleep(level=self._judge_sleep_level)
            _LOGGER.info("local vLLM engine sleep(level=%s)", self._judge_sleep_level)
            self._debug(f"local vllm sleep level={self._judge_sleep_level}")
        except Exception as exc:
            _LOGGER.warning("local vLLM engine sleep failed: %s", exc)

    def _judge_batch_local(self, prompts: list[str]) -> tuple[list[float], list[float], list[str]]:
        self._ensure_vllm_engine()
        if not prompts:
            return [], [], []
        assert self._vllm_engine is not None
        assert self._vllm_sampling_params is not None
        chat_prompts = [self._build_local_chat_prompt(prompt) for prompt in prompts]
        for idx, (prompt, chat_prompt) in enumerate(zip(prompts, chat_prompts)):
            self._save_judge_model_input_debug(
                backend="vllm_engine",
                user_prompt=prompt,
                model_input=chat_prompt,
                batch_index=idx,
            )
        _LOGGER.info("LLM judge generating for %d prompts via local vLLM engine", len(prompts))
        self._debug(f"local vllm generate start batch={len(prompts)}")
        self._wake_llm()
        try:
            outputs = self._vllm_engine.generate(
                chat_prompts,
                sampling_params=self._vllm_sampling_params,
                use_tqdm=False,
            )
        finally:
            self._sleep_llm()
        scaled_scores: list[float] = []
        raw_scores: list[float] = []
        texts: list[str] = []
        for output in outputs:
            text = ""
            if output.outputs:
                text = (output.outputs[0].text or "").strip()
            raw = self._extract_score(text)
            scaled = self._scale_score(raw)
            scaled_scores.append(scaled)
            raw_scores.append(raw if raw is not None else 0.0)
            texts.append(text)
            self._debug(f"parsed score raw={raw} scaled={scaled}")
        _LOGGER.info("LLM judge local vLLM generation complete")
        self._debug("local vllm generate done")
        return scaled_scores, raw_scores, texts

    def _post_server_action(self, action: str, params: dict[str, str] | None = None) -> None:
        if not self._judge_server_root:
            return
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self._judge_server_root}/{action}{query}"
        try:
            headers = {}
            if self._judge_api_key:
                headers["Authorization"] = f"Bearer {self._judge_api_key}"
            req = urllib.request.Request(url, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=self._judge_timeout) as resp:
                _LOGGER.info("vLLM server %s -> %s", action, resp.status)
        except Exception as exc:
            _LOGGER.warning("vLLM server %s failed: %s", action, exc)

    def _post_sleep_wake(self, action: str, level: int | None = None, tags: list[str] | None = None) -> None:
        if action not in {"sleep", "wake_up"}:
            return
        params = {}
        if level is not None:
            params["level"] = str(level)
        if tags:
            params["tags"] = ",".join(tags)
        self._post_server_action(action, params=params)

    def _is_sleeping(self) -> bool | None:
        if not self._judge_server_root:
            return None
        url = f"{self._judge_server_root}/is_sleeping"
        try:
            with urllib.request.urlopen(url, timeout=self._judge_timeout) as resp:
                data = resp.read().decode("utf-8")
            payload = json.loads(data)
            if isinstance(payload, dict):
                value = payload.get("sleeping")
                if isinstance(value, bool):
                    return value
            if isinstance(payload, bool):
                return payload
        except Exception as exc:
            _LOGGER.warning("vLLM server is_sleeping failed: %s", exc)
        return None

    def _wait_sleep_state(self, expect_sleeping: bool, timeout: float | None = None) -> None:
        if not self._judge_server_root:
            return
        if timeout is None:
            timeout = self._judge_timeout
        start = time.time()
        last = None
        while True:
            last = self._is_sleeping()
            if last is None:
                break
            if last == expect_sleeping:
                break
            if time.time() - start > timeout:
                _LOGGER.warning(
                    "Timed out waiting for vLLM sleep state=%s (last=%s)",
                    expect_sleeping,
                    last,
                )
                break
            time.sleep(0.2)

    def _wake_llm(self) -> None:
        if not self._judge_enable_sleep:
            return
        if self._judge_backend == "vllm_engine":
            self._wake_local_llm()
            return
        self._post_sleep_wake("wake_up")
        self._wait_sleep_state(expect_sleeping=False)
        self._debug("wake_up")

    def _sleep_llm(self) -> None:
        if not self._judge_enable_sleep:
            return
        if self._judge_backend == "vllm_engine":
            self._sleep_local_llm()
            return
        # self._post_server_action("reset_prefix_cache")
        # self._post_server_action("reset_mm_cache")
        _LOGGER.info("vLLM judge sleep(level=%s)", self._judge_sleep_level)
        self._post_sleep_wake("sleep", level=self._judge_sleep_level)
        self._wait_sleep_state(expect_sleeping=True)
        self._debug(f"sleep level={self._judge_sleep_level}")

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            role = str(msg.get("role", "")).strip() or "unknown"
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(parts)

    def _normalize_message_dict(self, message: Any) -> dict[str, Any] | None:
        if hasattr(message, "model_dump"):
            try:
                message = message.model_dump()
            except Exception:
                pass
        if isinstance(message, dict):
            if "role" not in message and "content" not in message:
                return None
            return {
                "role": str(message.get("role", "unknown")),
                "content": message.get("content", ""),
            }
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None and content is None:
            return None
        return {
            "role": str(role or "unknown"),
            "content": content if content is not None else "",
        }

    def _get_message_list_from_non_tensor(self, data_item) -> tuple[list[dict[str, Any]] | None, str | None]:
        for source in ("messages", "prompt", "raw_prompt"):
            raw = data_item.non_tensor_batch.get(source)
            if raw is None:
                continue
            if hasattr(raw, "model_dump"):
                try:
                    raw = raw.model_dump()
                except Exception:
                    pass
            if isinstance(raw, dict) and "messages" in raw:
                raw = raw["messages"]
            if isinstance(raw, tuple):
                raw = list(raw)
            if isinstance(raw, list):
                normalized: list[dict[str, Any]] = []
                for item in raw:
                    msg = self._normalize_message_dict(item)
                    if msg is not None:
                        normalized.append(msg)
                if normalized:
                    return normalized, source
        return None, None

    def _build_conversation_from_messages(self, messages: list[dict[str, Any]]) -> tuple[str, bool, bool]:
        normalized = list(messages)
        removed_first_system = False
        dropped_last_assistant = False

        for idx, msg in enumerate(normalized):
            role = str(msg.get("role", "")).strip().lower()
            if role == "system":
                normalized = normalized[:idx] + normalized[idx + 1 :]
                removed_first_system = True
                break

        if normalized and str(normalized[-1].get("role", "")).strip().lower() == "assistant":
            normalized = normalized[:-1]
            dropped_last_assistant = True

        return self._format_messages(normalized), removed_first_system, dropped_last_assistant

    def _strip_system_prefix_with_anchors(self, text: str) -> tuple[str, bool]:
        anchors = [
            re.compile(r"(^|\n)\s*(?P<anchor>user\s*\n)", re.IGNORECASE),
            re.compile(r"(?P<anchor><\|im_start\|>\s*user\s*\n)", re.IGNORECASE),
            re.compile(r"(?P<anchor><\|start_header_id\|>\s*user\s*<\|end_header_id\|>)", re.IGNORECASE),
        ]
        best_start: int | None = None
        for pattern in anchors:
            match = pattern.search(text)
            if not match:
                continue
            start = match.start("anchor")
            if best_start is None or start < best_start:
                best_start = start
        if best_start is None:
            return text, False
        return text[best_start:], True

    def _strip_trailing_assistant_marker(self, text: str) -> tuple[str, bool]:
        stripped = text.rstrip()
        patterns = [
            re.compile(r"\n\s*assistant\s*:?\s*$", re.IGNORECASE),
            re.compile(r"<\|im_start\|>\s*assistant\s*$", re.IGNORECASE),
            re.compile(r"<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>\s*$", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(stripped)
            if not match:
                continue
            return stripped[: match.start()].rstrip(), True
        return text, False

    def _get_conversation_text(self, data_item, fallback_prompt: str) -> tuple[str, str, bool, bool]:
        messages, source = self._get_message_list_from_non_tensor(data_item)
        if messages is not None:
            conversation, removed_first_system, dropped_last_assistant = self._build_conversation_from_messages(messages)
            return conversation, str(source), removed_first_system, dropped_last_assistant
        conversation, stripped = self._strip_system_prefix_with_anchors(fallback_prompt)
        conversation, dropped_assistant_marker = self._strip_trailing_assistant_marker(conversation)
        return conversation, "decoded_fallback", stripped, dropped_assistant_marker

    def _build_judge_prompt(self, conversation: str, output: str, label: str) -> str:
        prompt = self._judge_prompt_template
        prompt = prompt.replace("{input}", str(conversation))
        prompt = prompt.replace("{output}", str(output))
        prompt = prompt.replace("{label}", str(label))
        self._debug(f"built judge prompt len={len(prompt)}")
        return prompt

    def _extract_score(self, text: str) -> float | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            payload = match.group(0)
            try:
                data = json.loads(payload)
                score = data.get("SCORE") if isinstance(data, dict) else None
                if score is None and isinstance(data, dict):
                    score = data.get("score")
                if score is not None:
                    return float(score)
            except Exception:
                pass

        match = re.search(r"\"?SCORE\"?\s*:\s*\"?([0-9]+(?:\.[0-9]+)?)\"?", text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        match = re.search(r"\b(10|[1-9])(?:\.\d+)?\b", text)
        if match:
            return float(match.group(1))

        return None

    def _scale_score(self, raw_score: float | None) -> float:
        if raw_score is None:
            return 0.0
        raw_score = max(1.0, min(10.0, float(raw_score)))
        scale = str(self._judge_score_scale).lower()
        if scale in {"raw", "none"}:
            return raw_score
        if scale in {"0-1", "0_to_1", "zero_to_one"}:
            return (raw_score - 1.0) / 9.0
        if scale in {"-1-1", "-1_to_1", "minus1_to_1"}:
            return (raw_score - 5.5) / 4.5
        return (raw_score - 1.0) / 9.0

    def _is_max_context_length_bad_request(self, exc: Exception) -> bool:
        # Keep this matcher intentionally strict so only the known
        # context-length related 400 errors are skipped.
        if exc.__class__.__name__ != "BadRequestError":
            return False

        status_code = getattr(exc, "status_code", None)
        if status_code is not None and int(status_code) != 400:
            return False

        message = str(exc).lower()
        return (
            "maximum context length is" in message
            and "input tokens" in message
        )

    def _is_api_timeout_error(self, exc: Exception) -> bool:
        # Keep timeout matching conservative and targeted to the OpenAI client
        # timeout surface used by the online judge path.
        current = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            name = current.__class__.__name__
            if name in {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}:
                return True
            message = str(current).lower()
            if "request timed out" in message or "timed out" in message:
                return True
            next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
            if not isinstance(next_exc, Exception):
                break
            current = next_exc
        return False

    def _save_judge_model_input_debug(
        self,
        *,
        backend: str,
        user_prompt: str,
        model_input: Any,
        batch_index: int | None = None,
    ) -> None:
        self._maybe_save_debug(
            {
                "stage": "judge_model_input",
                "backend": backend,
                "batch_index": batch_index,
                "judge_prompt_user_content": user_prompt,
                "judge_model_input": model_input,
            },
            force=True,
        )

    async def _judge_one(self, prompt: str, semaphore: asyncio.Semaphore) -> tuple[float, float, str]:
        async with semaphore:
            request_messages = [{"role": "user", "content": prompt}]
            self._save_judge_model_input_debug(
                backend="openai_server",
                user_prompt=prompt,
                model_input={"messages": request_messages},
            )
            try:
                if self._async_client is not None:
                    response = await self._async_client.chat.completions.create(
                        model=self._judge_model_name,
                        messages=request_messages,
                        extra_body=self._extra_body if self._extra_body else None,
                        **self._sampling_params,
                    )
                else:
                    response = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self._judge_model_name,
                        messages=request_messages,
                        extra_body=self._extra_body if self._extra_body else None,
                        **self._sampling_params,
                    )
            except Exception as exc:
                if self._is_api_timeout_error(exc):
                    _LOGGER.warning(
                        "Skipping LLM judge due to API timeout; "
                        "fallback score=0. prompt_chars=%d",
                        len(prompt),
                    )
                    print(f"[llm_judge] API timeout; fallback score=0 prompt_chars={len(prompt)}", flush=True)
                    self._debug("judge skipped due to API timeout; fallback score=0")
                    return 0.0, 0.0, "SKIPPED_API_TIMEOUT"
                if self._is_max_context_length_bad_request(exc):
                    _LOGGER.warning(
                        "Skipping LLM judge due to max context length exceeded; "
                        "fallback score=0. prompt_chars=%d",
                        len(prompt),
                    )
                    self._debug("judge skipped due to max context length exceeded; fallback score=0")
                    return 0.0, 0.0, "SKIPPED_MAX_CONTEXT_LENGTH"
                raise
        text = response.choices[0].message.content or ""
        raw_score = self._extract_score(text)
        scaled_score = self._scale_score(raw_score)
        self._debug(f"parsed score raw={raw_score} scaled={scaled_score}")
        return scaled_score, (raw_score if raw_score is not None else 0.0), text

    async def _judge_batch_async(self, prompts: list[str]) -> tuple[list[float], list[float], list[str]]:
        if self._judge_backend != "openai_server":
            raise RuntimeError("_judge_batch_async only supports backend=openai_server.")
        self._ensure_client()
        _LOGGER.info("LLM judge generating for %d prompts via OpenAI API", len(prompts))
        self._debug(f"generate start batch={len(prompts)}")
        self._wake_llm()

        max_concurrent = int(self.llm_judge_kwargs.get("max_concurrent_requests", 8))
        if max_concurrent <= 0:
            max_concurrent = 1
        semaphore = asyncio.Semaphore(max_concurrent)

        try:
            results = await asyncio.gather(*(self._judge_one(prompt, semaphore) for prompt in prompts))
        finally:
            self._sleep_llm()
        _LOGGER.info("LLM judge generation complete")
        self._debug("generate done")
        scaled_scores, raw_scores, texts = zip(*results) if results else ([], [], [])
        return list(scaled_scores), list(raw_scores), list(texts)

    def _judge_batch(self, prompts: list[str]) -> tuple[list[float], list[float], list[str]]:
        if self._judge_backend == "vllm_engine":
            start = time.perf_counter()
            try:
                return self._judge_batch_local(prompts)
            finally:
                elapsed = time.perf_counter() - start
                avg = (elapsed / len(prompts)) if prompts else 0.0
                _LOGGER.info(
                    "LLM judge batch complete (local vllm) total_s=%.3f avg_s=%.3f prompts=%d",
                    elapsed,
                    avg,
                    len(prompts),
                )
                self._debug(f"judge batch (local vllm) total_s={elapsed:.3f} avg_s={avg:.3f} prompts={len(prompts)}")

        # Use thread-compatible async loop management instead of asyncio.run()
        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._judge_batch_async(prompts))
        finally:
            elapsed = time.perf_counter() - start
            avg = (elapsed / len(prompts)) if prompts else 0.0
            _LOGGER.info(
                "LLM judge batch complete total_s=%.3f avg_s=%.3f prompts=%d",
                elapsed,
                avg,
                len(prompts),
            )
            self._debug(f"judge batch total_s={elapsed:.3f} avg_s={avg:.3f} prompts={len(prompts)}")
            loop.close()

    def _maybe_save_debug(self, payload: dict[str, Any], *, force: bool = False) -> None:
        if not self._debug_enabled or self._debug_file_path is None:
            return
        if not force:
            if self._debug_saved >= self._debug_max_items:
                return
            if (self._debug_saved % self._debug_every) != 0:
                self._debug_saved += 1
                return
        payload = dict(payload)
        payload["ts"] = time.time()
        payload["pid"] = os.getpid()
        with open(self._debug_file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if not force:
            self._debug_saved += 1

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        batch_size = len(data)
        reward_extra_info["llm_judge_score"] = [None] * batch_size
        reward_extra_info["llm_judge_reward"] = [None] * batch_size
        if self._judge_log_outputs:
            reward_extra_info["llm_judge_output"] = [None] * batch_size

        llm_indices = []
        llm_prompts = []
        llm_meta = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            reward_type = None
            if isinstance(extra_info, dict):
                reward_type = extra_info.get("reward_type")
                if isinstance(reward_type, str):
                    reward_type = reward_type.strip().lower()

            _LOGGER.debug("Using compute_score for data_source=%s", data_source)
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score = None
            accuracy = None
            format_reward = None
            if isinstance(result, dict):
                score = result.get("score")
                accuracy = result.get("accuracy")
                format_reward = result.get("format")
                if score is None and isinstance(accuracy, (int, float)):
                    score = accuracy
                if not self._format_score_checked:
                    if all(isinstance(value, (int, float)) for value in (accuracy, format_reward, score)):
                        denom = format_reward - accuracy
                        if abs(denom) > 1e-8:
                            inferred = (score - accuracy) / denom
                            if not math.isfinite(inferred) or abs(inferred - self._format_score) > 1e-3:
                                raise ValueError(
                                    "format_score mismatch: compute_score appears to use "
                                    f"{inferred:.6f} but reward_model.reward_kwargs.format_score="
                                    f"{self._format_score:.6f}"
                                )
                    self._format_score_checked = True
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                accuracy = result
                reward_extra_info["acc"].append(score)

            if reward_type in _LLM_JUDGE_REWARD_TYPES:
                _LOGGER.info("Routing to LLM judge reward_type=%s data_source=%s", reward_type, data_source)
                self._debug(f"route to llm_judge reward_type={reward_type} data_source={data_source}")
                conversation, conversation_source, system_removed, dropped_last_assistant = self._get_conversation_text(
                    data_item, prompt_str
                )
                parsed_answer = _extract_answer(response_str)
                judge_prompt = self._build_judge_prompt(
                    conversation=conversation,
                    output=parsed_answer,
                    label=ground_truth,
                )
                self._maybe_save_debug(
                    {
                        "stage": "judge_prompt",
                        "uid": data_item.non_tensor_batch.get("uid"),
                        "data_source": data_source,
                        "reward_type": reward_type,
                        "prompt": judge_prompt,
                        "conversation": conversation,
                        "conversation_source": conversation_source,
                        "system_removed": system_removed,
                        "dropped_last_assistant": dropped_last_assistant,
                        "response": response_str,
                        "parsed_answer": parsed_answer,
                        "ground_truth": ground_truth,
                        "format_score": self._format_score,
                        "instruction_following_reward_weight": self._instruction_following_reward_weight,
                    }
                )
                llm_indices.append(i)
                llm_prompts.append(judge_prompt)
                llm_meta.append(
                    {
                        "prompt_str": prompt_str,
                        "response_str": response_str,
                        "parsed_answer": parsed_answer,
                        "ground_truth": ground_truth,
                        "data_source": data_source,
                        "reward_type": reward_type,
                        "valid_response_length": int(valid_response_length),
                        "base_accuracy": accuracy,
                        "base_score": score,
                        "base_format": format_reward,
                        "format_score": self._format_score,
                        "instruction_following_reward_weight": self._instruction_following_reward_weight,
                    }
                )
                continue

            reward = score
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                self._debug(
                    f"overlong_penalty applied len={valid_response_length} exceed={exceed_len} "
                    f"penalty={overlong_reward}"
                )
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if llm_prompts:
            _LOGGER.info("Scoring %d items with LLM judge", len(llm_prompts))
            scaled_scores, raw_scores, outputs = self._judge_batch(llm_prompts)
            for idx, scaled_score, raw_score, output_text, meta in zip(
                llm_indices, scaled_scores, raw_scores, outputs, llm_meta
            ):
                base_accuracy = meta.get("base_accuracy")
                if not isinstance(base_accuracy, (int, float)):
                    base_accuracy = 0.0

                format_reward = meta.get("base_format")
                format_weight = self._format_score

                reward_type = meta.get("reward_type")
                inst_weight = 0.0
                if reward_type == "llm_judge":
                    combined_accuracy = scaled_score
                else:
                    inst_weight = self._instruction_following_reward_weight
                    combined_accuracy = inst_weight * base_accuracy + (1.0 - inst_weight) * scaled_score

                if not isinstance(format_reward, (int, float)):
                    combined_score = combined_accuracy
                else:
                    combined_score = (1.0 - format_weight) * combined_accuracy + format_weight * format_reward

                reward = combined_score
                if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_resp_len - overlong_buffer_len
                    exceed_len = meta["valid_response_length"] - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    reward += overlong_reward
                    self._debug(
                        f"overlong_penalty(llm) len={meta['valid_response_length']} exceed={exceed_len} "
                        f"penalty={overlong_reward}"
                    )
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)

                reward_tensor[idx, meta["valid_response_length"] - 1] = reward
                if "accuracy" in reward_extra_info and idx < len(reward_extra_info["accuracy"]):
                    reward_extra_info["accuracy"][idx] = combined_accuracy
                if "score" in reward_extra_info and idx < len(reward_extra_info["score"]):
                    reward_extra_info["score"][idx] = combined_score
                if "acc" in reward_extra_info and idx < len(reward_extra_info["acc"]):
                    reward_extra_info["acc"][idx] = combined_accuracy
                reward_extra_info["llm_judge_score"][idx] = raw_score
                reward_extra_info["llm_judge_reward"][idx] = scaled_score
                self._maybe_save_debug(
                    {
                        "stage": "judge_output",
                        "data_source": meta["data_source"],
                        "reward_type": reward_type,
                        "raw_score": raw_score,
                        "scaled_score": scaled_score,
                        "format_score": format_weight,
                        "base_accuracy": base_accuracy,
                        "base_format": format_reward,
                        "instruction_following_reward_weight": inst_weight,
                        "combined_accuracy": combined_accuracy,
                        "combined_score": combined_score,
                        "judge_output": output_text,
                        "judge_output_raw": output_text,
                        "llm_judge_output": output_text,
                        "prompt": meta["prompt_str"],
                        "response": meta["response_str"],
                        "parsed_answer": meta["parsed_answer"],
                        "ground_truth": meta["ground_truth"],
                    },
                    force=True,
                )
                if self._judge_log_outputs:
                    reward_extra_info["llm_judge_output"][idx] = output_text

                data_source = meta["data_source"]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", meta["prompt_str"])
                    print("[response]", meta["response_str"])
                    print("[ground_truth]", meta["ground_truth"])
                    print("[llm_judge_score]", raw_score)
                    print("[llm_judge_reward]", scaled_score)
                    if self._judge_log_outputs:
                        print("[llm_judge_output]", output_text)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
