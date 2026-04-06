"""Shared vLLM judge utilities for local scoring and extraction tasks."""
from __future__ import annotations

import atexit
import gc
import json
import os
import random
import socket
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional
from openai import DefaultHttpxClient, OpenAI

from vllm import LLM, SamplingParams

_DEFAULT_SYSTEM_PROMPT = ""

_GPT_MODEL_NAME = ["gpt-4o-2024-05-13"]
_VLM_HINTS = (
    "qwen3-vl",
    "qwen3vl",
    "qwen2.5-vl",
    "qwen2.5vl",
    "qwen25vl",
    "qwen-vl",
    "vision",
    "multimodal",
    "vl-instruct",
)


def _pick_unused_port() -> int:
    for _ in range(200):
        port = random.randint(20000, 65000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
        return port
    raise RuntimeError("Failed to find a free TCP port for vLLM judge server.")


def _strip_chat_completions(url: str) -> str:
    url = url.rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url



def _resolve_model_path(model_hint: Optional[str]) -> str:
    """Resolve the model path for the local vLLM judge."""
    for env_var in ("CHARXIV_JUDGE_MODEL_PATH", "VLLM_JUDGE_MODEL_PATH"):
        candidate = os.getenv(env_var)
        if candidate:
            return candidate
    if model_hint:
        return model_hint
    env_model = os.getenv("MODEL_VERSION")
    if env_model:
        return env_model
    raise RuntimeError(
        "Unable to resolve judge model path. Set CHARXIV_JUDGE_MODEL_PATH or MODEL_VERSION."
    )


def resolve_judge_mode(model_hint: Optional[str] = None) -> str:
    """Resolve judge mode: 'vlm' uses image+text, 'llm' uses text only."""
    override = os.getenv("LMMS_EVAL_JUDGE_MODE", "").strip().lower()
    if override in {"vlm", "llm"}:
        return override
    model_path = _resolve_model_path(model_hint).lower()
    if any(hint in model_path for hint in _VLM_HINTS):
        return "vlm"
    return "llm"


def _is_multimodal_model_path(model_path: str) -> bool:
    lowered = (model_path or "").lower()
    return any(hint in lowered for hint in _VLM_HINTS)


@dataclass(frozen=True)
class JudgeConfig:
    model_path: str
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    max_num_seqs: Optional[int]
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    gpu_memory_utilization: float
    system_prompt: Optional[str]


def _load_config(model_hint: Optional[str]) -> JudgeConfig:
    model_path = _resolve_model_path(model_hint)
    tensor_parallel = int(os.getenv("CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE", os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    dtype = os.getenv("CHARXIV_JUDGE_DTYPE", os.getenv("VLLM_JUDGE_DTYPE", "bfloat16"))
    max_model_len = int(os.getenv("CHARXIV_JUDGE_MAX_MODEL_LEN", "18384"))
    max_num_seqs = int(os.getenv("VLLM_JUDGE_MAX_NUM_SEQS", "256"))
    temperature = float(os.getenv("CHARXIV_JUDGE_TEMPERATURE", "0.0"))
    top_p = float(os.getenv("CHARXIV_JUDGE_TOP_P", "0.8"))
    top_k = int(os.getenv("CHARXIV_JUDGE_TOP_K", "20"))
    presence_penalty = float(os.getenv("CHARXIV_JUDGE_PRESENCE_PENALTY", "0.0"))
    max_tokens = int(os.getenv("CHARXIV_JUDGE_MAX_TOKENS", os.getenv("VLLM_JUDGE_MAX_TOKENS", "2048")))
    gpu_memory_utilization = float(
        os.getenv("CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION", os.getenv("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", "0.9"))
    )
    system_prompt = os.getenv("CHARXIV_JUDGE_SYSTEM_PROMPT", os.getenv("VLLM_JUDGE_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT))
    return JudgeConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        system_prompt=system_prompt,
    )


class VLLMJudgeEngine:
    """Thin wrapper around vLLM for JSON generation with batched prompts."""

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config
        self._llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
        )
        self.supports_multimodal = hasattr(self._llm, "chat")
        self._tokenizer = self._llm.get_tokenizer()
        sampling_kwargs = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.top_k > 0:
            sampling_kwargs["top_k"] = config.top_k
        if config.presence_penalty != 0.0:
            sampling_kwargs["presence_penalty"] = config.presence_penalty
        self._base_sampling_kwargs = dict(sampling_kwargs)
        self._sampling_params = SamplingParams(**self._base_sampling_kwargs)
        raw_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._system_prompt = raw_prompt.strip()

    def _build_chat_prompt(self, prompt: str) -> str:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _resolve_sampling_params(self, max_tokens: Optional[int]) -> SamplingParams:
        if max_tokens is None or max_tokens == self._base_sampling_kwargs.get("max_tokens"):
            return self._sampling_params
        override_kwargs = dict(self._base_sampling_kwargs)
        override_kwargs["max_tokens"] = max_tokens
        return SamplingParams(**override_kwargs)

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: Optional[int] = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        chat_prompts = [self._build_chat_prompt(prompt) for prompt in prompts]
        sampling_params = self._resolve_sampling_params(max_tokens)
        outputs = self._llm.generate(
            chat_prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )
        responses: list[str] = []
        for output in outputs:
            if not output.outputs:
                responses.append("")
                continue
            responses.append(output.outputs[0].text.strip())
        return responses

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        batched = self.generate_json_batch([prompt], max_tokens=max_tokens)
        return batched[0] if batched else ""

    def generate_json_messages(self, messages: list[dict[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        if not self.supports_multimodal:
            raise RuntimeError("This local judge backend does not support multimodal message inputs.")
        payload_messages = list(messages or [])
        if self._system_prompt and not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in payload_messages):
            payload_messages = [{"role": "system", "content": self._system_prompt}] + payload_messages
        sampling_params = self._resolve_sampling_params(max_tokens)
        outputs = self._llm.chat(
            messages=[payload_messages],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def shutdown(self) -> None:
        """Gracefully shut down the vLLM engine to avoid 'died unexpectedly' errors."""
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None:
                core = getattr(engine, "engine_core", None)
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                elif hasattr(engine, "shutdown"):
                    engine.shutdown()
            elif hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        self._llm = None


class VLLMJudgeEngineQwen3:
    """Thin wrapper around vLLM for JSON generation with batched prompts."""

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config
        template_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "examples",
                "prompts",
                "qwen3_nonthinking.jinja",
            )
        )
        self._chat_template = None
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as handle:
                self._chat_template = handle.read()
        self._llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
        )
        self.supports_multimodal = hasattr(self._llm, "chat")
        self._tokenizer = self._llm.get_tokenizer()
        if self._chat_template and hasattr(self._tokenizer, "chat_template"):
            self._tokenizer.chat_template = self._chat_template
        sampling_kwargs = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 2048,
            "top_k": 20,
            "min_p": 0.0,
        }
        self._base_sampling_kwargs = dict(sampling_kwargs)
        self._sampling_params = SamplingParams(**self._base_sampling_kwargs)
        # raw_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        # self._system_prompt = raw_prompt.strip()

    def _build_chat_prompt(self, prompt: str) -> str:
        messages = []
        # if self._system_prompt:
        #     messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _resolve_sampling_params(self, max_tokens: Optional[int]) -> SamplingParams:
        if max_tokens is None or max_tokens == self._base_sampling_kwargs.get("max_tokens"):
            return self._sampling_params
        override_kwargs = dict(self._base_sampling_kwargs)
        override_kwargs["max_tokens"] = max_tokens
        return SamplingParams(**override_kwargs)

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: Optional[int] = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        chat_prompts = [self._build_chat_prompt(prompt) for prompt in prompts]
        sampling_params = self._resolve_sampling_params(max_tokens)
        outputs = self._llm.generate(
            chat_prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )
        responses: list[str] = []
        for output in outputs:
            if not output.outputs:
                responses.append("")
                continue
            responses.append(output.outputs[0].text.strip())
        return responses

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        batched = self.generate_json_batch([prompt], max_tokens=max_tokens)
        return batched[0] if batched else ""

    def generate_json_messages(self, messages: list[dict[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        if not self.supports_multimodal:
            raise RuntimeError("This local judge backend does not support multimodal message inputs.")
        sampling_params = self._resolve_sampling_params(max_tokens)
        chat_kwargs: dict[str, Any] = {}
        if self._chat_template:
            chat_kwargs["chat_template"] = self._chat_template
        outputs = self._llm.chat(
            messages=[list(messages or [])],
            sampling_params=sampling_params,
            use_tqdm=False,
            **chat_kwargs,
        )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def shutdown(self) -> None:
        """Gracefully shut down the vLLM engine to avoid 'died unexpectedly' errors."""
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None:
                core = getattr(engine, "engine_core", None)
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                elif hasattr(engine, "shutdown"):
                    engine.shutdown()
            elif hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        self._llm = None


class GPTJudgeEngine:
    """Call the hosted GPT API for JSON responses."""

    def __init__(self, config: JudgeConfig) -> None:
        api_key = os.getenv("GPT_API_KEY")
        if not api_key:
            raise RuntimeError("GPT_API_KEY environment variable is required for GPT judge model.")
        self.supports_multimodal = True
        self._client = OpenAI(api_key=api_key)
        self._model = config.model_path
        self._max_retries = 10
        self._default_max_tokens = min(config.max_tokens, 1024)
        self._max_token_cap = min(config.max_tokens, 1024)
        raw_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._system_prompt = raw_prompt.strip()

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _run_completion(self, prompt: str, max_tokens: Optional[int]) -> str:
        retries = 0
        tokens = min(max_tokens or self._default_max_tokens, self._max_token_cap)
        while retries < self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    messages=self._build_messages(prompt),
                    model=self._model,
                    response_format={"type": "json_object"},
                    n=1,
                    max_tokens=tokens,
                    temperature=0,
                    top_p=1,
                    seed=42,
                ).choices[0].message.content
                if response is None:
                    raise ValueError("Empty response from GPT judge.")
                json.loads(response)
                return response.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"Error during GPT judge request: {exc}")
                if "Unterminated string starting at" in str(exc):
                    if tokens >= self._max_token_cap:
                        break
                    tokens = min(self._max_token_cap, tokens * 2)
                    print(f"Retrying with max_tokens: {tokens}")
                retries += 1
        print(f"Failed to get response for prompt: {prompt}")
        return ""

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: Optional[int] = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        iterator = prompts
        if use_tqdm:
            try:
                from tqdm import tqdm

                iterator = tqdm(prompts, desc="Querying GPT judge")
            except Exception:
                iterator = prompts
        responses: list[str] = []
        for prompt in iterator:
            responses.append(self._run_completion(prompt, max_tokens))
        return responses

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        batched = self.generate_json_batch([prompt], max_tokens=max_tokens)
        return batched[0] if batched else ""

    def generate_json_messages(self, messages: list[dict[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        retries = 0
        tokens = min(max_tokens or self._default_max_tokens, self._max_token_cap)
        payload_messages = list(messages or [])
        if self._system_prompt:
            payload_messages = [{"role": "system", "content": self._system_prompt}] + payload_messages
        while retries < self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=payload_messages,
                    max_tokens=tokens,
                    temperature=0,
                    top_p=1,
                    seed=42,
                ).choices[0].message.content
                if response:
                    return response.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"Error during GPT judge request: {exc}")
            retries += 1
        return ""


class VLLMServerJudge:
    """OpenAI-compatible vLLM server judge with wake/sleep support."""

    _MODEL_NAME = "qwen_vllm"
    _API_KEY = "qwen"
    _TIMEOUT = 120.0
    _ENABLE_SLEEP = True
    _SLEEP_LEVEL = 1
    _WAKE_TAGS = ["kv_cache", "weights"]
    _MAX_CONCURRENT_REQUESTS = 256

    _TEMPERATURE = 0.7
    _TOP_P = 0.8
    _TOP_K = 20
    _MIN_P = 0.0
    _MAX_TOKENS = 2048
    _PRESENCE_PENALTY = 0.0
    _FREQUENCY_PENALTY = 0.0

    _CHAT_TEMPLATE_PATH = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "examples",
                "prompts",
                "qwen3_nonthinking.jinja",
            )
        )

    def __init__(self, config: JudgeConfig) -> None:
        _ = config
        self.supports_multimodal = True
        base_url = os.getenv("VLLM_JUDGE_BASE_URL")
        if not base_url:
            api_url = os.getenv("OPENAI_API_URL")
            if api_url:
                base_url = _strip_chat_completions(api_url)
        if not base_url:
            env_port = os.getenv("VLLM_JUDGE_PORT") or os.getenv("VLLM_PORT")
            port = None
            if env_port and env_port.strip().lower() != "auto":
                try:
                    port = int(env_port)
                except ValueError:
                    port = None
            if port is None:
                port = _pick_unused_port()
                os.environ["VLLM_JUDGE_PORT"] = str(port)
            base_url = f"http://localhost:{port}/v1"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            server_root = base_url[:-3]
        else:
            server_root = base_url
            base_url = f"{base_url}/v1"
        self._base_url = base_url
        self._server_root = server_root
        self._client = OpenAI(
            api_key=self._API_KEY,
            base_url=self._base_url,
            timeout=self._TIMEOUT,
            http_client=DefaultHttpxClient(trust_env=False),
        )
        self._chat_template_path = self._CHAT_TEMPLATE_PATH
        self._sampling_params = {
            "temperature": self._TEMPERATURE,
            "top_p": self._TOP_P,
            "max_tokens": self._MAX_TOKENS,
            "presence_penalty": self._PRESENCE_PENALTY,
            "frequency_penalty": self._FREQUENCY_PENALTY,
        }
        self._extra_body = {"top_k": self._TOP_K, "min_p": self._MIN_P}

    def _post_sleep_wake(self, action: str, level: Optional[int] = None, tags: Optional[list[str]] = None) -> None:
        if action not in {"sleep", "wake_up"}:
            return
        params = {}
        if level is not None:
            params["level"] = str(level)
        if tags:
            params["tags"] = ",".join(tags)
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self._server_root}/{action}{query}"
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=self._TIMEOUT):
                pass
        except Exception:
            return

    def _is_sleeping(self) -> Optional[bool]:
        url = f"{self._server_root}/is_sleeping"
        try:
            with urllib.request.urlopen(url, timeout=self._TIMEOUT) as resp:
                data = resp.read().decode("utf-8")
            payload = json.loads(data)
            if isinstance(payload, dict):
                value = payload.get("sleeping")
                if isinstance(value, bool):
                    return value
            if isinstance(payload, bool):
                return payload
        except Exception:
            return None
        return None

    def _wait_sleep_state(self, expect_sleeping: bool, timeout: float = 30.0) -> None:
        start = time.time()
        while True:
            current = self._is_sleeping()
            if current is None or current == expect_sleeping:
                break
            if time.time() - start > timeout:
                break
            time.sleep(0.2)

    def _wake_llm(self) -> None:
        if not self._ENABLE_SLEEP:
            return
        self._post_sleep_wake("wake_up")
        self._wait_sleep_state(expect_sleeping=False)

    def _sleep_llm(self) -> None:
        if not self._ENABLE_SLEEP:
            return
        self._post_sleep_wake("sleep", level=self._SLEEP_LEVEL)
        self._wait_sleep_state(expect_sleeping=True)

    def _run_completion(self, prompt: str, max_tokens: Optional[int]) -> str:
        return self._run_messages([{"role": "user", "content": prompt}], max_tokens=max_tokens)

    def _run_messages(self, messages: list[dict[str, Any]], max_tokens: Optional[int]) -> str:
        params = dict(self._sampling_params)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        retries = 0
        while retries < 10:
            try:
                response = self._client.chat.completions.create(
                    model=self._MODEL_NAME,
                    messages=messages,
                    extra_body=self._extra_body if self._extra_body else None,
                    **params,
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"Error during vLLM server request: {exc}")
            retries += 1
        return ""

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: Optional[int] = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        iterator = prompts
        if use_tqdm:
            try:
                from tqdm import tqdm

                iterator = tqdm(prompts, desc="Querying vLLM server judge")
            except Exception:
                iterator = prompts
        responses: list[str] = []
        self._wake_llm()
        try:
            for prompt in iterator:
                responses.append(self._run_completion(prompt, max_tokens))
        finally:
            self._sleep_llm()
        return responses

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        batched = self.generate_json_batch([prompt], max_tokens=max_tokens)
        return batched[0] if batched else ""

    def generate_json_messages(self, messages: list[dict[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        self._wake_llm()
        try:
            return self._run_messages(messages, max_tokens=max_tokens)
        finally:
            self._sleep_llm()


_ENGINE_CACHE: dict[str, object] = {}


def _shutdown_all_engines() -> None:
    """atexit hook: gracefully shut down all cached vLLM judge engines."""
    for engine in _ENGINE_CACHE.values():
        if hasattr(engine, "shutdown"):
            try:
                engine.shutdown()
            except Exception:
                pass
    _ENGINE_CACHE.clear()


atexit.register(_shutdown_all_engines)


def get_judge_engine(model_hint: Optional[str] = None):
    config = _load_config(model_hint)
    use_server_env = os.getenv("VLLM_SERVER_JUDGE", "").strip().lower() in {"1", "true", "yes"}
    use_server = use_server_env or config.model_path == VLLMServerJudge._MODEL_NAME
    cache_key = f"{config.model_path}|server={int(use_server)}"
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        model_lower = config.model_path.lower()
        if config.model_path in _GPT_MODEL_NAME:
            engine = GPTJudgeEngine(config)
        elif config.model_path == VLLMServerJudge._MODEL_NAME or use_server:
            engine = VLLMServerJudge(config)
        elif "qwen3" in model_lower and ("32b" in model_lower or "14b" in model_lower) and not _is_multimodal_model_path(config.model_path):
            engine = VLLMJudgeEngineQwen3(config)
        else:
            engine = VLLMJudgeEngine(config)
        _ENGINE_CACHE[cache_key] = engine
    return engine


def judge_supports_multimodal(engine: object) -> bool:
    return bool(getattr(engine, "supports_multimodal", False))


def extract_json_candidate(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
