import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm

from lmms_eval.api.registry import register_model

from dotenv import load_dotenv
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.openai_compatible import (
    OpenAICompatible as OpenAICompatibleSimple,
)
from lmms_eval.protocol import ChatMessages

load_dotenv(verbose=True)


@register_model("openai_compatible_chat")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def _process_single_request(self, reg_args):
        """Process a single request: build payload, encode images, call API. Thread-safe."""
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = reg_args

        if self.continual_mode is True and self.cache_mode == "resume":
            doc_uuid = f"{task}___{split}___{doc_id}"
            if doc_uuid in self.response_cache:
                cached = self.response_cache[doc_uuid]
                if cached:
                    return cached, 0, 0, True

        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})

        payload = {"messages": chat_messages.to_openai_messages()}
        payload["model"] = self.model_version

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0

        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]

        if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version or "gpt-5" in self.model_version:
            del payload["temperature"]
            payload.pop("max_tokens")
            payload["reasoning_effort"] = os.environ.get("REASONING_EFFORT", "medium")
            payload["response_format"] = {"type": "text"}
            payload["max_completion_tokens"] = gen_kwargs["max_new_tokens"]

        verbose = os.environ.get("API_VERBOSE", "0") == "1"

        # API call with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**payload)
                latency = time.time() - start_time
                response_text = response.choices[0].message.content
                tokens = response.usage.completion_tokens if hasattr(response, "usage") else len(response_text.split())

                if verbose:
                    usage = response.usage
                    finish_reason = response.choices[0].finish_reason
                    reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None) if usage else None
                    eval_logger.info(
                        f"[API] doc_id={doc_id} task={task} | finish={finish_reason} | "
                        f"completion_tokens={usage.completion_tokens if usage else '?'} reasoning_tokens={reasoning_tokens} | "
                        f"latency={latency:.1f}s | response_text={repr(response_text[:200]) if response_text else repr(response_text)}"
                    )

                return response_text, latency, tokens, False
            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {e}")
                if attempt == self.max_retries - 1:
                    eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {e}")
                    return "", 0, 0, False
                else:
                    time.sleep(self.timeout)

    def generate_until(self, requests) -> List[str]:
        max_workers = int(os.environ.get("API_MAX_WORKERS", "8"))
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0
        total_tokens = 0

        all_args = [reg.args for reg in requests]

        if max_workers <= 1:
            for idx, args in enumerate(all_args):
                response_text, latency, tokens, _cached = self._process_single_request(args)
                res[idx] = response_text
                e2e_latency += latency
                total_tokens += tokens
                pbar.update(1)
                if self.continual_mode is True and not _cached:
                    _, _, _, doc_id, task, split = args
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    self.response_cache[doc_uuid] = response_text
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)
        else:
            eval_logger.info(f"Making {len(all_args)} API calls with {max_workers} parallel workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for idx, args in enumerate(all_args):
                    future = executor.submit(self._process_single_request, args)
                    future_to_idx[future] = (idx, args)

                for future in as_completed(future_to_idx):
                    idx, args = future_to_idx[future]
                    response_text, latency, tokens, _cached = future.result()
                    res[idx] = response_text
                    e2e_latency += latency
                    total_tokens += tokens
                    pbar.update(1)
                    if self.continual_mode is True and not _cached:
                        _, _, _, doc_id, task, split = args
                        doc_uuid = f"{task}___{split}___{doc_id}"
                        self.response_cache[doc_uuid] = response_text
                        with open(self.response_persistent_file, "w") as f:
                            json.dump(self.response_cache, f)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(total_tokens=total_tokens, e2e_latency=e2e_latency, avg_speed=avg_speed)

        pbar.close()
        return res
