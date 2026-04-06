# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
from io import BytesIO
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, Sampler
try:
    from torchdata.stateful_dataloader.stateful import Stateful
except Exception:  # pragma: no cover - fallback when torchdata is unavailable
    class Stateful:  # type: ignore[no-redef]
        pass
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.molmo2_prompt_instructions import (
    MOLMO2_LEGACY_REASONING_SUFFIX,
    MOLMO2_REASONING_INSTRUCTION_VARIANTS,
    MOLMO2_REASONING_REWARD_TYPES,
)
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

_OVERLONG_FILTER_CACHE_FORMAT_VERSION = 1
_OVERLONG_FILTER_CACHE_SUBDIR = "overlong_filter_cache"
_REMOTE_DATASET_PREFIXES = ("hdfs://", "s3://", "http://", "https://")


def _build_missing_local_dataset_error(path: str) -> FileNotFoundError:
    message = (
        f"Dataset file '{path}' was not found.\n"
        "The model-run bash launchers under 'vero-rl/examples/model_runs' now expect formatted local data "
        "under 'vero-rl/data' by default.\n"
        "Run 'python scripts/download_and_format_vero_600k.py' from the repository root to prepare:\n"
        "  - vero-rl/data/vero_600k_train.verl.jsonl\n"
        "  - vero-rl/data/vero_600k_val.verl.jsonl\n"
        "  - vero-rl/data/images/...\n"
        "See README.md and vero-rl/README.md for the full data setup instructions."
    )
    return FileNotFoundError(message)


def _to_jsonable(value):
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)

def _parse_per_batch_domain_weights(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        weights = dict(raw)
    elif isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        if raw.startswith("{") or raw.startswith("["):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                weights = parsed
            elif isinstance(parsed, list):
                weights = {}
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("domain")
                        weight = item.get("weight") or item.get("value")
                        if name is None or weight is None:
                            raise ValueError(f"Invalid domain weight entry: {item!r}")
                        weights[str(name)] = float(weight)
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        weights[str(item[0])] = float(item[1])
                    else:
                        raise ValueError(f"Invalid domain weight entry: {item!r}")
            else:
                raise ValueError("per_batch_domain_weights JSON must be dict or list")
        else:
            weights = {}
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        "per_batch_domain_weights must be 'name:weight' pairs separated by commas"
                    )
                name, value = part.split(":", 1)
                name = name.strip()
                value = value.strip()
                if not name:
                    raise ValueError("per_batch_domain_weights contains empty domain name")
                weights[name] = float(value)
    elif isinstance(raw, (list, tuple)):
        weights = {}
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("domain")
                weight = item.get("weight") or item.get("value")
                if name is None or weight is None:
                    raise ValueError(f"Invalid domain weight entry: {item!r}")
                weights[str(name)] = float(weight)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                weights[str(item[0])] = float(item[1])
            elif isinstance(item, str):
                if ":" not in item:
                    raise ValueError(f"Invalid domain weight entry: {item!r}")
                name, value = item.split(":", 1)
                weights[name.strip()] = float(value.strip())
            else:
                raise ValueError(f"Invalid domain weight entry: {item!r}")
    else:
        raise TypeError(f"Unsupported per_batch_domain_weights type: {type(raw).__name__}")

    filtered = {}
    for name, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid weight for domain '{name}': {value!r}") from None
        if not np.isfinite(weight):
            raise ValueError(f"Non-finite weight for domain '{name}': {weight!r}")
        if weight > 0:
            filtered[str(name)] = weight
    if not filtered:
        raise ValueError("per_batch_domain_weights must contain at least one positive weight")
    total = sum(filtered.values())
    return {name: weight / total for name, weight in filtered.items()}


class PerBatchDomainSampler(Sampler[int], Stateful):
    def __init__(self, domain2indices, domain_weights, batch_size, total_size, seed=18, shuffle=True):
        self.domain_names = list(domain_weights.keys())
        self.domain_weights = [domain_weights[name] for name in self.domain_names]
        self.domain2indices = {
            name: np.array(domain2indices[name], dtype=np.int64, copy=True) for name in self.domain_names
        }
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(self.seed)
        self._batch_counts = self._compute_batch_counts()
        self._domain_perm = {}
        self._domain_pos = {}
        self.epoch = 0
        self._batch_cursor = 0
        self._epoch_started = False
        for name in self.domain_names:
            self._reshuffle(name)
        self._num_samples = (int(total_size) // self.batch_size) * self.batch_size

    def _compute_batch_counts(self):
        raw = np.array(self.domain_weights, dtype=np.float64) * self.batch_size
        counts = np.floor(raw).astype(np.int64)
        remainder = raw - counts
        leftover = int(self.batch_size - counts.sum())
        if leftover > 0:
            order = np.argsort(-remainder)
            for i in range(leftover):
                counts[order[i % len(order)]] += 1
        return counts.tolist()

    def _reshuffle(self, name):
        if self.shuffle:
            self._domain_perm[name] = self.rng.permutation(self.domain2indices[name])
        else:
            self._domain_perm[name] = self.domain2indices[name].copy()
        self._domain_pos[name] = 0

    def _take(self, name, count):
        out = []
        while len(out) < count:
            perm = self._domain_perm[name]
            pos = self._domain_pos[name]
            remaining = len(perm) - pos
            take = min(count - len(out), remaining)
            if take > 0:
                out.extend(perm[pos : pos + take])
                pos += take
            if pos >= len(perm):
                self._reshuffle(name)
                pos = self._domain_pos[name]
            self._domain_pos[name] = pos
        return out

    def _start_epoch(self):
        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.epoch)
        for name in self.domain_names:
            self._reshuffle(name)
        self._batch_cursor = 0
        self._epoch_started = True

    def __iter__(self):
        if not self._epoch_started:
            self._start_epoch()
        num_batches = self._num_samples // self.batch_size
        for _ in range(self._batch_cursor, num_batches):
            batch = []
            for name, count in zip(self.domain_names, self._batch_counts, strict=True):
                if count:
                    batch.extend(self._take(name, int(count)))
            if self.shuffle:
                self.rng.shuffle(batch)
            self._batch_cursor += 1
            for idx in batch:
                yield int(idx)
        self.epoch += 1
        self._epoch_started = False

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._batch_cursor = 0
        self._epoch_started = False

    def state_dict(self):
        return {
            "domain_names": list(self.domain_names),
            "epoch": self.epoch,
            "batch_cursor": self._batch_cursor,
            "epoch_started": self._epoch_started,
            "rng_state": self.rng.bit_generator.state,
            "domain_pos": dict(self._domain_pos),
            "domain_perm": {k: v.copy() for k, v in self._domain_perm.items()},
            "shuffle": self.shuffle,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state_dict):
        if list(state_dict.get("domain_names", [])) != list(self.domain_names):
            raise ValueError("PerBatchDomainSampler domain names do not match the saved state")
        self.epoch = int(state_dict.get("epoch", 0))
        self._batch_cursor = int(state_dict.get("batch_cursor", 0))
        self._epoch_started = bool(state_dict.get("epoch_started", False))
        self.shuffle = bool(state_dict.get("shuffle", self.shuffle))
        self.seed = int(state_dict.get("seed", self.seed))
        self.batch_size = int(state_dict.get("batch_size", self.batch_size))
        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        domain_pos = state_dict.get("domain_pos", {})
        domain_perm = state_dict.get("domain_perm", {})
        for name in self.domain_names:
            if name in domain_perm:
                self._domain_perm[name] = np.array(domain_perm[name], dtype=np.int64, copy=True)
            if name in domain_pos:
                self._domain_pos[name] = int(domain_pos[name])


# Task-type routing is keyed off per-sample metadata because the new datasets
# share a single `data_source`. We therefore rely on the sample ID format to
# distinguish between counting/grounding/clicking/search tasks. Extend these
# maps whenever a new dataset is added.
_TASK_TYPE_BY_ID_PREFIX = {
    "tallyqa": "counting",
    "oodvqa": "counting",
    "pixmo": "counting",
    "multihop": "counting",
    "objects365": "grounding",
    "refcocog": "grounding",
    "aerialvg": "grounding",
    "groundui": "clicking",
    "mobile": "clicking",
    "desktop": "clicking",
    "web": "clicking",
    "pixelreasoner": "search",
    "visual": "search",
    "visionprobe": "search",
}

_TASK_TYPE_BY_IMAGE_PREFIX = {
    "tallyqa": "counting",
    "oodvqa": "counting",
    "objects365": "grounding",
    "groundui": "clicking",
    "pixelreasoner": "search",
}


def infer_task_type(
    data_source: Optional[str],
    extra_info: Optional[dict],
    reward_model: Optional[dict],
) -> str:
    """Infer task type for routing reward functions."""
    if extra_info is None:
        extra_info = {}

    sample_id = (extra_info.get("id") or "").lower()
    if sample_id:
        if "_grounding_" in sample_id:
            return "grounding"
        if "_counting_" in sample_id:
            return "counting"
        if sample_id.startswith("visual_probe"):
            return "search"
        prefix = sample_id.split("_")[0]
        if prefix in _TASK_TYPE_BY_ID_PREFIX:
            return _TASK_TYPE_BY_ID_PREFIX[prefix]

    ground_truth = ""
    if reward_model and isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth") or ""
    if not isinstance(ground_truth, str):
        try:
            ground_truth = json.dumps(ground_truth)
        except Exception:
            ground_truth = str(ground_truth)
    if "bbox_2d" in ground_truth:
        return "grounding"

    images = extra_info.get("images") if isinstance(extra_info, dict) else None
    if images:
        image_name = os.path.basename(images[0]).lower()
        prefix = image_name.split("_")[0]
        if prefix in _TASK_TYPE_BY_IMAGE_PREFIX:
            return _TASK_TYPE_BY_IMAGE_PREFIX[prefix]

    question = (extra_info.get("question") or "").lower()
    if "bbox" in question or "box" in question:
        return "grounding"
    if any(token in question for token in ["search", "locate", "find"]):
        return "search"

    return "counting"


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_root = config.get("image_root", None)
        self.max_pixels = config.get("max_pixels", None)
        self.normalize_bbox_to_1000 = config.get("normalize_bbox_to_1000", False)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # system prompt
        self.system_prompt = config.get("system_prompt", None)
        if isinstance(self.system_prompt, str):
            if os.path.exists(self.system_prompt):
                with open(self.system_prompt, "r") as f:
                    self.system_prompt = f.read()

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.per_batch_domain_weights = _parse_per_batch_domain_weights(
            config.get("per_batch_domain_weights", None)
        )
        self.domain2indices = None
        self.domain_weights = None
        self._model_family = self._detect_model_family()
        if self._model_family == "molmo2":
            if self.system_prompt is not None:
                logger.warning("system_prompt is disabled for model family 'molmo2'; overriding configured value with None.")
            else:
                logger.warning("system_prompt is disabled for model family 'molmo2'; forcing None.")
            self.system_prompt = None

        self._download()
        self._read_files_and_tokenize()
        self._prepare_domain_sampling()

    @staticmethod
    def _load_model_type_from_config_path(path: str | None) -> str | None:
        if not isinstance(path, str) or not path:
            return None
        candidate = path.strip()
        if not candidate:
            return None
        candidate = os.path.expanduser(candidate)
        if os.path.isfile(candidate):
            config_path = candidate if os.path.basename(candidate) == "config.json" else None
        elif os.path.isdir(candidate):
            config_path = os.path.join(candidate, "config.json")
        else:
            return None
        if config_path is None or not os.path.isfile(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        model_type = payload.get("model_type")
        return str(model_type).strip().lower() if isinstance(model_type, str) else None

    def _detect_model_family(self) -> str | None:
        candidate_paths: list[str] = []
        tokenizer_path = getattr(self.tokenizer, "name_or_path", None)
        if isinstance(tokenizer_path, str):
            candidate_paths.append(tokenizer_path)
        init_kwargs = getattr(self.tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            init_path = init_kwargs.get("name_or_path")
            if isinstance(init_path, str):
                candidate_paths.append(init_path)
        processor_tokenizer = getattr(self.processor, "tokenizer", None)
        processor_tokenizer_path = getattr(processor_tokenizer, "name_or_path", None)
        if isinstance(processor_tokenizer_path, str):
            candidate_paths.append(processor_tokenizer_path)

        seen_model_type = False
        for path in dict.fromkeys(candidate_paths):
            model_type = self._load_model_type_from_config_path(path)
            if not model_type:
                continue
            seen_model_type = True
            if model_type == "molmo2":
                return "molmo2"
            # Config-derived model_type is authoritative; avoid heuristic fallback.
            return None

        if seen_model_type:
            return None

        # Fallback only when config-based detection is unavailable.
        processor_name = self.processor.__class__.__name__ if self.processor is not None else ""
        image_processor_name = ""
        if self.processor is not None and getattr(self.processor, "image_processor", None) is not None:
            image_processor_name = self.processor.image_processor.__class__.__name__
        tokenizer_name = self.tokenizer.__class__.__name__
        if "Molmo2Processor" in processor_name or "Molmo2ImageProcessor" in image_processor_name:
            return "molmo2"
        if "Molmo" in tokenizer_name and "Qwen" not in tokenizer_name:
            return "molmo2"
        return None

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            candidate = os.path.expanduser(parquet_file)
            if not parquet_file.startswith(_REMOTE_DATASET_PREFIXES) and not os.path.exists(candidate):
                raise _build_missing_local_dataset_error(parquet_file)
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            # Determine backend based on file suffix so we can accept parquet and JSONL inputs.
            data_file_lower = data_file.lower()
            if data_file_lower.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
                dataframe = datasets.load_dataset("json", data_files=data_file, split="train")
            else:
                dataframe = datasets.load_dataset("parquet", data_files=data_file, split="train")

            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}", file=sys.stderr, flush=True)

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

        print(f"after filtering, dataset len: {len(self.dataframe)}", file=sys.stderr, flush=True)

    def _prepare_domain_sampling(self):
        if not self.per_batch_domain_weights:
            return
        if "ability" not in self.dataframe.column_names:
            raise KeyError("per_batch_domain_weights requires an 'ability' column in the dataset")
        abilities = np.asarray(self.dataframe["ability"], dtype=object)
        if abilities.size == 0:
            raise ValueError("Dataset is empty; cannot build per-batch domain sampler")

        available_domains = set(np.unique(abilities))
        filtered_weights = {}
        for domain, weight in self.per_batch_domain_weights.items():
            if domain not in available_domains:
                logger.warning("per_batch_domain_weights domain '%s' not found in dataset", domain)
                continue
            filtered_weights[domain] = weight
        if not filtered_weights:
            raise ValueError("No per_batch_domain_weights domains were found in the dataset")

        # Renormalize after filtering missing domains.
        total = sum(filtered_weights.values())
        self.domain_weights = {name: weight / total for name, weight in filtered_weights.items()}

        self.domain2indices = {}
        for domain in self.domain_weights:
            self.domain2indices[domain] = np.where(abilities == domain)[0]

        missing = available_domains - set(self.domain_weights.keys())
        if missing:
            logger.warning(
                "per_batch_domain_weights does not include domains %s; they will be ignored",
                sorted(missing),
            )

    def build_domain_sampler(self, batch_size, seed=18, shuffle=True):
        if self.domain2indices is None or self.domain_weights is None:
            raise RuntimeError("per_batch_domain_weights is not configured for this dataset")
        return PerBatchDomainSampler(
            domain2indices=self.domain2indices,
            domain_weights=self.domain_weights,
            batch_size=batch_size,
            total_size=len(self.dataframe),
            seed=seed,
            shuffle=shuffle,
        )

    def _get_overlong_filter_cache_root(self) -> Optional[str]:
        cache_root = os.environ.get("HF_DATASETS_CACHE")
        if not cache_root:
            logger.warning("HF_DATASETS_CACHE is not set; skipping overlong prompt filter cache.")
            print("Overlong filter cache bypassed: HF_DATASETS_CACHE is not set.", file=sys.stderr, flush=True)
            return None
        try:
            cache_root = os.path.join(os.path.expanduser(cache_root), _OVERLONG_FILTER_CACHE_SUBDIR)
            os.makedirs(cache_root, exist_ok=True)
            return cache_root
        except OSError as exc:
            logger.warning("Failed to prepare overlong prompt filter cache at %s: %s", cache_root, exc)
            print(
                f"Overlong filter cache bypassed: failed to prepare cache directory '{cache_root}': {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    @staticmethod
    def _build_file_cache_descriptor(requested_path: str, resolved_path: str) -> dict:
        stat_result = os.stat(resolved_path)
        return {
            "requested_path": requested_path,
            "resolved_path": resolved_path,
            "size": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        }

    @staticmethod
    def _build_processing_identity(obj) -> Optional[dict]:
        if obj is None:
            return None

        init_kwargs = getattr(obj, "init_kwargs", None)
        tokenizer = getattr(obj, "tokenizer", None)
        image_processor = getattr(obj, "image_processor", None)
        identity = {
            "class_name": obj.__class__.__name__,
            "name_or_path": getattr(obj, "name_or_path", None),
            "init_name_or_path": init_kwargs.get("name_or_path") if isinstance(init_kwargs, dict) else None,
            "chat_template": getattr(obj, "chat_template", None),
        }
        if tokenizer is not None:
            identity["tokenizer"] = {
                "class_name": tokenizer.__class__.__name__,
                "name_or_path": getattr(tokenizer, "name_or_path", None),
                "chat_template": getattr(tokenizer, "chat_template", None),
            }
        if image_processor is not None:
            identity["image_processor"] = {
                "class_name": image_processor.__class__.__name__,
            }
        return _to_jsonable(identity)

    def _build_overlong_filter_cache_spec(self, dataframe: datasets.Dataset) -> dict:
        data_files = []
        for requested_path, resolved_path in zip(self.original_data_files, self.data_files, strict=True):
            data_files.append(self._build_file_cache_descriptor(requested_path=requested_path, resolved_path=resolved_path))

        return {
            "format_version": _OVERLONG_FILTER_CACHE_FORMAT_VERSION,
            "data_files": data_files,
            "max_prompt_length": int(self.max_prompt_length),
            "max_pixels": None if self.max_pixels is None else int(self.max_pixels),
            "prompt_key": self.prompt_key,
            "image_key": self.image_key,
            "video_key": self.video_key,
            "system_prompt": self.system_prompt,
            "apply_chat_template_kwargs": _to_jsonable(self.apply_chat_template_kwargs),
            "image_root": self.image_root,
            "has_processor": self.processor is not None,
            "tokenizer": self._build_processing_identity(self.tokenizer),
            "processor": self._build_processing_identity(self.processor),
            "model_family": self._model_family,
            "source_length": int(len(dataframe)),
        }

    @staticmethod
    def _hash_overlong_filter_cache_spec(filter_spec: dict) -> str:
        payload = json.dumps(_to_jsonable(filter_spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_overlong_filter_cache_paths(self, cache_key: str) -> Optional[dict[str, str]]:
        cache_root = self._get_overlong_filter_cache_root()
        if cache_root is None:
            return None
        cache_dir = os.path.join(cache_root, cache_key)
        return {
            "cache_dir": cache_dir,
            "meta_path": os.path.join(cache_dir, "meta.json"),
            "indices_path": os.path.join(cache_dir, "kept_indices.npy"),
            "lock_path": os.path.join(cache_root, f"{cache_key}.lock"),
        }

    @staticmethod
    def _validate_kept_indices(kept_indices: np.ndarray, source_length: int) -> np.ndarray:
        if kept_indices.ndim != 1:
            raise ValueError(f"Expected kept indices to be 1D, got shape {kept_indices.shape!r}")
        if kept_indices.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Expected integer kept indices, got dtype {kept_indices.dtype}")
        if kept_indices.size:
            if kept_indices.min() < 0 or kept_indices.max() >= source_length:
                raise ValueError("Kept indices are out of bounds for the source dataset")
            if not np.all(kept_indices[1:] >= kept_indices[:-1]):
                raise ValueError("Kept indices must be sorted in non-decreasing order")
        return kept_indices.astype(np.int64, copy=False)

    def _load_overlong_filter_cache(
        self,
        cache_paths: Optional[dict[str, str]],
        filter_spec: dict,
    ) -> Optional[np.ndarray]:
        if cache_paths is None:
            return None

        meta_path = cache_paths["meta_path"]
        indices_path = cache_paths["indices_path"]
        if not os.path.exists(meta_path) or not os.path.exists(indices_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("format_version") != _OVERLONG_FILTER_CACHE_FORMAT_VERSION:
                raise ValueError(f"Unsupported cache format version: {meta.get('format_version')!r}")
            if meta.get("filter_spec") != filter_spec:
                raise ValueError("Stored filter spec does not match the current configuration")

            kept_indices = np.load(indices_path, allow_pickle=False)
            kept_indices = self._validate_kept_indices(kept_indices, int(filter_spec["source_length"]))

            if int(meta.get("source_length", -1)) != int(filter_spec["source_length"]):
                raise ValueError("Stored source length does not match current dataset length")
            if int(meta.get("kept_count", -1)) != int(kept_indices.size):
                raise ValueError("Stored kept_count does not match cached kept indices")

            return kept_indices
        except Exception as exc:
            logger.warning("Ignoring invalid overlong prompt filter cache at %s: %s", cache_paths["cache_dir"], exc)
            print(
                f"Overlong filter cache bypassed: invalid cache at '{cache_paths['cache_dir']}': {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _save_overlong_filter_cache(
        self,
        cache_paths: dict[str, str],
        filter_spec: dict,
        cache_key: str,
        kept_indices: np.ndarray,
    ) -> None:
        os.makedirs(cache_paths["cache_dir"], exist_ok=True)

        meta = {
            "format_version": _OVERLONG_FILTER_CACHE_FORMAT_VERSION,
            "cache_key": cache_key,
            "source_length": int(filter_spec["source_length"]),
            "kept_count": int(kept_indices.size),
            "filter_spec": filter_spec,
        }

        fd, meta_tmp_path = tempfile.mkstemp(dir=cache_paths["cache_dir"], prefix="meta.", suffix=".tmp")
        os.close(fd)
        try:
            with open(meta_tmp_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, sort_keys=True)
            os.replace(meta_tmp_path, cache_paths["meta_path"])
        finally:
            if os.path.exists(meta_tmp_path):
                os.remove(meta_tmp_path)

        fd, indices_tmp_path = tempfile.mkstemp(dir=cache_paths["cache_dir"], prefix="kept_indices.", suffix=".tmp")
        os.close(fd)
        try:
            with open(indices_tmp_path, "wb") as handle:
                np.save(handle, kept_indices, allow_pickle=False)
            os.replace(indices_tmp_path, cache_paths["indices_path"])
        finally:
            if os.path.exists(indices_tmp_path):
                os.remove(indices_tmp_path)

    @staticmethod
    def _get_overlong_filter_temp_index_column(dataframe: datasets.Dataset) -> str:
        column_name = "__overlong_filter_cache_idx__"
        while column_name in dataframe.column_names:
            column_name = f"_{column_name}"
        return column_name

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        image_key = self.image_key
        video_key = self.video_key
        filter_spec = self._build_overlong_filter_cache_spec(dataframe)
        cache_key = self._hash_overlong_filter_cache_spec(filter_spec)
        cache_paths = self._get_overlong_filter_cache_paths(cache_key)

        cached_indices = self._load_overlong_filter_cache(cache_paths, filter_spec)
        if cached_indices is not None:
            print(
                f"Overlong filter cache hit: using kept indices from '{cache_paths['cache_dir']}'.",
                file=sys.stderr,
                flush=True,
            )
            return dataframe.select(cached_indices.tolist())

        if cache_paths is not None:
            print(
                f"Overlong filter cache miss: computing kept indices for key {cache_key}.",
                file=sys.stderr,
                flush=True,
            )

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc2len(doc) -> int:
                messages = self._build_messages(doc, remove_prompt=False)
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                images = (
                    [process_image(image, self.image_root, self.max_pixels) for image in doc[image_key]]
                    if image_key in doc and doc[image_key]
                    else None
                )
                videos = (
                    [process_video(video) for video in doc[video_key]]
                    if video_key in doc and doc[video_key]
                    else None
                )

                try:
                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])
                except ValueError:
                    # Treat invalid image/video inputs as overlong so they get filtered out.
                    return self.max_prompt_length + 1

        else:

            def doc2len(doc) -> int:
                messages = self._build_messages(doc, remove_prompt=False)
                return len(
                    tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, **self.apply_chat_template_kwargs
                    )
                )

        print("Start filtering overlong prompts...", file=sys.stderr, flush=True)
        len_before = len(dataframe)
        index_column = self._get_overlong_filter_temp_index_column(dataframe)
        dataframe_with_index = dataframe.add_column(index_column, list(range(len_before)))

        dataframe = dataframe_with_index.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        kept_indices = np.asarray(dataframe[index_column], dtype=np.int64)
        dataframe = dataframe.remove_columns([index_column])

        if cache_paths is not None:
            try:
                from filelock import FileLock

                with FileLock(cache_paths["lock_path"], timeout=60):
                    cached_indices = self._load_overlong_filter_cache(cache_paths, filter_spec)
                    if cached_indices is not None:
                        print(
                            f"Overlong filter cache hit after compute: reusing kept indices from '{cache_paths['cache_dir']}'.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return dataframe_with_index.remove_columns([index_column]).select(cached_indices.tolist())

                    self._save_overlong_filter_cache(cache_paths, filter_spec, cache_key, kept_indices)
                    print(
                        f"Overlong filter cache saved: wrote kept indices to '{cache_paths['cache_dir']}'.",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                logger.warning("Failed to save overlong prompt filter cache at %s: %s", cache_paths["cache_dir"], exc)
                print(
                    f"Overlong filter cache bypassed: failed to save cache at '{cache_paths['cache_dir']}': {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        print(f"filter dataset len: {len(dataframe)}", file=sys.stderr, flush=True)
        print(f"Filtered {len_before - len(dataframe)} overlong prompts.", file=sys.stderr, flush=True)
        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict, *, remove_prompt: bool = True) -> list[dict]:
        """Normalize raw prompt payloads into a chat-style message list.

        Args:
            example: Single sample dictionary that contains the prompt payload.
            remove_prompt: Whether the prompt field should be removed from the sample.

        Returns:
            A list of message dictionaries compatible with the tokenizer chat template.
        """

        if self.prompt_key not in example:
            raise KeyError(f"Prompt key '{self.prompt_key}' not found in example")

        raw_messages = example.pop(self.prompt_key) if remove_prompt else example[self.prompt_key]

        messages = self._normalize_messages(raw_messages)

        # If the sample references images/videos we need to split inline tokens
        # such as <image>/<video> into multimodal content blocks.
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    # Already processed into multimodal segments.
                    continue
                if not isinstance(content, str):
                    raise TypeError(
                        "Expected message 'content' to be a string or list when images/videos are present, "
                        f"got {type(content).__name__}"
                    )

                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _normalize_messages(self, raw_messages) -> list[dict]:
        """Convert various prompt payload formats into a list of message dicts."""

        if isinstance(raw_messages, dict):
            if self.system_prompt:
                return [self._ensure_message_dict({"role": "system", "content": self.system_prompt}), self._ensure_message_dict(raw_messages)]
            else:
                return [self._ensure_message_dict(raw_messages)]

        if isinstance(raw_messages, str):
            if self.system_prompt:
                return [self._ensure_message_dict({"role": "system", "content": self.system_prompt}), self._ensure_message_dict({"role": "user", "content": raw_messages})]
            else:
                return [self._ensure_message_dict({"role": "user", "content": raw_messages})]

        if isinstance(raw_messages, (list, tuple)):
            normalized: list[dict] = []
            for idx, message in enumerate(raw_messages):
                default_role = "user" if idx == 0 else "assistant"
                if idx == 0 and self.system_prompt:
                    normalized.append(self._ensure_message_dict({"role": "system", "content": self.system_prompt}))
                if isinstance(message, dict):
                    normalized.append(self._ensure_message_dict(message, default_role=default_role))
                elif isinstance(message, str):
                    normalized.append(self._ensure_message_dict({"role": default_role, "content": message}))
                else:
                    normalized.append(
                        self._ensure_message_dict({"role": default_role, "content": str(message)})
                    )

            return normalized

        raise TypeError(
            "Unsupported prompt payload type. Expected dict, list, tuple, or str, "
            f"but received {type(raw_messages).__name__}"
        )

    @staticmethod
    def _ensure_message_dict(message: dict, default_role: str = "user") -> dict:
        msg = dict(message)
        msg.setdefault("role", default_role)
        if "content" not in msg:
            raise KeyError("Message dict must contain a 'content' field")
        return msg

    def _rewrite_coordinate_instruction_text(self, text: str, reward_type: str) -> str:
        if not text:
            return text
        reward_type = reward_type.lower()
        if self._model_family == "molmo2":
            if reward_type == "grounding":
                new_text = (
                    'Output format in answer tags (strict JSON): { "results": [ "x_min": , "y_min": , "x_max": , "y_max": }, ... ] }. '
                    "x_min, y_min, x_max, y_max should be integer coordinates normalized to a 0-1000 scale."
                )
                pattern = r"Output the bounding box\(es\) in a JSON array format:.*?array\."
                if re.search(pattern, text, flags=re.DOTALL):
                    return re.sub(pattern, new_text, text, flags=re.DOTALL)
                pattern = r"Output the bounding box\(es\) in a JSON array format:.*"
                if re.search(pattern, text, flags=re.DOTALL):
                    return re.sub(pattern, new_text, text, flags=re.DOTALL)
                return text
            if reward_type == "clicking":
                new_text = (
                    "Point to the region using an XML format."
                )
                pattern = r"Output the point in a JSON array format:.*?origin at top-left\."
                if re.search(pattern, text, flags=re.DOTALL):
                    return re.sub(pattern, new_text, text, flags=re.DOTALL)
                pattern = r"Output the point in a JSON array format:.*"
                if re.search(pattern, text, flags=re.DOTALL):
                    return re.sub(pattern, new_text, text, flags=re.DOTALL)
                text = re.sub(
                    r"\s*Use absolute pixel coordinates \(origin at top-left\)\.",
                    "",
                    text,
                    flags=re.DOTALL,
                )
                return text
            return text
        if reward_type == "grounding":
            new_text = (
                'Output the bounding box(es) in a JSON array format: [{\"bbox_2d\": [x_min, y_min, x_max, y_max]. '
                "For a single object, return one array element; for multiple objects, return one element per object in the array."
            )
            pattern = r"Output the bounding box\(es\) in a JSON array format:.*?array\."
            if re.search(pattern, text, flags=re.DOTALL):
                return re.sub(pattern, new_text, text, flags=re.DOTALL)
            pattern = r"Output the bounding box\(es\) in a JSON array format:.*"
            if re.search(pattern, text, flags=re.DOTALL):
                return re.sub(pattern, new_text, text, flags=re.DOTALL)
            return text
        if reward_type == "clicking":

            new_text = 'Output the point in a JSON array format: [{\"point_2d\": [x, y]}].'

            pattern = r"Output the point in a JSON array format:.*?origin at top-left\."
            if re.search(pattern, text, flags=re.DOTALL):
                return re.sub(pattern, new_text, text, flags=re.DOTALL)
            pattern = r"Output the point in a JSON array format:.*"
            if re.search(pattern, text, flags=re.DOTALL):
                return re.sub(pattern, new_text, text, flags=re.DOTALL)
            # Remove absolute-coordinate phrasing even if the pattern didn't match.
            text = re.sub(
                r"\s*Use absolute pixel coordinates \(origin at top-left\)\.",
                "",
                text,
                flags=re.DOTALL,
            )
            return text
        return text

    def _rewrite_prompt_payload(self, payload, reward_type: str):
        if isinstance(payload, str):
            return self._rewrite_coordinate_instruction_text(payload, reward_type)
        if isinstance(payload, dict):
            updated = dict(payload)
            if "content" in updated:
                updated["content"] = self._rewrite_prompt_payload(updated["content"], reward_type)
            elif updated.get("type") == "text" and isinstance(updated.get("text"), str):
                updated["text"] = self._rewrite_coordinate_instruction_text(updated["text"], reward_type)
            return updated
        if isinstance(payload, list):
            return [self._rewrite_prompt_payload(item, reward_type) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self._rewrite_prompt_payload(item, reward_type) for item in payload)
        return payload

    def _maybe_rewrite_question(self, row_dict: dict) -> None:
        extra_info = row_dict.get("extra_info")
        if not isinstance(extra_info, dict):
            return
        reward_type = extra_info.get("reward_type")
        if not isinstance(reward_type, str):
            return
        reward_type = reward_type.lower()
        if reward_type not in {"grounding", "clicking"}:
            return

        if self.prompt_key in row_dict:
            row_dict[self.prompt_key] = self._rewrite_prompt_payload(
                row_dict[self.prompt_key],
                reward_type,
            )

        question = extra_info.get("question")
        if isinstance(question, str):
            extra_info["question"] = self._rewrite_coordinate_instruction_text(question, reward_type)

    @staticmethod
    def _has_molmo2_reasoning_instruction(text: str) -> bool:
        if not text:
            return False
        if MOLMO2_LEGACY_REASONING_SUFFIX.strip() in text:
            return True
        for suffix in MOLMO2_REASONING_INSTRUCTION_VARIANTS:
            if suffix.strip() in text:
                return True
        return False

    def _append_molmo2_reasoning_suffix(self, text: str, suffix: str | None = None) -> str:
        if not text:
            return text
        if self._has_molmo2_reasoning_instruction(text):
            return text
        selected_suffix = suffix or random.choice(MOLMO2_REASONING_INSTRUCTION_VARIANTS)
        return text.rstrip() + " " + selected_suffix.lstrip()

    def _append_instruction_to_user_payload(self, payload, suffix: str | None = None):
        if isinstance(payload, str):
            updated = self._append_molmo2_reasoning_suffix(payload, suffix=suffix)
            return updated, updated != payload

        if isinstance(payload, dict):
            updated = dict(payload)
            role = updated.get("role")
            if role is not None and str(role).lower() != "user":
                return payload, False

            if "content" in updated:
                updated_content, changed = self._append_instruction_to_user_payload(
                    updated["content"], suffix=suffix
                )
                if changed:
                    updated["content"] = updated_content
                    return updated, True
                return payload, False

            if updated.get("type") == "text" and isinstance(updated.get("text"), str):
                old_text = updated["text"]
                new_text = self._append_molmo2_reasoning_suffix(old_text, suffix=suffix)
                if new_text != old_text:
                    updated["text"] = new_text
                    return updated, True
            return payload, False

        if isinstance(payload, list):
            updated = []
            appended = False
            for item in payload:
                if not appended:
                    updated_item, changed = self._append_instruction_to_user_payload(item, suffix=suffix)
                    if changed:
                        updated.append(updated_item)
                        appended = True
                        continue
                updated.append(item)
            return updated, appended

        if isinstance(payload, tuple):
            updated = []
            appended = False
            for item in payload:
                if not appended:
                    updated_item, changed = self._append_instruction_to_user_payload(item, suffix=suffix)
                    if changed:
                        updated.append(updated_item)
                        appended = True
                        continue
                updated.append(item)
            return tuple(updated), appended

        return payload, False

    def _maybe_append_molmo2_reasoning_instruction(self, row_dict: dict) -> None:
        extra_info = row_dict.get("extra_info")
        if not isinstance(extra_info, dict):
            return
        selected_suffix = random.choice(MOLMO2_REASONING_INSTRUCTION_VARIANTS)

        if self.prompt_key in row_dict:
            updated_prompt, _ = self._append_instruction_to_user_payload(
                row_dict[self.prompt_key], suffix=selected_suffix
            )
            row_dict[self.prompt_key] = updated_prompt

        question = extra_info.get("question")
        if isinstance(question, str):
            extra_info["question"] = self._append_molmo2_reasoning_suffix(
                question, suffix=selected_suffix
            )

    def _maybe_rewrite_question_for_molmo2(self, row_dict: dict) -> None:
        if self._model_family != "molmo2":
            return
        extra_info = row_dict.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
            row_dict["extra_info"] = extra_info
        extra_info["model_family"] = "molmo2"
        reward_type = extra_info.get("reward_type")
        normalized_reward_type = reward_type.lower() if isinstance(reward_type, str) else None
        if normalized_reward_type in {"grounding", "clicking"}:
            self._maybe_rewrite_question(row_dict)

        if normalized_reward_type in MOLMO2_REASONING_REWARD_TYPES:
            self._maybe_append_molmo2_reasoning_instruction(row_dict)

    def _prepare_image_with_size(self, image):
        image_for_processing = image
        image_size = None

        if isinstance(image, str):
            image_path = os.path.join(self.image_root, image) if self.image_root else image
            try:
                with Image.open(image_path) as img:
                    image_size = (img.width, img.height)
                    image_for_processing = img.convert("RGB")
            except Exception:
                image_for_processing = image
        elif isinstance(image, Image.Image):
            image_size = (image.width, image.height)
        elif isinstance(image, dict):
            raw_image = image.get("image")
            if isinstance(raw_image, Image.Image):
                image_size = (raw_image.width, raw_image.height)
            elif "bytes" in image:
                try:
                    with Image.open(BytesIO(image["bytes"])) as img:
                        image_size = (img.width, img.height)
                except Exception:
                    image_size = None

        return image_for_processing, image_size

    def _maybe_normalize_bbox_ground_truth(self, ground_truth, image_size):
        if ground_truth is None or not image_size:
            return None

        width, height = image_size
        if not width or not height:
            return None

        parsed = None
        output_as_str = False
        if isinstance(ground_truth, str):
            try:
                parsed = json.loads(ground_truth)
                output_as_str = True
            except Exception:
                return None
        elif isinstance(ground_truth, dict):
            parsed = [ground_truth]
        elif isinstance(ground_truth, list):
            parsed = ground_truth
        else:
            return None

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return None
        if not any(isinstance(item, dict) and "bbox_2d" in item for item in parsed):
            return None

        changed = False
        for item in parsed:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                coords = [float(val) for val in bbox]
            except (TypeError, ValueError):
                continue
            x1, y1, x2, y2 = coords
            # Normalize to Qwen 0-1000 space with integer rounding and clamping.
            normalized = [
                round(x1 / width * 1000),
                round(y1 / height * 1000),
                round(x2 / width * 1000),
                round(y2 / height * 1000),
            ]
            normalized = [max(0, min(val, 1000)) for val in normalized]
            # Ensure bbox is ordered after rounding/clamping.
            normalized[0], normalized[2] = sorted((normalized[0], normalized[2]))
            normalized[1], normalized[3] = sorted((normalized[1], normalized[3]))
            item["bbox_2d"] = normalized
            changed = True

        if not changed:
            return None

        if output_as_str:
            return json.dumps(parsed, ensure_ascii=True)
        if isinstance(ground_truth, dict):
            return parsed[0]
        return parsed

    def _maybe_rescale_bbox_ground_truth(self, ground_truth, src_size, dst_size):
        """Rescale bbox_2d values from source image size to destination image size."""
        if ground_truth is None or not src_size or not dst_size:
            return None

        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        if not src_w or not src_h or not dst_w or not dst_h:
            return None
        if src_w == dst_w and src_h == dst_h:
            return None

        parsed = None
        output_as_str = False
        output_as_dict = False

        if isinstance(ground_truth, str):
            try:
                parsed = json.loads(ground_truth)
                output_as_str = True
            except Exception:
                return None
        elif isinstance(ground_truth, dict):
            parsed = copy.deepcopy(ground_truth)
            output_as_dict = True
        elif isinstance(ground_truth, list):
            parsed = copy.deepcopy(ground_truth)
        else:
            return None

        if isinstance(parsed, dict):
            parsed_items = [parsed]
            output_as_dict = True
        elif isinstance(parsed, list):
            parsed_items = parsed
        else:
            return None

        scale_x = float(dst_w) / float(src_w)
        scale_y = float(dst_h) / float(src_h)
        changed = False

        for item in parsed_items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(val) for val in bbox]
            except (TypeError, ValueError):
                continue

            rescaled = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            # Keep bbox well-formed after scaling.
            rescaled[0], rescaled[2] = sorted((rescaled[0], rescaled[2]))
            rescaled[1], rescaled[3] = sorted((rescaled[1], rescaled[3]))
            item["bbox_2d"] = rescaled
            changed = True

        if not changed:
            return None

        if output_as_str:
            return json.dumps(parsed_items, ensure_ascii=True)
        if output_as_dict:
            return parsed_items[0]
        return parsed_items

    def _maybe_pad_auxiliary_token_type_ids(self, type_ids, target_len, truncation):
        current_len = type_ids.shape[-1]
        if current_len < target_len:
            padding_len = target_len - current_len
            pad_tensor = type_ids.new_zeros((type_ids.shape[0], padding_len))
            type_ids = torch.cat([pad_tensor, type_ids], dim=-1)
        elif current_len > target_len:
            if truncation == "left":
                type_ids = type_ids[:, -target_len:]
            elif truncation == "right":
                type_ids = type_ids[:, :target_len]
            elif truncation == "middle":
                left_half = target_len // 2; right_half = target_len - left_half
                type_ids = torch.cat([type_ids[:, :left_half], type_ids[:, -right_half:]], dim=-1)
            else:
                raise ValueError(f"prompt longer than max_prompt_length\ntruncation {truncation} is not supported")
        return type_ids[0]

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        self._maybe_rewrite_question_for_molmo2(row_dict)
        messages = self._build_messages(row_dict)
        model_inputs = {}
        token_type_ids = None
        mm_token_type_ids = None

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # Preserve original image paths for downstream reward functions.
                if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                    row_dict["extra_info"] = {}
                if "images" not in row_dict["extra_info"]:
                    row_dict["extra_info"]["images"] = row_dict_images

                processed_images = []
                image_sizes = []
                processed_image_sizes = []
                for image in row_dict_images:
                    image_for_processing, image_size = self._prepare_image_with_size(image)
                    processed_image = process_image(image_for_processing, self.image_root, self.max_pixels)
                    processed_images.append(processed_image)
                    image_sizes.append(image_size)
                    if isinstance(processed_image, Image.Image):
                        processed_image_sizes.append((processed_image.width, processed_image.height))
                    else:
                        processed_image_sizes.append(None)

                images = processed_images
                if any(image_sizes):
                    row_dict["extra_info"].setdefault("image_sizes", image_sizes)

                if not self.normalize_bbox_to_1000:
                    reward_type = row_dict["extra_info"].get("reward_type")
                    if isinstance(reward_type, str) and reward_type.lower() in {"grounding", "clicking"}:
                        reward_model = row_dict.get("reward_model")
                        if isinstance(reward_model, dict):
                            source_size = image_sizes[0] if image_sizes else None
                            target_size = processed_image_sizes[0] if processed_image_sizes else None
                            updated_ground_truth = self._maybe_rescale_bbox_ground_truth(
                                reward_model.get("ground_truth"),
                                source_size,
                                target_size,
                            )
                            if updated_ground_truth is not None:
                                reward_model["ground_truth"] = updated_ground_truth
                                row_dict["extra_info"]["ground_truth_rescaled_to_processed_image"] = True

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

                if self.normalize_bbox_to_1000:
                    # Keep GT in absolute pixels; reward layer handles 0-1000 conversion.
                    row_dict["extra_info"].setdefault("normalize_bbox_to_1000", True)

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            processor_kwargs = {
                "text": [raw_prompt],
                "images": images,
                "videos": videos,
                "return_tensors": "pt",
            }
            if self.processor.__class__.__name__ in {"Qwen2VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLProcessor"}:
                processor_kwargs["return_mm_token_type_ids"] = True
            model_inputs = self.processor(**processor_kwargs)

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            token_type_ids = model_inputs.pop("token_type_ids", None)
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids", None)

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            token_type_ids = model_inputs.pop("token_type_ids", None)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        if token_type_ids is not None:
            row_dict["token_type_ids"] = self._maybe_pad_auxiliary_token_type_ids(
                token_type_ids, self.max_prompt_length, self.truncation
            )

        if mm_token_type_ids is not None:
            row_dict["mm_token_type_ids"] = self._maybe_pad_auxiliary_token_type_ids(
                mm_token_type_ids, self.max_prompt_length, self.truncation
            )
       
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        if "task_type" not in row_dict["extra_info"]:
            row_dict["extra_info"]["task_type"] = infer_task_type(
                row_dict.get("data_source"),
                row_dict["extra_info"],
                row_dict.get("reward_model"),
            )
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
