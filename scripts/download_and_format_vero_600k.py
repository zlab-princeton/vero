#!/usr/bin/env python3
"""Download or reuse a cached Vero-600k HF dataset and export local veRL JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import datasets
from datasets import Image as HFImage
from huggingface_hub import snapshot_download
from PIL import Image as PILImage


DEFAULT_REPO_ID = "gsarch/Vero-600k"
DEFAULT_TRAIN_FILENAME = "vero_600k_train.verl.jsonl"
DEFAULT_VAL_FILENAME = "vero_600k_val.verl.jsonl"
def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "vero-rl" / "data"

    parser = argparse.ArgumentParser(description="Download and format Vero-600k into repo-local veRL JSONL files.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id or local snapshot path.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Output directory for JSONL and images.")
    parser.add_argument("--train-split", default="train", help="Training split name in the HF dataset.")
    parser.add_argument("--val-split", default="val", help="Validation split name in the HF dataset.")
    parser.add_argument("--configs", help="Optional comma-separated subset of dataset configs to export.")
    parser.add_argument("--limit-per-split", type=int, help="Optional row cap per config and split for debugging.")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL and exported image files.")
    return parser.parse_args()


def sanitize_component(value: str, fallback: str = "item") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or fallback


def ensure_output_targets(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / DEFAULT_TRAIN_FILENAME
    val_jsonl = output_dir / DEFAULT_VAL_FILENAME

    if not overwrite:
        existing = [path for path in (train_jsonl, val_jsonl) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing output files: {joined}. "
                "Re-run with --overwrite to replace them."
            )
    return train_jsonl, val_jsonl


def resolve_dataset_source(repo_id: str, cache_dir: str | None) -> str:
    local_candidate = Path(os.path.expanduser(repo_id))
    if local_candidate.exists():
        return str(local_candidate.resolve())
    return snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir)


def resolve_config_names(dataset_source: str, requested_configs: str | None) -> list[str]:
    if requested_configs:
        configs = [item.strip() for item in requested_configs.split(",") if item.strip()]
        if not configs:
            raise ValueError("--configs was provided but no valid config names were parsed.")
        return configs

    config_names = datasets.get_dataset_config_names(dataset_source)
    if not config_names:
        raise ValueError(f"No dataset configs found in {dataset_source!r}.")
    return config_names


def stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    fragments.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    fragments.append("<image>")
                elif item.get("type") == "video":
                    fragments.append("<video>")
                else:
                    fragments.append(json.dumps(item, ensure_ascii=False))
            else:
                fragments.append(str(item))
        return "".join(fragments)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return "" if content is None else str(content)


def normalize_prompt(prompt_payload, fallback_question: str) -> list[dict[str, str]]:
    if isinstance(prompt_payload, list):
        messages = []
        for message in prompt_payload:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = stringify_content(message.get("content", ""))
            messages.append({"role": role, "content": content})
        if messages:
            return messages

    if isinstance(prompt_payload, dict):
        roles = prompt_payload.get("role")
        contents = prompt_payload.get("content")
        if isinstance(roles, list) and isinstance(contents, list):
            messages = []
            for role, content in zip(roles, contents, strict=False):
                messages.append({"role": str(role), "content": stringify_content(content)})
            if messages:
                return messages
        if "role" in prompt_payload or "content" in prompt_payload:
            return [
                {
                    "role": str(prompt_payload.get("role", "user")),
                    "content": stringify_content(prompt_payload.get("content", "")),
                }
            ]

    return [{"role": "user", "content": fallback_question}]


def choose_question(prompt_messages: list[dict[str, str]], raw_extra_info: dict) -> str:
    question = raw_extra_info.get("question")
    if isinstance(question, str) and question:
        return question
    for message in prompt_messages:
        if message.get("role") == "user" and message.get("content"):
            return message["content"]
    return ""


def normalize_reward_model(reward_model_payload) -> dict[str, str]:
    reward_model_payload = reward_model_payload if isinstance(reward_model_payload, dict) else {}
    style = reward_model_payload.get("style")
    ground_truth = reward_model_payload.get("ground_truth")
    return {
        "style": "" if style is None else str(style),
        "ground_truth": "" if ground_truth is None else str(ground_truth),
    }


def normalize_extra_info(raw_extra_info, *, record_id: str, split_name: str, row_index: int, question: str) -> dict:
    raw_extra_info = raw_extra_info if isinstance(raw_extra_info, dict) else {}
    tolerance = raw_extra_info.get("tolerance")
    if tolerance is not None:
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError):
            tolerance = None

    raw_index = raw_extra_info.get("index")
    if raw_index is None:
        normalized_index = row_index
    else:
        try:
            normalized_index = int(raw_index)
        except (TypeError, ValueError):
            normalized_index = row_index

    answer = raw_extra_info.get("answer")
    reward_type = raw_extra_info.get("reward_type")
    return {
        "id": record_id,
        "split": split_name,
        "index": normalized_index,
        "question": question,
        "answer": "" if answer is None else str(answer),
        "reward_type": "" if reward_type is None else str(reward_type),
        "tolerance": tolerance,
    }


def detect_image_suffix(image_dict: dict) -> str:
    raw_path = image_dict.get("path")
    if isinstance(raw_path, str):
        suffix = Path(raw_path).suffix.lower()
        if suffix:
            return suffix
    return ".png"


def export_image(image_dict: dict, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return

    image_bytes = image_dict.get("bytes")
    if isinstance(image_bytes, bytes):
        destination.write_bytes(image_bytes)
        return

    raw_path = image_dict.get("path")
    if isinstance(raw_path, str):
        source_path = Path(raw_path)
        if source_path.exists():
            destination.write_bytes(source_path.read_bytes())
            return

    decoded = image_dict.get("decoded")
    if isinstance(decoded, PILImage.Image):
        decoded.save(destination)
        return

    raise ValueError(f"Unable to export image to {destination}: no bytes or readable source path were found.")


def make_relative_image_path(config_name: str, split_name: str, row_index: int, record_id: str, image_dict: dict) -> Path:
    suffix = detect_image_suffix(image_dict)
    filename = f"{row_index:08d}_{sanitize_component(record_id, fallback='sample')}{suffix}"
    return Path("images") / split_name / sanitize_component(config_name, fallback="config") / filename


def iter_rows(
    dataset_source: str,
    config_name: str,
    split_name: str,
    cache_dir: str | None,
    limit_per_split: int | None,
) -> Iterable[dict]:
    dataset = datasets.load_dataset(
        dataset_source,
        name=config_name,
        split=split_name,
        cache_dir=cache_dir,
        verification_mode="no_checks",
    )
    if limit_per_split is not None:
        dataset = dataset.select(range(min(len(dataset), limit_per_split)))
    dataset = dataset.cast_column("image", HFImage(decode=False))
    for row in dataset:
        yield row


def format_row(row: dict, *, config_name: str, split_name: str, row_index: int, output_dir: Path, overwrite: bool) -> dict:
    raw_extra_info = row.get("extra_info")
    raw_prompt = row.get("prompt")
    prompt_messages = normalize_prompt(raw_prompt, fallback_question="")

    record_id = row.get("id")
    if record_id is None and isinstance(raw_extra_info, dict):
        record_id = raw_extra_info.get("id")
    record_id = str(record_id) if record_id is not None else f"{config_name}_{split_name}_{row_index}"

    question = choose_question(prompt_messages, raw_extra_info if isinstance(raw_extra_info, dict) else {})
    if question and prompt_messages == [{"role": "user", "content": ""}]:
        prompt_messages = [{"role": "user", "content": question}]

    image_dict = row.get("image")
    if not isinstance(image_dict, dict):
        raise TypeError(f"Expected 'image' to be a dict for config {config_name}, split {split_name}, row {row_index}.")

    relative_image_path = make_relative_image_path(config_name, split_name, row_index, record_id, image_dict)
    export_image(image_dict, output_dir / relative_image_path, overwrite=overwrite)

    return {
        "id": record_id,
        "data_source": "" if row.get("data_source") is None else str(row.get("data_source")),
        "prompt": prompt_messages,
        "images": [relative_image_path.as_posix()],
        "ability": "" if row.get("ability") is None else str(row.get("ability")),
        "reward_model": normalize_reward_model(row.get("reward_model")),
        "extra_info": normalize_extra_info(
            raw_extra_info,
            record_id=record_id,
            split_name=split_name,
            row_index=row_index,
            question=question,
        ),
    }


def write_split_jsonl(
    output_path: Path,
    dataset_source: str,
    config_names: list[str],
    split_name: str,
    output_dir: Path,
    cache_dir: str | None,
    limit_per_split: int | None,
    overwrite: bool,
) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for config_name in config_names:
            print(f"[format] config={config_name} split={split_name}", flush=True)
            rows = iter_rows(
                dataset_source=dataset_source,
                config_name=config_name,
                split_name=split_name,
                cache_dir=cache_dir,
                limit_per_split=limit_per_split,
            )
            for row_index, row in enumerate(rows):
                record = format_row(
                    row,
                    config_name=config_name,
                    split_name=split_name,
                    row_index=row_index,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_jsonl, val_jsonl = ensure_output_targets(output_dir, overwrite=args.overwrite)

    dataset_source = resolve_dataset_source(args.repo_id, args.cache_dir)
    config_names = resolve_config_names(dataset_source, args.configs)
    print(f"[source] using dataset source: {dataset_source}", flush=True)
    print(f"[source] configs: {', '.join(config_names)}", flush=True)

    train_count = write_split_jsonl(
        output_path=train_jsonl,
        dataset_source=dataset_source,
        config_names=config_names,
        split_name=args.train_split,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        limit_per_split=args.limit_per_split,
        overwrite=args.overwrite,
    )
    val_count = write_split_jsonl(
        output_path=val_jsonl,
        dataset_source=dataset_source,
        config_names=config_names,
        split_name=args.val_split,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        limit_per_split=args.limit_per_split,
        overwrite=args.overwrite,
    )

    print("", flush=True)
    print("Formatting complete.", flush=True)
    print(f"train rows: {train_count}", flush=True)
    print(f"val rows:   {val_count}", flush=True)
    print(f"train jsonl: {train_jsonl}", flush=True)
    print(f"val jsonl:   {val_jsonl}", flush=True)
    print(f"image root:  {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
