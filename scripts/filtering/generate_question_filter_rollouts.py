#!/usr/bin/env python3
"""Generate question-filter rollouts with vLLM from a plain JSONL dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("question_filter_rollouts")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True, help="Filtering prompt text file.")
    parser.add_argument("--pretrained", required=True, help="Model name or path.")
    parser.add_argument("--data-file", required=True, help="Input JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where rollouts are saved.")
    parser.add_argument("--save-tag", required=True, help="Subdirectory name under --output-dir.")
    parser.add_argument("--image-root", default="", help="Optional root for relative image paths.")
    parser.add_argument(
        "--allowed-local-media-path",
        default=None,
        help="Optional directory that vLLM may access for local media.",
    )
    parser.add_argument("--load-folder", default=None, help="Resume from an existing rollout dir.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-batch-size", type=int, default=1000)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset loading without initializing vLLM.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def load_prompt_text(prompt_file: str) -> str:
    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return path.read_text(encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            records.append(payload)
    return records


def get_sample_id(sample: Dict[str, Any], fallback_index: int) -> str:
    sample_id = sample.get("id")
    if sample_id is None or str(sample_id).strip() == "":
        return f"sample_{fallback_index:06d}"
    return str(sample_id)


def get_question_text(sample: Dict[str, Any]) -> str:
    for key in ("input_query", "question", "problem"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("<image>", "").strip()
    return ""


def prepare_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_file = Path(args.data_file)
    if not data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    records = load_jsonl_records(data_file)
    start_index = max(args.start_index, 0)
    end_index = len(records) if args.end_index is None else min(args.end_index, len(records))
    if end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")

    sliced = records[start_index:end_index]
    if args.max_samples is not None:
        sliced = sliced[: args.max_samples]

    dataset: List[Dict[str, Any]] = []
    for index, sample in enumerate(sliced, start=start_index):
        payload = dict(sample)
        payload["id"] = get_sample_id(payload, index)
        dataset.append(payload)
    return dataset


def load_existing_ids(load_folder: str) -> set[str]:
    folder = Path(load_folder)
    if not folder.exists():
        LOGGER.warning("Load folder %s does not exist; ignoring resume state.", load_folder)
        return set()

    existing: set[str] = set()
    for path in sorted(folder.glob("rollouts_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = payload.get("id")
                if sample_id is not None:
                    existing.add(str(sample_id))
    return existing


def create_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) / args.save_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    args_path = output_dir / "args.json"
    args_path.write_text(json.dumps(serialize_args(args), indent=2, sort_keys=True), encoding="utf-8")
    return output_dir


def resolve_allowed_local_media_path(args: argparse.Namespace, data_file: Path) -> Optional[str]:
    if args.allowed_local_media_path:
        allowed_path = Path(args.allowed_local_media_path).expanduser().resolve()
    elif args.image_root:
        allowed_path = Path(args.image_root).expanduser().resolve()
    else:
        allowed_path = data_file.parent.resolve()

    if not allowed_path.exists() or not allowed_path.is_dir():
        return None
    return str(allowed_path)


def resolve_image_path(image_path: str, image_root: str, data_file: Path) -> str:
    if image_path.startswith(("http://", "https://", "file://")):
        return image_path
    if os.path.isabs(image_path):
        return image_path
    if image_root:
        return str((Path(image_root) / image_path).expanduser().resolve())
    return str((data_file.parent / image_path).resolve())


def init_llm(args: argparse.Namespace, allowed_local_media_path: Optional[str]):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit("vLLM is required for filtering generation. Please install vLLM.") from exc

    engine_kwargs: Dict[str, Any] = {
        "model": args.pretrained,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
    }
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if allowed_local_media_path:
        engine_kwargs["allowed_local_media_path"] = allowed_local_media_path

    LOGGER.info("Initializing vLLM with %s", engine_kwargs)
    return LLM(**engine_kwargs)


def create_sampling_params(args: argparse.Namespace):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise SystemExit("vLLM is required for filtering generation. Please install vLLM.") from exc

    return SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        skip_special_tokens=True,
    )


def build_question_prompts(
    tokenizer: Any,
    samples: Sequence[Dict[str, Any]],
    prompt_text: str,
    args: argparse.Namespace,
    data_file: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for question filtering with images.") from exc

    prompts: List[Dict[str, Any]] = []
    valid_samples: List[Dict[str, Any]] = []

    for sample in samples:
        question = get_question_text(sample)
        image_path = sample.get("image")

        user_content: List[Dict[str, Any]] = []
        multimodal_payload: Optional[Dict[str, Any]] = None

        if isinstance(image_path, str) and image_path.strip():
            resolved_path = resolve_image_path(image_path, args.image_root, data_file)
            local_path = resolved_path[len("file://") :] if resolved_path.startswith("file://") else resolved_path
            if not resolved_path.startswith(("http://", "https://")):
                try:
                    with Image.open(local_path) as image:
                        multimodal_payload = {"image": image.convert("RGB")}
                except (FileNotFoundError, OSError, Image.DecompressionBombError) as exc:
                    LOGGER.warning(
                        "Skipping sample %s because image %s could not be opened: %s",
                        sample["id"],
                        local_path,
                        exc,
                    )
                    continue
                image_url = resolved_path if resolved_path.startswith("file://") else f"file://{resolved_path}"
            else:
                image_url = resolved_path

            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if question:
            user_content.append({"type": "text", "text": f"Question: {question}"})
        else:
            user_content.append({"type": "text", "text": "Question:"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_payload: Dict[str, Any] = {"prompt": prompt}
        if multimodal_payload is not None:
            prompt_payload["multi_modal_data"] = multimodal_payload

        prompts.append(prompt_payload)
        valid_samples.append(sample)

    return prompts, valid_samples


def flush_records(records: List[Dict[str, Any]], output_dir: Path, save_tag: str, batch_index: int) -> None:
    if not records:
        return
    output_path = output_dir / f"rollouts_{save_tag}_batch{batch_index}_final.jsonl"
    write_jsonl(output_path, records)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.save_batch_size <= 0:
        raise SystemExit("--save-batch-size must be greater than 0")
    if args.n_rollouts <= 0:
        raise SystemExit("--n-rollouts must be greater than 0")

    prompt_text = load_prompt_text(args.prompt_file)
    data_file = Path(args.data_file).resolve()
    output_dir = create_output_dir(args)
    dataset = prepare_dataset(args)

    existing_ids = load_existing_ids(args.load_folder) if args.load_folder else set()
    remaining_samples = [sample for sample in dataset if sample["id"] not in existing_ids]
    allowed_media_path = resolve_allowed_local_media_path(args, data_file)

    print(f"[setup] Prompt file: {args.prompt_file}")
    print(f"[setup] Data file: {data_file}")
    print(f"[setup] Output dir: {output_dir}")
    print(f"[setup] Loaded {len(dataset)} samples; {len(remaining_samples)} remaining after resume filtering.")
    if allowed_media_path:
        print(f"[setup] Allowed local media path: {allowed_media_path}")

    if args.dry_run:
        print("[dry-run] Configuration validated. Skipping vLLM initialization.")
        return

    if not remaining_samples:
        print("[info] No remaining samples to process.")
        return

    llm = init_llm(args, allowed_media_path)
    tokenizer = llm.get_tokenizer()
    sampling_params = create_sampling_params(args)

    current_records: List[Dict[str, Any]] = []
    file_batch_index = 0
    samples_in_current_file = 0

    for batch_start in range(0, len(remaining_samples), args.batch_size):
        sample_batch = remaining_samples[batch_start : batch_start + args.batch_size]
        prompts, valid_samples = build_question_prompts(
            tokenizer=tokenizer,
            samples=sample_batch,
            prompt_text=prompt_text,
            args=args,
            data_file=data_file,
        )
        if not prompts:
            continue

        print(
            f"[batch] Generating question-filter rollouts for "
            f"{batch_start + 1}-{batch_start + len(valid_samples)} of {len(remaining_samples)}"
        )
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

        for output, sample in zip(outputs, valid_samples, strict=True):
            question = get_question_text(sample)
            image_path = sample.get("image")
            true_answer = sample.get("true_answer") or sample.get("answer")

            for candidate in output.outputs:
                current_records.append(
                    {
                        "question": question,
                        "image": image_path,
                        "true_answer": true_answer,
                        "thoughts": [],
                        "final_answer": candidate.text,
                        "prompt_file": args.prompt_file,
                        "judge_score": 1.0,
                        "scale_factor": 1.0,
                        "id": sample["id"],
                    }
                )

            samples_in_current_file += 1
            if samples_in_current_file >= args.save_batch_size:
                flush_records(current_records, output_dir, args.save_tag, file_batch_index)
                print(
                    f"[checkpoint] Wrote {len(current_records)} rollouts to "
                    f"{output_dir / f'rollouts_{args.save_tag}_batch{file_batch_index}_final.jsonl'}"
                )
                current_records = []
                samples_in_current_file = 0
                file_batch_index += 1

    if current_records:
        flush_records(current_records, output_dir, args.save_tag, file_batch_index)
        print(
            f"[final] Wrote {len(current_records)} rollouts to "
            f"{output_dir / f'rollouts_{args.save_tag}_batch{file_batch_index}_final.jsonl'}"
        )

    print(f"[done] Finished question filtering for {len(remaining_samples)} samples.")


if __name__ == "__main__":
    main(sys.argv[1:])
