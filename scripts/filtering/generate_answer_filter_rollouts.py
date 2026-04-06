#!/usr/bin/env python3
"""Generate answer-filter rollouts with vLLM from a plain JSONL dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from generate_question_filter_rollouts import (
    configure_logging,
    create_output_dir,
    create_sampling_params,
    get_question_text,
    init_llm,
    load_existing_ids,
    load_prompt_text,
    prepare_dataset,
    write_jsonl,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True, help="Filtering prompt template file.")
    parser.add_argument("--pretrained", required=True, help="Model name or path.")
    parser.add_argument("--data-file", required=True, help="Input JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where rollouts are saved.")
    parser.add_argument("--save-tag", required=True, help="Subdirectory name under --output-dir.")
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
    parser.add_argument("--max-new-tokens", type=int, default=12096)
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


def build_answer_filter_prompts(
    tokenizer: Any,
    samples: Sequence[Dict[str, Any]],
    template_text: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    prompts: List[str] = []
    valid_samples: List[Dict[str, Any]] = []
    formatted_prompts: List[str] = []

    for sample in samples:
        question = get_question_text(sample)
        ground_truth = sample.get("true_answer") or sample.get("answer") or ""
        filled_prompt = template_text.replace("{QUESTION}", str(question)).replace(
            "{GROUND_TRUTH}",
            str(ground_truth),
        )
        messages = [{"role": "user", "content": filled_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        valid_samples.append(sample)
        formatted_prompts.append(filled_prompt)

    return prompts, valid_samples, formatted_prompts


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

    template_text = load_prompt_text(args.prompt_file)
    output_dir = create_output_dir(args)
    dataset = prepare_dataset(args)
    existing_ids = load_existing_ids(args.load_folder) if args.load_folder else set()
    remaining_samples = [sample for sample in dataset if sample["id"] not in existing_ids]

    print(f"[setup] Prompt file: {args.prompt_file}")
    print(f"[setup] Data file: {Path(args.data_file).resolve()}")
    print(f"[setup] Output dir: {output_dir}")
    print(f"[setup] Loaded {len(dataset)} samples; {len(remaining_samples)} remaining after resume filtering.")

    if args.dry_run:
        print("[dry-run] Configuration validated. Skipping vLLM initialization.")
        return

    if not remaining_samples:
        print("[info] No remaining samples to process.")
        return

    llm = init_llm(args, allowed_local_media_path=None)
    tokenizer = llm.get_tokenizer()
    sampling_params = create_sampling_params(args)

    current_records: List[Dict[str, Any]] = []
    file_batch_index = 0
    samples_in_current_file = 0

    for batch_start in range(0, len(remaining_samples), args.batch_size):
        sample_batch = remaining_samples[batch_start : batch_start + args.batch_size]
        prompts, valid_samples, formatted_prompts = build_answer_filter_prompts(
            tokenizer=tokenizer,
            samples=sample_batch,
            template_text=template_text,
        )

        print(
            f"[batch] Generating answer-filter rollouts for "
            f"{batch_start + 1}-{batch_start + len(valid_samples)} of {len(remaining_samples)}"
        )
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

        for output, sample, formatted_prompt in zip(outputs, valid_samples, formatted_prompts, strict=True):
            question = get_question_text(sample)
            image_path = sample.get("image")
            ground_truth = sample.get("true_answer") or sample.get("answer")

            for candidate in output.outputs:
                current_records.append(
                    {
                        "question": question,
                        "image": image_path,
                        "true_answer": ground_truth,
                        "prompt_text": formatted_prompt,
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

    print(f"[done] Finished answer filtering for {len(remaining_samples)} samples.")


if __name__ == "__main__":
    main(sys.argv[1:])
