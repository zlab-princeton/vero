# Vero Evaluation Guide

## Overview

Vero is evaluated on **VeroEvalSuite**, a comprehensive benchmark of **30 diverse benchmarks** across all 6 task categories. The evaluation harness is in `vero-eval/`, a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

> **In a hurry / using an AI coding agent?** Follow [docs/AGENTS_SETUP.md](AGENTS_SETUP.md) — it walks an agent (Claude Code or Codex) or a human through environment setup, cache paths, HF login, the judge model, and a smoke eval, end to end.

## Environment & Data Setup

Before running any benchmark you need (1) the `verovlm` environment, (2) Hugging Face cache paths on a roomy disk, (3) an HF login for gated datasets, and (4) an LLM judge — but only for the judge-based tasks (11 of the 30 benchmarks; see the **LLM Judge** column in the tables below).

### 1. Install the environment

```bash
bash scripts/setup_env.sh          # creates the verovlm conda env
```

### 2. Set cache paths and the judge (one file)

Benchmark datasets, the model under test, and the judge **all download automatically on first run** via Hugging Face. Where they land is controlled by the standard `HF_HOME` / `HF_DATASETS_CACHE` / `HF_HUB_CACHE` variables. Copy the template, point it at a disk with plenty of space, and source it in every shell:

```bash
cp scripts/set_paths.sh.example set_paths.sh   # at the repo root
$EDITOR set_paths.sh                           # set ROOT_PATH (a disk with space)
source set_paths.sh                            # sets HF_HOME, caches, JUDGE_MODEL_PATH
```

`set_paths.sh` is git-ignored, so your paths and tokens never get committed. To override the download location ad hoc, just export the variables yourself:

```bash
export HF_HOME=/path/to/hf_cache               # everything caches under here
```

### 3. Log in to Hugging Face

Several datasets are gated (e.g. `MMMU/MMMU_Pro`). Authenticate once:

```bash
huggingface-cli login        # or: export HF_TOKEN=hf_xxx
```

### 4. Pick (and optionally pre-download) the judge

The judge-based reasoning tasks use a local **LLM judge** to extract and/or grade the final answer (see [Judge Setup](#judge-setup)). Set `JUDGE_MODEL_PATH` to a local directory or an HF id; it downloads on first use. `set_paths.sh` defaults it to `Qwen/Qwen3-32B`. Rule-based tasks never touch the judge, so you can skip this step if you only run those.

### 5. Preflight (recommended)

Verify everything — env imports, GPUs, HF login, judge — and optionally pre-fetch the judge, without launching a GPU run:

```bash
cd vero-eval
bash examples/preflight.sh                  # checks only
bash examples/preflight.sh --download-judge # also pre-downloads the judge
```

## VeroEvalSuite Benchmarks

**LLM Judge column legend:**
- **No** — fully rule-based scoring: the final answer is parsed from `<answer>…</answer>` / `\boxed{…}` and compared programmatically. No judge, runs on **1 GPU**.
- **Yes** — an **LLM judge** extracts and/or grades the final answer. Needs `JUDGE_MODEL_PATH` and **2 GPUs** (model + judge) — see [Judge Setup](#judge-setup).
- **Yes (VLM-capable)** — same as **Yes**, and the judge additionally receives the image when `JUDGE_MODEL_PATH` points at a vision-language model — see [VLM (image-aware) judge](#vlm-image-aware-judge).

### Chart & OCR (6 benchmarks)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| ChartQA | `chartqa_reasoning` | [`lmms-lab/ChartQA`](https://huggingface.co/datasets/lmms-lab/ChartQA) | No |
| ChartQA-Pro | `chartqa_pro_reasoning` | [`ahmed-masry/ChartQAPro`](https://huggingface.co/datasets/ahmed-masry/ChartQAPro) | No |
| InfoVQA | `infovqa_val_reasoning` | [`lmms-lab/DocVQA`](https://huggingface.co/datasets/lmms-lab/DocVQA) | No |
| CharXiv | `charxiv_reasoning` | [`princeton-nlp/CharXiv`](https://huggingface.co/datasets/princeton-nlp/CharXiv) | Yes |
| ChartMuseum | `chartmuseum_reasoning` | [`gsarch/ChartMuseum`](https://huggingface.co/datasets/gsarch/ChartMuseum) | Yes |
| EvoChart | `evochart_reasoning` | [`gsarch/EvoChart-QA`](https://huggingface.co/datasets/gsarch/EvoChart-QA) | No |

### STEM (4 benchmarks)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| MMMU-PRO Standard | `mmmu_pro_standard_reasoning` | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) | No |
| MMMU-PRO Vision | `mmmu_pro_vision_reasoning` | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) | No |
| MathVision | `mathvision_test_reasoning` | [`MathLLMs/MathVision`](https://huggingface.co/datasets/MathLLMs/MathVision) | Yes |
| MathVista | `mathvista_testmini_reasoning` | [`AI4Math/MathVista`](https://huggingface.co/datasets/AI4Math/MathVista) | Yes |

### Spatial & Action (5 benchmarks)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| CVBench | `cv_bench_reasoning` | [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | No |
| EmbSpatial | `embspatial_reasoning` | [`FlagEval/EmbSpatial-Bench`](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) | No |
| ERQA | `erqa_reasoning` | [`FlagEval/ERQA`](https://huggingface.co/datasets/FlagEval/ERQA) | No |
| GameQA-Lite | `game_qa_lite_reasoning` | [`gsarch/Game-QA-Lite`](https://huggingface.co/datasets/gsarch/Game-QA-Lite) | No |
| Blink | `blink_reasoning` | [`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | No |

### Knowledge & Recognition (4 benchmarks)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| RealWorldQA | `realworldqa_reasoning` | [`lmms-lab/RealWorldQA`](https://huggingface.co/datasets/lmms-lab/RealWorldQA) | No |
| SimpleVQA (English) | `simplevqa_en_reasoning` | [`gsarch/SimpleVQA-EN`](https://huggingface.co/datasets/gsarch/SimpleVQA-EN) | Yes |
| FVQA | `fvqa_reasoning` | [`lmms-lab/FVQA`](https://huggingface.co/datasets/lmms-lab/FVQA) | Yes |
| MM-Vet V2 | `mmvetv2_group_img_reasoning` | [`whyu/mm-vet-v2`](https://huggingface.co/datasets/whyu/mm-vet-v2) | Yes |

### Grounding, Counting & Visual Search (8 benchmarks — VisualProbe is split into 3 difficulty levels)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| CountBenchQA | `countbenchqa_reasoning` | [`vikhyatk/CountBenchQA`](https://huggingface.co/datasets/vikhyatk/CountBenchQA) | No |
| CountQA | `countqa_reasoning` | [`Jayant-Sravan/CountQA`](https://huggingface.co/datasets/Jayant-Sravan/CountQA) | No |
| MME-RealWorld-Lite | `mme_realworld_lite_reasoning` | [`yifanzhang114/MME-RealWorld-lite-lmms-eval`](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval) | No |
| V*Bench | `vstar_bench_reasoning` | [`lmms-lab/vstar-bench`](https://huggingface.co/datasets/lmms-lab/vstar-bench) | No |
| AerialVG | `aerialvg_bbox_reasoning` | [`IPEC-COMMUNITY/AerialVG`](https://huggingface.co/datasets/IPEC-COMMUNITY/AerialVG) | No |
| VisualProbe (Easy) | `visual_probe_easy_reasoning` | [`Mini-o3/VisualProbe_Easy`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Easy) | Yes |
| VisualProbe (Medium) | `visual_probe_medium_reasoning` | [`Mini-o3/VisualProbe_Medium`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Medium) | Yes |
| VisualProbe (Hard) | `visual_probe_hard_reasoning` | [`Mini-o3/VisualProbe_Hard`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Hard) | Yes |
| ScreenSpot | `screenspot_point_in_box_reasoning` | [`rootsautomation/ScreenSpot`](https://huggingface.co/datasets/rootsautomation/ScreenSpot) | No |
| ScreenSpotPro | `screenspotpro_point_in_box_reasoning` | [`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | No |

### Captioning & Instruction Following (3 benchmarks)
| Benchmark | Task Name | Dataset | LLM Judge |
|-----------|-----------|---------|-----------|
| MM-MTBench | `mm_mt_bench_reasoning` | [`mistralai/MM-MT-Bench`](https://huggingface.co/datasets/mistralai/MM-MT-Bench) | Yes (VLM-capable) |
| MIABench | `mia_bench_reasoning` | [`lmms-lab/MIA-Bench`](https://huggingface.co/datasets/lmms-lab/MIA-Bench) | Yes (VLM-capable) |
| MMIFEval | `mmifeval_reasoning` | [`lscpku/MMIFEval`](https://huggingface.co/datasets/lscpku/MMIFEval) | Yes (VLM-capable) |

## Quick Start

The `examples/eval.sh` wrapper is the simplest entry point — it wires up the judge for you:

```bash
cd vero-eval

# chartqa_reasoning uses rule-based extraction (no judge needed)
bash examples/eval.sh \
    --model-path zlab-princeton/Vero-Qwen3I-8B \
    --tasks chartqa_reasoning \
    --limit 5                       # drop --limit for the full set

# mathvista_testmini_reasoning needs a judge — pass it with --judge-model.
# Judge-based tasks need 2 GPUs (model + judge), so use --num-gpus 2.
bash examples/eval.sh \
    --model-path zlab-princeton/Vero-Qwen3I-8B \
    --tasks mathvista_testmini_reasoning \
    --judge-model Qwen/Qwen3-32B \
    --num-gpus 2
```

Equivalent direct invocation (the judge is read from `JUDGE_MODEL_PATH`, not a CLI flag):

```bash
cd vero-eval
export JUDGE_MODEL_PATH=Qwen/Qwen3-32B   # only needed for judge-based tasks

python -m lmms_eval \
    --model vllm \
    --model_args model=zlab-princeton/Vero-Qwen3I-8B,tensor_parallel_size=1 \
    --tasks mathvista_testmini_reasoning \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/
```

## Using eval_domain.sh

The domain evaluation script is `vero-eval/examples/eval_domain.sh`. It expands a
`--domain` + `--variant` pair into the full task list and runs them in one
`lmms_eval` invocation. The model defaults to `zlab-princeton/Vero-Qwen3I-8B` and the
variant to `reasoning`; the judge is read from `JUDGE_MODEL_PATH` (set by `set_paths.sh`,
or pass `--judge-model`).

**Full reproduction** — all 30 benchmarks for the default model, no `--limit`:

```bash
bash examples/eval_domain.sh --domain all --num-gpus 2
```

Pick the `--variant` that matches the checkpoint type (instruct vs thinking):

| Vero checkpoint | Type | `--variant` |
|-----------------|------|-------------|
| `Vero-Qwen25-7B`, `Vero-Qwen3I-8B` | instruct | `reasoning` (default) |
| `Vero-Qwen3T-8B`, `Vero-MiMo-7B`, `Vero-Qwen35-9B`, `Vero-Qwen35-9B-Base` | thinking | `reasoning_samplingq3` |

```bash
# A single domain (reasoning domains use the judge → --num-gpus 2)
bash examples/eval_domain.sh --domain chart_ocr --variant reasoning --num-gpus 2

# A thinking checkpoint over all domains
bash examples/eval_domain.sh \
    --model-path zlab-princeton/Vero-Qwen3T-8B \
    --domain all --variant reasoning_samplingq3 --num-gpus 2
```

### Task Presets

The eval script supports preset groups for running benchmarks by category:
- Chart & OCR benchmarks
- STEM benchmarks
- Spatial & Action benchmarks
- Knowledge & Recognition benchmarks
- Grounding, Counting & Search benchmarks
- Captioning & Instruction Following benchmarks

## Batch Evaluation

To evaluate a whole category (or everything) in one run, use `eval_domain.sh` with
`--domain all`:

```bash
bash examples/eval_domain.sh \
    --model-path /path/to/checkpoint \
    --domain all \
    --variant reasoning \
    --judge-model Qwen/Qwen3-32B
```

To sweep several checkpoints, loop over `eval.sh` / `eval_domain.sh` with a distinct
`--output-path` per checkpoint:

```bash
for ckpt in /path/to/checkpoints/global_step_*; do
  bash examples/eval_domain.sh \
      --model-path "$ckpt" \
      --domain all --variant reasoning \
      --judge-model Qwen/Qwen3-32B \
      --output-path "./eval_results/$(basename "$ckpt")"
done
```

## System Prompt

The Vero system prompt (which defines the `<think>` / `<answer>` output format) is **already baked into the chat template**. If you do not pass a system message, the chat template uses it by default — no extra setup is needed. The chat template also appends a `<think>` token at the start of the assistant turn to kick off reasoning, so **do not add `<think>` manually** to the prompt.

**If you want to use a custom system prompt**, you have two options:

1. **Recommended:** keep the chat template's default system prompt and place your custom instructions in the user message.
2. Override the system role entirely — in this case, make sure your custom system prompt still instructs the model to produce the `<think>` / `<answer>` format, or append the Vero prompt from [`vero-eval/examples/prompts/vero_system_prompt.txt`](../vero-eval/examples/prompts/vero_system_prompt.txt) to yours.

## Chain-of-Thought Evaluation

Vero models generate reasoning traces in `<think>` tags. Tasks fall into two groups:

- **Rule-based scoring** (no judge, 19 of 30 benchmarks): the final answer is parsed from `<answer>…</answer>` / `\boxed{…}` and compared programmatically — e.g. `chartqa_reasoning`, the ScreenSpot point/bbox tasks. See the **LLM Judge = No** rows in the benchmark tables above.
- **Judge-based scoring** (needs a judge, 11 of 30 benchmarks): an **LLM judge** reads the response and extracts and/or grades the final answer. What the judge does varies by task:

| Task | Judge role |
|------|-----------|
| `mathvista_testmini_reasoning` | Extracts the final answer from the response; comparison to ground truth is then rule-based |
| `mathvision_test_reasoning` | Grades the answer against ground truth (binary 0/1) |
| `charxiv_reasoning` | Extracts the final answer and grades it in one call |
| `chartmuseum_reasoning` | Grades the answer against ground truth (correct/incorrect) |
| `fvqa_reasoning` | Grades the answer against the reference and candidate answers (correct/incorrect) |
| `simplevqa_en_reasoning` | Grades the answer against candidate answers (correct/incorrect) |
| `mmvetv2_group_img_reasoning` | Assigns a numeric score following the official MM-Vet grading protocol |
| `visual_probe_{easy,medium,hard}_reasoning` | Grades the answer against the reference (correct/incorrect) |
| `mm_mt_bench_reasoning` | Rates response quality 1–10 following the official protocol (**VLM-capable**) |
| `mia_bench_reasoning` | Scores instruction adherence per instruction (**VLM-capable**) |
| `mmifeval_reasoning` | Hybrid: rule-based constraint checks plus judge calls for constraints that need semantic/visual verification (**VLM-capable**) |

**VLM-capable** means the judge also receives the image when the configured judge is a vision-language model — see [VLM (image-aware) judge](#vlm-image-aware-judge) below.

### Judge Setup

The judge is selected **by environment variables**, not CLI flags:

```bash
export JUDGE_MODEL_PATH=Qwen/Qwen3-32B    # local dir or HF id (downloads on first use)
export API_TYPE=vllm                      # use the local judge (see note below)
```

`examples/eval.sh --judge-model <model>` and `examples/eval_domain.sh --judge-model <model>` export both of these for you (and `set_paths.sh` sets them globally).

> **`API_TYPE` picks the backend.** Most judge tasks default to the local vLLM judge, but a few (`mmvetv2`, `mm_mt_bench`, `mia_bench`, `mmifeval`) default to the **OpenAI** backend — so set `API_TYPE=vllm` to keep everything local. To use an OpenAI `gpt-4o-*` judge instead, set `API_TYPE=openai`, `JUDGE_MODEL_PATH=gpt-4o-2024-05-13`, and `GPT_API_KEY`.

> **If `JUDGE_MODEL_PATH` is unset, judge tasks fall back to OpenAI `gpt-4o-*`** and require `GPT_API_KEY`. Set a local judge (above) to stay fully offline/free.

**GPU requirement.** Judge-based tasks need **2 GPUs** — one for the model under test and one for the judge. Run them with `--num-gpus 2` (the wrappers give the judge the same GPU count, i.e. `CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=$NUM_GPUS`). **Rule-based tasks** (e.g. `chartqa_reasoning`, the ScreenSpot tasks) use no judge and run on **1 GPU**.

The judge loads in-process as a separate vLLM engine. Tune it with:

```bash
export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=2     # judge GPUs (defaults to --num-gpus)
export CHARXIV_JUDGE_MAX_MODEL_LEN=18384        # cap judge context
```

### VLM (image-aware) judge

Three tasks — `mm_mt_bench_reasoning`, `mia_bench_reasoning`, and `mmifeval_reasoning` — can send the **image along with the text** to the judge, so grading also accounts for visual content. The mode is picked automatically from the judge model name: if `JUDGE_MODEL_PATH` looks like a vision-language model (e.g. contains `qwen3-vl`, `qwen2.5-vl`, …) these tasks run the judge in image+text mode; otherwise they fall back to text-only judging. All other judge tasks are text-only regardless of the judge model.

```bash
# Text-only LLM judging for all judge tasks (the default):
export JUDGE_MODEL_PATH=Qwen/Qwen3-32B

# Image-aware judging for mm_mt_bench / mia_bench / mmifeval (paper protocol):
export JUDGE_MODEL_PATH=Qwen/Qwen3-VL-32B-Instruct
```

To force a mode regardless of the model name, set `LMMS_EVAL_JUDGE_MODE=vlm` or `LMMS_EVAL_JUDGE_MODE=llm`.

> **Reproducing the paper's numbers:** the paper uses **Qwen3-32B** (thinking disabled) as the LLM judge for all text-only judge tasks and **Qwen3-VL-32B-Instruct** as the VLM judge for `mm_mt_bench`, `mia_bench`, and `mmifeval`. With the default `Qwen/Qwen3-32B` those three tasks still run, but judge text-only — convenient for quick runs, not identical to the paper protocol.

### Judge server mode (optional)

To run the judge as a standalone OpenAI-compatible vLLM server instead of in-process (e.g. to pin it to dedicated GPUs), launch `vllm serve <judge>` yourself and set:

```bash
export VLLM_SERVER_JUDGE=1
export VLLM_JUDGE_BASE_URL=http://127.0.0.1:<port>/v1   # your vllm serve endpoint
```

> Note: `--launcher_args` in `lmms_eval` controls a *different* (sglang-based) judge framework and is **not** used by VeroEvalSuite's reasoning tasks. Use `JUDGE_MODEL_PATH` as above.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unrecognized arguments: --judge_model_name ...` | Old wrapper passed a non-existent CLI flag | Update to the current `eval.sh`/`eval_domain.sh`; the judge is set via `JUDGE_MODEL_PATH` / `--judge-model`, not a CLI arg. |
| `RuntimeError: Unable to resolve judge model path` / unexpected calls to `gpt-4o` / `GPT_API_KEY` errors | `JUDGE_MODEL_PATH` not set, so a judge task fell back to OpenAI | `export JUDGE_MODEL_PATH=Qwen/Qwen3-32B` (or pass `--judge-model`). |
| `mmvetv2`/`mm_mt_bench`/`mia_bench`/`mmifeval` still call `api.openai.com` despite a local judge | These tasks default to `API_TYPE=openai` | `export API_TYPE=vllm` (set automatically by `--judge-model` and `set_paths.sh`). |
| Judge task OOMs / hangs on a single GPU | Judge has no GPU of its own | Judge tasks need **2 GPUs** — run with `--num-gpus 2`. |
| Gated dataset / `401` / `you need to be authenticated` | Not logged in to Hugging Face | `huggingface-cli login` or `export HF_TOKEN=...`. |
| Disk fills up in `~/.cache/huggingface` | `HF_HOME` not redirected | `source set_paths.sh` (sets `HF_HOME`), or `export HF_HOME=/path/with/space`. |
| vLLM CUDA OOM loading the model | GPU memory fraction too high | Lower it: `--gpu-mem-util 0.7` (or add `gpu_memory_utilization=0.7` to `--model_args`). |
| OOM when the **judge** loads | Judge too big for one card | `export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=2`, use a higher-memory GPU, or a smaller judge. |
| Want to verify setup without a full run | — | `bash examples/preflight.sh` (add `--download-judge` to pre-fetch the judge). |

## Adding New Benchmarks

Task definitions live in `lmms_eval/tasks/<task_name>/`:

1. Create a directory under `lmms_eval/tasks/`
2. Add a YAML config defining the dataset, prompts, and metrics
3. Add a `utils.py` with task-specific preprocessing and metric functions
4. Register variants (CoT, direct answer, etc.) as separate YAML files

See `lmms_eval/tasks/chartqa/` for a complete example.
