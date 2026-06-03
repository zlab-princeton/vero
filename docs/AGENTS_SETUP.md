# Agent Setup Guide — Vero Evaluation, End to End

This is a single, self-contained runbook for setting up **Vero** and running
**VeroEvalSuite** from a clean machine. It is written so a coding agent
(**Claude Code**, **Codex**, or similar) can execute it step by step — but a human
can follow it just as well.

**How to use it with an agent:** point your agent at this file, e.g.

- Claude Code: `claude` then *"Follow docs/AGENTS_SETUP.md to set up this repo and run a smoke evaluation."*
- Codex: *"Read docs/AGENTS_SETUP.md and execute the steps to set up the env and run examples/preflight.sh."*

The agent should run the steps in order, stop at any `[FAIL]`, and use the
[Troubleshooting](#troubleshooting) table to recover.

> **Scope: this guide is for evaluation only.** Evaluation and training are
> independent — if you only want to *run the benchmarks* you never need to touch the
> training stack. RL training has its own setup in [docs/TRAINING.md](TRAINING.md).

---

## What you'll have at the end

- A `verovlm` conda env with PyTorch, vLLM, transformers, and `vero-eval` installed.
- Hugging Face caches pointed at a roomy disk (so datasets/models don't fill `$HOME`).
- An LLM judge configured for the reasoning benchmarks.
- A verified smoke evaluation, plus the exact commands for full runs.

## Where does the data download to? (Short answer)

Everything Hugging Face pulls — **benchmark datasets, the model under test, and the
judge** — downloads **automatically on first use**. The location is the standard
Hugging Face cache, controlled by `HF_HOME`:

```bash
export HF_HOME=/path/to/hf_cache    # datasets, models, and judge all cache here
```

`scripts/set_paths.sh.example` sets `HF_HOME` (and `HF_DATASETS_CACHE`,
`HF_HUB_CACHE`) for you. You never pre-download datasets by hand; you only choose
*where* they land and make sure you're logged in for gated ones.

---

## Prerequisites

- **conda / miniconda** on `PATH`.
- **CUDA toolkit** with `nvcc` (needed to compile flash-attn). On clusters:
  `module load cudatoolkit`.
- **GPUs.** Rule-based tasks run on **1 GPU**; **judge-based tasks need 2 GPUs**
  (one for the model under test, one for the judge) — see [GPU sizing](#gpu-sizing).
- **Disk:** enough free space for the benchmark datasets, plus the judge model if
  you cache one locally.

---

## Step 0 — Clone

```bash
git clone https://github.com/zlab-princeton/vero.git
cd vero
```

## Step 1 — Create the environment

```bash
bash scripts/setup_env.sh        # creates conda env "verovlm"; compiles flash-attn (10–20 min)
```

> Override the env name with `CONDA_ENV_NAME=...`, or its location with
> `CONDA_ENV_PATH=...` (useful on shared clusters).

## Step 2 — Set cache paths + judge (one file)

```bash
cp scripts/set_paths.sh.example set_paths.sh
$EDITOR set_paths.sh             # set ROOT_PATH to a disk with plenty of space
source set_paths.sh              # activates verovlm; sets HF_HOME, caches, JUDGE_MODEL_PATH
```

`set_paths.sh` (no `.example`) is git-ignored, so your paths/tokens stay local.
**Source it in every new shell** (and inside SLURM jobs) before evaluating.

Key variables it sets:

| Variable | Meaning |
|----------|---------|
| `HF_HOME` | Root of all Hugging Face downloads (datasets, models, judge). |
| `HF_DATASETS_CACHE`, `HF_HUB_CACHE` | Dataset / model-hub sub-caches under `HF_HOME`. |
| `JUDGE_MODEL_PATH` | Eval judge model for reasoning tasks (default `Qwen/Qwen3-32B`). |
| `API_TYPE` | Judge backend; `vllm` runs the local judge (some tasks otherwise default to OpenAI). |

## Step 3 — Log in to Hugging Face

Some datasets are gated (e.g. `MMMU/MMMU_Pro`):

```bash
huggingface-cli login            # or: export HF_TOKEN=hf_xxx (add it to set_paths.sh)
```

## Step 4 — Choose (and optionally pre-download) the judge

The reasoning variants extract/score the final answer with a local **LLM judge**.
`JUDGE_MODEL_PATH` (set in Step 2) selects it. To avoid a surprise multi-GB
download mid-eval, pre-fetch it now:

```bash
cd vero-eval
bash examples/preflight.sh --download-judge
```

> **If `JUDGE_MODEL_PATH` is unset**, judge tasks silently fall back to OpenAI
> `gpt-4o-*` and require `GPT_API_KEY`. Set a local judge to stay offline/free.

## Step 5 — Preflight (verify before burning GPU time)

```bash
cd vero-eval
bash examples/preflight.sh
```

This checks: env imports (`torch`, `vllm`, `lmms_eval`), visible GPUs, HF login,
and the judge. Fix every `[FAIL]` (and ideally `[WARN]`) before continuing.

## Step 6 — Smoke evaluation (tiny, validates the whole path)

```bash
cd vero-eval

# Rule-based task (no judge) — fastest sanity check
bash examples/eval.sh \
  --model-path zlab-princeton/Vero-Qwen3I-8B \
  --tasks chartqa_reasoning \
  --limit 5

# Judge-based task — confirms the judge works end to end (needs 2 GPUs)
bash examples/eval.sh \
  --model-path zlab-princeton/Vero-Qwen3I-8B \
  --tasks mathvista_testmini_reasoning \
  --judge-model Qwen/Qwen3-32B \
  --num-gpus 2 \
  --limit 5
```

Results (per-sample JSONL + aggregate JSON) land in `./eval_results/`.

## Step 7 — Full runs

```bash
# One full benchmark (judge task → 2 GPUs)
bash examples/eval.sh \
  --model-path zlab-princeton/Vero-Qwen3I-8B \
  --tasks mathvista_testmini_reasoning \
  --judge-model Qwen/Qwen3-32B \
  --num-gpus 2

# A whole domain
bash examples/eval_domain.sh \
  --model-path zlab-princeton/Vero-Qwen3I-8B \
  --domain chart_ocr --variant reasoning \
  --judge-model Qwen/Qwen3-32B \
  --num-gpus 2

# Everything
bash examples/eval_domain.sh \
  --model-path zlab-princeton/Vero-Qwen3I-8B \
  --domain all --variant reasoning \
  --judge-model Qwen/Qwen3-32B \
  --num-gpus 2
```

---

## Reference

### Task variants (the suffix on each task name)

| Variant suffix | Use for |
|----------------|---------|
| `_reasoning` | **Vero / trained checkpoints** (Qwen2.5-VL family). Emits `<think>`/`<answer>`. |
| `_reasoning_samplingq3` | Vero checkpoints trained on Qwen3-VL (relative 1000×1000 coords). |
| `_qwen25_zs` | Qwen2.5-VL-Instruct zero-shot baseline. |
| `_qwen3_zs` / `_qwen3_thinking_zs` | Qwen3-VL-Instruct / -Thinking zero-shot baselines. |
| `_mimo_zs`, `_gpt5nano_zs` | MiMo-VL / GPT-5-nano baselines. |

### Domains (for `eval_domain.sh --domain`)

`chart_ocr`, `stem`, `spatial_action`, `knowledge_rec`,
`grounding_counting_search`, `instruction_following`, `all`.

### Tasks that need a judge

`mathvista`, `mathvision`, `mmvetv2`, `mm_mt_bench`, `mia_bench`, `mmifeval`,
`fvqa`, `simplevqa`, `charxiv`, `chartmuseum`, `visual_probe`. Everything else
(e.g. `chartqa_reasoning`, ScreenSpot point/bbox) uses rule-based extraction.

### GPU sizing

**Judge-based tasks need 2 GPUs** — one for the model under test and one for the
judge. Run them with `--num-gpus 2`; the wrapper gives the judge the same GPU count
(`CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=$NUM_GPUS`). Rule-based tasks need only 1 GPU.
If the **eval model** itself runs out of memory, lower its memory fraction with
`--gpu-mem-util` (e.g. `0.7`). See [docs/EVALUATION.md](EVALUATION.md#judge-setup)
for more judge knobs.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unrecognized arguments: --judge_model_name ...` | Stale wrapper / non-existent CLI flag | Use the current `eval.sh`/`eval_domain.sh`; judge is set via `--judge-model` / `JUDGE_MODEL_PATH`. |
| Unexpected `gpt-4o` calls or `GPT_API_KEY` errors | `JUDGE_MODEL_PATH` unset → OpenAI fallback | `export JUDGE_MODEL_PATH=Qwen/Qwen3-32B` or pass `--judge-model`. |
| `mmvetv2`/`mm_mt_bench`/`mia_bench`/`mmifeval` hit `api.openai.com` despite a local judge | Those tasks default to `API_TYPE=openai` | `export API_TYPE=vllm` (set for you by `--judge-model` and `set_paths.sh`). |
| Judge task OOMs / hangs on a single GPU | Judge needs its own GPU | Run judge tasks with `--num-gpus 2`. |
| `401` / "you need to be authenticated" / gated dataset | Not logged in to Hugging Face | `huggingface-cli login` or `export HF_TOKEN=...`. |
| Home disk fills during download | `HF_HOME` not redirected | `source set_paths.sh`, or `export HF_HOME=/path/with/space`. |
| `ModuleNotFoundError: torch / vllm / lmms_eval` | env not active | `conda activate verovlm` (or re-`source set_paths.sh`). |
| vLLM CUDA OOM loading the model | GPU fraction too high | `--gpu-mem-util 0.7`. |
| `nvcc not found` during `setup_env.sh` | No CUDA toolkit | `module load cudatoolkit` (or install CUDA) and re-run. |

---

## Agent checklist (condensed)

```text
[ ] git clone + cd vero
[ ] bash scripts/setup_env.sh                       # conda env "verovlm"
[ ] cp scripts/set_paths.sh.example set_paths.sh    # edit ROOT_PATH
[ ] source set_paths.sh                             # HF_HOME + judge set
[ ] huggingface-cli login                           # gated datasets
[ ] cd vero-eval && bash examples/preflight.sh --download-judge
[ ] bash examples/eval.sh --model-path zlab-princeton/Vero-Qwen3I-8B \
        --tasks chartqa_reasoning --limit 5         # smoke (no judge, 1 GPU)
[ ] bash examples/eval.sh --model-path zlab-princeton/Vero-Qwen3I-8B \
        --tasks mathvista_testmini_reasoning --judge-model Qwen/Qwen3-32B --num-gpus 2 --limit 5
[ ] scale up: eval_domain.sh --domain all --variant reasoning --judge-model Qwen/Qwen3-32B --num-gpus 2
```

See [docs/EVALUATION.md](EVALUATION.md) for the full benchmark list and judge details.
