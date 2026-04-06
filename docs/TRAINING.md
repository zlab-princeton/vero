# Vero Training Guide

## Overview

Vero uses **GSPO** (Group-relative Sequence Policy Optimization) with task-routed reward functions. GSPO is a variant of GRPO that replaces per-token importance ratios with a sequence-level ratio, uses asymmetric clipping (epsilon_high > epsilon_low), and removes the KL penalty for less-restricted updates. Training is single-stage RL on top of existing instruction-tuned models — no SFT warm start required.

## Prerequisites

- **Hardware**: 8x A100-80GB or H100-80GB GPUs (1 node)
- **Software**: `verovlm` conda environment (see [setup_env.sh](../scripts/setup_env.sh))
- **Data**: Training data in veRL JSONL format (see [DATA.md](DATA.md))

## Quick Start

```bash
# 1. Set required environment variables
export TRAIN_FILES="/path/to/train.verl.jsonl"
export VAL_FILES="/path/to/val.verl.jsonl"
export ROOT_PATH="/path/to/data_root"  # for datasets and checkpoints

# 2. Navigate to RL code
cd vero-rl

# 3. Launch training (REPO_ROOT is auto-detected)
bash examples/model_runs/run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh
```

## Training Scripts

| Script | Model | Base |
|--------|-------|------|
| [`run_gspo_qwen25vl_instruct_mix_all_llmjudge.sh`](../vero-rl/examples/model_runs/run_gspo_qwen25vl_instruct_mix_all_llmjudge.sh) | Vero-Qwen25-7B | Qwen2.5-VL-7B-Instruct |
| [`run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh`](../vero-rl/examples/model_runs/run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh) | Vero-Qwen3I-8B | Qwen3-VL-8B-Instruct |
| [`run_gspo_mimovl_mix_all_llmjudge.sh`](../vero-rl/examples/model_runs/run_gspo_mimovl_mix_all_llmjudge.sh) | Vero-MiMo-7B | MiMo-VL-7B-SFT |

All scripts are in `vero-rl/examples/model_runs/`.

## Configuration

Configs use Hydra with OmegaConf. The shared base config is at:
```
vero-rl/examples/model_runs/config/gspo_llmjudge_shared.yaml
```

Model-specific overrides are in the same directory (e.g., `gspo_qwen3vl.yaml`, `gspo_qwen25vl.yaml`).

### Key Parameters

**Data**:
| Parameter | Value |
|-----------|-------|
| Train batch size | 256 |
| Validation batch size | 512 |
| Max prompt length | 12,288–18,432 tokens (model-dependent) |
| Max response length | 10,240–18,432 tokens |
| Max image pixels | 9,437,184–16,777,216 |
| Rollout samples per prompt (n) | 8 |

**Optimizer**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| Warmup steps | 40 |
| Total training steps | 4,000 |
| Save frequency | Every 40 steps |

**GSPO Algorithm**:
| Parameter | Value |
|-----------|-------|
| Clip ratio low | 0.0003 |
| Clip ratio high | 0.0004 |
| Entropy coefficient | 0 |
| Loss aggregation | seq-mean-token-mean |

### Required Environment Variables

The training configs use the following environment variables via `${oc.env:...}`:

| Variable | Purpose |
|----------|---------|
| `TRAIN_FILES` | Path to training data JSONL |
| `VAL_FILES` | Path to validation data JSONL |
| `ROOT_PATH` | Root directory for datasets and checkpoints |
| `REPO_ROOT` | Repository root (auto-detected by training scripts) |

Set these before launching training:
```bash
export TRAIN_FILES="/path/to/train.verl.jsonl"
export VAL_FILES="/path/to/val.verl.jsonl"
export ROOT_PATH="/path/to/data_root"
```

## Reward

The reward stack lives in [`vero-rl/vero_reward/`](../vero-rl/vero_reward/). The reward combines format checking with task-specific accuracy verifiers:

**R(y, y\*) = 0.8 · R_acc(y, y\*) + 0.2 · R_fmt(y) + R_overlong(y)**

### Format Reward (R_fmt)

Checks valid `<think>...</think><answer>...</answer>` structure. For discrete symbolic answers, a single `\boxed{}` is required for full score. Implemented in the format-checking logic of [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py).

### Accuracy Verifiers (R_acc)

The main entrypoint [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) routes scoring by `reward_type`:

| Reward Type | Module | Description |
|-------------|--------|-------------|
| `string_match` | [`string_match.py`](../vero-rl/vero_reward/string_match.py) | Normalized exact-string equality |
| `multiple_choice` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Extract A–Z letter match |
| `numeric` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | SymPy-backed math_verify with optional tolerance |
| `list_match` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Any-match across reference strings |
| `ordering` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Exact order match; 0.2 discount for correct set wrong order |
| `web_action` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Weighted match over JSON action fields |
| `grounding` | [`grounding_reward.py`](../vero-rl/vero_reward/grounding_reward.py) | Hungarian matching of bounding boxes, IoU threshold 0.5 |
| `clicking` | [`click_reward.py`](../vero-rl/vero_reward/click_reward.py) | Point-in-box check |
| `instruction_following` | [`instructions.py`](../vero-rl/vero_reward/instructions.py) | Proportion of programmatic constraints satisfied |
| `llm_judge` | [`vero_vllm_judge.py`](../vero-rl/verl/workers/reward_manager/vero_vllm_judge.py) | Qwen3-32B scores 1–10 via OpenAI-compatible API |

### Overlong Penalty (R_overlong)

Linearly ramps from 0 to -1 over the final 2,048 tokens before the context limit.

### LLM Judge

The LLM judge runs as a vLLM server alongside training. The reward manager [`vero_vllm_judge.py`](../vero-rl/verl/workers/reward_manager/vero_vllm_judge.py) calls the judge using the prompt in [`llm_judge_reference.txt`](../vero-rl/examples/prompts/llm_judge_reference.txt).

The judge server is started automatically by the training scripts via [`llm_judge_server.sh`](../vero-rl/examples/model_runs/shared/llm_judge_server.sh), which launches a local `vllm serve` process, waits for readiness, and prepares the endpoint for reward calls.

| Parameter | Value |
|-----------|-------|
| Judge model | Qwen3-32B |
| Temperature | 0.7 |
| API endpoint | `http://localhost:51001/v1` |
| IF blend weight | configurable in `gspo_llmjudge_shared.yaml` |

## SLURM Usage

The training scripts include SBATCH headers. To submit as a SLURM job:

```bash
sbatch examples/model_runs/run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh
```

Edit the SBATCH headers at the top of each script to match your cluster configuration (partition, account, GPU type, etc.).

## Monitoring

Training logs to Weights & Biases. To use offline mode:
```bash
export WANDB_MODE=offline
# Sync later:
wandb sync wandb/offline-run-*
```

## Checkpointing

Checkpoints are saved every 40 steps to `$ROOT_PATH/checkpoints/verl/...`. To resume from a checkpoint, set the `resume_from` parameter in the Hydra config or pass it as a CLI override.
