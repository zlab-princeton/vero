# Vero RL Training (vero-rl)

Reinforcement learning training framework for the Vero project, which is a clean fork from [veRL](https://github.com/volcengine/verl).

## Usage

See [docs/TRAINING.md](../docs/TRAINING.md) for the full training guide.

## Dataset Setup

The bash launchers in [`examples/model_runs/`](examples/model_runs/) now expect formatted local data under [`data/`](data/) by default.

From the repository root, run:

```bash
python scripts/download_and_format_vero_600k.py
```

This prepares:

```text
vero-rl/data/vero_600k_train.verl.jsonl
vero-rl/data/vero_600k_val.verl.jsonl
vero-rl/data/images/...
```

The shared model-run config automatically resolves:

- `TRAIN_FILES` -> `${REPO_ROOT}/data/vero_600k_train.verl.jsonl`
- `VAL_FILES` -> `${REPO_ROOT}/data/vero_600k_val.verl.jsonl`
- `IMAGE_ROOT` -> `${REPO_ROOT}/data`

You can still override those defaults by exporting `TRAIN_FILES`, `VAL_FILES`, or `IMAGE_ROOT` before launching any script in `examples/model_runs/`.

## Key Components

- `vero_reward` - reward code
- `verl/workers/reward_manager/vero_vllm_judge.py` - reward manager with llm judge code
- `verl/trainer/` — PPO/GRPO training logic and configs
- `verl/models/transformers/` — VLM model adapters
- `verl/workers/` — Distributed worker implementations
- `verl/utils/dataset/` — Dataset loading and processing
- `examples/model_runs/` — Training launch scripts

## Supported Models

- Qwen3.5-VL (9B)
- Qwen2.5-VL (7B)
- Qwen3-VL (8B)
- MiMo-VL (7B)
- Bee (8B)
- Molmo2 (8B, 7B)
