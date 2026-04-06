# Ablations With Shared Hydra Config

This folder keeps the long runtime logic in each script, while moving common trainer settings into one shared Hydra config:

- `config/gspo_llmjudge_shared.yaml`

Each run script hardcodes only the non-shared values for that ablation (dataset paths, `max_pixels`, `format_score`, `NUM_GPUS`, experiment suffix).

## Usage

From `vero-rl` root:

```bash
sbatch examples/ablations_shared_config/run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh
sbatch examples/ablations_shared_config/run_gspo_qwen3vl_instruct_IF_llmjudge.sh
```

The first positional argument is rollout engine (`vllm` by default). Extra Hydra overrides can be appended:

```bash
bash examples/ablations_shared_config/run_gspo_qwen3vl_instruct_IF_llmjudge.sh vllm trainer.total_training_steps=200
```
