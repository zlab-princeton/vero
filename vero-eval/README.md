# Vero Evaluation (vero-eval)

Evaluation harness for the Vero project, forked from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Installation

```bash
pip install -e .
```

See the main [Vero README](../README.md) for full setup instructions.

## Usage

See [docs/EVALUATION.md](../docs/EVALUATION.md) for the full evaluation guide.

## Key Directories

| Directory | Description |
|-----------|-------------|
| `lmms_eval/tasks/` | Task definitions (YAML + utilities) for all benchmarks |
| `lmms_eval/models/simple/` | Direct model wrappers (vLLM, HuggingFace, Accelerate) |
| `lmms_eval/models/chat/` | OpenAI-compatible chat model wrappers |
| `lmms_eval/llm_judge/` | LLM-based answer extraction for CoT evaluation |
| `lmms_eval/evaluator.py` | Main evaluation orchestration engine |
| `lmms_eval/loggers/` | Result logging (JSON outputs, Weights & Biases) |
| `examples/` | Evaluation scripts (single task, full domain sweep) |

## Task Structure

Task definitions live in `lmms_eval/tasks/<domain>/`. Each task has:
- YAML configs for different evaluation modes (standard, CoT, zero-shot)
- `utils.py` for task-specific preprocessing and metrics

Each model type has its own YAML file with a separate task name. For example, ChartQA has:
- `chartqa_mimo_zs.yaml` — MIMO zero-shot evaluation
- `chartqa_gpt5nano_zs.yaml` — GPT-5 Nano zero-shot evaluation

## Evaluation Workflows

### Single Task Evaluation

```bash
# Run a single checkpoint on a task
bash examples/eval.sh
```

Output artifacts (JSONL samples + aggregate JSON results) are saved to the `--output-path` directory.

### Full Domain Evaluation

```bash
# Evaluate across an entire domain (e.g., all Chart & OCR tasks)
bash examples/eval_domain.sh
```

## Evaluation Domains

### Chart & OCR
ChartQA-Pro, ChartQA, InfoVQA, CharXiv, ChartMuseum, EvoChart

### Grounding, Counting & Visual Search
CountBenchQA, CountQA, MME-RealWorld, V*Bench, AerialVG, VisualProbe, ScreenSpot, ScreenSpotPro

### STEM
MMMU-PRO (Standard & Vision), MathVision, MathVista

### Spatial & Action
Blink, ERQA, GameQA, EmbSpatial, CVBench

### Knowledge & Recognition
RealWorldQA, SimpleVQA, FVQA, MM-Vet V2

### Captioning & Instruction Following
MM-MTBench, MIABench, MMIFEval

## Troubleshooting

- **vLLM CUDA OOM**: Lower `max_seq_len` or adjust `--gpu-memory-utilization` in the eval script.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
