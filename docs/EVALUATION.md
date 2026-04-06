# Vero Evaluation Guide

## Overview

Vero is evaluated on **VeroEvalSuite**, a comprehensive benchmark of **32 diverse benchmarks** across all 6 task categories. The evaluation harness is in `vero-eval/`, a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## VeroEvalSuite Benchmarks

### Chart & OCR (6 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| ChartQA | `chartqa_reasoning` |
| ChartQA-Pro | `chartqa_pro_reasoning` |
| InfoVQA | `infovqa_val_reasoning` |
| CharXiv | `charxiv_reasoning` |
| ChartMuseum | `chartmuseum_reasoning` |
| EvoChart | `evochart_reasoning` |

### STEM (4 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| MMMU-PRO Standard | `mmmu_pro_standard_reasoning` |
| MMMU-PRO Vision | `mmmu_pro_vision_reasoning` |
| MathVision | `mathvision_test_reasoning` |
| MathVista | `mathvista_testmini_reasoning` |

### Spatial & Action (5 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| CVBench | `cv_bench_reasoning` |
| EmbSpatial | `embspatial_reasoning` |
| ERQA | `erqa_reasoning` |
| GameQA | `game_qa_lite_reasoning` |
| Blink | `blink_reasoning` |

### Knowledge & Recognition (4 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| RealWorldQA | `realworldqa_reasoning` |
| SimpleVQA (English) | `simplevqa_en_reasoning` |
| FVQA | `fvqa_reasoning` |
| MM-Vet V2 | `mmvetv2_group_img_reasoning` |

### Grounding, Counting & Visual Search (10 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| CountBenchQA | `countbenchqa_reasoning` |
| CountQA | `countqa_reasoning` |
| MME-RealWorld | `mme_realworld_lite_reasoning` |
| V*Bench | `vstar_bench_reasoning` |
| AerialVG | `aerialvg_bbox_reasoning` |
| VisualProbe (Easy) | `visual_probe_easy_reasoning` |
| VisualProbe (Medium) | `visual_probe_medium_reasoning` |
| VisualProbe (Hard) | `visual_probe_hard_reasoning` |
| ScreenSpot | `screenspot_point_in_box_reasoning` |
| ScreenSpotPro | `screenspotpro_point_in_box_reasoning` |

### Captioning & Instruction Following (3 benchmarks)
| Benchmark | Task Name |
|-----------|-----------|
| MM-MTBench | `mm_mt_bench_reasoning` |
| MIABench | `mia_bench_reasoning` |
| MMIFEval | `mmifeval_reasoning` |

## Quick Start

```bash
cd vero-eval

# Evaluate a single model on a single benchmark
python -m lmms_eval \
    --model vllm \
    --model_args model=zlab-princeton/Vero-Qwen3I-8B,tensor_parallel_size=1 \
    --tasks chartqa_reasoning \
    --batch_size 2048 \
    --output_path ./eval_results/
```

## Using eval_domain.sh

The domain evaluation script is `vero-eval/examples/eval_domain.sh`. It handles:
- Dynamic port allocation for judge servers
- Model-specific decoding configurations
- Multiple benchmark presets

```bash
# Evaluate on a single domain
bash examples/eval_domain.sh \
    --model-path /path/to/model/checkpoint \
    --domain chart_ocr \
    --variant reasoning

# Evaluate on all domains
bash examples/eval_domain.sh \
    --model-path /path/to/model/checkpoint \
    --domain all \
    --variant reasoning
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

To evaluate multiple checkpoints (e.g., from a training run):

```bash
bash examples/submit_eval_array.sh \
    --ckpt-dir /path/to/checkpoints/ \
    --tasks "chartqa_reasoning,mathvista_testmini_reasoning" \
    --model-family qwen3
```

## Chain-of-Thought Evaluation

Vero models generate reasoning traces in `<think>` tags. The evaluation harness uses an **LLM judge** to extract the final answer from the reasoning trace.

### Judge Setup

The judge runs as a vLLM server (OpenAI-compatible):
- **Default model**: Qwen3-32B
- **Thinking mode**: Disabled (for consistent answer extraction)
- **Temperature**: 0.7

The judge server starts automatically during evaluation. To configure:
```bash
export JUDGE_MODEL_PATH="/path/to/Qwen3-32B"
export JUDGE_BACKEND="engine"  # or "server"
```

## Adding New Benchmarks

Task definitions live in `lmms_eval/tasks/<task_name>/`:

1. Create a directory under `lmms_eval/tasks/`
2. Add a YAML config defining the dataset, prompts, and metrics
3. Add a `utils.py` with task-specific preprocessing and metric functions
4. Register variants (CoT, direct answer, etc.) as separate YAML files

See `lmms_eval/tasks/chartqa/` for a complete example.
