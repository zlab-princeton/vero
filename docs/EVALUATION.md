# Vero Evaluation Guide

## Overview

Vero is evaluated on **VeroEvalSuite**, a comprehensive benchmark of **32 diverse benchmarks** across all 6 task categories. The evaluation harness is in `vero-eval/`, a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## VeroEvalSuite Benchmarks

### Chart & OCR (6 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| ChartQA | `chartqa_reasoning` | [`lmms-lab/ChartQA`](https://huggingface.co/datasets/lmms-lab/ChartQA) |
| ChartQA-Pro | `chartqa_pro_reasoning` | [`ahmed-masry/ChartQAPro`](https://huggingface.co/datasets/ahmed-masry/ChartQAPro) |
| InfoVQA | `infovqa_val_reasoning` | [`lmms-lab/DocVQA`](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| CharXiv | `charxiv_reasoning` | [`princeton-nlp/CharXiv`](https://huggingface.co/datasets/princeton-nlp/CharXiv) |
| ChartMuseum | `chartmuseum_reasoning` | [`gsarch/ChartMuseum`](https://huggingface.co/datasets/gsarch/ChartMuseum) |
| EvoChart | `evochart_reasoning` | [`gsarch/EvoChart-QA`](https://huggingface.co/datasets/gsarch/EvoChart-QA) |

### STEM (4 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| MMMU-PRO Standard | `mmmu_pro_standard_reasoning` | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| MMMU-PRO Vision | `mmmu_pro_vision_reasoning` | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| MathVision | `mathvision_test_reasoning` | [`MathLLMs/MathVision`](https://huggingface.co/datasets/MathLLMs/MathVision) |
| MathVista | `mathvista_testmini_reasoning` | [`AI4Math/MathVista`](https://huggingface.co/datasets/AI4Math/MathVista) |

### Spatial & Action (5 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| CVBench | `cv_bench_reasoning` | [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| EmbSpatial | `embspatial_reasoning` | [`FlagEval/EmbSpatial-Bench`](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) |
| ERQA | `erqa_reasoning` | [`FlagEval/ERQA`](https://huggingface.co/datasets/FlagEval/ERQA) |
| GameQA-Lite | `game_qa_lite_reasoning` | [`gsarch/Game-QA-Lite`](https://huggingface.co/datasets/gsarch/Game-QA-Lite) |
| Blink | `blink_reasoning` | [`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) |

### Knowledge & Recognition (4 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| RealWorldQA | `realworldqa_reasoning` | [`lmms-lab/RealWorldQA`](https://huggingface.co/datasets/lmms-lab/RealWorldQA) |
| SimpleVQA (English) | `simplevqa_en_reasoning` | [`gsarch/SimpleVQA-EN`](https://huggingface.co/datasets/gsarch/SimpleVQA-EN) |
| FVQA | `fvqa_reasoning` | [`lmms-lab/FVQA`](https://huggingface.co/datasets/lmms-lab/FVQA) |
| MM-Vet V2 | `mmvetv2_group_img_reasoning` | [`whyu/mm-vet-v2`](https://huggingface.co/datasets/whyu/mm-vet-v2) |

### Grounding, Counting & Visual Search (10 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| CountBenchQA | `countbenchqa_reasoning` | [`vikhyatk/CountBenchQA`](https://huggingface.co/datasets/vikhyatk/CountBenchQA) |
| CountQA | `countqa_reasoning` | [`Jayant-Sravan/CountQA`](https://huggingface.co/datasets/Jayant-Sravan/CountQA) |
| MME-RealWorld-Lite | `mme_realworld_lite_reasoning` | [`yifanzhang114/MME-RealWorld-lite-lmms-eval`](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval) |
| V*Bench | `vstar_bench_reasoning` | [`lmms-lab/vstar-bench`](https://huggingface.co/datasets/lmms-lab/vstar-bench) |
| AerialVG | `aerialvg_bbox_reasoning` | [`IPEC-COMMUNITY/AerialVG`](https://huggingface.co/datasets/IPEC-COMMUNITY/AerialVG) |
| VisualProbe (Easy) | `visual_probe_easy_reasoning` | [`Mini-o3/VisualProbe_Easy`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Easy) |
| VisualProbe (Medium) | `visual_probe_medium_reasoning` | [`Mini-o3/VisualProbe_Medium`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Medium) |
| VisualProbe (Hard) | `visual_probe_hard_reasoning` | [`Mini-o3/VisualProbe_Hard`](https://huggingface.co/datasets/Mini-o3/VisualProbe_Hard) |
| ScreenSpot | `screenspot_point_in_box_reasoning` | [`rootsautomation/ScreenSpot`](https://huggingface.co/datasets/rootsautomation/ScreenSpot) |
| ScreenSpotPro | `screenspotpro_point_in_box_reasoning` | [`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) |

### Captioning & Instruction Following (3 benchmarks)
| Benchmark | Task Name | Dataset |
|-----------|-----------|---------|
| MM-MTBench | `mm_mt_bench_reasoning` | [`mistralai/MM-MT-Bench`](https://huggingface.co/datasets/mistralai/MM-MT-Bench) |
| MIABench | `mia_bench_reasoning` | [`lmms-lab/MIA-Bench`](https://huggingface.co/datasets/lmms-lab/MIA-Bench) |
| MMIFEval | `mmifeval_reasoning` | [`lscpku/MMIFEval`](https://huggingface.co/datasets/lscpku/MMIFEval) |

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

## System Prompt

The Vero system prompt (which defines the `<think>` / `<answer>` output format) is **already baked into the chat template**. If you do not pass a system message, the chat template uses it by default — no extra setup is needed. The chat template also appends a `<think>` token at the start of the assistant turn to kick off reasoning, so **do not add `<think>` manually** to the prompt.

**If you want to use a custom system prompt**, you have two options:

1. **Recommended:** keep the chat template's default system prompt and place your custom instructions in the user message.
2. Override the system role entirely — in this case, make sure your custom system prompt still instructs the model to produce the `<think>` / `<answer>` format, or append the Vero prompt from [`vero-eval/examples/prompts/vero_system_prompt.txt`](../vero-eval/examples/prompts/vero_system_prompt.txt) to yours.

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
