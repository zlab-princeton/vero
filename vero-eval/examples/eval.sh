#!/bin/bash
# eval.sh — Run VeroEvalSuite evaluation on a single task or domain
#
# Usage:
#   bash examples/eval.sh --tasks <TASK_LIST> [--model-path <HF_MODEL_ID_OR_PATH>]
#
# The model defaults to zlab-princeton/Vero-Qwen3I-8B (Vero's Qwen3-VL-8B-Instruct
# checkpoint). Override it with --model-path.
#
# Examples:
#   # Single task
#   bash examples/eval.sh \
#     --model-path Qwen/Qwen2.5-VL-7B-Instruct \
#     --tasks countbenchqa_qwen25_zs
#
#   # Multiple tasks (comma-separated)
#   bash examples/eval.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --tasks countbenchqa_reasoning,countqa_reasoning,vstar_bench_reasoning
#
#   # With custom output directory and batch size
#   bash examples/eval.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --tasks chartqa_reasoning \
#     --output-path ./results \
#     --batch-size 4
#
#   # Run with a local LLM judge (for tasks that need CoT answer extraction)
#   # --judge-model exports JUDGE_MODEL_PATH, which every judge task reads.
#   bash examples/eval.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --tasks mathvision_test_reasoning \
#     --judge-model Qwen/Qwen3-32B
#
#   # Quick smoke test (5 samples) — --limit is passed straight to lmms_eval
#   bash examples/eval.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --tasks chartqa_reasoning \
#     --limit 5
#
# Judge note: tasks like mathvista/mathvision/mmvet/mm_mt_bench/mia_bench/
# mmifeval/fvqa/simplevqa/charxiv/chartmuseum/visual_probe score answers with an
# LLM judge. Set the judge via --judge-model (or `export JUDGE_MODEL_PATH=...`,
# e.g. in set_paths.sh). If unset, those tasks fall back to OpenAI gpt-4o and
# need GPT_API_KEY. Judge-based tasks need 2 GPUs (one for the model under test,
# one for the judge) — run them with `--num-gpus 2`. Rule-based tasks (e.g.
# chartqa_reasoning, the ScreenSpot tasks) need no judge and run on 1 GPU.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────
MODEL_PATH="zlab-princeton/Vero-Qwen3I-8B"   # default: Vero Qwen3-VL-8B-Instruct
TASKS=""
OUTPUT_PATH="./eval_results"
BATCH_SIZE=1
NUM_GPUS=1
JUDGE_MODEL=""
GPU_MEM_UTIL=""
EXTRA_ARGS=""

# ── parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2"; shift 2 ;;
    --tasks)        TASKS="$2"; shift 2 ;;
    --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --num-gpus)     NUM_GPUS="$2"; shift 2 ;;
    --judge-model)  JUDGE_MODEL="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
  esac
done

if [[ -z "$TASKS" ]]; then
  echo "Usage: bash examples/eval.sh --tasks <TASKS> [--model-path <MODEL>]"
  echo "  --model-path   HuggingFace model ID or local path (default: zlab-princeton/Vero-Qwen3I-8B)"
  echo "  --tasks        Comma-separated task names (e.g. countbenchqa_reasoning)"
  echo "  --output-path  Results directory (default: ./eval_results)"
  echo "  --batch-size   Batch size (default: 1)"
  echo "  --num-gpus     Number of GPUs for tensor parallelism (default: 1)"
  echo "  --judge-model  Local LLM judge for CoT answer extraction (sets JUDGE_MODEL_PATH)"
  echo "  --gpu-mem-util vLLM gpu_memory_utilization for the eval model (e.g. 0.8)"
  exit 1
fi

# ── build model args ──────────────────────────────────────────────────
MODEL_ARGS="model=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS}"
if [[ -n "$GPU_MEM_UTIL" ]]; then
  MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEM_UTIL}"
fi

# ── configure judge ───────────────────────────────────────────────────
# Judge tasks read the JUDGE_MODEL_PATH env var (NOT a CLI flag). --judge-model
# simply exports it for this run; you can also set it in set_paths.sh.
if [[ -n "$JUDGE_MODEL" ]]; then
  export JUDGE_MODEL_PATH="$JUDGE_MODEL"
fi
# Route judge-based tasks to the local vLLM judge. A few tasks (mmvetv2/
# mm_mt_bench/mia_bench/mmifeval) default to the OpenAI backend, so set API_TYPE
# whenever a local judge is configured. (For an OpenAI gpt-4o judge instead, set
# API_TYPE=openai and GPT_API_KEY yourself.) The judge loads on its own GPU(s),
# so judge-based tasks need 2 GPUs — give it the same count as the model.
if [[ -n "${JUDGE_MODEL_PATH:-}" ]]; then
  export API_TYPE="${API_TYPE:-vllm}"
  export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE="${CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
fi

# ── run evaluation ────────────────────────────────────────────────────
echo "Model:   $MODEL_PATH"
echo "Tasks:   $TASKS"
echo "Output:  $OUTPUT_PATH"
if [[ -n "${JUDGE_MODEL_PATH:-}" ]]; then
  echo "Judge:   ${JUDGE_MODEL_PATH}"
else
  echo "Judge:   (none — judge tasks fall back to OpenAI gpt-4o; set --judge-model or JUDGE_MODEL_PATH)"
fi
echo ""

python -m lmms_eval \
  --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  $EXTRA_ARGS
