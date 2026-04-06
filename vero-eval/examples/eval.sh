#!/bin/bash
# eval.sh — Run VeroEvalSuite evaluation on a single task or domain
#
# Usage:
#   bash examples/eval.sh --model-path <HF_MODEL_ID_OR_PATH> --tasks <TASK_LIST>
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
#   # Run with LLM judge (for tasks that need CoT answer extraction)
#   bash examples/eval.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --tasks mathvision_test_reasoning \
#     --judge-model Qwen/Qwen3-32B

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────
MODEL_PATH=""
TASKS=""
OUTPUT_PATH="./eval_results"
BATCH_SIZE=1
NUM_GPUS=1
JUDGE_MODEL=""
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
    *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$TASKS" ]]; then
  echo "Usage: bash examples/eval.sh --model-path <MODEL> --tasks <TASKS>"
  echo "  --model-path   HuggingFace model ID or local path"
  echo "  --tasks        Comma-separated task names (e.g. countbenchqa_reasoning)"
  echo "  --output-path  Results directory (default: ./eval_results)"
  echo "  --batch-size   Batch size (default: 1)"
  echo "  --num-gpus     Number of GPUs for tensor parallelism (default: 1)"
  echo "  --judge-model  LLM judge model path (optional, for CoT evaluation)"
  exit 1
fi

# ── build model args ──────────────────────────────────────────────────
MODEL_ARGS="model=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS}"

# ── build judge args ──────────────────────────────────────────────────
JUDGE_ARGS=""
if [[ -n "$JUDGE_MODEL" ]]; then
  JUDGE_ARGS="--judge_model_name vllm --judge_model_args model=${JUDGE_MODEL}"
fi

# ── run evaluation ────────────────────────────────────────────────────
echo "Model:   $MODEL_PATH"
echo "Tasks:   $TASKS"
echo "Output:  $OUTPUT_PATH"
echo ""

python -m lmms_eval \
  --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  $JUDGE_ARGS \
  $EXTRA_ARGS
