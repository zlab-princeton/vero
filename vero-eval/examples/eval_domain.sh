#!/bin/bash
# eval_domain.sh — Run VeroEvalSuite by domain
#
# Usage:
#   bash examples/eval_domain.sh --model-path <MODEL> --domain <DOMAIN> --variant <VARIANT>
#
# Examples:
#   # Evaluate Vero model on all Chart & OCR tasks
#   bash examples/eval_domain.sh \
#     --model-path zlab-princeton/Vero-Qwen3I-8B \
#     --domain chart_ocr \
#     --variant reasoning
#
#   # Evaluate Qwen2.5-VL base model on STEM tasks
#   bash examples/eval_domain.sh \
#     --model-path Qwen/Qwen2.5-VL-7B-Instruct \
#     --domain stem \
#     --variant qwen25_zs
#
#   # Evaluate MiMo model on all domains
#   bash examples/eval_domain.sh \
#     --model-path XiaomiMiMo/MiMo-VL-7B-RL \
#     --domain all \
#     --variant mimo_zs

set -euo pipefail

# ── domain → task mappings ────────────────────────────────────────────
# Each domain maps to a comma-separated list of task prefixes.
# The variant suffix (_reasoning, _qwen25_zs, etc.) is appended automatically.

declare -A DOMAIN_TASKS
DOMAIN_TASKS[chart_ocr]="chartqa,infovqa_val,charxiv,chartmuseum,evochart,chartqa_pro"
DOMAIN_TASKS[stem]="mmmu_pro,mathvision_test,mathvista_testmini"
DOMAIN_TASKS[spatial_action]="cv_bench,embspatial,erqa,game_qa_lite,blink"
DOMAIN_TASKS[knowledge_rec]="realworldqa,simplevqa_en,fvqa,mmvetv2_group_img"
DOMAIN_TASKS[grounding_counting_search]="countbenchqa,countqa,mme_realworld_lite,vstar_bench,aerialvg_bbox,visual_probe_easy,visual_probe_medium,visual_probe_hard,screenspot_point_in_box,screenspotpro_point_in_box"
DOMAIN_TASKS[instruction_following]="mia_bench,mm_mt_bench,mmifeval"

# ── defaults ──────────────────────────────────────────────────────────
MODEL_PATH=""
DOMAIN=""
VARIANT=""
OUTPUT_PATH="./eval_results"
BATCH_SIZE=1
NUM_GPUS=1
JUDGE_MODEL=""

# ── parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2"; shift 2 ;;
    --domain)       DOMAIN="$2"; shift 2 ;;
    --variant)      VARIANT="$2"; shift 2 ;;
    --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --num-gpus)     NUM_GPUS="$2"; shift 2 ;;
    --judge-model)  JUDGE_MODEL="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$DOMAIN" || -z "$VARIANT" ]]; then
  echo "Usage: bash examples/eval_domain.sh --model-path <MODEL> --domain <DOMAIN> --variant <VARIANT>"
  echo ""
  echo "Domains:  chart_ocr, stem, spatial_action, knowledge_rec, grounding_counting_search, instruction_following, all"
  echo "Variants: reasoning, reasoning_samplingq3, qwen25_zs, qwen3_zs, qwen3_thinking_zs, gpt5nano_zs, mimo_zs"
  exit 1
fi

# ── resolve domain list ───────────────────────────────────────────────
if [[ "$DOMAIN" == "all" ]]; then
  DOMAINS=(chart_ocr stem spatial_action knowledge_rec grounding_counting_search instruction_following)
else
  DOMAINS=("$DOMAIN")
fi

# ── build full task list ──────────────────────────────────────────────
ALL_TASKS=""
for d in "${DOMAINS[@]}"; do
  prefixes="${DOMAIN_TASKS[$d]}"
  IFS=',' read -ra PREFIX_ARR <<< "$prefixes"
  for prefix in "${PREFIX_ARR[@]}"; do
    task="${prefix}_${VARIANT}"
    if [[ -n "$ALL_TASKS" ]]; then
      ALL_TASKS="${ALL_TASKS},${task}"
    else
      ALL_TASKS="${task}"
    fi
  done
done

# ── build judge args ──────────────────────────────────────────────────
JUDGE_ARGS=""
if [[ -n "$JUDGE_MODEL" ]]; then
  JUDGE_ARGS="--judge_model_name vllm --judge_model_args model=${JUDGE_MODEL}"
fi

# ── run ───────────────────────────────────────────────────────────────
echo "Model:   $MODEL_PATH"
echo "Domain:  $DOMAIN"
echo "Variant: $VARIANT"
echo "Tasks:   $ALL_TASKS"
echo "Output:  $OUTPUT_PATH"
echo ""

python -m lmms_eval \
  --model vllm \
  --model_args "model=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS}" \
  --tasks "$ALL_TASKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  $JUDGE_ARGS
