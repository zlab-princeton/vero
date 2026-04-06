#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_DIR="$SCRIPT_DIR/prompts"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/filtering/run_answer_filtering.sh \
    --domain <chart_ocr|stem|spatial_action|knowledge_recognition|counting_grounding_search> \
    --model <model_name_or_path> \
    --data-file <input_jsonl> \
    --output-dir <output_dir> \
    [options]

Common options:
  --save-tag <tag>
  --load-folder <dir>
  --batch-size <int>
  --save-batch-size <int>
  --max-samples <int>
  --start-index <int>
  --end-index <int>
  --max-model-len <int>
  --tensor-parallel-size <int>
  --pipeline-parallel-size <int>
  --gpu-memory-utilization <float>
  --dtype <dtype>
  --trust-remote-code
  --dry-run
  --help
EOF
}

resolve_prompt() {
  case "$1" in
    knowledge_recognition) echo "$PROMPT_DIR/answer_filter_knowledge_recognition.txt" ;;
    chart_ocr|stem|spatial_action|counting_grounding_search) echo "$PROMPT_DIR/answer_filter_default.txt" ;;
    *)
      echo "Unknown domain: $1" >&2
      exit 1
      ;;
  esac
}

DOMAIN=""
MODEL=""
DATA_FILE=""
OUTPUT_DIR=""
SAVE_TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --data-file)
      DATA_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save-tag)
      SAVE_TAG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DOMAIN" || -z "$MODEL" || -z "$DATA_FILE" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 1
fi

PROMPT_FILE="$(resolve_prompt "$DOMAIN")"
if [[ -z "$SAVE_TAG" ]]; then
  SAVE_TAG="answer_filter_${DOMAIN}_$(date +%Y%m%d_%H%M%S)"
fi

python "$SCRIPT_DIR/generate_answer_filter_rollouts.py" \
  --prompt-file "$PROMPT_FILE" \
  --pretrained "$MODEL" \
  --data-file "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --save-tag "$SAVE_TAG" \
  "${EXTRA_ARGS[@]}"
