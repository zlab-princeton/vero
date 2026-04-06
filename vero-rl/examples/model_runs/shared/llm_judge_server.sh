#!/bin/bash

cleanup_vllm_server() {
  echo "[INFO] Cleaning up resources..."
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] Killing vLLM server process..."
    pkill -f "vllm serve"
  fi
}

handle_interrupt_and_exit() {
  echo "[INFO] Interrupt received, exiting..."
  trap - EXIT INT TERM
  cleanup_vllm_server
  exit 130
}

llm_judge_setup_and_start() {
  local shared_dir repo_root
  shared_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="${REPO_ROOT:-$(cd "$shared_dir/../.." && pwd)}"

  GPU_MEMORY_UTILIZATION_LLM_JUDGE="${GPU_MEMORY_UTILIZATION_LLM_JUDGE:-0.35}"
  VLLM_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-Qwen/Qwen3-32B}"
  VLLM_JUDGE_CHAT_TEMPLATE="${VLLM_JUDGE_CHAT_TEMPLATE:-$repo_root/examples/prompts/chat_template_no_think.jinja}"

  trap cleanup_vllm_server EXIT
  trap handle_interrupt_and_exit INT TERM

  if [ "${START_VLLM_JUDGE_SERVER:-0}" -eq 1 ]; then
    VLLM_JUDGE_LOG_DIR="${VLLM_JUDGE_LOG_DIR:-vllm_logs}"
    VLLM_LOG_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-no_suffix}"
    VLLM_LOG_SUFFIX="${VLLM_LOG_SUFFIX//[^[:alnum:]_.-]/_}"
    VLLM_LOG_RANDOM_ID="$(printf "%04d" "$((RANDOM % 10000))")"
    VLLM_JUDGE_LOG_FILE="${VLLM_JUDGE_LOG_DIR}/vllm_judge_${VLLM_JUDGE_PORT}_${DATE_TIME}_${VLLM_LOG_SUFFIX}_${VLLM_LOG_RANDOM_ID}.txt"
    mkdir -p "$VLLM_JUDGE_LOG_DIR"
    echo "Starting vLLM judge server at port ${VLLM_JUDGE_PORT}"
    echo "vLLM judge log file: ${VLLM_JUDGE_LOG_FILE}"
    echo "Using current conda env for vLLM judge server: ${CONDA_PREFIX:-system}"

    VLLM_JUDGE_TP="${VLLM_JUDGE_TENSOR_PARALLEL_SIZE:-${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}}"
    if ! [[ "$VLLM_JUDGE_TP" =~ ^[0-9]+$ ]] || [ "$VLLM_JUDGE_TP" -lt 1 ]; then
      echo "Warning: invalid vLLM judge tensor parallel size '$VLLM_JUDGE_TP'; defaulting to 1"
      VLLM_JUDGE_TP=1
    fi
    echo "Using vLLM judge tensor parallel size: ${VLLM_JUDGE_TP}"

    # Avoid transformers deprecation warning for TRANSFORMERS_CACHE in the judge subprocess.
    PREV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE-__RVLM_UNSET__}"
    if [ -n "${HF_HOME:-}" ] && [ "${PREV_TRANSFORMERS_CACHE}" != "__RVLM_UNSET__" ]; then
      unset TRANSFORMERS_CACHE
    fi

    VLLM_SERVER_DEV_MODE=1 vllm serve "$VLLM_JUDGE_MODEL_PATH" \
      --port "$VLLM_JUDGE_PORT" \
      --served-model-name "$VLLM_JUDGE_MODEL_NAME" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION_LLM_JUDGE" \
      --tensor-parallel-size "$VLLM_JUDGE_TP" \
      --uvicorn-log-level info \
      --max-model-len 32768 \
      --enable-sleep-mode \
      --enforce-eager \
      --dtype bfloat16 \
      --chat-template "$VLLM_JUDGE_CHAT_TEMPLATE" \
      --api-key "$VLLM_JUDGE_API_KEY" > "$VLLM_JUDGE_LOG_FILE" 2>&1 &

    if [ "${PREV_TRANSFORMERS_CACHE}" != "__RVLM_UNSET__" ]; then
      export TRANSFORMERS_CACHE="$PREV_TRANSFORMERS_CACHE"
    fi
    echo "Waiting for vLLM judge server to be ready..."
    VLLM_READY=0
    for i in $(seq 1 90); do
      if curl --noproxy "*" -fsS \
        -H "Authorization: Bearer ${VLLM_JUDGE_API_KEY}" \
        "http://127.0.0.1:${VLLM_JUDGE_PORT}/v1/models" >/dev/null; then
        VLLM_READY=1
        echo "vLLM judge server is ready."
        break
      else
        echo "Judge server not ready yet... (attempt $i/90)"
      fi
      sleep 10
    done

    if [ "$VLLM_READY" -ne 1 ]; then
      echo "Error: vLLM judge server did not become ready on port ${VLLM_JUDGE_PORT}"
      exit 1
    fi

    curl --noproxy "*" -fsS -X POST \
      -H "Authorization: Bearer ${VLLM_JUDGE_API_KEY}" \
      "http://127.0.0.1:${VLLM_JUDGE_PORT}/reset_prefix_cache" >/dev/null

    curl --noproxy "*" -fsS -X POST \
      -H "Authorization: Bearer ${VLLM_JUDGE_API_KEY}" \
      "http://127.0.0.1:${VLLM_JUDGE_PORT}/reset_mm_cache" >/dev/null

    curl --noproxy "*" -fsS -X POST \
      -H "Authorization: Bearer ${VLLM_JUDGE_API_KEY}" \
      "http://127.0.0.1:${VLLM_JUDGE_PORT}/sleep?level=1" >/dev/null

    echo "Waiting for vLLM judge server to enter sleep..."
    SLEEP_READY=0
    for i in $(seq 1 60); do
      status=$(curl --noproxy "*" -fsS \
        -H "Authorization: Bearer ${VLLM_JUDGE_API_KEY}" \
        "http://127.0.0.1:${VLLM_JUDGE_PORT}/is_sleeping" | tr -d '[:space:]')
      if echo "$status" | grep -Eq '(^true$|"sleeping":true|"is_sleeping":true)'; then
        SLEEP_READY=1
        echo "vLLM judge server is sleeping."
        break
      fi
      sleep 1
    done

    if [ "$SLEEP_READY" -ne 1 ]; then
      echo "Warning: timed out waiting for vLLM judge server sleep state; last status: ${status}"
    fi
  fi
}
