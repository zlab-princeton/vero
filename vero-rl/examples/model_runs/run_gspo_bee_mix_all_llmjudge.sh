#!/bin/bash
# Train Bee-8B-SFT with GSPO + LLM Judge
set -x
set -o pipefail

DATE_TIME="$(date +%Y%m%d_%H%M%S)"
TRAINER_EXPERIMENT_SUFFIX="bee_sft_8b_mix_all"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"
export TRAINER_EXPERIMENT_NAME

# LLM Judge server
NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
VLLM_JUDGE_TENSOR_PARALLEL_SIZE="$NUM_GPUS"
START_VLLM_JUDGE_SERVER="1"
VLLM_JUDGE_PORT="51001"
VLLM_JUDGE_API_KEY="qwen"
VLLM_JUDGE_MODEL_NAME="qwen_vllm"
VLLM_JUDGE_MODEL_PATH="$HF_HOME/Qwen/Qwen3.5-27B"
VLLM_JUDGE_CHAT_TEMPLATE="$REPO_ROOT/examples/prompts/chat_template_no_think.jinja"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/shared/llm_judge_server.sh"
llm_judge_setup_and_start

python3 -m verl.trainer.main_ppo \
  --config-path="$SCRIPT_DIR/config" \
  --config-name='gspo_bee'
