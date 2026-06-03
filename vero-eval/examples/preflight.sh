#!/usr/bin/env bash
# preflight.sh — verify a machine is ready to run VeroEvalSuite, and optionally
# pre-download the LLM judge. This DOES NOT launch a GPU eval; it only checks
# the environment and prints the exact command to run next.
#
# Scope: EVALUATION only. If you just want to run the benchmarks you never need
# the training stack. RL training is independent and has its own setup (see
# docs/TRAINING.md); its run scripts start/verify the reward judge themselves, so
# there is no separate training preflight.
#
# Usage:
#   bash examples/preflight.sh                 # run all checks
#   bash examples/preflight.sh --download-judge   # also fetch the judge model
#   bash examples/preflight.sh --judge-model Qwen/Qwen3-32B --download-judge
#
# Tip: `source set_paths.sh` first (or let this script source it) so HF_HOME and
# JUDGE_MODEL_PATH are set the same way your eval run will see them.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOWNLOAD_JUDGE=0
JUDGE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-judge) DOWNLOAD_JUDGE=1; shift ;;
    --judge-model)    JUDGE_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Source per-user paths if present (guarded: conda hooks can reference unset vars).
if [[ -f "${REPO_ROOT}/set_paths.sh" ]]; then
  echo "==> Sourcing ${REPO_ROOT}/set_paths.sh"
  set +u
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/set_paths.sh" || echo "   (set_paths.sh returned non-zero; continuing)"
  set -u
else
  echo "==> No set_paths.sh found at repo root."
  echo "    Create one:  cp scripts/set_paths.sh.example set_paths.sh  (then edit ROOT_PATH)"
fi

[[ -n "$JUDGE_OVERRIDE" ]] && export JUDGE_MODEL_PATH="$JUDGE_OVERRIDE"

PASS=0; WARN=0; FAIL=0
ok()   { echo "  [ OK ] $*"; PASS=$((PASS+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

PY="$(command -v python || true)"

echo ""
echo "=== 1. Python environment ==="
if [[ -z "$PY" ]]; then
  fail "no 'python' on PATH — activate the verovlm env (bash scripts/setup_env.sh)."
else
  ok "python: $PY ($("$PY" --version 2>&1))"
  "$PY" - <<'PYEOF'
import importlib, sys
def check(mod, attr="__version__"):
    try:
        m = importlib.import_module(mod)
        print(f"  [ OK ] {mod} {getattr(m, attr, '?')}")
        return True
    except Exception as e:
        print(f"  [FAIL] import {mod}: {e}")
        return False
check("torch")
check("vllm")
check("lmms_eval", attr="__file__")
check("datasets")
try:
    import torch
    print(f"  [{'OK' if torch.cuda.is_available() else 'WARN'}] torch.cuda.is_available()={torch.cuda.is_available()}, "
          f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as e:
    print(f"  [WARN] could not query CUDA: {e}")
PYEOF
fi

echo ""
echo "=== 2. GPUs ==="
NGPU=0
if command -v nvidia-smi &>/dev/null; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)"
  GPU_RC=$?
  if [[ $GPU_RC -eq 0 && -n "$GPU_LIST" ]]; then
    NGPU="$(printf '%s\n' "$GPU_LIST" | grep -c .)"
    ok "${NGPU} GPU(s) visible:"
    printf '%s\n' "$GPU_LIST" | sed 's/^/         /'
  else
    warn "nvidia-smi present but no GPUs visible / driver not loaded (login node? request a GPU before evaluating)."
  fi
else
  warn "nvidia-smi not found — evaluation needs a CUDA GPU."
fi

echo ""
echo "=== 3. Hugging Face cache + auth ==="
echo "  HF_HOME=${HF_HOME:-<unset, defaults to ~/.cache/huggingface>}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-<unset>}"
[[ -n "${HF_HOME:-}" ]] && ok "HF_HOME set (datasets/models download here)." \
                         || warn "HF_HOME unset — downloads go to ~/.cache/huggingface (often a small home quota)."
if [[ -n "$PY" ]]; then
  WHO="$("$PY" -c 'from huggingface_hub import whoami; print(whoami().get("name",""))' 2>/dev/null)"
  if [[ -n "$WHO" ]]; then
    ok "logged in to Hugging Face as '${WHO}'."
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is set."
  else
    warn "not logged in to Hugging Face — gated datasets (e.g. MMMU_Pro) will fail. Run: huggingface-cli login"
  fi
fi

echo ""
echo "=== 4. LLM judge ==="
JUDGE="${JUDGE_MODEL_PATH:-}"
if [[ -z "$JUDGE" ]]; then
  warn "JUDGE_MODEL_PATH unset — judge tasks fall back to OpenAI gpt-4o (needs GPT_API_KEY)."
  warn "Set a local judge:  export JUDGE_MODEL_PATH=Qwen/Qwen3-32B"
else
  echo "  JUDGE_MODEL_PATH=${JUDGE}"
  if [[ -d "$JUDGE" ]]; then
    ok "judge is a local directory."
  else
    ok "judge is an HF id — downloads on first use (under HF_HOME). Use --download-judge to pre-fetch."
  fi
  if [[ "${NGPU:-0}" -lt 2 ]]; then
    warn "judge-based tasks need 2 GPUs (model + judge); only ${NGPU} visible. Run them with --num-gpus 2."
  fi
  if [[ "$DOWNLOAD_JUDGE" -eq 1 && -n "$PY" && ! -d "$JUDGE" ]]; then
    echo "  ==> Pre-downloading judge '${JUDGE}' (this is a large download)..."
    "$PY" - "$JUDGE" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo_id=repo)
print(f"  [ OK ] judge cached at: {path}")
PYEOF
  fi
fi

echo ""
echo "=== Summary: ${PASS} ok, ${WARN} warn, ${FAIL} fail ==="
echo ""
echo "Next — smoke test one task (tiny, validates the full path incl. judge; needs 2 GPUs):"
echo "  bash examples/eval.sh \\"
echo "    --model-path zlab-princeton/Vero-Qwen3I-8B \\"
echo "    --tasks mathvista_testmini_reasoning \\"
echo "    --judge-model \"${JUDGE:-Qwen/Qwen3-32B}\" \\"
echo "    --num-gpus 2 \\"
echo "    --limit 5"
echo ""
echo "Then a full domain (or --domain all):"
echo "  bash examples/eval_domain.sh \\"
echo "    --model-path zlab-princeton/Vero-Qwen3I-8B \\"
echo "    --domain chart_ocr --variant reasoning \\"
echo "    --judge-model \"${JUDGE:-Qwen/Qwen3-32B}\" \\"
echo "    --num-gpus 2"

[[ "$FAIL" -gt 0 ]] && exit 1 || exit 0
