#!/usr/bin/env bash
# setup_env.sh — Create the verovlm conda environment for Vero
#
# Installs PyTorch, vLLM, transformers, flash-attn, and both
# vero-eval and vero-rl packages in editable mode.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Prerequisites:
#   - conda/miniconda installed
#   - CUDA toolkit available (nvcc accessible via PATH or CUDA_HOME)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verovlm}"

# Optional: set a custom conda env path (useful on shared clusters)
# export CONDA_ENV_PATH="/path/to/conda/envs/${CONDA_ENV_NAME}"

echo "=== Vero Environment Setup ==="
echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"

# ---- Create conda env if needed ----
if [ -n "${CONDA_ENV_PATH:-}" ]; then
    if ! conda env list | grep -q "${CONDA_ENV_PATH}"; then
        echo "==> Creating conda env '${CONDA_ENV_NAME}' at ${CONDA_ENV_PATH}"
        conda create -y -p "${CONDA_ENV_PATH}" python=3.10
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_PATH}"
else
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "==> Creating conda env '${CONDA_ENV_NAME}'"
        conda create -y -n "${CONDA_ENV_NAME}" python=3.10
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
fi

PY="$(which python)"
PIP="$(which pip)"
echo "==> Python: ${PY} ($(python --version))"

# Helper: uv pip install into the conda env
uvpip() {
    if command -v uv &> /dev/null; then
        uv pip install --python "${PY}" "$@"
    else
        "${PIP}" install "$@"
    fi
}

# ---- Install uv (fast pip alternative) ----
echo "==> Installing uv"
"${PIP}" install uv

# ---- PyTorch (cu130) ----
echo "==> Installing PyTorch 2.10.0+cu130"
uvpip torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- vLLM ----
echo "==> Installing vLLM 0.17.0"
uvpip vllm==0.17.0

# ---- Core packages ----
# Install transformers>=5.0.0 AFTER vllm (vllm pins transformers<5, but 5.x works at runtime).
echo "==> Installing core packages (transformers 5.x, etc.)"
uvpip \
    "transformers>=5.0.0" \
    accelerate datasets peft safetensors \
    "flash-linear-attention>=0.4.0" \
    "qwen-vl-utils==0.0.14"

# ---- verl dependencies ----
echo "==> Installing verl dependencies"
uvpip \
    codetiming dill hydra-core "numpy>=2.0.0" pandas \
    "pyarrow>=19.0.0" pybind11 pylatexenc \
    "ray[default]>=2.41.0" torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    wandb "packaging>=20.0" tensorboard \
    deepspeed liger-kernel \
    math-verify mathruler nltk langdetect \
    "latex2sympy2-extended>=1.11.0" \
    "python-levenshtein>=0.27.3" \
    "uvloop==0.21.0" "setuptools>=82.0.0"

# ---- GPU-compiled deps (flash-attn, causal-conv1d) ----
echo "==> Compiling causal-conv1d (requires nvcc)"
MAX_JOBS="${MAX_JOBS:-4}" "${PIP}" install causal-conv1d --no-build-isolation --no-cache-dir

echo "==> Compiling flash-attn (this may take 10-20 minutes)"
MAX_JOBS="${MAX_JOBS:-4}" "${PIP}" install flash-attn --no-build-isolation --no-cache-dir

# ---- Install vero-eval (lmms-eval fork) ----
echo "==> Installing vero-eval (editable)"
cd "${REPO_ROOT}/vero-eval"
"${PIP}" install -e . --no-deps

# ---- Install vero-rl (veRL fork) ----
echo "==> Installing vero-rl (editable)"
cd "${REPO_ROOT}/vero-rl"
"${PIP}" install -e . --no-deps

# ---- Fix vLLM Qwen3.5 config bug if present ----
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
VLLM_CFG_DIR="${SITE_PKGS}/vllm/transformers_utils/configs"
for cfg_file in "${VLLM_CFG_DIR}/qwen3_5.py" "${VLLM_CFG_DIR}/qwen3_5_moe.py"; do
    cfg_name="$(basename "${cfg_file}")"
    if [[ ! -f "${cfg_file}" ]]; then
        continue
    fi
    if ! grep -q 'ignore_keys_at_rope_validation.*\[' "${cfg_file}" 2>/dev/null; then
        continue
    fi
    echo "==> Fixing ${cfg_name} (list->set for ignore_keys_at_rope_validation)"
    python -c "
import pathlib, sys
p = pathlib.Path('${cfg_file}')
t = p.read_text()
old = 'kwargs[\"ignore_keys_at_rope_validation\"] = [\n            \"mrope_section\",\n            \"mrope_interleaved\",\n        ]'
new = 'kwargs[\"ignore_keys_at_rope_validation\"] = {\n            \"mrope_section\",\n            \"mrope_interleaved\",\n        }'
if old in t:
    t = t.replace(old, new)
    p.write_text(t)
    print('  Fixed!')
else:
    print('  Pattern not found -- may already be fixed', file=sys.stderr)
"
done

# ---- Freeze ----
echo "==> Freezing requirements"
"${PIP}" freeze > "${REPO_ROOT}/requirements-frozen.txt"

echo ""
echo "=== Vero environment ready ==="
echo "Activate with:  conda activate ${CONDA_ENV_NAME}"
