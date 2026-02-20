#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"

if [[ -n "${CONDA_SH:-}" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "CONDA_SH is set, but file was not found: ${CONDA_SH}" >&2
    exit 1
  fi
else
  for candidate in \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CONDA_SH:-}" || ! -f "${CONDA_SH}" ]]; then
  MINIFORGE_DIR="${HOME}/miniforge3"
  MINIFORGE_SH="${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  MINIFORGE_INSTALLER="${TMPDIR:-/tmp}/Miniforge3-Linux-x86_64.sh"

  echo "No conda installation detected. Installing Miniforge to ${MINIFORGE_DIR}..."
  if [[ -f "${MINIFORGE_SH}" ]]; then
    echo "Found existing Miniforge at ${MINIFORGE_DIR}; reusing it."
  else
    if command -v wget >/dev/null 2>&1; then
      wget -O "${MINIFORGE_INSTALLER}" "${MINIFORGE_URL}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -o "${MINIFORGE_INSTALLER}" "${MINIFORGE_URL}"
    else
      echo "Could not download Miniforge: neither wget nor curl is available." >&2
      exit 1
    fi
    bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_DIR}"
    rm -f "${MINIFORGE_INSTALLER}"
  fi

  CONDA_SH="${MINIFORGE_SH}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Miniforge installation did not produce conda.sh at ${CONDA_SH}" >&2
    exit 1
  fi
fi

source "${CONDA_SH}"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
fi

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "Creating conda environment at ${ENV_PATH}"
  conda create -y -p "${ENV_PATH}" python=3.12
else
  echo "Conda environment already exists at ${ENV_PATH}"
fi

conda activate "${ENV_PATH}"
python -m pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.4 wheels"
if python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.10" torchvision torchaudio; then
  echo "Installed stable torch>=2.10 build from CUDA 12.4 index."
elif python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 "torch>=2.10" torchvision torchaudio; then
  echo "Installed nightly torch>=2.10 build from CUDA 12.4 nightly index."
else
  echo "Falling back to torch 2.6.0+cu124 (known-good on Engaging)."
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124
fi

echo "Installing NanoGPT + modded-nanogpt dependencies"
python -m pip install \
  numpy \
  transformers \
  datasets \
  tiktoken \
  wandb \
  tqdm \
  huggingface-hub \
  setuptools \
  typing-extensions==4.15.0

echo "Installing optional modded-nanogpt extras"
if ! python -m pip install kernels; then
  echo "Optional package 'kernels' could not be installed; continuing."
fi

cat <<EOF

Environment setup complete.
To activate manually:
  source ${CONDA_SH}
  conda activate ${ENV_PATH}

EOF
