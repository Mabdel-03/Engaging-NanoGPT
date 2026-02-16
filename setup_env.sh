#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh"
ENV_PATH="/home/mabdel03/conda_envs/nanogpt_env"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Could not find conda initialization script at: ${CONDA_SH}" >&2
  exit 1
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

cat <<'EOF'

Environment setup complete.
To activate manually:
  source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
  conda activate /home/mabdel03/conda_envs/nanogpt_env

EOF
