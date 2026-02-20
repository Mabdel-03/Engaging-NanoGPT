#!/usr/bin/env bash
#SBATCH --job-name=build-flash-attn
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Default request is 1x H100. Override at submission time, for example:
#   GPU_TYPE=h100 sbatch --gres=gpu:${GPU_TYPE}:1 slurm/modded/build_flash_attn.sh
GPU_TYPE="${GPU_TYPE:-h100}"
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"
MAX_JOBS="${MAX_JOBS:-32}"

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Could not find conda initialization script at: ${CONDA_SH}" >&2
  echo "Set CONDA_SH to your conda.sh path before submitting this job." >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_PATH}"

echo "Building flash-attn from source on $(hostname)"
echo "Target GPU type: ${GPU_TYPE}"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(nvcc --version | tail -1)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
MAX_JOBS="${MAX_JOBS}" pip install flash-attn --no-build-isolation

echo "flash-attn build complete"
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
