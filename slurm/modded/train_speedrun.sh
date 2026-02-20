#!/usr/bin/env bash
#SBATCH --job-name=modded-speedrun
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Default to 8x H100. Override at submission time, for example:
#   GPU_TYPE=h100 NUM_GPUS=2 sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh
GPU_TYPE="${GPU_TYPE:-h100}"
NUM_GPUS="${NUM_GPUS:-8}"
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"

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

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}/modded_nanogpt"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU_TYPE=${GPU_TYPE} NUM_GPUS=${NUM_GPUS}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
nvidia-smi

if [[ ! -f data/fineweb10B/fineweb_train_000001.bin ]]; then
  echo "FineWeb token cache not found. Downloading first 9 chunks..."
  python data/cached_fineweb10B.py 9
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" train_gpt.py
