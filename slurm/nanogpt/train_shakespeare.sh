#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-train-shakespeare
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-h100}"
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
cd "${REPO_ROOT}/nanogpt"

echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU_TYPE=${GPU_TYPE}"
nvidia-smi

if [[ ! -f data/shakespeare_char/train.bin ]]; then
  echo "Shakespeare data not found. Preparing now..."
  python data/shakespeare_char/prepare.py
fi

python train.py config/train_shakespeare_char.py --out_dir="${REPO_ROOT}/out/nanogpt-shakespeare"
