#!/usr/bin/env bash
#SBATCH --job-name=modded-speedrun-2gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
fi

source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda activate /home/mabdel03/conda_envs/nanogpt_env

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
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
nvidia-smi

if [[ ! -f data/fineweb10B/fineweb_train_000001.bin ]]; then
  echo "FineWeb token cache not found. Downloading first 9 chunks..."
  python data/cached_fineweb10B.py 9
fi

torchrun --standalone --nproc_per_node=2 train_gpt.py
