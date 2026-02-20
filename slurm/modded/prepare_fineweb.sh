#!/usr/bin/env bash
#SBATCH --job-name=modded-prepare-fineweb
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

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

FINEWEB_CHUNKS="${FINEWEB_CHUNKS:-9}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
echo "Downloading FineWeb GPT-2 token cache with ${FINEWEB_CHUNKS} train chunks"
python data/cached_fineweb10B.py "${FINEWEB_CHUNKS}"

echo "Done. Files are in modded_nanogpt/data/fineweb10B/"
