#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-prepare-openwebtext
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
cd "${REPO_ROOT}/nanogpt"

echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
echo "Preparing OpenWebText dataset in ${PWD}"
python data/openwebtext/prepare.py

echo "Done. Files are in nanogpt/data/openwebtext/"
