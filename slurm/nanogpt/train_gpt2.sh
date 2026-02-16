#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-train-gpt2
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-20}"
WANDB_LOG="${WANDB_LOG:-False}"
OUT_DIR="${REPO_ROOT}/out/nanogpt-gpt2"

echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "SLURM_NNODES=${SLURM_NNODES:-1}, GPUS_PER_NODE=${GPUS_PER_NODE}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
nvidia-smi

if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
  # Multi-node variant: launch this script with srun on one task per node.
  # Example:
  #   srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash slurm/nanogpt/train_gpt2.sh
  read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_NODELIST}")
  MASTER_PORT="${MASTER_PORT:-29500}"
  NODE_RANK="${SLURM_NODEID:-0}"

  torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train.py config/train_gpt2.py \
    --gradient_accumulation_steps="${GRAD_ACC_STEPS}" \
    --wandb_log="${WANDB_LOG}" \
    --out_dir="${OUT_DIR}"
else
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
    train.py config/train_gpt2.py \
    --gradient_accumulation_steps="${GRAD_ACC_STEPS}" \
    --wandb_log="${WANDB_LOG}" \
    --out_dir="${OUT_DIR}"
fi
