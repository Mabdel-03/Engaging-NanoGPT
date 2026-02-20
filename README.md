# Engaging-NanoGPT

This repository is a self-contained Engaging playbook for researchers who want to:

1. Run Andrej Karpathy's baseline NanoGPT training flow end-to-end.
2. Run Keller Jordan's modded-nanogpt speedrun flow end-to-end.
3. Use both as a launchpad for new modded NanoGPT research.

The scripts in this repo are now modularized for GPU configuration:

- Default GPU type is `h100` (configurable).
- Baseline NanoGPT can run with flexible GPU counts through DDP.
- Modded run is supported for `NUM_GPUS` in `{1, 2, 4, 8}` in this vendored snapshot.

---

## 1) Overview and Motivation

### What is NanoGPT?

NanoGPT is Andrej Karpathy's minimal, readable GPT training codebase for reproducing GPT-style language model training with modern PyTorch and DDP.

- Upstream: https://github.com/karpathy/nanoGPT
- It is intentionally simple and hackable.
- It is a strong baseline for controlled experiments.

### Why Modded NanoGPT?

The modded-nanogpt community focuses on one concrete optimization target:

- Train a GPT-2-scale model to `<= 3.28` validation loss on FineWeb.
- Do it as fast as possible (official speedrun framing uses 8 Hopper GPUs, commonly H100).

Compared with baseline NanoGPT, modded runs combine:

- Architecture changes (attention/head/layout choices, gating, schedule design).
- Optimizer innovations (Muon/NorMuon style updates).
- Systems and kernel work (FlashAttention, Triton kernels, FP8 paths, communication scheduling).

### Why this matters

This challenge is useful in two ways:

- **Engineering**: push wall-clock training speed down dramatically.
- **Science**: understand which algorithmic/system choices matter most, and why.

### World record leaderboard

The canonical leaderboard for this project is maintained in the modded-nanogpt README under **World record history**:

- https://github.com/KellerJordan/modded-nanogpt#world-record-history

---

## 2) Repository Layout

- `nanogpt/`: vendored baseline NanoGPT training/data scripts.
- `modded_nanogpt/`: vendored modded-nanogpt training code.
- `slurm/`: job scripts for baseline and modded workflows.
- `setup_env.sh`: one-time environment bootstrap.
- `activate_env.sh`: fast env activation helper.

---

## 3) Prerequisites on Engaging

You need:

- Access to MIT Engaging and an allocation with GPU partitions.
- Scratch/project space for datasets, logs, and checkpoints.

Before setup, check whether you already have conda:

```bash
conda --version
```

- If that prints a version, you can use your existing conda installation.
- If it does not, use the quick Miniforge install path below.

For users with existing conda, identify:

- `CONDA_SH`: your conda init script (example: `$HOME/miniforge3/etc/profile.d/conda.sh`).
- `ENV_PATH`: your env location (default in scripts is `$HOME/conda_envs/nanogpt_env`).

Quick Miniforge install (if `conda` is not available):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
```

---

## 4) One-Time Environment Setup

### 4.1 Clone the repo

```bash
cd /path/to/your/scratch
git clone https://github.com/Mabdel-03/Engaging-NanoGPT.git
cd Engaging-NanoGPT
```

### 4.2 Create the environment

If you already have conda, use your installation:

```bash
conda --version
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export ENV_PATH="$HOME/conda_envs/nanogpt_env"
bash setup_env.sh
```

If you do not have conda yet, install Miniforge quickly, then run setup:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
export CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
export ENV_PATH="$HOME/conda_envs/nanogpt_env"
bash setup_env.sh
```

Note: `setup_env.sh` can also auto-install Miniforge to `$HOME/miniforge3` if `CONDA_SH` is not set and no conda installation is detected.

### 4.3 Activate for interactive commands

```bash
export CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
export ENV_PATH="$HOME/conda_envs/nanogpt_env"
source activate_env.sh
```

---

## 5) Baseline NanoGPT on Engaging (Karpathy Flow)

This section gets you from zero to a working baseline run.

### 5.1 What this baseline is

The baseline uses the original NanoGPT training stack:

- Standard GPT architecture (`nanogpt/model.py`).
- DDP training (`nanogpt/train.py`).
- OpenWebText / Shakespeare data prep scripts.

Use this to establish a reference before advanced speedrun modifications.

### 5.2 Prepare data

Quick sanity dataset:

```bash
sbatch slurm/nanogpt/prepare_shakespeare.sh
```

Larger GPT-2-style dataset:

```bash
sbatch slurm/nanogpt/prepare_openwebtext.sh
```

### 5.3 Train a fast sanity baseline (1 GPU)

```bash
GPU_TYPE=h100 sbatch --gres=gpu:${GPU_TYPE}:1 slurm/nanogpt/train_shakespeare.sh
```

Outputs land in:

- `out/nanogpt-shakespeare/`

### 5.4 Train baseline GPT-2 with configurable GPU count

Default script launch (2x H100 by default in script headers):

```bash
sbatch slurm/nanogpt/train_gpt2.sh
```

Custom GPU count/type (single-node examples):

```bash
# 1x H100
GPU_TYPE=h100 GPUS_PER_NODE=1 GRAD_ACC_STEPS=40 \
sbatch --gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE} slurm/nanogpt/train_gpt2.sh

# 2x H100
GPU_TYPE=h100 GPUS_PER_NODE=2 GRAD_ACC_STEPS=20 \
sbatch --gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE} slurm/nanogpt/train_gpt2.sh

# 4x H100
GPU_TYPE=h100 GPUS_PER_NODE=4 GRAD_ACC_STEPS=10 \
sbatch --gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE} slurm/nanogpt/train_gpt2.sh

# 8x H100
GPU_TYPE=h100 GPUS_PER_NODE=8 GRAD_ACC_STEPS=5 \
sbatch --gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE} slurm/nanogpt/train_gpt2.sh
```

Notes:

- In baseline NanoGPT, `gradient_accumulation_steps` must be divisible by `WORLD_SIZE`.
- The examples above keep total effective accumulation aligned with the default baseline behavior.

### 5.5 Sample from a trained checkpoint

```bash
cd nanogpt
python sample.py --out_dir=../out/nanogpt-shakespeare
```

### 5.6 What you can do with baseline NanoGPT (modularity)

Baseline NanoGPT is ideal for controlled experiments:

- Edit architecture in `nanogpt/model.py`.
- Change optimization schedule and training knobs in `nanogpt/train.py` and `nanogpt/config/*.py`.
- Swap datasets or tokenization workflows in `nanogpt/data/*`.
- Benchmark changes with `nanogpt/bench.py`.

Useful references:

- NanoGPT upstream: https://github.com/karpathy/nanoGPT
- Karpathy llm.c baseline context: https://github.com/karpathy/llm.c

---

## 6) Modded NanoGPT on Engaging (Keller Jordan Flow)

This section gets you through the speedrun-style path.

### 6.1 What this modded run is

The vendored modded pipeline in `modded_nanogpt/train_gpt.py` includes advanced design choices such as:

- Parameter-bank model layout and communication scheduling.
- NorMuon + Adam hybrid optimizer logic.
- Triton fused kernels (`modded_nanogpt/triton_kernels.py`).
- FP8/flash-attention-oriented execution paths.
- Dynamic training schedule and windowing strategy.

Upstream reference:

- https://github.com/KellerJordan/modded-nanogpt

### 6.2 Prepare FineWeb token cache

Default (first 9 train chunks + val chunk):

```bash
sbatch slurm/modded/prepare_fineweb.sh
```

Custom number of train chunks:

```bash
FINEWEB_CHUNKS=3 sbatch slurm/modded/prepare_fineweb.sh
```

### 6.3 Optional: build flash-attn from source

```bash
GPU_TYPE=h100 sbatch --gres=gpu:${GPU_TYPE}:1 slurm/modded/build_flash_attn.sh
```

### 6.4 Train modded NanoGPT with configurable GPU count

Default script launch (8x H100 by default in script headers):

```bash
sbatch slurm/modded/train_speedrun.sh
```

Custom GPU count/type:

```bash
# Supported GPU counts in this snapshot: 1, 2, 4, 8

# 1x H100
GPU_TYPE=h100 NUM_GPUS=1 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh

# 2x H100
GPU_TYPE=h100 NUM_GPUS=2 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh

# 4x H100
GPU_TYPE=h100 NUM_GPUS=4 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh

# 8x H100 (recommended speedrun target setup)
GPU_TYPE=h100 NUM_GPUS=8 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh
```

If your allocation has H200 instead of H100, you can switch only `GPU_TYPE`:

```bash
GPU_TYPE=h200 NUM_GPUS=8 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} slurm/modded/train_speedrun.sh
```

Important:

- This vendored modded script expects `world_size` in `{1, 2, 4, 8}`.
- That comes from internal sharding/schedule assumptions in `train_gpt.py`.

### 6.5 What you can do with modded NanoGPT (modularity)

This is your high-performance experimentation surface:

- **Model structure**: attention blocks, skip behavior, gating, embeddings.
- **Optimizer behavior**: NorMuon/Adam hyperparameters, schedule logic.
- **Kernel pathing**: Triton kernels, flash-attention versions, compile flags.
- **Distributed systems**: sharding layout, comms order, NCCL settings.
- **Ablations**: isolate one change per run; compare wall-clock and val loss.

Useful references:

- Modded upstream: https://github.com/KellerJordan/modded-nanogpt
- Muon overview: https://kellerjordan.github.io/posts/muon/
- Polar Express sign method paper: https://arxiv.org/pdf/2505.16932

---

## 7) End-to-End Operational Checklist

For a new Engaging researcher, the minimal flow is:

1. Clone repo.
2. Set `CONDA_SH` and `ENV_PATH`.
3. Run `bash setup_env.sh`.
4. Validate baseline path:
   - `sbatch slurm/nanogpt/prepare_shakespeare.sh`
   - `sbatch slurm/nanogpt/train_shakespeare.sh`
5. Validate modded path:
   - `sbatch slurm/modded/prepare_fineweb.sh`
   - `GPU_TYPE=h100 NUM_GPUS=1 sbatch --gres=gpu:h100:1 slurm/modded/train_speedrun.sh`
6. Scale to target hardware:
   - Baseline: tune `GPUS_PER_NODE` + `GRAD_ACC_STEPS`
   - Modded: increase `NUM_GPUS` to `2`, `4`, then `8`

---

## 8) Join the Speedrun

Track current records in the official world record table:

- https://github.com/KellerJordan/modded-nanogpt#world-record-history

Use this repository to:

- Try to beat the current best speedrun result.
- Run structured studies on why certain changes accelerate training so strongly.
- Build your own modded NanoGPT variants on top of reproducible Engaging workflows.

---

## Appendix A: Useful Cluster Commands

```bash
sinfo -o "%P %G %N %a" | rg gpu
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## Appendix B: Troubleshooting

- **Conda activation fails**
  - Verify your `CONDA_SH`:
    - `echo "$CONDA_SH"`
    - `ls "$CONDA_SH"`
  - Verify your env path:
    - `echo "$ENV_PATH"`
    - `ls "$ENV_PATH"`
- **`torchrun: command not found`**
  - Check environment activation:
    - `python -c "import torch; print(torch.__version__)"`
- **NCCL hangs / multi-node issues**
  - Keep `NCCL_IB_DISABLE=1` unless InfiniBand setup is confirmed.
  - Add `NCCL_DEBUG=INFO` for diagnostics.
- **OOM**
  - Reduce batch size / sequence length / model size / accumulation.
- **FineWeb download hiccups**
  - Retry in a fresh job (transient network issues happen).
