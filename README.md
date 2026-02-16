# Engaging-NanoGPT: A Hacker's Guide to Training Modded NanoGPT on MIT's Engaging Cluster

This repository is a hands-on walkthrough for MIT researchers who want to understand and run modern GPT training on Engaging.  
You will start with Andrej Karpathy's baseline NanoGPT setup, then move to Keller Jordan's modded speedrun setup, then branch into your own training ideas.

Follow this guide in order:
1. Read and understand the challenge
2. Clone the repo on MIT Engaging
3. Set up the environment
4. Set up and train Andrej Karpathy's original run (baseline)
5. Set up and train Keller Jordan's modded run (advanced)
6. Explore your own experiments

## Step 1) Read and Understand the Challenge

The core challenge is from the `modded-nanogpt` speedrun community:
- Train a GPT-2-scale model to `<= 3.28` validation loss on FineWeb
- Do it as fast as possible on 8 Hopper-class GPUs (official benchmark uses 8x H100)

Why `3.28`?
- That target comes from Karpathy's GPT-2 replication baseline in `llm.c` (about 45 minutes on 8x H100)
- The speedrun community iteratively pushed that down to around 1.5 minutes through model, optimizer, and systems changes

What changed in the advanced setup:
- Modernized architecture choices (e.g., rotary-style ideas, attention changes, extra shortcuts)
- Faster/more specialized optimization techniques
- Low-level kernel and distributed systems improvements

Upstream references:
- Karpathy baseline: https://github.com/karpathy/nanoGPT
- Modded speedrun: https://github.com/KellerJordan/modded-nanogpt

## Step 2) Clone This Repo on Engaging

Use scratch storage if possible because datasets and checkpoints can get large.

```bash
cd /path/to/your/scratch
git clone https://github.com/Mabdel-03/Engaging-NanoGPT.git
cd Engaging-NanoGPT
```

Quick structure:
- `nanogpt/`: vendored baseline training and data prep files
- `modded_nanogpt/`: vendored modded speedrun training files
- `slurm/`: ready-to-submit SLURM scripts
- `setup_env.sh`: one-time environment bootstrap
- `activate_env.sh`: quick environment activation helper

## Step 3) Set Up the Environment

Run setup once:

```bash
bash setup_env.sh
```

This installs:
- PyTorch (CUDA 12.4 wheels; nightly fallback if needed)
- Baseline + modded dependencies in one env
- Conda environment at `/home/mabdel03/conda_envs/nanogpt_env`

Manual activation:

```bash
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda activate /home/mabdel03/conda_envs/nanogpt_env
```

Shortcut:

```bash
source activate_env.sh
```

## Step 4) Baseline: Andrej Karpathy's Original NanoGPT Run

This is the baseline phase. You reproduce the original style first so you can compare future improvements against a known reference.

### 4A. Prepare baseline data

Small sanity-check dataset:

```bash
sbatch slurm/nanogpt/prepare_shakespeare.sh
```

Larger GPT-2 dataset:

```bash
sbatch slurm/nanogpt/prepare_openwebtext.sh
```

### 4B. Train a quick baseline on 1 GPU

```bash
sbatch slurm/nanogpt/train_shakespeare.sh
```

Output checkpoints will be written under:
- `out/nanogpt-shakespeare/`

### 4C. Train a multi-GPU baseline with DDP

Default run:

```bash
sbatch slurm/nanogpt/train_gpt2.sh
```

This script launches:

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```

Optional multi-node variant:

```bash
# Request nodes/resources, then launch one task per node
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:4 slurm/nanogpt/train_gpt2.sh
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash slurm/nanogpt/train_gpt2.sh
```

Sample from a trained checkpoint:

```bash
cd nanogpt
python sample.py --out_dir=../out/nanogpt-shakespeare
```

Context: full GPT-2 reproduction in the original setup is a long run (often measured in days on large GPU nodes). This is exactly why the speedrun challenge became interesting.

## Step 5) Advanced: Keller Jordan's Modded-NanoGPT Run

Now move to the advanced speedrun-style path. This is where most performance innovations live.

### 5A. Prepare FineWeb token cache

Default (first 900M tokens, 9 chunks):

```bash
sbatch slurm/modded/prepare_fineweb.sh
```

Override chunk count:

```bash
FINEWEB_CHUNKS=3 sbatch slurm/modded/prepare_fineweb.sh
```

### 5B. Launch the 8x H200 speedrun path

```bash
sbatch slurm/modded/train_speedrun.sh
```

This script requests:
- `--partition=mit_normal_gpu`
- `--gres=gpu:h200:8`
- `torchrun --standalone --nproc_per_node=8 train_gpt.py`

What to expect:
- First run may pay a compile/startup overhead
- Throughput and final loss depend on software versions and node state
- Logs should make it easy to compare against your baseline behavior

## Step 6) Explore Your Own Training Techniques

This repo is intended to be a launchpad for your own ideas. Good first experiment directions:

- **Model changes**: edit `modded_nanogpt/train_gpt.py` model blocks and attention behavior
- **Optimizer/schedule changes**: adjust optimizer setup, LR schedule, warmup/cooldown, batch schedule
- **Systems changes**: tune DDP/NCCL settings and data pipeline behavior
- **Cluster script changes**: copy and modify `slurm/modded/train_speedrun.sh` or `slurm/nanogpt/train_gpt2.sh`
- **Ablation workflow**: change one variable at a time and keep run notes for reproducibility

Suggested experiment loop:
1. Clone an existing SLURM script into a new experiment script
2. Add a clear job name and output path
3. Make one focused code change
4. Run and compare wall-clock + validation loss against your previous run
5. Repeat

## Appendix: Engaging Cluster Context

Useful partitions for this repo:
- `mit_normal_gpu` (6h): includes 8x H200 nodes (good for modded speedruns)
- `mit_preemptable` (up to 2 days): useful for longer baseline training

Useful checks:

```bash
sinfo -o "%P %G %N %a" | rg gpu
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## Appendix: SLURM Script Index

- Baseline (`nanogpt`):
  - `slurm/nanogpt/prepare_shakespeare.sh`
  - `slurm/nanogpt/prepare_openwebtext.sh`
  - `slurm/nanogpt/train_shakespeare.sh`
  - `slurm/nanogpt/train_gpt2.sh`
- Advanced (`modded_nanogpt`):
  - `slurm/modded/prepare_fineweb.sh`
  - `slurm/modded/train_speedrun.sh`

## Appendix: Troubleshooting

- **Conda activation fails**
  - Check: `source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh`
  - Check env path exists: `ls /home/mabdel03/conda_envs/nanogpt_env`
- **`torchrun: command not found`**
  - Verify environment and torch install: `python -c "import torch; print(torch.__version__)"`
- **NCCL hangs / multi-node issues**
  - Keep `NCCL_IB_DISABLE=1` unless you confirm InfiniBand setup
  - Try `NCCL_DEBUG=INFO` for more detail
- **OOM errors**
  - Reduce batch size, context length, model size, or accumulation settings
- **Dataset download failures**
  - Retry in a fresh job; transient network issues happen

## Appendix: Upstream Sources

- NanoGPT source: https://github.com/karpathy/nanoGPT
- Modded source: https://github.com/KellerJordan/modded-nanogpt

These are vendored snapshots for onboarding and reproducible cluster setup.
