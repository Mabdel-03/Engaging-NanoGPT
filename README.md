# Engaging-NanoGPT

This repository is a self-contained Engaging playbook for researchers who want to:

1. Run Andrej Karpathy's baseline NanoGPT training flow end-to-end.
2. Run Keller Jordan's modded-nanogpt speedrun flow end-to-end.
3. Use both as a launchpad for new modded NanoGPT research.

The scripts in this repo are now modularized for GPU configuration:

- Default GPU type is `h100` (configurable).
- Baseline NanoGPT can run with flexible GPU counts through DDP.
- Modded run is supported for `NUM_GPUS` in `{1, 2, 4, 8}` in this vendored snapshot.

> **Command convention:** after you `cd Engaging-NanoGPT`, run all commands in this README from the repository root unless a section explicitly tells you to `cd` elsewhere.

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
- Any writable filesystem location (scratch, project, or home) with enough free space.

Storage requirement (generous lower bound):

- Work in a location with **at least 120 GB free** before setup.
- Why this much:
  - OpenWebText prep can use ~54 GB in HuggingFace cache (`nanogpt/data/openwebtext/prepare.py`).
  - Baseline `train.bin` is ~17 GB.
  - Modded default FineWeb cache (9 train chunks + val) is ~2 GB.
  - Conda environment + wheels + build artifacts can take tens of GB.
  - Logs/checkpoints/temporary files add additional overhead.

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
cd /path/to/your/workspace
git clone https://github.com/Mabdel-03/Engaging-NanoGPT.git
cd Engaging-NanoGPT
```

From this point onward, run commands from `Engaging-NanoGPT/` unless the README explicitly says otherwise.

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

### 5.1 What this baseline is (exact training flow)

The baseline uses the original NanoGPT stack:

- Standard GPT-style architecture in `nanogpt/model.py`.
- DDP training loop in `nanogpt/train.py`.
- OpenWebText/Shakespeare preparation scripts in `nanogpt/data/*`.

Exact baseline GPT-2 training defaults in this repo (`nanogpt/train.py` + `nanogpt/config/train_gpt2.py`):

- **Optimizer**: AdamW with `learning_rate=6e-4`, `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`, and `grad_clip=1.0`.
- **LR schedule**: 2,000-step warmup, then cosine decay to `min_lr=6e-5` across 600,000 iterations.
- **Batching**: `batch_size=12`, `block_size=1024`, and `gradient_accumulation_steps=5*8=40` (about 491,520 tokens per optimizer step at 8 GPUs).
- **Total token budget**: 600,000 iterations, about 300B tokens total.
- **Data path**: OpenWebText tokenized with GPT-2 BPE (`tiktoken`) into `train.bin`/`val.bin`.
- **Precision path**: bfloat16 autocast when available (otherwise float16 + GradScaler).
- **Compile path**: `compile=True` by default in `nanogpt/train.py`.
- **DDP behavior**: `gradient_accumulation_steps` is divided by `WORLD_SIZE`, and it must be divisible by `WORLD_SIZE`.
- **Checkpointing/eval**: eval every 1,000 iters in `train_gpt2.py`, and checkpoints are written to `out/`.
- **Typical target**: NanoGPT's GPT-2 config comment targets about `~2.85` validation loss in about 5 days on 8x A100 40GB.

Use this baseline to establish a clean reference before advanced speedrun changes.

### 5.2 GPT-2 scale model size and architecture

The baseline target is GPT-2 small (124M parameters). In this repo, the canonical shape is:

- **Layers (`n_layer`)**: 12
- **Attention heads (`n_head`)**: 12
- **Embedding/model width (`n_embd`)**: 768
- **Head dimension**: 64 (`768 / 12`)
- **MLP hidden width**: 3072 (`4 * n_embd`)
- **Context length (`block_size`)**: 1024 tokens
- **Vocabulary**: 50,257 GPT-2 BPE tokens, padded to 50,304 in default scratch config for efficiency
- **Block structure**: pre-norm residual block, `x + Attn(LN(x))`, then `x + MLP(LN(x))`
- **MLP activation**: GELU
- **Position signal**: learned absolute position embeddings (`wpe`)
- **Weight tying**: token embedding and `lm_head` weights are tied

GPT-2 family sizes (from `from_pretrained` config mapping in `nanogpt/model.py`):

- `gpt2`: 124M (`12/12/768`)
- `gpt2-medium`: 350M (`24/16/1024`)
- `gpt2-large`: 774M (`36/20/1280`)
- `gpt2-xl`: 1558M (`48/25/1600`)

### 5.3 Prepare data

Quick sanity dataset:

```bash
sbatch slurm/nanogpt/prepare_shakespeare.sh
```

Larger GPT-2-style dataset:

```bash
sbatch slurm/nanogpt/prepare_openwebtext.sh
```

### 5.4 Train a fast sanity baseline (1 GPU)

```bash
GPU_TYPE=h100 sbatch --gres=gpu:${GPU_TYPE}:1 slurm/nanogpt/train_shakespeare.sh
```

Outputs land in:

- `out/nanogpt-shakespeare/`

### 5.5 Train baseline GPT-2 with configurable GPU count

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

### 5.6 Sample from a trained checkpoint

This is the one place in the baseline flow where you intentionally leave repo root:

```bash
cd nanogpt
python sample.py --out_dir=../out/nanogpt-shakespeare
```

### 5.7 What you can do with baseline NanoGPT (modularity)

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

### 6.1 What this modded run is (exact changes and training details)

The vendored speedrun pipeline in `modded_nanogpt/train_gpt.py` is not just "NanoGPT + tuning"; it changes architecture, optimizer, scheduling, and kernel stack in a coupled way.

#### 6.1.1 Exact model architecture differences vs baseline NanoGPT

- **Core width**: still 768 model dim, but with `num_heads=6` and `head_dim=128` (instead of `12 x 64`).
- **Depth/layout**: `num_layers=11`; attention is skipped in layer index 6, so only 10 layers perform attention.
- **Normalization**: RMSNorm (`F.rms_norm`) replaces LayerNorm.
- **MLP nonlinearity**: ReLU-squared (`relu(x)^2`) replaces GELU, executed through a fused Triton kernel.
- **Attention kernel path**: uses `flash_attn_varlen_func` with windowed causal attention masks.
- **RoPE path**: half-truncated RoPE (only half head dims rotated), with YaRN updates when long-window size changes.
- **Paired-head layers**: layers `[0, 2, 5, 9]` use paired-head attention mechanics.
- **Key-offset trick**: long-window layers shift stationary key dims by one token to improve induction behavior.
- **Value embeddings**: a bank of `5 * vocab_size` learned value embeddings injected in selected layers.
- **Bigram embeddings**: hash-based bigram embedding table with `bigram_vocab_size = 50304 * 5`.
- **Gated extras**: smear gate (inject token `t-1` into token `t`), skip gate (U-net style skip from layer 3 to layer 6), attention/value gates.
- **Backout term**: subtracts scaled contribution from early-stack context branch (`x -= backout_lambda * x_backout`).
- **Parameter banks**:
  - attention bank shape `(10, 3072, 768)` and reshaped `(40, 768, 768)` for sharding
  - MLP bank shape `(12, 2, 3072, 768)` and reshaped `(24, 3072, 768)` for sharding
- **LM head path**: transposed-weight linear with FP8-capable matmul scaling path.
- **Output/loss path**: fused softcapped cross-entropy (`23 * sigmoid((logits + 5) / 7.5)` logic in eval path).

#### 6.1.2 Exact optimizer and distributed step changes

- **Hybrid optimizer split**:
  - NorMuon on projection-heavy parameter banks (`attn`, `mlp`)
  - Adam on embeddings, gates, scalars, and other non-bank params
- **NorMuon defaults**: `lr=0.023`, `momentum=0.95`, `beta2=0.95`, `weight_decay=1.2`.
- **Adam defaults**: `lr=0.008`, `eps=1e-10`, `weight_decay=0.005` (with per-group multipliers/betas).
- **Polar Express orthogonalization**: used in NorMuon updates to speed orthogonalized steps.
- **Alternating optimizer cadence**: Adam updates only on odd-numbered steps; NorMuon updates every step.
- **Communication scheduling**: explicit `scatter_order` and `work_order` to overlap comms with compute and process small params first.
- **World-size assumption**: `world_size` must divide 8; this snapshot supports practical runs at 1/2/4/8 GPUs.

#### 6.1.3 Exact modded training schedule

- **Data**: FineWeb10B cached binaries (`fineweb_train_*.bin`, `fineweb_val_*.bin`), with fixed validation token budget.
- **Sequence length**: `train_max_seq_len = 128 * 16 = 2048`.
- **Gradient accumulation**: `grad_accum_steps = 8 // world_size`.
- **Schedule length**: 1,515 scheduled iterations + 40 extension iterations.
- **Stage 1 (first third)**: batch `8 * 2048 * 8`, windows `(1, 3)`, MTP weights `[1.0, 0.5, 0.25 -> 0.0]`, `lr_mul=1.0`.
- **Stage 2 (second third)**: batch `16 * 2048 * 8`, windows `(3, 7)`, MTP weights `[1.0, 0.5 -> 0.0]`, `lr_mul=1.52`.
- **Stage 3 (third third)**: batch `24 * 2048 * 8`, windows `(5, 11)`, MTP weights `[1.0]`, `lr_mul=1.73`.
- **Extension stage**: batch `24 * 2048 * 8`, windows `(6, 13)`, then final long-window extension to 20 for YaRN post-extension path.
- **LR cooldown**: final `cooldown_frac=0.55` portion decays toward `0.1x` multiplier.
- **Embed/lm_head transition**: tied early, then split at the stage-2 boundary (forced to an odd step).

#### 6.1.4 Why this is thought to be faster

- Fused Triton kernels reduce memory traffic and kernel launch overhead.
- FP8-capable lm-head path cuts math/IO cost for the largest output projection.
- Windowed FlashAttention avoids full quadratic attention over long contexts.
- ReLU-squared + fused MLP path is cheaper and easier to fuse than GELU-heavy paths.
- RMSNorm is lighter than LayerNorm.
- Parameter banks improve sharding regularity and communication overlap.
- Progressive schedule (batch/window growth + MTP simplification) spends early training on cheaper steps.
- Alternating Adam cadence cuts optimizer-side overhead on non-bank parameters.
- Slightly shallower attention path (10 attention layers active) reduces per-step compute.
- Fullgraph `torch.compile` is used to optimize end-to-end execution.

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

Backend note:

- `modded_nanogpt/train_gpt.py` first tries the speedrun kernel interface (`varunneal/flash-attention-3`) and falls back to local `flash_attn` if unavailable.
- This improves portability for general users.
- For strict speedrun comparability and reproducibility, pin the exact FlashAttention backend/version and keep it fixed across runs.

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
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} --cpus-per-task=16 slurm/modded/train_speedrun.sh

# 2x H100
GPU_TYPE=h100 NUM_GPUS=2 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} --cpus-per-task=32 slurm/modded/train_speedrun.sh

# 4x H100
GPU_TYPE=h100 NUM_GPUS=4 \
sbatch --gres=gpu:${GPU_TYPE}:${NUM_GPUS} --cpus-per-task=64 slurm/modded/train_speedrun.sh

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
- The script header defaults to `--cpus-per-task=120` (sized for 8 GPUs / full node). When requesting fewer GPUs, override `--cpus-per-task` on the command line to avoid QOS CPU limits. Suggested values: 1 GPU = 16, 2 GPUs = 32, 4 GPUs = 64.

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

1. Clone repo and `cd Engaging-NanoGPT` (run all following commands from repo root unless explicitly stated).
2. Set `CONDA_SH` and `ENV_PATH`.
3. Run `bash setup_env.sh`.
4. Validate baseline path:
   - `sbatch slurm/nanogpt/prepare_shakespeare.sh`
   - `sbatch slurm/nanogpt/train_shakespeare.sh`
5. Validate modded path:
   - `sbatch slurm/modded/prepare_fineweb.sh`
   - `GPU_TYPE=h100 NUM_GPUS=1 sbatch --gres=gpu:h100:1 --cpus-per-task=16 slurm/modded/train_speedrun.sh`
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
