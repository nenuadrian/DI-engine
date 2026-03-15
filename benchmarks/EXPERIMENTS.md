# Control Experiments: V-MPO + GTrXL via DI-engine

This document describes the experiment plan for validating V-MPO with GTrXL (Gated Transformer-XL) on control and Atari tasks. These experiments complement the LLM experiments in the TRL fork and provide the "classic RL" evidence for the GEMPI paper.

All scripts generate SBATCH job files in `jobs/`. Submit via `sbatch` on a cluster or run locally with `run_jobs.sh --local`.

---

## 1. Core Benchmark: VMPO vs PPO vs R2D2

**Purpose:** Demonstrate that V-MPO with GTrXL matches or exceeds PPO and R2D2 (off-policy) on standard benchmarks. This validates the GEMPI framework's claim that EM-based methods are competitive with policy gradient methods in the transformer-based control setting.

**Scripts:**
- `gtrxl_vmpo.sh <suffix>` -- generates VMPO jobs (5 configs)
- `gtrxl_ppo.sh <suffix>` -- generates PPO jobs (3 configs)
- `gtrxl_r2d2.sh <suffix>` -- generates R2D2 jobs (3 configs)
- `gtrxl_all_algos.sh <suffix>` -- generates all at once

**Environments:**

| Environment | Type | Action Space | Key Challenge |
|------------|------|-------------|---------------|
| Pendulum | Classic control | Discrete (11) | Continuous → discrete, partial observability proxy |
| Pong | Atari | Discrete (6) | Temporal credit assignment over long episodes |
| SpaceInvaders | Atari | Discrete (6) | Varying reward density, strategic movement |

**Configurations per algorithm:**

| Algorithm | Pendulum | Pong | SpaceInvaders |
|-----------|----------|------|---------------|
| VMPO-GTrXL | pendulum_vmpo_gtrxl_config | pong_vmpo_gtrxl_config | spaceinvaders_vmpo_gtrxl_config |
| VMPO-GTrXL-Dropout | -- | pong_vmpo_gtrxl_dropout_config | spaceinvaders_vmpo_gtrxl_dropout_config |
| PPO-GTrXL | pendulum_ppo_gtrxl_config | pong_ppo_gtrxl_config | spaceinvaders_ppo_gtrxl_config |
| R2D2-GTrXL | pendulum_r2d2_gtrxl_config | pong_r2d2_gtrxl_config | spaceinvaders_r2d2_gtrxl_config |

**What to measure:**
- Evaluator reward mean over train iterations (learning curves)
- Max reward achieved across seeds
- Iterations to reach best score (sample efficiency)
- Reward variance across seeds (stability)

**Existing results** (from `reports/`): VMPO outperforms on Pendulum (-150 vs -225 R2D2, -202 PPO) and Pong (19.0 vs -15.0 R2D2, 15.1 PPO). VMPO-Dropout is best on SpaceInvaders (629 vs 439 VMPO, 548 PPO).

### How to Run

```bash
# Generate job files for a benchmark run named "v1"
bash benchmarks/gtrxl_all_algos.sh v1

# Submit to SLURM cluster
bash benchmarks/run_jobs.sh

# Or run locally (one at a time, with timeout)
bash benchmarks/run_jobs.sh --local

# Generate report from logs
python benchmarks/gtrxl_benchmark_report.py --base-dir . --recursive
```

---

## 2. Dropout Ablation

**Purpose:** Test whether dropout in the GTrXL backbone improves generalisation or hurts sample efficiency. The GEMPI paper hypothesises that dropout acts as implicit regularisation in the M-step, complementing the explicit KL trust region.

**Configs already available:**
- `pong_vmpo_gtrxl_dropout_config.py` (dropout=0.1)
- `spaceinvaders_vmpo_gtrxl_dropout_config.py` (dropout=0.1)

**What to compare:**
- VMPO vs VMPO-Dropout on same environment
- Does dropout slow early learning but improve final performance?
- Does dropout reduce overfitting on SpaceInvaders (where it helped in prior runs)?

**Status:** These are already included in `gtrxl_vmpo.sh`. The report script handles them as a separate algorithm (`vmpo_dropout`).

---

## 3. Extended Environments

**Purpose:** Test V-MPO on additional environments to broaden the evidence base. These environments test different aspects: long-term memory (BSuite), different Atari games (Q*bert), and different physics (LunarLander).

**Script:** `gtrxl_vmpo_extended.sh <suffix>`

**Additional environments:**

| Environment | Config | Key Challenge |
|------------|--------|---------------|
| Q*bert | qbert_vmpo_gtrxl_config | Complex scoring, multi-objective |
| CartPole | cartpole_vmpo_gtrxl_config | Simple baseline (should solve easily) |
| LunarLander | lunarlander_vmpo_gtrxl_config | Continuous physics, sparse reward |
| BSuite Memory-15 | memory_len_15_vmpo_gtrxl_config | Long-term memory (GTrXL advantage) |

**What to measure:**
- Does VMPO solve CartPole quickly? (sanity check)
- Does GTrXL help on BSuite Memory-15? (transformer memory advantage)
- LunarLander convergence speed vs Pendulum

---

## 4. Multi-Seed Runs

**Purpose:** All results should be averaged over 3+ seeds for statistical significance. The existing benchmark infrastructure handles this via directory naming (`<config>_seed0/`, `<config>_seed1/`, etc.).

**How to run multiple seeds:**

Edit the config files to set different seeds, or use the SBATCH array feature. Example for 3 seeds:

```bash
# In each SBATCH file, add:
#SBATCH -a 0-2

# And in the python command, pass the seed:
python -u ${config} --seed ${SLURM_ARRAY_TASK_ID}
```

The `run_jobs.sh --local` script already handles `#SBATCH -a` arrays by iterating over task IDs.

The report script (`gtrxl_benchmark_report.py`) automatically discovers multi-seed runs and computes mean +/- std.

---

## 5. Video Recording

**Purpose:** Generate replay videos of trained agents for qualitative evaluation and paper figures.

**Script:** `video.sh` (currently hardcoded to Pong)

```bash
# Record best checkpoint on Pong
ding -m eval \
  -c dizoo/atari/config/serial/pong/pong_vmpo_gtrxl_config.py \
  -s 0 \
  --load-path log/ckpt_best.pth.tar \
  --replay-path log/replay_pong_best
```

Adjust `-c` and `--load-path` for other environments/algorithms.

---

## 6. Report Generation

**Script:** `gtrxl_benchmark_report.py`

Generates a timestamped report in `reports/report_YYYYMMDD_HHMMSS/` containing:
- `README.md` with learning curves, max score tables, and coverage stats
- `score_over_time.png` plot with mean +/- std bands per algorithm
- `summary.json` with all raw data for downstream analysis

```bash
# From the DI-engine root directory
python benchmarks/gtrxl_benchmark_report.py --base-dir . --recursive

# Or from a specific experiment directory
python benchmarks/gtrxl_benchmark_report.py --base-dir /path/to/experiment/logs --recursive
```

---

## VMPO Hyperparameters (from configs)

Key VMPO hyperparameters used across environments:

| Parameter | Pendulum | Pong | SpaceInvaders | Q*bert |
|-----------|----------|------|---------------|--------|
| learning_rate | 1e-3 | 3e-4 | 1e-4 | 2.5e-4 |
| batch_size | 64 | 320 | 256 | 256 |
| epoch_per_collect | 10 | 10 | 4 | 4 |
| topk_fraction | 0.5 | 0.5 | 0.5 | 0.5 |
| epsilon_eta | 0.1 | 0.1 | 0.1 | 0.1 |
| epsilon_kl | 0.02 | 0.02 | 0.02 | 0.02 |
| GTrXL hidden | 64 | 1024 | 2048 | 1024 |
| GTrXL heads | 4 | 8 | 512 (head_dim) | 2 |
| GTrXL layers | 3 | 3 | 5 | 3 |

---

## File Index

| File | Description |
|------|-------------|
| `gtrxl_vmpo.sh` | Core VMPO benchmark (5 configs) |
| `gtrxl_ppo.sh` | PPO baseline benchmark (3 configs) |
| `gtrxl_r2d2.sh` | R2D2 baseline benchmark (3 configs) |
| `gtrxl_all_algos.sh` | Run all algorithms at once |
| `gtrxl_vmpo_extended.sh` | Extended VMPO (9 configs including Q*bert, CartPole, LunarLander, BSuite) |
| `run_jobs.sh` | Submit SBATCH jobs or run locally |
| `video.sh` | Record evaluation replay |
| `gtrxl_benchmark_report.py` | Generate report from logs |

---

## Cluster Configuration

Default SBATCH settings (overridable via env vars):

```bash
SBATCH_PARTITION=gpuA      # GPU partition name
SBATCH_NTASKS=12            # CPUs per job
SBATCH_TIME=0-12            # 12 hours wall time
# Always 1 GPU per job
```

Override example:
```bash
SBATCH_PARTITION=gpuB SBATCH_TIME=1-0 bash benchmarks/gtrxl_vmpo.sh my-run
```

WandB settings:
```bash
WANDB_ENTITY=adrian-research   # Override with your entity
WANDB_PROJECT=minerva-rl-benchmark-<suffix>  # Auto-set by scripts
```
