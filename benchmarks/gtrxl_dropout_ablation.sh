#!/usr/bin/env bash
set -euo pipefail

# Dropout ablation: 2x2 design (PPO vs VMPO) x (no-dropout vs dropout=0.1)
# on Pong and SpaceInvaders with GTrXL backbone.
#
# This is the controlled experiment for the paper's dropout-compatibility claim
# (Section 5.7 / RQ4). VMPO's detached-weight M-step should be compatible with
# dropout; PPO's importance-ratio method should degrade.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/jobs"
mkdir -p "$JOBS_DIR"

configs=(
    # --- VMPO baseline (dropout=0.0) ---
    "dizoo/atari/config/serial/pong/pong_vmpo_gtrxl_config.py"
    "dizoo/atari/config/serial/spaceinvaders/spaceinvaders_vmpo_gtrxl_config.py"
    # --- VMPO + dropout (dropout=0.1) ---
    "dizoo/atari/config/serial/pong/pong_vmpo_gtrxl_dropout_config.py"
    "dizoo/atari/config/serial/spaceinvaders/spaceinvaders_vmpo_gtrxl_dropout_config.py"
    # --- PPO baseline (dropout=0.0) ---
    "dizoo/atari/config/serial/pong/pong_ppo_gtrxl_config.py"
    "dizoo/atari/config/serial/spaceinvaders/spaceinvaders_ppo_gtrxl_config.py"
    # --- PPO + dropout (dropout=0.1) ---
    "dizoo/atari/config/serial/pong/pong_ppo_gtrxl_dropout_config.py"
    "dizoo/atari/config/serial/spaceinvaders/spaceinvaders_ppo_gtrxl_dropout_config.py"
)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix>" >&2
    exit 1
fi

BENCHMARK_SUFFIX="$1"
WANDB_PROJECT_NAME="minerva-rl-benchmark-${BENCHMARK_SUFFIX}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpuA}"
SBATCH_NTASKS="${SBATCH_NTASKS:-12}"
SBATCH_TIME="${SBATCH_TIME:-1-00}"

benchmark_slug="${BENCHMARK_SUFFIX//\//-}"
benchmark_slug="${benchmark_slug// /-}"

for config in "${configs[@]}"; do
    config_slug="$(basename "$config" .py)"
    job_file="$JOBS_DIR/gtrxl_dropout_ablation_${benchmark_slug}_${config_slug}.sbatch.sh"

    cat > "$job_file" <<EOF
#!/bin/bash --login
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -n ${SBATCH_NTASKS}
#SBATCH -t ${SBATCH_TIME}
#SBATCH -G 1

WANDB_PROJECT="${WANDB_PROJECT_NAME}"
WANDB_ENTITY="\${WANDB_ENTITY:-adrian-research}"
WORKDIR="\${WORKDIR:-\$HOME/scratch/DI-engine}"


cd "\$WORKDIR"
source .venv/bin/activate
export WANDB_PROJECT
export WANDB_ENTITY

python -u ${config}
EOF

    chmod +x "$job_file"
    echo "Created $job_file"
done

echo ""
echo "Done. ${#configs[@]} dropout-ablation jobs generated."
echo "Submit jobs with: sbatch jobs/gtrxl_dropout_ablation_*.sbatch.sh"
echo ""
echo "Experiment design:"
echo "  Environment: Pong, SpaceInvaders (Atari, GTrXL backbone, memory_len=64)"
echo "  Conditions:  PPO (d=0.0), PPO (d=0.1), VMPO (d=0.0), VMPO (d=0.1)"
echo "  Walltime:    ${SBATCH_TIME} per job"
