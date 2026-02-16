#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/jobs"
mkdir -p "$JOBS_DIR"

configs=(
    "dizoo/classic_control/pendulum/config/pendulum_r2d2_gtrxl_config.py"
    "dizoo/atari/config/serial/pong/pong_r2d2_gtrxl_config.py"
    "dizoo/atari/config/serial/spaceinvaders/spaceinvaders_r2d2_gtrxl_config.py"
)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix>" >&2
    exit 1
fi

BENCHMARK_SUFFIX="$1"
WANDB_PROJECT_NAME="minerva-rl-benchmark-${BENCHMARK_SUFFIX}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpuA}"
SBATCH_NTASKS="${SBATCH_NTASKS:-12}"
SBATCH_TIME="${SBATCH_TIME:-4-0}"

benchmark_slug="${BENCHMARK_SUFFIX//\//-}"
benchmark_slug="${benchmark_slug// /-}"

for config in "${configs[@]}"; do
    config_slug="$(basename "$config" .py)"
    job_file="$JOBS_DIR/gtrxl_r2d2_${benchmark_slug}_${config_slug}.sbatch.sh"

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

echo "Done. Submit jobs with: sbatch jobs/<job-file>.sbatch.sh"
