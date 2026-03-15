#!/usr/bin/env bash
set -euo pipefail

# Run ALL algorithm benchmarks (VMPO, PPO, R2D2) in one go.
# Generates all job files then optionally submits them.
#
# Usage:
#   ./gtrxl_all_algos.sh <benchmark-suffix>              # Generate job files only
#   ./gtrxl_all_algos.sh <benchmark-suffix> --submit      # Generate and sbatch submit
#   ./gtrxl_all_algos.sh <benchmark-suffix> --local       # Generate and run locally

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix> [--submit|--local]" >&2
    exit 1
fi

SUFFIX="$1"
ACTION="${2:-}"

echo "=== Generating VMPO jobs ==="
bash "${SCRIPT_DIR}/gtrxl_vmpo.sh" "${SUFFIX}"

echo ""
echo "=== Generating PPO jobs ==="
bash "${SCRIPT_DIR}/gtrxl_ppo.sh" "${SUFFIX}"

echo ""
echo "=== Generating R2D2 jobs ==="
bash "${SCRIPT_DIR}/gtrxl_r2d2.sh" "${SUFFIX}"

echo ""
echo "=== All job files generated ==="

if [[ "${ACTION}" == "--submit" ]]; then
    echo "Submitting all jobs via sbatch..."
    bash "${SCRIPT_DIR}/run_jobs.sh"
elif [[ "${ACTION}" == "--local" ]]; then
    echo "Running all jobs locally..."
    bash "${SCRIPT_DIR}/run_jobs.sh" --local
else
    echo "To submit: ./benchmarks/run_jobs.sh"
    echo "To run locally: ./benchmarks/run_jobs.sh --local"
fi
