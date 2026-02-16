#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/jobs"

for job in "$JOBS_DIR"/*.sbatch.sh; do
    [ -e "$job" ] || { echo "No job files found in $JOBS_DIR/"; break; }
    echo "Submitting $job"
    sbatch "$job"
done
