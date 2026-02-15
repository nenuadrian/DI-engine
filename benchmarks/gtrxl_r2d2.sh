#!/usr/bin/env bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix> [extra main.py args...]" >&2
    exit 1
fi
BENCHMARK_SUFFIX="$1"
shift


export WANDB_PROJECT=minerva-rl-benchmark-${BENCHMARK_SUFFIX}
export WANDB_ENTITY=adrian-research

python -u dizoo/classic_control/pendulum/config/pendulum_r2d2_gtrxl_config.py

python -u dizoo/atari/config/serial/pong/pong_r2d2_gtrxl_config.py

python -u dizoo/atari/config/serial/spaceinvaders/spaceinvaders_r2d2_gtrxl_config.py