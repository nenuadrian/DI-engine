# GTrXL Benchmark Report

- Generated: `2026-02-15 10:34:51`
- Base directory: `/net/scratch/mbax2an2/DI-engine`

## Score Over Time

X-axis uses train iteration as a time-weighted proxy.
Each line is the seed-aggregated mean evaluator reward for one algorithm.

![Score over time](score_over_time.png)

## Max Score by Algorithm

| Environment | R2D2-GTrXL | VMPO-GTrXL |
| --- | ---: | ---: |
| Pendulum | -437.016 | -197.642 |
| Pong | 19.600 | 19.250 |
| SpaceInvaders | 277.500 | 285.000 |

## Coverage

| Environment | Algorithm | Parsed Runs |
| --- | --- | ---: |
| Pendulum | R2D2-GTrXL | 2 |
| Pendulum | VMPO-GTrXL | 1 |
| Pong | R2D2-GTrXL | 2 |
| Pong | VMPO-GTrXL | 2 |
| SpaceInvaders | R2D2-GTrXL | 2 |
| SpaceInvaders | VMPO-GTrXL | 1 |

## Parsed Logs

- `/net/scratch/mbax2an2/DI-engine/pendulum_r2d2_gtrxl_seed0/log/evaluator/evaluator_logger.txt` (Pendulum, R2D2-GTrXL, seed=0, points=1)
- `/net/scratch/mbax2an2/DI-engine/pendulum_r2d2_gtrxl_seed0_260215_100544/log/evaluator/evaluator_logger.txt` (Pendulum, R2D2-GTrXL, seed=0, points=105)
- `/net/scratch/mbax2an2/DI-engine/pendulum_vmpo_gtrxl_seed0/log/evaluator/evaluator_logger.txt` (Pendulum, VMPO-GTrXL, seed=0, points=22)
- `/net/scratch/mbax2an2/DI-engine/pong_r2d2_gtrxl_seed0_260214_225402/log/evaluator/evaluator_logger.txt` (Pong, R2D2-GTrXL, seed=0, points=1)
- `/net/scratch/mbax2an2/DI-engine/pong_r2d2_gtrxl_seed0_260214_225557/log/evaluator/evaluator_logger.txt` (Pong, R2D2-GTrXL, seed=0, points=39)
- `/net/scratch/mbax2an2/DI-engine/pong_vmpo_gtrxl_seed0/log/evaluator/evaluator_logger.txt` (Pong, VMPO-GTrXL, seed=0, points=62)
- `/net/scratch/mbax2an2/DI-engine/pong_vmpo_gtrxl_seed0_260215_094909/log/evaluator/evaluator_logger.txt` (Pong, VMPO-GTrXL, seed=0, points=1)
- `/net/scratch/mbax2an2/DI-engine/spaceinvaders_r2d2_gtrxl_seed0/log/evaluator/evaluator_logger.txt` (SpaceInvaders, R2D2-GTrXL, seed=0, points=1)
- `/net/scratch/mbax2an2/DI-engine/spaceinvaders_r2d2_gtrxl_seed0_260215_102631/log/evaluator/evaluator_logger.txt` (SpaceInvaders, R2D2-GTrXL, seed=0, points=1)
- `/net/scratch/mbax2an2/DI-engine/spaceinvaders_vmpo_gtrxl_seed0/log/evaluator/evaluator_logger.txt` (SpaceInvaders, VMPO-GTrXL, seed=0, points=7)

_Report path: `/net/scratch/mbax2an2/DI-engine/reports/report_20260215_103434`_
