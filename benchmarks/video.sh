ding -m eval \                 
  -c dizoo/atari/config/serial/pong/pong_vmpo_gtrxl_config.py \
  -s 0 \
  --load-path log/ckpt_best.pth.tar \
  --replay-path log/replay_pong_best