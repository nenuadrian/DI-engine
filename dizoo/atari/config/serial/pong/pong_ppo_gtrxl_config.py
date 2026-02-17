from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
pong_ppo_gtrxl_config = dict(
    exp_name='pong_ppo_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        # 'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            type='gtrxl_vac',
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size=1024,
            encoder_hidden_size_list=[128, 512, 1024],
            att_head_dim=16,
            att_head_num=8,
            att_mlp_num=2,
            att_layer_num=3,
            memory_len=0,
            dropout=0.0,
            gru_gating=True,
            gru_bias=2.0,
            actor_head_hidden_size=1024,
            critic_head_hidden_size=1024,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            ppo_param_init=False,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            kl_beta=0.01,
            kl_type='k1',
            pretrained_model_path=None,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
    wandb_logger=dict(
        enabled=True,
        sync_tensorboard=True,
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False,
    ),
)
pong_ppo_gtrxl_config = EasyDict(pong_ppo_gtrxl_config)
main_config = pong_ppo_gtrxl_config
pong_ppo_gtrxl_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
pong_ppo_gtrxl_create_config = EasyDict(pong_ppo_gtrxl_create_config)
create_config = pong_ppo_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c pong_ppo_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
