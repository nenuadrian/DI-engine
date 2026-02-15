from easydict import EasyDict

collector_env_num = 10
evaluator_env_num = 5
pendulum_r2d2_gtrxl_config = dict(
    exp_name='pendulum_r2d2_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
        continuous=False,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=3,
            action_shape=11,  # 11 discrete actions in Pendulum when continuous=False.
            memory_len=5,
            hidden_size=64,
            gru_bias=2.0,
            att_layer_num=3,
            dropout=0.0,
            att_head_num=4,
        ),
        discount_factor=0.97,
        nstep=3,
        burnin_step=4,
        unroll_len=11,
        seq_len=8,
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=5e-4,
            target_update_theta=0.001,
            value_rescale=True,
            init_memory='old',
        ),
        collect=dict(
            n_sample=32,
            traj_len_inf=True,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=40)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
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
pendulum_r2d2_gtrxl_config = EasyDict(pendulum_r2d2_gtrxl_config)
main_config = pendulum_r2d2_gtrxl_config
pendulum_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2_gtrxl'),
)
pendulum_r2d2_gtrxl_create_config = EasyDict(pendulum_r2d2_gtrxl_create_config)
create_config = pendulum_r2d2_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c pendulum_r2d2_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
