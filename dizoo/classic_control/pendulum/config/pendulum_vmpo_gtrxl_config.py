from easydict import EasyDict

collector_env_num = 10
evaluator_env_num = 5
pendulum_vmpo_gtrxl_config = dict(
    exp_name='pendulum_vmpo_gtrxl_seed0',
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
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=3,
            action_shape=11,  # 11 discrete actions in Pendulum when continuous=False.
            hidden_size=64,
            encoder_hidden_size_list=[128, 128, 64],
            att_head_dim=16,
            att_head_num=4,
            att_mlp_num=2,
            att_layer_num=3,
            memory_len=0,
            dropout=0.0,
            gru_bias=2.0,
            actor_head_hidden_size=64,
            critic_head_hidden_size=64,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=1e-3,
            ppo_param_init=False,
            value_weight=0.5,
            entropy_weight=0.01,
            adv_norm=True,
            value_norm=True,
            ignore_done=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            topk_fraction=0.5,
            epsilon_eta=0.1,
            epsilon_kl=0.02,
            temperature_init=1.0,
            temperature_lr=1e-4,
            alpha_init=1.0,
            alpha_lr=1e-4,
        ),
        collect=dict(
            n_sample=5000,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=200, )),
    ),
    wandb_logger=dict(
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False,
    ),
)
pendulum_vmpo_gtrxl_config = EasyDict(pendulum_vmpo_gtrxl_config)
main_config = pendulum_vmpo_gtrxl_config
pendulum_vmpo_gtrxl_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='vmpo'),
)
pendulum_vmpo_gtrxl_create_config = EasyDict(pendulum_vmpo_gtrxl_create_config)
create_config = pendulum_vmpo_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c pendulum_vmpo_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
