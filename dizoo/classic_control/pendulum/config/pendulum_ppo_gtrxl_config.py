from easydict import EasyDict
import torch.nn as nn

collector_env_num = 10
evaluator_env_num = 5
pendulum_ppo_gtrxl_config = dict(
    exp_name='pendulum_ppo_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        recompute_adv=True,
        model=dict(
            type='gtrxl_vac',
            obs_shape=3,
            action_shape=1,
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
            action_space='continuous',
            actor_head_layer_num=0,
            critic_head_layer_num=0,
            sigma_type='independent',
            activation=nn.Tanh(),
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=1e-3,
            ppo_param_init=False,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            ignore_done=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
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
        enabled=True,
        sync_tensorboard=True,
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False,
    ),
)
pendulum_ppo_gtrxl_config = EasyDict(pendulum_ppo_gtrxl_config)
main_config = pendulum_ppo_gtrxl_config
pendulum_ppo_gtrxl_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
pendulum_ppo_gtrxl_create_config = EasyDict(pendulum_ppo_gtrxl_create_config)
create_config = pendulum_ppo_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c pendulum_ppo_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
