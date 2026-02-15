from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
cartpole_vmpo_gtrxl_config = dict(
    exp_name='cartpole_vmpo_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=4,
            action_shape=2,
            hidden_size=64,
            encoder_hidden_size_list=[64, 64, 64],
            att_head_dim=16,
            att_head_num=2,
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
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=1e-3,
            ppo_param_init=False,
            value_weight=0.5,
            entropy_weight=0.01,
            adv_norm=True,
            value_norm=True,
            ignore_done=False,
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
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=20, )),
    ),
    wandb_logger=dict(
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False,
    ),
)
cartpole_vmpo_gtrxl_config = EasyDict(cartpole_vmpo_gtrxl_config)
main_config = cartpole_vmpo_gtrxl_config
cartpole_vmpo_gtrxl_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='vmpo'),
)
cartpole_vmpo_gtrxl_create_config = EasyDict(cartpole_vmpo_gtrxl_create_config)
create_config = cartpole_vmpo_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_vmpo_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
