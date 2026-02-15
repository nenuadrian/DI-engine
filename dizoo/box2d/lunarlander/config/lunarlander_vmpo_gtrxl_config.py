from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_vmpo_gtrxl_config = dict(
    exp_name='lunarlander_vmpo_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=200,
        env_id='LunarLander-v2',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=8,
            action_shape=4,
            hidden_size=256,
            encoder_hidden_size_list=[256, 256, 256],
            att_head_dim=16,
            att_head_num=2,
            att_mlp_num=2,
            att_layer_num=3,
            memory_len=0,
            dropout=0.1,
            gru_bias=1.0,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            ppo_param_init=False,
            value_weight=0.5,
            entropy_weight=0.001,
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
            n_sample=512,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
    wandb_logger=dict(
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False,
    ),
)
lunarlander_vmpo_gtrxl_config = EasyDict(lunarlander_vmpo_gtrxl_config)
main_config = lunarlander_vmpo_gtrxl_config
lunarlander_vmpo_gtrxl_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='vmpo'),
)
lunarlander_vmpo_gtrxl_create_config = EasyDict(lunarlander_vmpo_gtrxl_create_config)
create_config = lunarlander_vmpo_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c lunarlander_vmpo_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
