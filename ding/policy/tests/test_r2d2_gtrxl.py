import pytest
import torch
from easydict import EasyDict

from ding.policy.r2d2_gtrxl import R2D2GTrXLPolicy


OBS_SPACE = 4
ACTION_SPACE = 2
ENV_NUM = 12
SEQ_LEN = 8

cfg = EasyDict(
    dict(
        cuda=False,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=OBS_SPACE,
            action_shape=ACTION_SPACE,
            memory_len=5,
            hidden_size=64,
            gru_bias=2.0,
            att_layer_num=3,
            dropout=0.0,
            att_head_num=4,
        ),
        discount_factor=0.99,
        nstep=3,
        burnin_step=4,
        unroll_len=11,
        seq_len=SEQ_LEN,
        learn=dict(
            update_per_collect=1,
            batch_size=8,
            learning_rate=5e-4,
            target_update_theta=0.001,
            value_rescale=True,
            init_memory='old',
            ignore_done=False,
        ),
        collect=dict(n_sample=8, traj_len_inf=True, env_num=ENV_NUM),
        eval=dict(env_num=ENV_NUM),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=1000),
        ),
    )
)


def get_batch(size: int = ENV_NUM):
    return {i: torch.zeros(OBS_SPACE) for i in range(size)}


@pytest.mark.unittest
def test_r2d2_gtrxl_collect_and_eval_decollate():
    # Regression test for env_num > seq_len: sequence-major fields must not be decollated as batch-major.
    assert ENV_NUM > SEQ_LEN
    policy = R2D2GTrXLPolicy(cfg, enable_field=['collect', 'eval'])
    batch = get_batch()

    collect_out = policy._forward_collect(batch, eps=0.1)
    assert len(collect_out) == ENV_NUM
    assert 'action' in collect_out[0]
    assert 'logit' in collect_out[0]
    assert 'memory' in collect_out[0]
    assert 'transformer_out' not in collect_out[0]
    assert 'input_seq' not in collect_out[0]

    eval_out = policy._forward_eval(batch)
    assert len(eval_out) == ENV_NUM
    assert 'action' in eval_out[0]
    assert 'logit' in eval_out[0]
    assert 'memory' in eval_out[0]
    assert 'transformer_out' not in eval_out[0]
    assert 'input_seq' not in eval_out[0]
