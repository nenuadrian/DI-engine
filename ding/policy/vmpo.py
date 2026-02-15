from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from ding.rl_utils import gae, gae_data
from ding.torch_utils import to_device, to_dtype
from ding.utils import POLICY_REGISTRY, split_data_generator

from .common_utils import default_preprocess_learn
from .ppo import PPOPolicy


@POLICY_REGISTRY.register('vmpo')
class VMPOPolicy(PPOPolicy):
    """
    Overview:
        On-policy discrete VMPO policy built on PPO's data flow (collector, train-sample processing,
        and actor-critic model interface). The learn loss uses VMPO-style:
        - top-k advantage weighting with temperature dual variable (eta)
        - KL trust-region penalty with adaptive dual variable (alpha)
    """

    config = dict(
        type='vmpo',
        cuda=False,
        on_policy=True,
        priority=False,
        priority_IS_weight=False,
        recompute_adv=True,
        action_space='discrete',
        nstep_return=False,
        multi_agent=False,
        transition_with_policy_data=True,
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            lr_scheduler=None,
            value_weight=0.5,
            entropy_weight=0.001,
            adv_norm=True,
            value_norm=True,
            ppo_param_init=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            ignore_done=False,
            topk_fraction=0.5,
            epsilon_eta=0.1,
            epsilon_kl=0.02,
            temperature_init=1.0,
            temperature_lr=1e-4,
            alpha_init=1.0,
            alpha_lr=1e-4,
        ),
        collect=dict(
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            VMPO for this setup defaults to a VAC-style actor-critic with GTrXL core.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and import path.
        """
        return 'gtrxl_vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        super()._init_learn()
        assert self._action_space == 'discrete', "Current VMPOPolicy implementation supports discrete action only."

        self._topk_fraction = float(self._cfg.learn.topk_fraction)
        if not 0.0 < self._topk_fraction <= 1.0:
            raise ValueError(f"`topk_fraction` must be in (0, 1], got {self._topk_fraction}.")

        self._epsilon_eta = float(self._cfg.learn.epsilon_eta)
        self._epsilon_kl = float(self._cfg.learn.epsilon_kl)

        eta_init = torch.tensor(float(self._cfg.learn.temperature_init), dtype=torch.float32, device=self._device)
        eta_init = torch.clamp(eta_init, min=1e-6)
        self._log_eta = torch.nn.Parameter(torch.log(torch.expm1(eta_init)))
        self._eta_optimizer = torch.optim.Adam([self._log_eta], lr=float(self._cfg.learn.temperature_lr))

        alpha_init = torch.tensor(float(self._cfg.learn.alpha_init), dtype=torch.float32, device=self._device)
        alpha_init = torch.clamp(alpha_init, min=1e-6)
        self._log_alpha = torch.nn.Parameter(torch.log(torch.expm1(alpha_init)))
        self._alpha_optimizer = torch.optim.Adam([self._log_alpha], lr=float(self._cfg.learn.alpha_lr))

    @staticmethod
    def _positive(log_param: torch.Tensor) -> torch.Tensor:
        return F.softplus(log_param) + 1e-8

    def _forward_learn(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        data['obs'] = to_dtype(data['obs'], torch.float32)
        if 'next_obs' in data:
            data['next_obs'] = to_dtype(data['next_obs'], torch.float32)

        return_infos: List[Dict[str, Any]] = []
        self._learn_model.train()

        for _ in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    traj_flag = data.get('traj_flag', None)
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)
                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns
            else:
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                logits = output['logit']
                old_logits = batch['logit']

                adv = batch['adv'].reshape(-1)
                if self._adv_norm:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                sample_weight = batch.get('weight', None)
                if sample_weight is None:
                    sample_weight = torch.ones_like(adv)
                else:
                    sample_weight = sample_weight.float().reshape(-1)

                # Keep optimization numerically safe when all sample weights are 0.
                valid_mask = sample_weight > 0
                if not bool(valid_mask.any()):
                    valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)
                    sample_weight = torch.ones_like(sample_weight)

                eta = self._positive(self._log_eta)
                scaled_adv = adv / eta

                with torch.no_grad():
                    valid_scaled_adv = scaled_adv.detach()[valid_mask]
                    valid_num = int(valid_scaled_adv.numel())
                    k = max(1, int(self._topk_fraction * valid_num))
                    k = min(k, valid_num)
                    topk_vals, _ = torch.topk(valid_scaled_adv, k)
                    threshold = topk_vals.min()
                    selected_mask = valid_mask & (scaled_adv.detach() >= threshold)
                    if not bool(selected_mask.any()):
                        selected_mask = valid_mask

                selected_weight = sample_weight[selected_mask]

                # E-step dual update for eta.
                selected_scaled_det = (adv.detach() / eta)[selected_mask]
                max_scaled = selected_scaled_det.max().detach()
                exp_scaled = torch.exp(selected_scaled_det - max_scaled) * selected_weight
                sum_exp_scaled = exp_scaled.sum() + 1e-8
                sum_selected_weight = selected_weight.sum() + 1e-8
                log_mean_exp = torch.log(sum_exp_scaled / sum_selected_weight) + max_scaled
                eta_loss = eta * (self._epsilon_eta + log_mean_exp)

                self._eta_optimizer.zero_grad()
                eta_loss.backward()
                self._eta_optimizer.step()

                with torch.no_grad():
                    eta_det = self._positive(self._log_eta).detach()

                # Fixed VMPO policy weights for selected samples.
                selected_scaled = (adv.detach() / eta_det)[selected_mask]
                max_selected_scaled = selected_scaled.max()
                unnormalized_w = torch.exp(selected_scaled - max_selected_scaled) * selected_weight
                weights = unnormalized_w / (unnormalized_w.sum() + 1e-8)

                action = batch['action'].long().reshape(-1)
                new_log_prob_all = F.log_softmax(logits, dim=-1)
                new_log_prob_action = new_log_prob_all.gather(1, action.unsqueeze(-1)).squeeze(-1)
                policy_nll = -(weights * new_log_prob_action[selected_mask]).sum()

                old_log_prob_all = F.log_softmax(old_logits, dim=-1)
                old_prob_all = old_log_prob_all.exp()
                kl_all = (old_prob_all * (old_log_prob_all - new_log_prob_all)).sum(dim=-1)
                kl_selected = (kl_all[selected_mask] * selected_weight).sum() / sum_selected_weight

                alpha = self._positive(self._log_alpha)
                alpha_loss = alpha * (self._epsilon_kl - kl_selected.detach())
                self._alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self._alpha_optimizer.step()

                with torch.no_grad():
                    alpha_det = self._positive(self._log_alpha).detach()

                policy_loss = policy_nll + alpha_det * kl_selected

                value_pred = output['value'].reshape(-1)
                value_target = batch['return'].reshape(-1).detach()
                value_loss = 0.5 * ((value_pred - value_target).pow(2) * sample_weight).sum() / (
                    sample_weight.sum() + 1e-8
                )

                entropy_all = -(new_log_prob_all.exp() * new_log_prob_all).sum(dim=-1)
                entropy_loss = (entropy_all * sample_weight).sum() / (sample_weight.sum() + 1e-8)

                total_loss = policy_loss + self._value_weight * value_loss - self._entropy_weight * entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                if self._cfg.learn.lr_scheduler is not None:
                    cur_lr = sum(self._lr_scheduler.get_last_lr()) / len(self._lr_scheduler.get_last_lr())
                else:
                    cur_lr = self._optimizer.defaults['lr']

                return_infos.append(
                    {
                        'cur_lr': float(cur_lr),
                        'total_loss': float(total_loss.item()),
                        'policy_loss': float(policy_loss.item()),
                        'value_loss': float(value_loss.item()),
                        'entropy_loss': float(entropy_loss.item()),
                        'adv_max': float(adv.max().item()),
                        'adv_mean': float(adv.mean().item()),
                        'value_mean': float(value_pred.mean().item()),
                        'value_max': float(value_pred.max().item()),
                        'approx_kl': float(kl_all.mean().item()),
                        'clipfrac': 0.0,
                        'dual_eta': float(eta_det.item()),
                        'dual_alpha': float(alpha_det.item()),
                        'dual_eta_loss': float(eta_loss.item()),
                        'dual_alpha_loss': float(alpha_loss.item()),
                        'selected_frac': float(selected_mask.float().mean().item()),
                        'policy_nll': float(policy_nll.item()),
                        'kl_selected': float(kl_selected.item()),
                    }
                )

        if self._cfg.learn.lr_scheduler is not None:
            self._lr_scheduler.step()

        return return_infos

    def _state_dict_learn(self) -> Dict[str, Any]:
        state = super()._state_dict_learn()
        state.update(
            {
                'log_eta': self._log_eta.detach(),
                'log_alpha': self._log_alpha.detach(),
                'eta_optimizer': self._eta_optimizer.state_dict(),
                'alpha_optimizer': self._alpha_optimizer.state_dict(),
            }
        )
        return state

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        super()._load_state_dict_learn(state_dict)
        if 'log_eta' in state_dict:
            self._log_eta.data.copy_(state_dict['log_eta'].to(self._device))
        if 'log_alpha' in state_dict:
            self._log_alpha.data.copy_(state_dict['log_alpha'].to(self._device))
        if 'eta_optimizer' in state_dict:
            self._eta_optimizer.load_state_dict(state_dict['eta_optimizer'])
        if 'alpha_optimizer' in state_dict:
            self._alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'dual_eta',
            'dual_alpha',
            'dual_eta_loss',
            'dual_alpha_loss',
            'selected_frac',
            'policy_nll',
            'kl_selected',
        ]
