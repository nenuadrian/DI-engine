import os
import re
from datetime import datetime
from typing import Optional, Callable, List, Any

from ditk import logging
from ding.policy import PolicyFactory
from ding.worker import IMetric, MetricSerialEvaluator


class AccMetric(IMetric):

    def eval(self, inputs: Any, label: Any) -> dict:
        return {
            "Acc": (inputs["logit"].sum(dim=1) == label).sum().item() / label.shape[0]
        }

    def reduce_mean(self, inputs: List[Any]) -> Any:
        s = 0
        for item in inputs:
            s += item["Acc"]
        return {"Acc": s / len(inputs)}

    def gt(self, metric1: Any, metric2: Any) -> bool:
        if metric2 is None:
            return True
        if isinstance(metric2, dict):
            m2 = metric2["Acc"]
        else:
            m2 = metric2
        return metric1["Acc"] > m2


def mark_not_expert(ori_data: List[dict]) -> List[dict]:
    for i in range(len(ori_data)):
        # Set is_expert flag (expert 1, agent 0)
        ori_data[i]["is_expert"] = 0
    return ori_data


def mark_warm_up(ori_data: List[dict]) -> List[dict]:
    # for td3_vae
    for i in range(len(ori_data)):
        ori_data[i]["warm_up"] = True
    return ori_data


def _sanitize_name_piece(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else default
    if not text:
        text = default
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    return text or default


def _default_wandb_run_name(cfg: "EasyDict") -> str:  # noqa
    algo = _sanitize_name_piece(
       '-'.join(cfg.get("exp_name").split("_")[1:3]), default="algo"
    ).lower()
    env_name = cfg.get("env", {}).get("env_id", None)
    if env_name is None:
        env_name = cfg.get("env", {}).get("type", None)
    env = _sanitize_name_piece(env_name, default="env")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{algo}_{env}_{timestamp}"


def _to_wandb_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_wandb_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_wandb_config(v) for v in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_wandb_config(value.item())
        except Exception:
            pass
    return str(value)


def maybe_init_wandb(cfg: "EasyDict") -> Optional[Any]:  # noqa
    """
    Overview:
        Optionally initialize a wandb run for serial pipelines when enabled in config.
    """
    wandb_cfg = cfg.get("wandb_logger", None)
    if wandb_cfg is None or not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        logging.warning("wandb is not installed, skip wandb logging.")
        return None

    project_name = wandb_cfg.get(
        "project_name", os.getenv("WANDB_PROJECT", "DI-engine")
    )
    run_name = wandb_cfg.get("run_name", None) or _default_wandb_run_name(cfg)
    entity = wandb_cfg.get("entity", os.getenv("WANDB_ENTITY"))
    wandb_dir = os.path.abspath(wandb_cfg.get("dir", cfg.exp_name))
    os.makedirs(wandb_dir, exist_ok=True)

    init_kwargs = dict(
        project=project_name,
        name=run_name,
        sync_tensorboard=wandb_cfg.get("sync_tensorboard", True),
        reinit=True,
        dir=wandb_dir,
    )
    if wandb_cfg.get("log_config", True):
        init_kwargs["config"] = _to_wandb_config(cfg)
    if entity:
        init_kwargs["entity"] = entity
    if wandb_cfg.get("group", None) is not None:
        init_kwargs["group"] = wandb_cfg.get("group")
    if wandb_cfg.get("job_type", None) is not None:
        init_kwargs["job_type"] = wandb_cfg.get("job_type")
    if wandb_cfg.get("mode", None) is not None:
        init_kwargs["mode"] = wandb_cfg.get("mode")
    if wandb_cfg.get("notes", None) is not None:
        init_kwargs["notes"] = wandb_cfg.get("notes")
    if wandb_cfg.get("anonymous", None) is not None:
        init_kwargs["anonymous"] = wandb_cfg.get("anonymous")
    if wandb_cfg.get("tags", None) is not None:
        tags = wandb_cfg.get("tags")
        init_kwargs["tags"] = (
            list(tags) if isinstance(tags, (list, tuple, set)) else [str(tags)]
        )

    try:
        run = wandb.init(**init_kwargs)
        logging.info(
            "wandb logging enabled: project=%s, run=%s", project_name, run_name
        )
        return run
    except Exception as e:
        logging.warning("wandb init failed, continue without wandb logging: %s", e)
        return None


def maybe_finish_wandb(wandb_run: Optional[Any]) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as e:
        logging.warning("wandb finish failed: %s", e)


def random_collect(
    policy_cfg: "EasyDict",  # noqa
    policy: "Policy",  # noqa
    collector: "ISerialCollector",  # noqa
    collector_env: "BaseEnvManager",  # noqa
    commander: "BaseSerialCommander",  # noqa
    replay_buffer: "IBuffer",  # noqa
    postprocess_data_fn: Optional[Callable] = None,
) -> None:  # noqa
    assert policy_cfg.random_collect_size > 0
    if policy_cfg.get("transition_with_policy_data", False):
        collector.reset_policy(policy.collect_mode)
    else:
        action_space = collector_env.action_space
        random_policy = PolicyFactory.get_random_policy(
            policy.collect_mode, action_space=action_space
        )
        collector.reset_policy(random_policy)
    collect_kwargs = commander.step()
    if policy_cfg.collect.collector.type == "episode":
        new_data = collector.collect(
            n_episode=policy_cfg.random_collect_size, policy_kwargs=collect_kwargs
        )
    else:
        new_data = collector.collect(
            n_sample=policy_cfg.random_collect_size,
            random_collect=True,
            record_random_collect=False,
            policy_kwargs=collect_kwargs,
        )  # 'record_random_collect=False' means random collect without output log
    if postprocess_data_fn is not None:
        new_data = postprocess_data_fn(new_data)
    replay_buffer.push(new_data, cur_collector_envstep=0)
    collector.reset_policy(policy.collect_mode)
