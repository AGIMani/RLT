# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training updates for RL Token actor-critic learning.

The functions in this module consume the learner-facing batches produced by
``collate_rlt_batch`` and implement the RLT paper's update equations:

* critic TD backup over a C-step reward sequence, with a target critic;
* actor update that maximizes critic value while regularizing toward the stored
  VLA/human reference chunk;
* optional SwanLab metric logging.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from groot_rlt.networks import RLTActor, reference_regularization_loss

ActorQMode = Literal["q1", "min", "mean"]
MissingNextReferenceMode = Literal["zero", "current", "error"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RLTTrainConfig:
    """Hyperparameters for one RLT actor-critic update.

    ``gamma`` is intentionally explicit because the replay schema stores the
    C-step reward sequence as the source of truth. The scalar target reward is
    recomputed here as ``sum_i gamma**i * reward_chunk[i]``.
    """

    gamma: float
    actor_regularization_beta: float = 1.0
    actor_q_mode: ActorQMode = "q1"
    actor_update_deterministic: bool = False
    target_action_deterministic: bool = False
    target_next_reference_key: str = "next_actor_input_reference_chunk"
    missing_next_reference: MissingNextReferenceMode = "error"
    actor_grad_clip_norm: float | None = None
    critic_grad_clip_norm: float | None = None
    log_prefix: str = "rlt"

    def validate(self) -> None:
        _require(0.0 <= self.gamma < 1.0, "gamma must be in [0, 1)")
        _require(
            self.actor_regularization_beta >= 0.0,
            "actor_regularization_beta must be non-negative",
        )
        _require(
            self.actor_q_mode in ("q1", "min", "mean"),
            "actor_q_mode must be one of: q1, min, mean",
        )
        _require(
            self.missing_next_reference in ("zero", "current", "error"),
            "missing_next_reference must be one of: zero, current, error",
        )
        if self.actor_grad_clip_norm is not None:
            _require(self.actor_grad_clip_norm > 0.0, "actor_grad_clip_norm must be positive")
        if self.critic_grad_clip_norm is not None:
            _require(self.critic_grad_clip_norm > 0.0, "critic_grad_clip_norm must be positive")


def _infer_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is None:
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def _torch_batch(batch: Mapping[str, Any], *, device: torch.device) -> dict[str, Tensor]:
    float_keys = (
        "rl_token",
        "proprio",
        "next_rl_token",
        "next_proprio",
        "critic_action_chunk",
        "executed_action_chunk",
        "actor_input_reference_chunk",
        "regularizer_reference_chunk",
        "reward_chunk",
        "discounted_reward_sum",
        "discount",
        "next_actor_input_reference_chunk",
        "next_regularizer_reference_chunk",
    )
    bool_keys = (
        "valid_mask",
        "reference_valid_mask",
        "next_reference_valid_mask",
        "intervention_mask",
        "done",
        "reference_dropout_mask",
    )

    tensors: dict[str, Tensor] = {}
    for key in float_keys:
        if key in batch:
            tensors[key] = _as_tensor(batch[key], device=device, dtype=torch.float32)
    for key in bool_keys:
        if key in batch:
            tensors[key] = _as_tensor(batch[key], device=device, dtype=torch.bool)

    required = (
        "rl_token",
        "proprio",
        "next_rl_token",
        "next_proprio",
        "critic_action_chunk",
        "actor_input_reference_chunk",
        "regularizer_reference_chunk",
        "reward_chunk",
        "valid_mask",
        "discount",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise KeyError(f"RLT training batch is missing required tensor keys: {missing}")
    return tensors


def discounted_reward_sum_from_chunk(
    reward_chunk: Tensor,
    valid_mask: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Compute the C-step discounted reward sequence from Eq. 3."""

    _require(reward_chunk.ndim == 2, "reward_chunk must have shape [B, C]")
    _require(valid_mask.shape == reward_chunk.shape, "valid_mask/reward_chunk shape mismatch")
    _require(0.0 <= gamma < 1.0, "gamma must be in [0, 1)")
    weights = torch.pow(
        torch.as_tensor(gamma, device=reward_chunk.device, dtype=reward_chunk.dtype),
        torch.arange(reward_chunk.shape[1], device=reward_chunk.device, dtype=reward_chunk.dtype),
    )
    return (reward_chunk * valid_mask.to(dtype=reward_chunk.dtype) * weights).sum(dim=1)


def _critic_outputs(
    critic: nn.Module, rl_token: Tensor, proprio: Tensor, action: Tensor
) -> tuple[Tensor, ...]:
    output = critic(rl_token, proprio, action)
    if isinstance(output, tuple):
        return output
    return (output,)


def _min_critic_value(
    critic: nn.Module,
    rl_token: Tensor,
    proprio: Tensor,
    action: Tensor,
) -> Tensor:
    if hasattr(critic, "min_q"):
        return critic.min_q(rl_token, proprio, action)
    values = _critic_outputs(critic, rl_token, proprio, action)
    if len(values) == 1:
        return values[0]
    return torch.stack(values, dim=0).min(dim=0).values


def _select_actor_q(values: tuple[Tensor, ...], mode: ActorQMode) -> Tensor:
    if mode == "q1":
        return values[0]
    stacked = torch.stack(values, dim=0)
    if mode == "min":
        return stacked.min(dim=0).values
    if mode == "mean":
        return stacked.mean(dim=0)
    raise ValueError(f"unsupported actor_q_mode: {mode}")


def _target_next_reference(
    tensors: Mapping[str, Tensor],
    *,
    config: RLTTrainConfig,
) -> tuple[Tensor, bool]:
    if config.target_next_reference_key in tensors:
        return tensors[config.target_next_reference_key], False
    if "next_regularizer_reference_chunk" in tensors:
        return tensors["next_regularizer_reference_chunk"], False

    if config.missing_next_reference == "error":
        raise KeyError(
            f"missing next actor reference key {config.target_next_reference_key!r}; "
            "Eq. 3 bootstraps from x_next, so pass a reference chunk aligned to x_next "
            "or choose missing_next_reference='zero'/'current'"
        )
    if config.missing_next_reference == "current":
        return tensors["actor_input_reference_chunk"], True
    return torch.zeros_like(tensors["actor_input_reference_chunk"]), True


def _grad_norm(parameters: list[nn.Parameter]) -> Tensor:
    grads = [param.grad.detach().norm(2) for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.stack(grads).norm(2)


def _clip_or_measure_grad_norm(
    parameters: list[nn.Parameter],
    max_norm: float | None,
) -> float:
    if max_norm is not None:
        norm = nn.utils.clip_grad_norm_(parameters, max_norm)
        return float(norm.detach().cpu())
    return float(_grad_norm(parameters).detach().cpu())


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _mean(value: Tensor) -> float:
    return float(value.detach().mean().cpu())


def _std(value: Tensor) -> float:
    return float(value.detach().std(unbiased=False).cpu())


def _fraction(mask: Tensor) -> float:
    return float(mask.detach().to(dtype=torch.float32).mean().cpu())


def _prefix(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _log_swanlab(
    swanlab_run: Any | None,
    metrics: Mapping[str, float],
    *,
    step: int | None,
) -> None:
    if swanlab_run is None:
        return
    log = getattr(swanlab_run, "log", None)
    if log is None:
        raise TypeError("swanlab_run must expose a log(metrics, step=...) method")
    if step is None:
        log(dict(metrics))
    else:
        log(dict(metrics), step=step)


@contextmanager
def _module_modes(*modules: tuple[nn.Module, bool]):
    previous = [module.training for module, _ in modules]
    for module, training in modules:
        module.train(training)
    try:
        yield
    finally:
        for (module, _), training in zip(modules, previous):
            module.train(training)


@contextmanager
def _temporarily_freeze(module: nn.Module):
    parameters = list(module.parameters())
    requires_grad = [param.requires_grad for param in parameters]
    for param in parameters:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, value in zip(parameters, requires_grad):
            param.requires_grad_(value)


def update_critic(
    *,
    actor: RLTActor,
    critic: nn.Module,
    target_critic: nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    config: RLTTrainConfig,
    device: torch.device | str | None = None,
    swanlab_run: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Run one Eq. 3 critic update and optionally log metrics to SwanLab."""

    config.validate()
    update_device = torch.device(device) if device is not None else _infer_device(critic)
    tensors = _torch_batch(batch, device=update_device)

    with _module_modes((actor, False), (critic, True), (target_critic, False)):
        with torch.no_grad():
            next_reference, next_reference_missing = _target_next_reference(tensors, config=config)
            next_action, _ = actor.sample_action(
                tensors["next_rl_token"],
                tensors["next_proprio"],
                next_reference,
                deterministic=config.target_action_deterministic,
            )
            if "next_reference_valid_mask" in tensors:
                next_action = next_action * tensors["next_reference_valid_mask"].to(
                    dtype=next_action.dtype
                ).unsqueeze(-1)
            target_q = _min_critic_value(
                target_critic,
                tensors["next_rl_token"],
                tensors["next_proprio"],
                next_action,
            )
            reward_sum = discounted_reward_sum_from_chunk(
                tensors["reward_chunk"],
                tensors["valid_mask"],
                gamma=config.gamma,
            )
            target = reward_sum + tensors["discount"] * target_q

        q_values = _critic_outputs(
            critic,
            tensors["rl_token"],
            tensors["proprio"],
            tensors["critic_action_chunk"],
        )
        losses = [F.mse_loss(q, target) for q in q_values]
        critic_loss = torch.stack(losses).sum()

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = [param for param in critic.parameters() if param.requires_grad]
        grad_norm = _clip_or_measure_grad_norm(critic_parameters, config.critic_grad_clip_norm)
        critic_optimizer.step()

    done = tensors.get("done")
    reference_dropout = tensors.get("reference_dropout_mask")
    discounted_reward_sum = tensors.get("discounted_reward_sum")
    metrics = {
        "critic/loss": _mean(critic_loss),
        "critic/target_mean": _mean(target),
        "critic/target_std": _std(target),
        "critic/target_q_mean": _mean(target_q),
        "critic/reward_sum_mean": _mean(reward_sum),
        "critic/discount_mean": _mean(tensors["discount"]),
        "critic/grad_norm": grad_norm,
        "critic/lr": _optimizer_lr(critic_optimizer),
        "batch/valid_fraction": _fraction(tensors["valid_mask"]),
        "batch/reward_nonzero_fraction": _fraction(tensors["reward_chunk"] != 0.0),
        "target/next_reference_missing": 1.0 if next_reference_missing else 0.0,
    }
    for index, (q_value, q_loss) in enumerate(zip(q_values, losses), start=1):
        metrics[f"critic/q{index}_mean"] = _mean(q_value)
        metrics[f"critic/q{index}_std"] = _std(q_value)
        metrics[f"critic/q{index}_loss"] = _mean(q_loss)
    if done is not None:
        metrics["batch/done_fraction"] = _fraction(done)
    if reference_dropout is not None:
        metrics["batch/reference_dropout_fraction"] = _fraction(reference_dropout)
    if discounted_reward_sum is not None:
        metrics["critic/reward_sum_stored_abs_diff_mean"] = _mean(
            (reward_sum - discounted_reward_sum).abs()
        )

    prefixed = _prefix(config.log_prefix, metrics)
    _log_swanlab(swanlab_run, prefixed, step=step)
    return prefixed


def update_actor(
    *,
    actor: RLTActor,
    critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    config: RLTTrainConfig,
    device: torch.device | str | None = None,
    swanlab_run: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Run one Eq. 5 actor update and optionally log metrics to SwanLab."""

    config.validate()
    update_device = torch.device(device) if device is not None else _infer_device(actor)
    tensors = _torch_batch(batch, device=update_device)

    with _module_modes((actor, True), (critic, False)):
        distribution = actor(
            tensors["rl_token"],
            tensors["proprio"],
            tensors["actor_input_reference_chunk"],
        )
        if config.actor_update_deterministic:
            action = distribution.mean
            log_prob = None
        else:
            action = distribution.rsample()
            log_prob = distribution.log_prob(action)

        valid_reference_mask = tensors["valid_mask"]
        if "reference_valid_mask" in tensors:
            valid_reference_mask = valid_reference_mask & tensors["reference_valid_mask"]

        with _temporarily_freeze(critic):
            q_values = _critic_outputs(critic, tensors["rl_token"], tensors["proprio"], action)
            actor_q = _select_actor_q(q_values, config.actor_q_mode)
            value_loss = -actor_q.mean()

        regularization_loss = reference_regularization_loss(
            action,
            tensors["regularizer_reference_chunk"],
            valid_mask=valid_reference_mask,
        )
        actor_loss = value_loss + config.actor_regularization_beta * regularization_loss

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_parameters = [param for param in actor.parameters() if param.requires_grad]
        grad_norm = _clip_or_measure_grad_norm(actor_parameters, config.actor_grad_clip_norm)
        actor_optimizer.step()

    reference_error = (action - tensors["regularizer_reference_chunk"]).pow(2).sum(dim=-1)
    has_valid_reference = bool(valid_reference_mask.any().detach().cpu())
    metrics = {
        "actor/loss": _mean(actor_loss),
        "actor/value_loss": _mean(value_loss),
        "actor/reference_regularization_loss": _mean(regularization_loss),
        "actor/reference_beta": float(config.actor_regularization_beta),
        "actor/q_mean": _mean(actor_q),
        "actor/action_mean": _mean(action),
        "actor/action_std": _std(action),
        "actor/reference_mse_per_valid_step": _mean(
            reference_error[valid_reference_mask].mean()
            if has_valid_reference
            else reference_error.mean()
        ),
        "actor/grad_norm": grad_norm,
        "actor/lr": _optimizer_lr(actor_optimizer),
        "batch/valid_fraction": _fraction(tensors["valid_mask"]),
    }
    if log_prob is not None:
        metrics["actor/log_prob_mean"] = _mean(log_prob)
    if "reference_dropout_mask" in tensors:
        metrics["batch/reference_dropout_fraction"] = _fraction(tensors["reference_dropout_mask"])
    for index, q_value in enumerate(q_values, start=1):
        metrics[f"actor/q{index}_mean"] = _mean(q_value)

    prefixed = _prefix(config.log_prefix, metrics)
    _log_swanlab(swanlab_run, prefixed, step=step)
    return prefixed


def soft_update_target_network(
    *,
    source: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    """Polyak-update a target network in-place."""

    _require(0.0 <= tau <= 1.0, "tau must be in [0, 1]")
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)
