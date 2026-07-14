# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight actor-critic networks for RL Token training.

These modules mirror the RLT paper's learner interface:

* actor: ``pi_theta(a_1:C | x, reference_1:C)``;
* critic: ``Q_psi(x, a_1:C)``;
* double critic: TD3-style two-Q ensemble, using the minimum target value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _activation(name: str) -> type[nn.Module]:
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"unsupported activation: {name}")


def _mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    _require(input_dim > 0, "input_dim must be positive")
    _require(output_dim > 0, "output_dim must be positive")
    _require(hidden_dim > 0, "hidden_dim must be positive")
    _require(num_hidden_layers > 0, "num_hidden_layers must be positive")

    activation_cls = _activation(activation)
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_cls())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _check_rank(name: str, value: Tensor, rank: int) -> None:
    _require(value.ndim == rank, f"{name} must be rank {rank}, got shape {tuple(value.shape)}")


def _check_last_dim(name: str, value: Tensor, dim: int) -> None:
    _require(
        value.shape[-1] == dim,
        f"{name} last dim must be {dim}, got shape {tuple(value.shape)}",
    )


def _check_chunk(name: str, value: Tensor, *, chunk_len: int, action_dim: int) -> None:
    _check_rank(name, value, 3)
    _require(
        value.shape[1:] == (chunk_len, action_dim),
        f"{name} must have shape [B, {chunk_len}, {action_dim}], got {tuple(value.shape)}",
    )


def _state_features(rl_token: Tensor, proprio: Tensor) -> Tensor:
    _check_rank("rl_token", rl_token, 2)
    _check_rank("proprio", proprio, 2)
    _require(rl_token.shape[0] == proprio.shape[0], "rl_token/proprio batch size mismatch")
    return torch.cat([rl_token, proprio], dim=-1)


@dataclass(frozen=True)
class RLTNetworkConfig:
    """Dimensions and MLP defaults used by RLT actor/critic networks."""

    rl_token_dim: int
    proprio_dim: int
    action_dim: int
    chunk_len: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    activation: str = "relu"
    fixed_std: float = 0.1
    action_limit: float | None = None
    residual_reference: bool = True

    @property
    def state_dim(self) -> int:
        return self.rl_token_dim + self.proprio_dim

    @property
    def action_chunk_dim(self) -> int:
        return self.chunk_len * self.action_dim

    def validate(self) -> None:
        _require(self.rl_token_dim > 0, "rl_token_dim must be positive")
        _require(self.proprio_dim > 0, "proprio_dim must be positive")
        _require(self.action_dim > 0, "action_dim must be positive")
        _require(self.chunk_len > 0, "chunk_len must be positive")
        _require(self.hidden_dim > 0, "hidden_dim must be positive")
        _require(self.num_hidden_layers > 0, "num_hidden_layers must be positive")
        _require(self.fixed_std > 0.0, "fixed_std must be positive")
        if self.action_limit is not None:
            _require(self.action_limit > 0.0, "action_limit must be positive")
        _activation(self.activation)


class RLTActor(nn.Module):
    """Gaussian chunk actor ``pi_theta(a_1:C | x, reference_1:C)``.

    By default the mean is parameterized as ``reference + residual``. This keeps
    the network's initial behavior near the VLA reference while still matching
    the paper's Gaussian actor form ``N(mu_theta(x, reference), sigma^2 I)``.
    """

    def __init__(self, config: RLTNetworkConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.net = _mlp(
            input_dim=config.state_dim + config.action_chunk_dim,
            output_dim=config.action_chunk_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
        )
        self.register_buffer(
            "_fixed_std",
            torch.tensor(float(config.fixed_std), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        rl_token: Tensor,
        proprio: Tensor,
        reference_action_chunk: Tensor,
    ) -> Independent:
        mean = self.mean(rl_token, proprio, reference_action_chunk)
        std = torch.ones_like(mean) * self._fixed_std.to(device=mean.device, dtype=mean.dtype)
        return Independent(Normal(mean, std), reinterpreted_batch_ndims=2)

    def mean(
        self,
        rl_token: Tensor,
        proprio: Tensor,
        reference_action_chunk: Tensor,
    ) -> Tensor:
        state = _state_features(rl_token, proprio)
        _check_last_dim("rl_token", rl_token, self.config.rl_token_dim)
        _check_last_dim("proprio", proprio, self.config.proprio_dim)
        _check_chunk(
            "reference_action_chunk",
            reference_action_chunk,
            chunk_len=self.config.chunk_len,
            action_dim=self.config.action_dim,
        )
        _require(
            reference_action_chunk.shape[0] == state.shape[0],
            "reference_action_chunk batch size mismatch",
        )

        flat_reference = reference_action_chunk.flatten(start_dim=1)
        raw = self.net(torch.cat([state, flat_reference], dim=-1))
        raw = raw.view(-1, self.config.chunk_len, self.config.action_dim)
        mean = reference_action_chunk + raw if self.config.residual_reference else raw
        if self.config.action_limit is not None:
            mean = torch.tanh(mean) * float(self.config.action_limit)
        return mean

    def sample_action(
        self,
        rl_token: Tensor,
        proprio: Tensor,
        reference_action_chunk: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        distribution = self(rl_token, proprio, reference_action_chunk)
        if deterministic:
            return distribution.mean, None
        action = distribution.rsample()
        return action, distribution.log_prob(action)

    def forward_batch(
        self,
        batch: dict[str, Tensor],
        *,
        reference_key: str = "actor_input_reference_chunk",
    ) -> Independent:
        return self(batch["rl_token"], batch["proprio"], batch[reference_key])

    def sample_batch(
        self,
        batch: dict[str, Tensor],
        *,
        reference_key: str = "actor_input_reference_chunk",
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        return self.sample_action(
            batch["rl_token"],
            batch["proprio"],
            batch[reference_key],
            deterministic=deterministic,
        )


class RLTCritic(nn.Module):
    """Chunk critic ``Q_psi(x, a_1:C) -> scalar``."""

    def __init__(self, config: RLTNetworkConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.net = _mlp(
            input_dim=config.state_dim + config.action_chunk_dim,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
        )

    def forward(self, rl_token: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        state = _state_features(rl_token, proprio)
        _check_last_dim("rl_token", rl_token, self.config.rl_token_dim)
        _check_last_dim("proprio", proprio, self.config.proprio_dim)
        _check_chunk(
            "action_chunk",
            action_chunk,
            chunk_len=self.config.chunk_len,
            action_dim=self.config.action_dim,
        )
        _require(action_chunk.shape[0] == state.shape[0], "action_chunk batch size mismatch")
        q = self.net(torch.cat([state, action_chunk.flatten(start_dim=1)], dim=-1))
        return q.squeeze(-1)

    def forward_batch(
        self,
        batch: dict[str, Tensor],
        *,
        action_key: str = "critic_action_chunk",
        state_prefix: str = "",
    ) -> Tensor:
        rl_key = f"{state_prefix}rl_token"
        proprio_key = f"{state_prefix}proprio"
        return self(batch[rl_key], batch[proprio_key], batch[action_key])


class RLTDoubleCritic(nn.Module):
    """TD3-style ensemble of two independent RLT critics."""

    def __init__(self, config: RLTNetworkConfig) -> None:
        super().__init__()
        self.q1 = RLTCritic(config)
        self.q2 = RLTCritic(config)

    def forward(
        self, rl_token: Tensor, proprio: Tensor, action_chunk: Tensor
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(rl_token, proprio, action_chunk),
            self.q2(rl_token, proprio, action_chunk),
        )

    def min_q(self, rl_token: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        q1, q2 = self(rl_token, proprio, action_chunk)
        return torch.minimum(q1, q2)

    def forward_batch(
        self,
        batch: dict[str, Tensor],
        *,
        action_key: str = "critic_action_chunk",
        state_prefix: str = "",
    ) -> tuple[Tensor, Tensor]:
        rl_key = f"{state_prefix}rl_token"
        proprio_key = f"{state_prefix}proprio"
        return self(batch[rl_key], batch[proprio_key], batch[action_key])

    def min_q_batch(
        self,
        batch: dict[str, Tensor],
        *,
        action_key: str = "critic_action_chunk",
        state_prefix: str = "",
    ) -> Tensor:
        q1, q2 = self.forward_batch(batch, action_key=action_key, state_prefix=state_prefix)
        return torch.minimum(q1, q2)


def reference_regularization_loss(
    action_chunk: Tensor,
    reference_action_chunk: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Batch mean of the Eq. 5 chunk squared norm to the stored reference."""

    _check_rank("action_chunk", action_chunk, 3)
    _check_rank("reference_action_chunk", reference_action_chunk, 3)
    _require(action_chunk.shape == reference_action_chunk.shape, "action/reference shape mismatch")
    per_step_error = (action_chunk - reference_action_chunk).pow(2).sum(dim=-1)
    if valid_mask is None:
        return per_step_error.sum(dim=1).mean()
    _require(valid_mask.shape == per_step_error.shape, "valid_mask shape mismatch")
    weights = valid_mask.to(device=per_step_error.device, dtype=per_step_error.dtype)
    return (per_step_error * weights).sum(dim=1).mean()


def td3_min_target(q1: Tensor, q2: Tensor) -> Tensor:
    """Return the TD3-style minimum of two target Q estimates."""

    _require(q1.shape == q2.shape, "q1/q2 shape mismatch")
    return torch.minimum(q1, q2)


def make_rlt_networks(
    config: RLTNetworkConfig,
    *,
    target_critic: bool = True,
) -> dict[str, nn.Module]:
    """Convenience constructor for the RLT actor, critic, and optional target critic."""

    networks: dict[str, nn.Module] = {
        "actor": RLTActor(config),
        "critic": RLTDoubleCritic(config),
    }
    if target_critic:
        networks["target_critic"] = RLTDoubleCritic(config)
    return networks


def gaussian_entropy_per_sample(distribution: Independent) -> Tensor:
    """Return entropy per batch item for the chunk Gaussian policy."""

    entropy = distribution.entropy()
    if entropy.ndim == 0:
        return entropy.unsqueeze(0)
    return entropy


def fixed_std_log_prob_normalizer(*, chunk_len: int, action_dim: int, fixed_std: float) -> float:
    """Analytic log-density normalizer for diagnostics with fixed-std Gaussian actors."""

    _require(chunk_len > 0, "chunk_len must be positive")
    _require(action_dim > 0, "action_dim must be positive")
    _require(fixed_std > 0.0, "fixed_std must be positive")
    event_dim = chunk_len * action_dim
    return -0.5 * event_dim * math.log(2.0 * math.pi * fixed_std * fixed_std)
