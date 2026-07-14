# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from groot_rlt.networks import RLTActor, RLTDoubleCritic, RLTNetworkConfig
from groot_rlt.train import (
    RLTTrainConfig,
    discounted_reward_sum_from_chunk,
    soft_update_target_network,
    update_actor,
    update_critic,
)


class _SwanLabRun:
    def __init__(self) -> None:
        self.logs = []

    def log(self, metrics, step=None):
        self.logs.append((dict(metrics), step))


def _network_config() -> RLTNetworkConfig:
    return RLTNetworkConfig(
        rl_token_dim=5,
        proprio_dim=3,
        action_dim=2,
        chunk_len=4,
        hidden_dim=16,
        num_hidden_layers=2,
        fixed_std=0.05,
    )


def _batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    config = _network_config()
    return {
        "rl_token": torch.randn(batch_size, config.rl_token_dim),
        "proprio": torch.randn(batch_size, config.proprio_dim),
        "next_rl_token": torch.randn(batch_size, config.rl_token_dim),
        "next_proprio": torch.randn(batch_size, config.proprio_dim),
        "critic_action_chunk": torch.randn(batch_size, config.chunk_len, config.action_dim),
        "actor_input_reference_chunk": torch.zeros(batch_size, config.chunk_len, config.action_dim),
        "regularizer_reference_chunk": torch.ones(batch_size, config.chunk_len, config.action_dim),
        "next_actor_input_reference_chunk": torch.full(
            (batch_size, config.chunk_len, config.action_dim),
            0.5,
        ),
        "next_regularizer_reference_chunk": torch.full(
            (batch_size, config.chunk_len, config.action_dim),
            0.5,
        ),
        "reward_chunk": torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "discounted_reward_sum": torch.tensor([0.81, 0.45, 0.0], dtype=torch.float32),
        "discount": torch.tensor([0.0, 0.9**4, 0.9**4], dtype=torch.float32),
        "done": torch.tensor([True, False, False]),
        "valid_mask": torch.tensor(
            [[True, True, True, False], [True, True, True, True], [True, True, True, True]]
        ),
        "reference_valid_mask": torch.tensor(
            [[True, True, True, False], [True, True, True, True], [True, True, True, True]]
        ),
        "next_reference_valid_mask": torch.tensor(
            [[False, False, False, False], [True, True, True, True], [True, True, True, True]]
        ),
        "reference_dropout_mask": torch.tensor([True, False, False]),
    }


def _zero_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


class _OnesActor(torch.nn.Module):
    def sample_action(self, rl_token, proprio, reference_action_chunk, *, deterministic=False):
        del rl_token, proprio, deterministic
        return torch.ones_like(reference_action_chunk), None


class _CaptureTargetCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_chunk = None

    def min_q(self, rl_token, proprio, action_chunk):
        del rl_token, proprio
        self.action_chunk = action_chunk.detach().clone()
        return torch.zeros(action_chunk.shape[0], device=action_chunk.device)


def test_discounted_reward_sum_from_chunk_uses_c_step_sequence_and_valid_mask():
    reward_chunk = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])

    reward_sum = discounted_reward_sum_from_chunk(reward_chunk, valid_mask, gamma=0.5)

    torch.testing.assert_close(reward_sum, torch.tensor([0.5, 0.25]))


def test_update_critic_updates_parameters_and_logs_to_swanlab():
    torch.manual_seed(0)
    net_config = _network_config()
    actor = RLTActor(net_config)
    critic = RLTDoubleCritic(net_config)
    target_critic = RLTDoubleCritic(net_config)
    target_critic.load_state_dict(critic.state_dict())
    optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
    run = _SwanLabRun()

    before = [parameter.detach().clone() for parameter in critic.parameters()]
    metrics = update_critic(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        critic_optimizer=optimizer,
        batch=_batch(),
        config=RLTTrainConfig(gamma=0.9),
        swanlab_run=run,
        step=7,
    )

    after = list(critic.parameters())
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert metrics["rlt/critic/loss"] > 0.0
    assert metrics["rlt/critic/reward_sum_stored_abs_diff_mean"] == pytest.approx(0.0, abs=1.0e-6)
    assert metrics["rlt/target/next_reference_missing"] == 0.0
    assert run.logs[-1][1] == 7
    assert "rlt/critic/loss" in run.logs[-1][0]


def test_update_critic_requires_next_reference_by_default_for_strict_paper_alignment():
    net_config = _network_config()
    actor = RLTActor(net_config)
    critic = RLTDoubleCritic(net_config)
    target_critic = RLTDoubleCritic(net_config)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
    batch = _batch()
    del batch["next_actor_input_reference_chunk"]
    del batch["next_regularizer_reference_chunk"]

    with pytest.raises(KeyError, match="missing next actor reference"):
        update_critic(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            critic_optimizer=optimizer,
            batch=batch,
            config=RLTTrainConfig(gamma=0.9),
        )


def test_update_critic_masks_target_actor_action_padding_tail():
    net_config = _network_config()
    actor = _OnesActor()
    critic = RLTDoubleCritic(net_config)
    target_critic = _CaptureTargetCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
    batch = _batch()
    batch["next_reference_valid_mask"] = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    update_critic(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        critic_optimizer=optimizer,
        batch=batch,
        config=RLTTrainConfig(gamma=0.9),
    )

    assert target_critic.action_chunk is not None
    torch.testing.assert_close(
        target_critic.action_chunk,
        batch["next_reference_valid_mask"].to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 2),
    )


def test_update_actor_uses_regularizer_reference_not_dropped_actor_input():
    torch.manual_seed(0)
    net_config = _network_config()
    actor = RLTActor(net_config)
    critic = RLTDoubleCritic(net_config)
    _zero_module(actor)
    _zero_module(critic)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    run = _SwanLabRun()

    metrics = update_actor(
        actor=actor,
        critic=critic,
        actor_optimizer=optimizer,
        batch=_batch(),
        config=RLTTrainConfig(
            gamma=0.9,
            actor_regularization_beta=2.0,
            actor_update_deterministic=True,
        ),
        swanlab_run=run,
        step=8,
    )

    assert metrics["rlt/actor/reference_regularization_loss"] == pytest.approx(22.0 / 3.0)
    assert metrics["rlt/actor/loss"] == pytest.approx(44.0 / 3.0)
    assert metrics["rlt/batch/reference_dropout_fraction"] == pytest.approx(1.0 / 3.0)
    assert run.logs[-1][1] == 8


def test_soft_update_target_network_polyak_updates_parameters():
    net_config = _network_config()
    source = RLTDoubleCritic(net_config)
    target = RLTDoubleCritic(net_config)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(2.0)
        for parameter in target.parameters():
            parameter.fill_(0.0)

    soft_update_target_network(source=source, target=target, tau=0.25)

    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
