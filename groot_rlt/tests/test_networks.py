# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from groot_rlt.networks import (
    RLTActor,
    RLTDoubleCritic,
    RLTNetworkConfig,
    fixed_std_log_prob_normalizer,
    make_rlt_networks,
    reference_regularization_loss,
    td3_min_target,
)


def _config() -> RLTNetworkConfig:
    return RLTNetworkConfig(
        rl_token_dim=5,
        proprio_dim=3,
        action_dim=2,
        chunk_len=4,
        hidden_dim=16,
        num_hidden_layers=2,
        fixed_std=0.2,
    )


def _inputs(batch_size: int = 3):
    config = _config()
    return (
        torch.randn(batch_size, config.rl_token_dim),
        torch.randn(batch_size, config.proprio_dim),
        torch.randn(batch_size, config.chunk_len, config.action_dim),
    )


def test_rlt_actor_outputs_fixed_std_chunk_distribution():
    config = _config()
    actor = RLTActor(config)
    rl_token, proprio, reference = _inputs()

    distribution = actor(rl_token, proprio, reference)
    action, log_prob = actor.sample_action(rl_token, proprio, reference)

    assert distribution.mean.shape == (3, config.chunk_len, config.action_dim)
    assert action.shape == (3, config.chunk_len, config.action_dim)
    assert log_prob.shape == (3,)
    torch.testing.assert_close(distribution.stddev, torch.full_like(distribution.stddev, 0.2))


def test_rlt_actor_can_run_deterministically_from_batch():
    config = _config()
    actor = RLTActor(config)
    rl_token, proprio, reference = _inputs(batch_size=2)
    batch = {
        "rl_token": rl_token,
        "proprio": proprio,
        "actor_input_reference_chunk": reference,
    }

    action, log_prob = actor.sample_batch(batch, deterministic=True)

    assert action.shape == (2, config.chunk_len, config.action_dim)
    assert log_prob is None
    torch.testing.assert_close(action, actor.mean(rl_token, proprio, reference))


def test_double_critic_returns_two_values_and_min_q():
    config = _config()
    critic = RLTDoubleCritic(config)
    rl_token, proprio, action_chunk = _inputs()

    q1, q2 = critic(rl_token, proprio, action_chunk)
    min_q = critic.min_q(rl_token, proprio, action_chunk)

    assert q1.shape == (3,)
    assert q2.shape == (3,)
    torch.testing.assert_close(min_q, torch.minimum(q1, q2))


def test_double_critic_forward_batch_uses_collated_keys():
    config = _config()
    critic = RLTDoubleCritic(config)
    rl_token, proprio, action_chunk = _inputs(batch_size=2)
    batch = {
        "rl_token": rl_token,
        "proprio": proprio,
        "critic_action_chunk": action_chunk,
    }

    q1, q2 = critic.forward_batch(batch)

    assert q1.shape == (2,)
    assert q2.shape == (2,)


def test_reference_regularization_loss_respects_valid_mask():
    action = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]],
        dtype=torch.float32,
    )
    reference = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[True, True, False]])

    loss = reference_regularization_loss(action, reference, valid_mask=valid_mask)

    assert loss.item() == pytest.approx(2.0 + 8.0)


def test_td3_min_target_and_network_factory():
    q1 = torch.tensor([1.0, 3.0])
    q2 = torch.tensor([2.0, 1.0])
    config = _config()

    networks = make_rlt_networks(config)

    torch.testing.assert_close(td3_min_target(q1, q2), torch.tensor([1.0, 1.0]))
    assert set(networks) == {"actor", "critic", "target_critic"}


def test_fixed_std_log_prob_normalizer_matches_event_dim():
    value = fixed_std_log_prob_normalizer(chunk_len=4, action_dim=2, fixed_std=0.2)
    expected = -0.5 * 8 * math.log(2.0 * math.pi * 0.2 * 0.2)

    assert isinstance(value, float)
    assert value == pytest.approx(expected)
