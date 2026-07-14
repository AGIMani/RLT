# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from groot_rlt.networks import RLTActor, RLTDoubleCritic, RLTNetworkConfig
from groot_rlt.replay_buffer import RLTReplayBuffer
from groot_rlt.replay_schema import (
    BehaviorSource,
    CollectionStage,
    CriticalPhaseSegment,
    ReferenceSource,
    RLTTransition,
    TerminalOutcome,
)
from groot_rlt.train import RLTTrainConfig
from groot_rlt.trainer import RLTTrainer, RLTTrainerConfig, RLTTrainerRollout

GAMMA = 0.9


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


def _segment() -> CriticalPhaseSegment:
    return CriticalPhaseSegment(
        segment_id="segment-0",
        episode_id="episode-0",
        task_id="insert",
        episode_start_t=0,
        base_vla_prefix_end_t=0,
        handoff_t=0,
        terminal_t=100,
        terminal_outcome=TerminalOutcome.SUCCESS,
    )


def _chunk(value: float) -> np.ndarray:
    config = _network_config()
    return np.full((config.chunk_len, config.action_dim), value, dtype=np.float32)


def _transition(transition_id: str, *, sample_t: int, value: float) -> RLTTransition:
    segment = _segment()
    return RLTTransition(
        transition_id=transition_id,
        segment_id=segment.segment_id,
        episode_id=segment.episode_id,
        collection_stage=CollectionStage.VLA_WARMUP,
        decision_t=sample_t,
        sample_t=sample_t,
        next_t=sample_t + 4,
        chunk_offset=0,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key=f"obs/{sample_t}",
        next_obs_key=f"obs/{sample_t + 4}",
        rl_token_key=f"rl_token/{sample_t}",
        proprio_key=f"proprio/{sample_t}",
        next_rl_token_key=f"rl_token/{sample_t + 4}",
        next_proprio_key=f"proprio/{sample_t + 4}",
        executed_action_chunk=_chunk(value),
        train_reference_chunk=_chunk(value),
        next_train_reference_chunk=_chunk(value + 0.5),
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        next_reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=_chunk(value),
        behavior_source=BehaviorSource.VLA,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )


def _tensor_store(transitions: list[RLTTransition]) -> dict[str, np.ndarray]:
    config = _network_config()
    store = {}
    for transition in transitions:
        for key in (transition.rl_token_key, transition.next_rl_token_key):
            store[key] = np.full(config.rl_token_dim, float(len(store) + 1), dtype=np.float32)
        for key in (transition.proprio_key, transition.next_proprio_key):
            store[key] = np.full(config.proprio_dim, float(len(store) + 1), dtype=np.float32)
    return store


def _trainer(tmp_path, *, swanlab_run=None):
    torch.manual_seed(0)
    config = _network_config()
    actor = RLTActor(config)
    critic = RLTDoubleCritic(config)
    target_critic = RLTDoubleCritic(config)
    target_critic.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
    replay = RLTReplayBuffer(seed=0)
    warmup = [
        _transition("warmup-0", sample_t=0, value=1.0),
        _transition("warmup-1", sample_t=4, value=2.0),
    ]
    replay.extend(warmup, segment=_segment(), gamma=GAMMA)
    trainer = RLTTrainer(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        replay_buffer=replay,
        train_config=RLTTrainConfig(gamma=GAMMA, actor_update_deterministic=True),
        trainer_config=RLTTrainerConfig(
            batch_size=2,
            update_to_data_ratio=5,
            critic_updates_per_actor_update=2,
            target_update_tau=0.1,
            reference_dropout_prob=0.5,
            checkpoint_dir=tmp_path,
        ),
        tensor_resolver=_tensor_store(warmup),
        swanlab_run=swanlab_run,
    )
    return trainer, warmup


def test_trainer_alternates_rollout_and_learning_with_paper_ratios(tmp_path):
    run = _SwanLabRun()
    trainer, warmup = _trainer(tmp_path, swanlab_run=run)
    rollout_transitions = [
        _transition("rollout-0", sample_t=8, value=3.0),
        _transition("rollout-1", sample_t=12, value=4.0),
    ]
    trainer.tensor_resolver = _tensor_store(warmup + rollout_transitions)

    result = trainer.train_iteration(
        lambda _: RLTTrainerRollout.from_transitions(
            rollout_transitions,
            update_units=1,
            segment=_segment(),
            metrics={"success": 1.0},
        )
    )

    assert len(trainer.replay_buffer) == 4
    assert trainer.state.rollout_iterations == 1
    assert trainer.state.learner_updates == 5
    assert trainer.state.critic_updates == 5
    assert trainer.state.actor_updates == 2
    assert trainer.state.target_updates == 2
    assert result["summary"]["rlt/trainer/requested_learner_updates"] == 5.0
    assert result["summary"]["rlt/rollout/success"] == 1.0
    assert len(result["learner_metrics"]) == 5
    assert any("rlt/trainer/update_to_data_ratio" in metrics for metrics, _ in run.logs)


def test_trainer_checkpoint_roundtrip_restores_state_and_replay(tmp_path):
    trainer, _ = _trainer(tmp_path)
    trainer.learn(2)

    checkpoint_path = trainer.save_checkpoint()

    restored, _ = _trainer(tmp_path)
    restored.replay_buffer.clear()
    restored.load_checkpoint(checkpoint_path)

    assert restored.state == trainer.state
    assert len(restored.replay_buffer) == len(trainer.replay_buffer)
    assert restored.replay_buffer.ids() == trainer.replay_buffer.ids()
