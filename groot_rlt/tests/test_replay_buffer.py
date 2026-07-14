# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from groot_rlt.collate import collate_rlt_batch
from groot_rlt.replay_buffer import RLTReplayBuffer
from groot_rlt.replay_schema import (
    BehaviorSource,
    CollectionStage,
    CriticalPhaseSegment,
    ReferenceSource,
    RLTTransition,
    TerminalOutcome,
)

GAMMA = 0.99


def _segment() -> CriticalPhaseSegment:
    return CriticalPhaseSegment(
        segment_id="segment-0",
        episode_id="episode-0",
        task_id="insert",
        episode_start_t=0,
        base_vla_prefix_end_t=0,
        handoff_t=0,
        terminal_t=20,
        terminal_outcome=TerminalOutcome.SUCCESS,
    )


def _chunk(value: float, *, chunk_len: int = 4, action_dim: int = 3) -> np.ndarray:
    return np.full((chunk_len, action_dim), value, dtype=np.float32)


def _transition(
    transition_id: str,
    *,
    sample_t: int,
    value: float,
    collection_stage: CollectionStage = CollectionStage.ONLINE_RL,
) -> RLTTransition:
    segment = _segment()
    vla_reference = _chunk(value)
    if collection_stage is CollectionStage.VLA_WARMUP:
        executed = vla_reference
        actor = None
        behavior_source = BehaviorSource.VLA
    else:
        executed = _chunk(value + 10.0)
        actor = executed
        behavior_source = BehaviorSource.ACTOR

    return RLTTransition(
        transition_id=transition_id,
        segment_id=segment.segment_id,
        episode_id=segment.episode_id,
        collection_stage=collection_stage,
        decision_t=sample_t,
        sample_t=sample_t,
        next_t=sample_t + 4,
        chunk_offset=0,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key=f"obs/{sample_t}",
        next_obs_key=f"obs/{sample_t + 4}",
        rl_token_key=f"z/{sample_t}",
        proprio_key=f"proprio/{sample_t}",
        next_rl_token_key=f"z/{sample_t + 4}",
        next_proprio_key=f"proprio/{sample_t + 4}",
        executed_action_chunk=executed,
        train_reference_chunk=vla_reference,
        next_train_reference_chunk=vla_reference,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_reference,
        actor_proposed_chunk=actor,
        behavior_source=behavior_source,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )


def test_replay_buffer_add_validates_against_segment_when_requested():
    segment = _segment()
    transition = _transition("tr-0", sample_t=0, value=1.0)
    buffer = RLTReplayBuffer(seed=0)

    evicted = buffer.add(transition, segment=segment, gamma=GAMMA)

    assert evicted is None
    assert len(buffer) == 1
    assert buffer.get("tr-0") is transition


def test_replay_buffer_rejects_duplicate_ids_unless_replacing():
    buffer = RLTReplayBuffer(seed=0)
    transition = _transition("tr-0", sample_t=0, value=1.0)
    buffer.add(transition)

    with pytest.raises(ValueError, match="duplicate transition_id"):
        buffer.add(transition)

    replacement = _transition("tr-0", sample_t=4, value=2.0)
    buffer.add(replacement, replace=True)
    assert len(buffer) == 1
    assert buffer.get("tr-0") is replacement


def test_replay_buffer_capacity_evicts_oldest_transition():
    buffer = RLTReplayBuffer(capacity=2, seed=0)
    tr0 = _transition("tr-0", sample_t=0, value=1.0)
    tr1 = _transition("tr-1", sample_t=4, value=2.0)
    tr2 = _transition("tr-2", sample_t=8, value=3.0)

    assert buffer.add(tr0) is None
    assert buffer.add(tr1) is None
    evicted = buffer.add(tr2)

    assert evicted is tr0
    assert buffer.ids() == ("tr-1", "tr-2")


def test_replay_sample_stacks_training_batch_and_applies_reference_dropout():
    buffer = RLTReplayBuffer(seed=0)
    buffer.add(_transition("tr-0", sample_t=0, value=1.0))
    batch = buffer.sample_training_batch(
        batch_size=1,
        reference_dropout_prob=1.0,
        replace=False,
    )

    assert batch["executed_action_chunk"].shape == (1, 4, 3)
    assert batch["regularizer_reference_chunk"].shape == (1, 4, 3)
    assert batch["actor_input_reference_chunk"].shape == (1, 4, 3)
    assert batch["next_actor_input_reference_chunk"].shape == (1, 4, 3)
    np.testing.assert_allclose(batch["regularizer_reference_chunk"][0], _chunk(1.0))
    np.testing.assert_allclose(batch["actor_input_reference_chunk"][0], np.zeros((4, 3)))
    np.testing.assert_allclose(batch["next_actor_input_reference_chunk"][0], _chunk(1.0))
    np.testing.assert_array_equal(batch["reference_dropout_mask"], [True])


def test_collate_rlt_batch_resolves_state_tensors_and_keeps_reference_split():
    transition = _transition("tr-0", sample_t=0, value=1.0)
    tensors = {
        "z/0": np.asarray([1.0, 2.0], dtype=np.float32),
        "proprio/0": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
        "z/4": np.asarray([6.0, 7.0], dtype=np.float32),
        "proprio/4": np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
    }

    batch = collate_rlt_batch(
        [transition],
        reference_dropout_mask=[True],
        tensor_resolver=tensors,
    )

    np.testing.assert_allclose(batch["rl_token"], [[1.0, 2.0]])
    np.testing.assert_allclose(batch["proprio"], [[3.0, 4.0, 5.0]])
    np.testing.assert_allclose(batch["next_rl_token"], [[6.0, 7.0]])
    np.testing.assert_allclose(batch["next_proprio"], [[8.0, 9.0, 10.0]])
    assert batch["x"]["rl_token"] is batch["rl_token"]
    assert batch["x_next"]["proprio"] is batch["next_proprio"]
    np.testing.assert_allclose(batch["regularizer_reference_chunk"][0], _chunk(1.0))
    np.testing.assert_allclose(batch["actor_input_reference_chunk"][0], np.zeros((4, 3)))
    np.testing.assert_allclose(batch["next_actor_input_reference_chunk"][0], _chunk(1.0))
    np.testing.assert_allclose(batch["critic_action_chunk"][0], _chunk(11.0))


def test_replay_sample_training_batch_can_resolve_state_tensors():
    buffer = RLTReplayBuffer(seed=0)
    buffer.add(_transition("tr-0", sample_t=0, value=1.0))
    tensors = {
        "z/0": np.asarray([1.0, 2.0], dtype=np.float32),
        "proprio/0": np.asarray([3.0], dtype=np.float32),
        "z/4": np.asarray([4.0, 5.0], dtype=np.float32),
        "proprio/4": np.asarray([6.0], dtype=np.float32),
    }

    batch = buffer.sample_training_batch(
        batch_size=1,
        replace=False,
        tensor_resolver=tensors,
    )

    np.testing.assert_allclose(batch["x"]["rl_token"], [[1.0, 2.0]])
    np.testing.assert_allclose(batch["x"]["proprio"], [[3.0]])
    np.testing.assert_allclose(batch["x_next"]["rl_token"], [[4.0, 5.0]])
    np.testing.assert_allclose(batch["x_next"]["proprio"], [[6.0]])


def test_replay_sample_without_replacement_requires_enough_records():
    buffer = RLTReplayBuffer(seed=0)
    buffer.add(_transition("tr-0", sample_t=0, value=1.0))

    with pytest.raises(ValueError, match="batch_size cannot exceed"):
        buffer.sample(batch_size=2, replace=False)


def test_replay_buffer_counts_collection_stages():
    buffer = RLTReplayBuffer(seed=0)
    buffer.add(
        _transition(
            "tr-warmup",
            sample_t=0,
            value=1.0,
            collection_stage=CollectionStage.VLA_WARMUP,
        )
    )
    buffer.add(_transition("tr-rl", sample_t=4, value=2.0))

    assert buffer.by_collection_stage() == {
        CollectionStage.VLA_WARMUP.value: 1,
        CollectionStage.ONLINE_RL.value: 1,
    }
