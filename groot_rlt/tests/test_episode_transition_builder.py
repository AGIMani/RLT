# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from groot_rlt.episode_schema import (
    ChunkBehaviorSource,
    ChunkDecisionRecord,
    EpisodeSource,
    EpisodeStepRecord,
    HandoffSource,
    RLTEpisodeRecord,
    StepBehaviorSource,
    StepReferenceSource,
    TerminalLabelSource,
)
from groot_rlt.episode_transition_builder import EpisodeTransitionBuilder
from groot_rlt.replay_schema import ReferenceSource, TerminalOutcome

GAMMA = 0.9
ACTION_DIM = 2
CHUNK_LEN = 3
VLA_HORIZON = 6


def _episode() -> RLTEpisodeRecord:
    return RLTEpisodeRecord(
        episode_id="episode_000000",
        episode_index=0,
        task_id="pick_place",
        instruction="pick up the bottle and place it in the box",
        source=EpisodeSource.ONLINE_RL,
        length=9,
        fps=10.0,
        robot_type="real_shadow.single_arm.v1",
        env_id="isaaclab",
        state_dim=26,
        action_dim=ACTION_DIM,
        data_path="data/chunk-000/episode_000000.parquet",
        video_paths={"ego_view": "videos/ego_view.mp4"},
        chunk_len=CHUNK_LEN,
        vla_horizon=VLA_HORIZON,
        chunk_stride=2,
        episode_start_t=0,
        base_vla_prefix_end_t=0,
        handoff_t=0,
        terminal_t=8,
        terminal_outcome=TerminalOutcome.SUCCESS,
        terminal_label_source=TerminalLabelSource.HUMAN,
        handoff_source=HandoffSource.DATASET_START,
    )


def _action(value: float) -> np.ndarray:
    return np.asarray([value, value + 0.25], dtype=np.float32)


def _horizon(base: float) -> np.ndarray:
    return np.stack([_action(base + offset) for offset in range(VLA_HORIZON)], axis=0)


def _step(t: int, *, intervention: bool = False) -> EpisodeStepRecord:
    decision_t = t - (t % 2)
    if intervention:
        behavior_source = StepBehaviorSource.HUMAN_INTERVENTION
        reference_source = StepReferenceSource.HUMAN_INTERVENTION
        human_correction_action_key = f"human/{t}"
        intervention_id = "intervention_000"
        actor_proposed_action_key = None
    else:
        behavior_source = StepBehaviorSource.ACTOR
        reference_source = StepReferenceSource.VLA
        human_correction_action_key = None
        intervention_id = None
        actor_proposed_action_key = f"actor/{t}"

    return EpisodeStepRecord(
        episode_id="episode_000000",
        t=t,
        frame_index=t,
        timestamp_s=0.1 * t,
        obs_key=f"obs/{t}",
        image_keys={"ego_view": f"videos/ego_view.mp4#{t}"},
        proprio_key=f"proprio/{t}",
        rl_token_key=f"rl_token/{t}",
        executed_action_key=f"executed/{t}",
        behavior_source=behavior_source,
        reference_source=reference_source,
        intervention=intervention,
        is_decision_step=t == decision_t,
        decision_t=decision_t,
        chunk_offset=t - decision_t,
        reward=1.0 if t == 8 else 0.0,
        done=t == 8,
        vla_reference_action_key=None if intervention else f"vla_step/{t}",
        actor_proposed_action_key=actor_proposed_action_key,
        human_correction_action_key=human_correction_action_key,
        intervention_id=intervention_id,
    )


def _steps(*, intervention_ts: set[int] | None = None) -> list[EpisodeStepRecord]:
    intervention_ts = intervention_ts or set()
    return [_step(t, intervention=t in intervention_ts) for t in range(9)]


def _chunk_decisions() -> list[ChunkDecisionRecord]:
    return [
        ChunkDecisionRecord(
            episode_id="episode_000000",
            decision_t=0,
            obs_key="obs/0",
            rl_token_key="rl_token/0",
            proprio_key="proprio/0",
            vla_reference_chunk_key="vla/0",
            vla_horizon=VLA_HORIZON,
            chunk_len=CHUNK_LEN,
            executed_prefix_len=2,
            behavior_source=ChunkBehaviorSource.ACTOR,
            actor_proposed_chunk_key="actor_horizon/0",
        ),
        ChunkDecisionRecord(
            episode_id="episode_000000",
            decision_t=2,
            obs_key="obs/2",
            rl_token_key="rl_token/2",
            proprio_key="proprio/2",
            vla_reference_chunk_key="vla/2",
            vla_horizon=VLA_HORIZON,
            chunk_len=CHUNK_LEN,
            executed_prefix_len=2,
            behavior_source=ChunkBehaviorSource.ACTOR,
            actor_proposed_chunk_key="actor_horizon/2",
        ),
    ]


def _tensor_store(*, intervention_ts: set[int] | None = None) -> dict[str, np.ndarray]:
    intervention_ts = intervention_ts or set()
    store: dict[str, np.ndarray] = {
        "vla/0": _horizon(10.0),
        "vla/2": _horizon(20.0),
    }
    for t in range(9):
        actor = _action(100.0 + t)
        human = _action(300.0 + t)
        store[f"actor/{t}"] = actor
        store[f"human/{t}"] = human
        store[f"executed/{t}"] = human if t in intervention_ts else actor
    return store


def _build(*, intervention_ts: set[int] | None = None, allow_terminal_padding: bool = True):
    return EpisodeTransitionBuilder(
        tensor_resolver=_tensor_store(intervention_ts=intervention_ts),
        gamma=GAMMA,
        allow_terminal_padding=allow_terminal_padding,
    ).build(
        episode=_episode(),
        steps=_steps(intervention_ts=intervention_ts),
        chunk_decisions=_chunk_decisions(),
        interventions=[],
    )


def test_builder_keeps_decision_t_sample_t_as_unique_transition_key():
    transitions = _build()
    by_key = {
        (transition.decision_t, transition.sample_t): transition for transition in transitions
    }

    assert (0, 2) in by_key
    assert (2, 2) in by_key
    assert sum(transition.sample_t == 2 for transition in transitions) == 2
    assert (0, 4) not in by_key

    np.testing.assert_allclose(
        by_key[(0, 2)].vla_reference_chunk,
        _horizon(10.0)[2:5],
    )
    np.testing.assert_allclose(
        by_key[(2, 2)].vla_reference_chunk,
        _horizon(20.0)[0:3],
    )
    np.testing.assert_allclose(
        by_key[(0, 2)].next_train_reference_chunk,
        _horizon(20.0)[3:6],
    )
    np.testing.assert_array_equal(by_key[(0, 2)].next_reference_valid_mask, [True, True, True])


def test_builder_allows_padding_only_on_terminal_tail():
    transitions = _build()
    by_key = {
        (transition.decision_t, transition.sample_t): transition for transition in transitions
    }
    terminal = by_key[(2, 6)]

    assert terminal.done
    assert terminal.next_t == 8
    np.testing.assert_array_equal(terminal.valid_mask, [True, True, False])
    np.testing.assert_allclose(terminal.vla_reference_chunk[:2], _horizon(20.0)[4:6])
    np.testing.assert_allclose(terminal.vla_reference_chunk[2], np.zeros(ACTION_DIM))
    np.testing.assert_allclose(terminal.next_train_reference_chunk, np.zeros((3, ACTION_DIM)))
    np.testing.assert_array_equal(terminal.next_reference_valid_mask, [False, False, False])
    np.testing.assert_allclose(terminal.reward_chunk, [0.0, 1.0, 0.0])
    assert terminal.discounted_reward_sum == pytest.approx(GAMMA)
    assert terminal.discount == 0.0


def test_builder_can_disable_terminal_padding():
    with pytest.raises(ValueError, match="terminal padding"):
        _build(allow_terminal_padding=False)


def test_builder_requires_valid_discount_gamma():
    with pytest.raises(ValueError, match="gamma must be in"):
        EpisodeTransitionBuilder(tensor_resolver={}, gamma=1.0)


def test_builder_replaces_reference_inside_intervention_steps():
    transitions = _build(intervention_ts={3, 5})
    by_key = {
        (transition.decision_t, transition.sample_t): transition for transition in transitions
    }
    transition = by_key[(0, 2)]

    np.testing.assert_array_equal(transition.intervention_mask, [False, True, False])
    assert transition.reference_source[1] is ReferenceSource.HUMAN
    np.testing.assert_allclose(transition.train_reference_chunk[0], _horizon(10.0)[2])
    np.testing.assert_allclose(transition.train_reference_chunk[1], _action(303.0))
    np.testing.assert_allclose(transition.train_reference_chunk[2], _horizon(10.0)[4])
    np.testing.assert_allclose(transition.executed_action_chunk[1], _action(303.0))
    np.testing.assert_allclose(transition.next_train_reference_chunk[0], _action(305.0))
    np.testing.assert_allclose(transition.next_train_reference_chunk[1:], _horizon(20.0)[4:6])
