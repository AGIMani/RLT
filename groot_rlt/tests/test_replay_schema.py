# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from groot_rlt.replay_schema import (
    BehaviorSource,
    CollectionStage,
    CriticalPhaseSegment,
    ReferenceSource,
    RLTTransition,
    TerminalOutcome,
    compose_train_reference_chunk,
)

GAMMA = 0.99


def _segment(outcome=TerminalOutcome.SUCCESS):
    return CriticalPhaseSegment(
        segment_id="seg-0",
        episode_id="episode-0",
        task_id="insert",
        episode_start_t=0,
        base_vla_prefix_end_t=8,
        handoff_t=10,
        terminal_t=30,
        terminal_outcome=outcome,
    )


def _chunk(value: float, chunk_len: int = 4, action_dim: int = 3):
    return np.full((chunk_len, action_dim), value, dtype=np.float32)


def test_online_actor_transition_keeps_vla_reference_for_training():
    seg = _segment()
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)

    tr = RLTTransition(
        transition_id="tr-actor",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.ONLINE_RL,
        decision_t=10,
        sample_t=10,
        next_t=14,
        chunk_offset=0,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/10",
        next_obs_key="obs/14",
        rl_token_key="z/10",
        proprio_key="proprio/10",
        next_rl_token_key="z/14",
        next_proprio_key="proprio/14",
        executed_action_chunk=actor,
        train_reference_chunk=vla_ref,
        next_train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_ref,
        actor_proposed_chunk=actor,
        behavior_source=BehaviorSource.ACTOR,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)
    np.testing.assert_allclose(tr.as_training_item()["regularizer_reference_chunk"], vla_ref)

    dropped = tr.actor_input_reference_chunk(drop_reference=True)
    np.testing.assert_allclose(dropped, np.zeros_like(vla_ref))
    np.testing.assert_allclose(tr.train_reference_chunk, vla_ref)


def test_vla_warmup_executes_and_references_vla_chunk():
    seg = _segment()
    vla_ref = _chunk(1.5)

    tr = RLTTransition(
        transition_id="tr-warmup",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.VLA_WARMUP,
        decision_t=10,
        sample_t=10,
        next_t=14,
        chunk_offset=0,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/10",
        next_obs_key="obs/14",
        rl_token_key="z/10",
        proprio_key="proprio/10",
        next_rl_token_key="z/14",
        next_proprio_key="proprio/14",
        executed_action_chunk=vla_ref,
        train_reference_chunk=vla_ref,
        next_train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_ref,
        behavior_source=BehaviorSource.VLA,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)


def test_human_intervention_replaces_training_reference():
    seg = _segment()
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)
    human = _chunk(3.0)
    intervention_mask = [False, True, True, False]
    train_ref = compose_train_reference_chunk(
        vla_reference_chunk=vla_ref,
        human_correction_chunk=human,
        intervention_mask=intervention_mask,
    )
    executed = actor.copy()
    executed[1:3] = human[1:3]

    tr = RLTTransition(
        transition_id="tr-human",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.ONLINE_RL,
        decision_t=10,
        sample_t=12,
        next_t=16,
        chunk_offset=2,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/12",
        next_obs_key="obs/16",
        rl_token_key="z/12",
        proprio_key="proprio/12",
        next_rl_token_key="z/16",
        next_proprio_key="proprio/16",
        executed_action_chunk=executed,
        train_reference_chunk=train_ref,
        next_train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_ref,
        actor_proposed_chunk=actor,
        human_correction_chunk=human,
        behavior_source=[
            BehaviorSource.ACTOR,
            BehaviorSource.HUMAN,
            BehaviorSource.HUMAN,
            BehaviorSource.ACTOR,
        ],
        reference_source=[
            ReferenceSource.VLA,
            ReferenceSource.HUMAN,
            ReferenceSource.HUMAN,
            ReferenceSource.VLA,
        ],
        intervention_mask=intervention_mask,
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)
    np.testing.assert_allclose(tr.train_reference_chunk[1:3], human[1:3])
    np.testing.assert_allclose(tr.train_reference_chunk[[0, 3]], vla_ref[[0, 3]])


def test_intervention_requires_human_reference_replacement():
    seg = _segment()
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)
    human = _chunk(3.0)
    executed = actor.copy()
    executed[1] = human[1]

    with pytest.raises(ValueError, match="train_reference_chunk\\[1\\] must match"):
        RLTTransition(
            transition_id="tr-bad-human",
            segment_id=seg.segment_id,
            episode_id=seg.episode_id,
            collection_stage=CollectionStage.ONLINE_RL,
            decision_t=10,
            sample_t=12,
            next_t=16,
            chunk_offset=2,
            chunk_stride=2,
            chunk_len=4,
            vla_horizon=8,
            obs_key="obs/12",
            next_obs_key="obs/16",
            rl_token_key="z/12",
            proprio_key="proprio/12",
            next_rl_token_key="z/16",
            next_proprio_key="proprio/16",
            executed_action_chunk=executed,
            train_reference_chunk=vla_ref,
            next_train_reference_chunk=vla_ref,
            reward_chunk=[0.0, 0.0, 0.0, 0.0],
            valid_mask=[True, True, True, True],
            reference_valid_mask=[True, True, True, True],
            vla_reference_chunk=vla_ref,
            actor_proposed_chunk=actor,
            human_correction_chunk=human,
            behavior_source=[
                BehaviorSource.ACTOR,
                BehaviorSource.HUMAN,
                BehaviorSource.ACTOR,
                BehaviorSource.ACTOR,
            ],
            reference_source=[
                ReferenceSource.VLA,
                ReferenceSource.HUMAN,
                ReferenceSource.VLA,
                ReferenceSource.VLA,
            ],
            intervention_mask=[False, True, False, False],
            discounted_reward_sum=0.0,
            discount=GAMMA**4,
            done=False,
        )


def test_terminal_sparse_success_label_with_clipped_chunk():
    seg = _segment(outcome=TerminalOutcome.SUCCESS)
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)

    tr = RLTTransition(
        transition_id="tr-terminal",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.ONLINE_RL,
        decision_t=26,
        sample_t=28,
        next_t=30,
        chunk_offset=2,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/28",
        next_obs_key="obs/30",
        rl_token_key="z/28",
        proprio_key="proprio/28",
        next_rl_token_key="z/30",
        next_proprio_key="proprio/30",
        executed_action_chunk=actor,
        train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 1.0, 0.0, 0.0],
        valid_mask=[True, True, False, False],
        reference_valid_mask=[True, True, False, False],
        vla_reference_chunk=vla_ref,
        actor_proposed_chunk=actor,
        behavior_source=BehaviorSource.ACTOR,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=GAMMA,
        discount=0.0,
        done=True,
        terminal_outcome_if_done=TerminalOutcome.SUCCESS,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)


def test_failure_terminal_reward_is_zero_but_done_is_true():
    seg = _segment(outcome=TerminalOutcome.FAILURE)
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)

    tr = RLTTransition(
        transition_id="tr-failure",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.ONLINE_RL,
        decision_t=26,
        sample_t=26,
        next_t=30,
        chunk_offset=0,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/26",
        next_obs_key="obs/30",
        rl_token_key="z/26",
        proprio_key="proprio/26",
        next_rl_token_key="z/30",
        next_proprio_key="proprio/30",
        executed_action_chunk=actor,
        train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_ref,
        actor_proposed_chunk=actor,
        behavior_source=BehaviorSource.ACTOR,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=0.0,
        done=True,
        terminal_outcome_if_done=TerminalOutcome.FAILURE,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)


def test_chunk_alignment_rejects_wrong_offset():
    seg = _segment()
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)

    with pytest.raises(ValueError, match="chunk_offset is misaligned"):
        RLTTransition(
            transition_id="tr-bad-offset",
            segment_id=seg.segment_id,
            episode_id=seg.episode_id,
            collection_stage=CollectionStage.ONLINE_RL,
            decision_t=10,
            sample_t=12,
            next_t=16,
            chunk_offset=0,
            chunk_stride=2,
            chunk_len=4,
            vla_horizon=8,
            obs_key="obs/12",
            next_obs_key="obs/16",
            rl_token_key="z/12",
            proprio_key="proprio/12",
            next_rl_token_key="z/16",
            next_proprio_key="proprio/16",
            executed_action_chunk=actor,
            train_reference_chunk=vla_ref,
            next_train_reference_chunk=vla_ref,
            reward_chunk=[0.0, 0.0, 0.0, 0.0],
            valid_mask=[True, True, True, True],
            reference_valid_mask=[True, True, True, True],
            vla_reference_chunk=vla_ref,
            actor_proposed_chunk=actor,
            behavior_source=BehaviorSource.ACTOR,
            reference_source=ReferenceSource.VLA,
            intervention_mask=[False, False, False, False],
            discounted_reward_sum=0.0,
            discount=GAMMA**4,
            done=False,
        )


def test_subsampled_chunk_may_extend_past_single_vla_horizon_when_reference_is_explicit():
    seg = _segment()
    vla_ref = _chunk(1.0)
    actor = _chunk(2.0)

    tr = RLTTransition(
        transition_id="tr-late-offset",
        segment_id=seg.segment_id,
        episode_id=seg.episode_id,
        collection_stage=CollectionStage.ONLINE_RL,
        decision_t=10,
        sample_t=16,
        next_t=20,
        chunk_offset=6,
        chunk_stride=2,
        chunk_len=4,
        vla_horizon=8,
        obs_key="obs/16",
        next_obs_key="obs/20",
        rl_token_key="z/16",
        proprio_key="proprio/16",
        next_rl_token_key="z/20",
        next_proprio_key="proprio/20",
        executed_action_chunk=actor,
        train_reference_chunk=vla_ref,
        next_train_reference_chunk=vla_ref,
        reward_chunk=[0.0, 0.0, 0.0, 0.0],
        valid_mask=[True, True, True, True],
        reference_valid_mask=[True, True, True, True],
        vla_reference_chunk=vla_ref,
        actor_proposed_chunk=actor,
        behavior_source=BehaviorSource.ACTOR,
        reference_source=ReferenceSource.VLA,
        intervention_mask=[False, False, False, False],
        discounted_reward_sum=0.0,
        discount=GAMMA**4,
        done=False,
    )

    tr.validate_against_segment(seg, gamma=GAMMA)
