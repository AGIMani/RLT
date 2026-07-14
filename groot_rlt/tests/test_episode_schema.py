# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from groot_rlt.episode_schema import (
    ChunkBehaviorSource,
    ChunkDecisionRecord,
    EpisodeSource,
    EpisodeStepRecord,
    HandoffSource,
    InterventionRecord,
    RLTEpisodeRecord,
    StepBehaviorSource,
    StepReferenceSource,
    TerminalLabelSource,
    validate_episode_components,
)
from groot_rlt.replay_schema import TerminalOutcome


def _episode(source=EpisodeSource.DEMO):
    return RLTEpisodeRecord(
        episode_id="episode_000000",
        episode_index=0,
        task_id="pick_place",
        instruction="pick up the bottle and place it in the box",
        source=source,
        length=6,
        fps=10.0,
        robot_type="real_shadow.single_arm.v1",
        env_id="offline_lerobot",
        state_dim=26,
        action_dim=19,
        data_path="data/chunk-000/episode_000000.parquet",
        video_paths={
            "ego_view": "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            "wrist_view": "videos/chunk-000/observation.images.wrist_view/episode_000000.mp4",
        },
        chunk_len=2,
        vla_horizon=4,
        chunk_stride=1,
        episode_start_t=0,
        base_vla_prefix_end_t=0,
        handoff_t=0,
        terminal_t=5,
        terminal_outcome=TerminalOutcome.SUCCESS,
        terminal_label_source=TerminalLabelSource.HUMAN,
        handoff_source=HandoffSource.DATASET_START,
    )


def _demo_step(t: int, *, done: bool = False, reward: float = 0.0):
    decision_t = t - (t % 2)
    return EpisodeStepRecord(
        episode_id="episode_000000",
        t=t,
        frame_index=t,
        timestamp_s=0.1 * t,
        obs_key=f"obs/{t}",
        image_keys={
            "ego_view": f"videos/ego/episode_000000.mp4#{t}",
            "wrist_view": f"videos/wrist/episode_000000.mp4#{t}",
        },
        proprio_key=f"proprio/{t}",
        rl_token_key=f"rl_token/{t}",
        executed_action_key=f"action/executed/{t}",
        behavior_source=StepBehaviorSource.HUMAN_DEMO,
        reference_source=StepReferenceSource.DATASET_ACTION,
        intervention=False,
        is_decision_step=t == decision_t,
        decision_t=decision_t,
        chunk_offset=t - decision_t,
        reward=reward,
        done=done,
    )


def test_episode_record_can_be_created_from_trimmed_lerobot_metadata():
    episode = RLTEpisodeRecord.from_trimmed_lerobot_metadata(
        episode_index=0,
        length=471,
        instruction="pick up the bottle and place it in the box",
        data_path="data/chunk-000/episode_000000.parquet",
        video_paths={
            "ego_view": "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            "wrist_view": "videos/chunk-000/observation.images.wrist_view/episode_000000.mp4",
        },
        robot_type="real_shadow.single_arm.v1",
        state_dim=26,
        action_dim=19,
        fps=10.0,
        outcome=TerminalOutcome.SUCCESS,
        chunk_len=10,
        vla_horizon=50,
        chunk_stride=2,
        raw_episode_id="episode_20260509T122113Z_000000",
    )

    assert episode.source is EpisodeSource.DEMO
    assert episode.handoff_t == 0
    assert episode.terminal_t == 470
    assert episode.critical_segment().terminal_outcome is TerminalOutcome.SUCCESS


def test_future_demo_episode_components_validate():
    episode = _episode()
    steps = [_demo_step(t) for t in range(5)] + [_demo_step(5, done=True, reward=1.0)]
    chunk_decisions = [
        ChunkDecisionRecord(
            episode_id=episode.episode_id,
            decision_t=0,
            obs_key="obs/0",
            rl_token_key="rl_token/0",
            proprio_key="proprio/0",
            vla_reference_chunk_key="chunks/vla/0",
            vla_horizon=episode.vla_horizon,
            chunk_len=episode.chunk_len,
            executed_prefix_len=episode.chunk_len,
            behavior_source=ChunkBehaviorSource.VLA,
        )
    ]

    validate_episode_components(episode, steps, chunk_decisions, interventions=[])


def test_intervention_step_requires_human_correction_and_id():
    with pytest.raises(ValueError, match="human_correction_action_key"):
        EpisodeStepRecord(
            episode_id="episode_000000",
            t=2,
            frame_index=2,
            timestamp_s=0.2,
            obs_key="obs/2",
            image_keys={"ego_view": "videos/ego/episode_000000.mp4#2"},
            proprio_key="proprio/2",
            rl_token_key="rl_token/2",
            executed_action_key="action/executed/2",
            behavior_source=StepBehaviorSource.HUMAN_INTERVENTION,
            reference_source=StepReferenceSource.HUMAN_INTERVENTION,
            intervention=True,
            is_decision_step=True,
            decision_t=2,
            chunk_offset=0,
            reward=0.0,
            done=False,
        )


def test_actor_chunk_decision_requires_actor_proposal_key():
    with pytest.raises(ValueError, match="actor_proposed_chunk_key"):
        ChunkDecisionRecord(
            episode_id="episode_000000",
            decision_t=0,
            obs_key="obs/0",
            rl_token_key="rl_token/0",
            proprio_key="proprio/0",
            vla_reference_chunk_key="chunks/vla/0",
            vla_horizon=4,
            chunk_len=2,
            executed_prefix_len=2,
            behavior_source=ChunkBehaviorSource.ACTOR,
        )


def test_intervention_record_uses_half_open_interval():
    episode = _episode(source=EpisodeSource.HUMAN_INTERVENTION)
    intervention = InterventionRecord(
        episode_id=episode.episode_id,
        intervention_id="intervention_000",
        start_t=1,
        end_t_exclusive=4,
        takeover_t=1,
        human_action_chunk_key="interventions/intervention_000/actions",
        control_source="human_intervention",
        authority_state="human_intervention",
    )

    intervention.validate_against_episode(episode)
