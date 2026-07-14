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
from groot_rlt.replay_schema import CollectionStage, TerminalOutcome
from groot_rlt.warmup import (
    RLTWarmupConfig,
    WarmupEpisodeComponents,
    build_warmup_replay,
    build_warmup_transitions,
)

GAMMA = 0.9
ACTION_DIM = 2
CHUNK_LEN = 2
VLA_HORIZON = 4


def _action(t: int) -> np.ndarray:
    return np.asarray([float(t), float(t) + 0.5], dtype=np.float32)


def _episode(*, source: EpisodeSource = EpisodeSource.VLA_WARMUP) -> RLTEpisodeRecord:
    return RLTEpisodeRecord(
        episode_id="warmup_episode_000000",
        episode_index=0,
        task_id="insert",
        instruction="insert the connector",
        source=source,
        length=7,
        fps=10.0,
        robot_type="real_shadow.single_arm.v1",
        env_id="isaaclab",
        state_dim=8,
        action_dim=ACTION_DIM,
        data_path="data/chunk-000/episode_000000.parquet",
        video_paths={"ego_view": "videos/ego_view.mp4"},
        chunk_len=CHUNK_LEN,
        vla_horizon=VLA_HORIZON,
        chunk_stride=2,
        episode_start_t=0,
        base_vla_prefix_end_t=0,
        handoff_t=0,
        terminal_t=6,
        terminal_outcome=TerminalOutcome.SUCCESS,
        terminal_label_source=TerminalLabelSource.HUMAN,
        handoff_source=HandoffSource.DATASET_START,
    )


def _steps() -> list[EpisodeStepRecord]:
    steps = []
    for t in range(7):
        decision_t = t - (t % 2)
        steps.append(
            EpisodeStepRecord(
                episode_id="warmup_episode_000000",
                t=t,
                frame_index=t,
                timestamp_s=0.1 * t,
                obs_key=f"obs/{t}",
                image_keys={"ego_view": f"videos/ego_view.mp4#{t}"},
                proprio_key=f"proprio/{t}",
                rl_token_key=f"rl_token/{t}",
                executed_action_key=f"executed/{t}",
                behavior_source=StepBehaviorSource.VLA,
                reference_source=StepReferenceSource.VLA,
                intervention=False,
                is_decision_step=t == decision_t,
                decision_t=decision_t,
                chunk_offset=t - decision_t,
                reward=1.0 if t == 6 else 0.0,
                done=t == 6,
                vla_reference_action_key=f"vla_step/{t}",
            )
        )
    return steps


def _decisions() -> list[ChunkDecisionRecord]:
    return [
        ChunkDecisionRecord(
            episode_id="warmup_episode_000000",
            decision_t=0,
            obs_key="obs/0",
            rl_token_key="rl_token/0",
            proprio_key="proprio/0",
            vla_reference_chunk_key="vla/0",
            vla_horizon=VLA_HORIZON,
            chunk_len=CHUNK_LEN,
            executed_prefix_len=CHUNK_LEN,
            behavior_source=ChunkBehaviorSource.VLA,
        ),
        ChunkDecisionRecord(
            episode_id="warmup_episode_000000",
            decision_t=2,
            obs_key="obs/2",
            rl_token_key="rl_token/2",
            proprio_key="proprio/2",
            vla_reference_chunk_key="vla/2",
            vla_horizon=VLA_HORIZON,
            chunk_len=CHUNK_LEN,
            executed_prefix_len=CHUNK_LEN,
            behavior_source=ChunkBehaviorSource.VLA,
        ),
    ]


def _tensor_store() -> dict[str, np.ndarray]:
    store: dict[str, np.ndarray] = {
        "vla/0": np.stack([_action(t) for t in range(4)], axis=0),
        "vla/2": np.stack([_action(t) for t in range(2, 6)], axis=0),
    }
    for t in range(7):
        store[f"executed/{t}"] = _action(t)
    return store


def _components(*, source: EpisodeSource = EpisodeSource.VLA_WARMUP) -> WarmupEpisodeComponents:
    return WarmupEpisodeComponents(
        episode=_episode(source=source),
        steps=_steps(),
        chunk_decisions=_decisions(),
        interventions=[],
        tensor_resolver=_tensor_store(),
    )


def test_build_warmup_replay_prefills_vla_warmup_stage():
    config = RLTWarmupConfig(gamma=GAMMA, seed=0)

    replay, summary = build_warmup_replay([_components()], config=config)

    assert len(replay) == 4
    assert summary.episode_count == 1
    assert summary.stored_transition_count == 4
    assert summary.terminal_transition_count == 1
    assert summary.by_collection_stage == {CollectionStage.VLA_WARMUP.value: 4}
    for transition in replay:
        assert transition.collection_stage is CollectionStage.VLA_WARMUP
        np.testing.assert_allclose(transition.executed_action_chunk, transition.vla_reference_chunk)


def test_warmup_requires_vla_warmup_source_by_default():
    config = RLTWarmupConfig(gamma=GAMMA)

    with pytest.raises(ValueError, match="episode.source == vla_warmup"):
        build_warmup_transitions(_components(source=EpisodeSource.ONLINE_RL), config=config)
