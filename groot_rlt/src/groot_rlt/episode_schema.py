# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical episode-level records for future RLT data collection.

The existing LeRobot-style datasets under ``outputs/IsaacLab/trimmed`` already
store the essential time series: frame indices, timestamps, state, action, reward,
done, and video assets. RLT collection needs a few more pieces of provenance so
that chunk-level replay transitions can be derived later without guessing:

* critical-phase handoff and terminal label metadata;
* per-step executed action source and reference source;
* VLA reference chunks and actor proposal chunks at decision steps;
* human intervention segments and human correction action chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from groot_rlt.replay_schema import CriticalPhaseSegment, TerminalOutcome


class EpisodeSource(str, Enum):
    """High-level source of an episode before it is cut into replay samples."""

    DEMO = "demo"
    VLA_WARMUP = "vla_warmup"
    ONLINE_RL = "online_rl"
    HUMAN_INTERVENTION = "human_intervention"
    MIXED = "mixed"


class HandoffSource(str, Enum):
    """How control enters the RLT critical phase."""

    HUMAN = "human"
    SCRIPTED = "scripted"
    LEARNED = "learned"
    DATASET_START = "dataset_start"


class TerminalLabelSource(str, Enum):
    """Who or what produced the sparse terminal success/failure label."""

    HUMAN = "human"
    REWARD_MODEL = "reward_model"
    SCRIPTED = "scripted"


class ChunkBehaviorSource(str, Enum):
    """Chunk-level action provenance, allowing mixed per-step sources."""

    VLA = "vla"
    ACTOR = "actor"
    HUMAN = "human"
    MIXED = "mixed"


class StepBehaviorSource(str, Enum):
    """Step-level executed-action provenance in raw episode logs."""

    VLA = "vla"
    ACTOR = "actor"
    HUMAN_DEMO = "human_demo"
    HUMAN_INTERVENTION = "human_intervention"
    SCRIPTED = "scripted"


class StepReferenceSource(str, Enum):
    """Step-level reference provenance before deriving replay transitions."""

    VLA = "vla_reference"
    DATASET_ACTION = "dataset_action"
    HUMAN_DEMO = "human_demo"
    HUMAN_INTERVENTION = "human_intervention"
    NONE = "none"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_non_empty(name: str, value: str | None) -> None:
    _require(bool(value), f"{name} must be non-empty")


def _normalize_str_map(name: str, value: Mapping[str, str]) -> dict[str, str]:
    result = dict(value)
    _require(result, f"{name} must be non-empty")
    for key, item in result.items():
        _require_non_empty(f"{name} key", key)
        _require_non_empty(f"{name}[{key!r}]", item)
    return result


@dataclass(frozen=True)
class RLTEpisodeRecord:
    """Episode metadata required to derive RLT replay transitions.

    Frame convention: ``terminal_t`` is the terminal observation frame. Replay
    transitions may use ``next_t == terminal_t``; action rows at ``terminal_t`` are
    not required for RLT training and may be ignored.
    """

    episode_id: str
    episode_index: int
    task_id: str
    instruction: str
    source: EpisodeSource | str

    length: int
    fps: float
    robot_type: str
    env_id: str

    state_dim: int
    action_dim: int
    data_path: str
    video_paths: Mapping[str, str]

    chunk_len: int
    vla_horizon: int
    chunk_stride: int

    episode_start_t: int
    base_vla_prefix_end_t: int
    handoff_t: int
    terminal_t: int
    terminal_outcome: TerminalOutcome | str

    terminal_label_source: TerminalLabelSource | str = TerminalLabelSource.HUMAN
    handoff_source: HandoffSource | str = HandoffSource.HUMAN

    observation_state_key: str = "observation.state"
    action_key: str = "action"
    reward_key: str = "next.reward"
    done_key: str = "next.done"

    raw_episode_id: str | None = None
    raw_episode_dir: str | None = None
    base_vla_checkpoint: str | None = None
    rl_token_encoder_checkpoint: str | None = None
    actor_checkpoint: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", EpisodeSource(self.source))
        object.__setattr__(self, "terminal_outcome", TerminalOutcome(self.terminal_outcome))
        object.__setattr__(
            self, "terminal_label_source", TerminalLabelSource(self.terminal_label_source)
        )
        object.__setattr__(self, "handoff_source", HandoffSource(self.handoff_source))
        object.__setattr__(self, "video_paths", _normalize_str_map("video_paths", self.video_paths))
        self.validate()

    def validate(self) -> None:
        for name in ("episode_id", "task_id", "instruction", "robot_type", "env_id", "data_path"):
            _require_non_empty(name, getattr(self, name))

        for name in ("observation_state_key", "action_key", "reward_key", "done_key"):
            _require_non_empty(name, getattr(self, name))

        _require(self.episode_index >= 0, "episode_index must be non-negative")
        _require(self.length > 1, "length must include at least start and terminal frames")
        _require(self.fps > 0.0, "fps must be positive")
        _require(self.state_dim > 0, "state_dim must be positive")
        _require(self.action_dim > 0, "action_dim must be positive")
        _require(self.chunk_len > 0, "chunk_len must be positive")
        _require(
            self.vla_horizon > self.chunk_len,
            "vla_horizon must be greater than chunk_len because RLT uses C < H",
        )
        _require(self.chunk_stride > 0, "chunk_stride must be positive")

        _require(self.episode_start_t == 0, "episode_start_t must be 0 for collected episodes")
        _require(
            0 <= self.base_vla_prefix_end_t <= self.handoff_t < self.terminal_t < self.length,
            "0 <= base_vla_prefix_end_t <= handoff_t < terminal_t < length is required",
        )

    def critical_segment(self) -> CriticalPhaseSegment:
        """Return the replay-schema segment represented by this episode."""

        return CriticalPhaseSegment(
            segment_id=f"{self.episode_id}:critical",
            episode_id=self.episode_id,
            task_id=self.task_id,
            episode_start_t=self.episode_start_t,
            base_vla_prefix_end_t=self.base_vla_prefix_end_t,
            handoff_t=self.handoff_t,
            terminal_t=self.terminal_t,
            terminal_outcome=self.terminal_outcome,
            terminal_label_source=self.terminal_label_source.value,
            handoff_source=self.handoff_source.value,
        )

    @classmethod
    def from_trimmed_lerobot_metadata(
        cls,
        *,
        episode_index: int,
        length: int,
        instruction: str,
        data_path: str,
        video_paths: Mapping[str, str],
        robot_type: str,
        state_dim: int,
        action_dim: int,
        fps: float,
        outcome: TerminalOutcome | str,
        chunk_len: int,
        vla_horizon: int,
        chunk_stride: int,
        task_id: str = "task_0",
        raw_episode_id: str | None = None,
        raw_episode_dir: str | None = None,
    ) -> "RLTEpisodeRecord":
        """Create an RLT episode record from the current trimmed dataset metadata.

        Trimmed demo episodes do not have a separate handoff point, so the whole
        trimmed clip is treated as the critical segment.
        """

        return cls(
            episode_id=f"episode_{episode_index:06d}",
            episode_index=episode_index,
            task_id=task_id,
            instruction=instruction,
            source=EpisodeSource.DEMO,
            length=length,
            fps=fps,
            robot_type=robot_type,
            env_id="offline_lerobot",
            state_dim=state_dim,
            action_dim=action_dim,
            data_path=data_path,
            video_paths=video_paths,
            chunk_len=chunk_len,
            vla_horizon=vla_horizon,
            chunk_stride=chunk_stride,
            episode_start_t=0,
            base_vla_prefix_end_t=0,
            handoff_t=0,
            terminal_t=length - 1,
            terminal_outcome=outcome,
            terminal_label_source=TerminalLabelSource.HUMAN,
            handoff_source=HandoffSource.DATASET_START,
            raw_episode_id=raw_episode_id,
            raw_episode_dir=raw_episode_dir,
        )


@dataclass(frozen=True)
class EpisodeStepRecord:
    """One frame of future RLT collection time series.

    The corresponding LeRobot parquet row should still contain the actual state
    and executed action arrays. This record adds stable tensor keys and control
    provenance needed by RLT.
    """

    episode_id: str
    t: int
    frame_index: int
    timestamp_s: float

    obs_key: str
    image_keys: Mapping[str, str]
    proprio_key: str
    rl_token_key: str
    executed_action_key: str

    behavior_source: StepBehaviorSource | str
    reference_source: StepReferenceSource | str
    intervention: bool

    is_decision_step: bool
    decision_t: int
    chunk_offset: int

    reward: float
    done: bool

    vla_reference_action_key: str | None = None
    actor_proposed_action_key: str | None = None
    human_correction_action_key: str | None = None
    intervention_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior_source", StepBehaviorSource(self.behavior_source))
        object.__setattr__(self, "reference_source", StepReferenceSource(self.reference_source))
        object.__setattr__(self, "image_keys", _normalize_str_map("image_keys", self.image_keys))
        self.validate()

    def validate(self) -> None:
        for name in ("episode_id", "obs_key", "proprio_key", "rl_token_key", "executed_action_key"):
            _require_non_empty(name, getattr(self, name))

        _require(self.t >= 0, "t must be non-negative")
        _require(self.frame_index >= 0, "frame_index must be non-negative")
        _require(self.timestamp_s >= 0.0, "timestamp_s must be non-negative")
        _require(self.decision_t <= self.t, "decision_t must be <= t")
        _require(self.chunk_offset == self.t - self.decision_t, "chunk_offset is misaligned")
        if self.is_decision_step:
            _require(self.chunk_offset == 0, "decision steps must have chunk_offset=0")

        if self.intervention:
            _require(
                self.behavior_source is StepBehaviorSource.HUMAN_INTERVENTION,
                "intervention steps must have HUMAN_INTERVENTION behavior_source",
            )
            _require(
                self.reference_source is StepReferenceSource.HUMAN_INTERVENTION,
                "intervention steps must have HUMAN_INTERVENTION reference_source",
            )
            _require_non_empty("human_correction_action_key", self.human_correction_action_key)
            _require_non_empty("intervention_id", self.intervention_id)
            return

        if self.reference_source is StepReferenceSource.VLA:
            _require_non_empty("vla_reference_action_key", self.vla_reference_action_key)
        elif self.reference_source in {
            StepReferenceSource.DATASET_ACTION,
            StepReferenceSource.HUMAN_DEMO,
            StepReferenceSource.NONE,
        }:
            pass
        else:
            raise ValueError("HUMAN_INTERVENTION reference_source requires intervention=True")

        if self.behavior_source is StepBehaviorSource.ACTOR:
            _require_non_empty("actor_proposed_action_key", self.actor_proposed_action_key)
        elif self.behavior_source in {
            StepBehaviorSource.VLA,
            StepBehaviorSource.HUMAN_DEMO,
            StepBehaviorSource.SCRIPTED,
        }:
            pass
        else:
            raise ValueError("HUMAN_INTERVENTION behavior_source requires intervention=True")

    def validate_against_episode(self, episode: RLTEpisodeRecord) -> None:
        _require(self.episode_id == episode.episode_id, "episode_id does not match episode")
        _require(self.t < episode.length, "step t must be inside the episode")
        _require(
            self.chunk_offset % episode.chunk_stride == 0 or not self.is_decision_step,
            "decision-step chunk_offset must align with episode chunk_stride",
        )
        expected_done = self.t == episode.terminal_t
        _require(self.done == expected_done, "done must be true only at terminal_t")
        if self.done:
            _require(
                self.reward == episode.terminal_outcome.reward,
                "terminal reward must match episode terminal_outcome",
            )
        else:
            _require(self.reward == 0.0, "non-terminal step reward must be 0")


@dataclass(frozen=True)
class ChunkDecisionRecord:
    """Action chunks produced at one policy/VLA decision boundary."""

    episode_id: str
    decision_t: int

    obs_key: str
    rl_token_key: str
    proprio_key: str

    vla_reference_chunk_key: str
    vla_horizon: int
    chunk_len: int
    executed_prefix_len: int

    behavior_source: ChunkBehaviorSource | str
    actor_proposed_chunk_key: str | None = None
    human_correction_chunk_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior_source", ChunkBehaviorSource(self.behavior_source))
        self.validate()

    def validate(self) -> None:
        for name in (
            "episode_id",
            "obs_key",
            "rl_token_key",
            "proprio_key",
            "vla_reference_chunk_key",
        ):
            _require_non_empty(name, getattr(self, name))

        _require(self.decision_t >= 0, "decision_t must be non-negative")
        _require(self.chunk_len > 0, "chunk_len must be positive")
        _require(self.vla_horizon > self.chunk_len, "vla_horizon must be greater than chunk_len")
        _require(
            0 < self.executed_prefix_len <= self.chunk_len,
            "executed_prefix_len must be in (0, chunk_len]",
        )

        if self.behavior_source is ChunkBehaviorSource.ACTOR:
            _require_non_empty("actor_proposed_chunk_key", self.actor_proposed_chunk_key)
        elif self.behavior_source is ChunkBehaviorSource.HUMAN:
            _require_non_empty("human_correction_chunk_key", self.human_correction_chunk_key)
        elif self.behavior_source is ChunkBehaviorSource.MIXED:
            _require(
                bool(self.actor_proposed_chunk_key) or bool(self.human_correction_chunk_key),
                "mixed chunks must keep actor or human chunk provenance",
            )

    def validate_against_episode(self, episode: RLTEpisodeRecord) -> None:
        _require(self.episode_id == episode.episode_id, "episode_id does not match episode")
        _require(
            episode.handoff_t <= self.decision_t < episode.terminal_t,
            "decision_t not in RLT segment",
        )
        _require(self.chunk_len == episode.chunk_len, "chunk_len does not match episode")
        _require(self.vla_horizon == episode.vla_horizon, "vla_horizon does not match episode")


@dataclass(frozen=True)
class InterventionRecord:
    """One contiguous human takeover/correction segment.

    The interval is half-open: ``[start_t, end_t_exclusive)``. Existing inclusive
    frame annotations should be converted with ``end_t_exclusive = end_frame + 1``.
    """

    episode_id: str
    intervention_id: str
    start_t: int
    end_t_exclusive: int

    human_action_chunk_key: str
    control_source: str
    authority_state: str

    takeover_t: int | None = None
    operator_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "episode_id",
            "intervention_id",
            "human_action_chunk_key",
            "control_source",
            "authority_state",
        ):
            _require_non_empty(name, getattr(self, name))

        _require(self.start_t >= 0, "start_t must be non-negative")
        _require(
            self.start_t < self.end_t_exclusive,
            "start_t must be before end_t_exclusive",
        )
        if self.takeover_t is not None:
            _require(
                self.start_t <= self.takeover_t < self.end_t_exclusive,
                "takeover_t must lie inside the intervention interval",
            )

    def validate_against_episode(self, episode: RLTEpisodeRecord) -> None:
        _require(self.episode_id == episode.episode_id, "episode_id does not match episode")
        _require(self.end_t_exclusive <= episode.length, "intervention ends outside episode")
        _require(
            self.start_t < episode.terminal_t,
            "intervention must start before terminal_t",
        )


def validate_episode_components(
    episode: RLTEpisodeRecord,
    steps: Sequence[EpisodeStepRecord],
    chunk_decisions: Sequence[ChunkDecisionRecord],
    interventions: Sequence[InterventionRecord],
) -> None:
    """Validate one collected episode and its sidecar records."""

    episode.validate()
    _require(len(steps) == episode.length, "steps length must match episode.length")
    for expected_t, step in enumerate(steps):
        _require(step.t == expected_t, "steps must be sorted and contiguous by t")
        step.validate_against_episode(episode)

    _require(chunk_decisions, "at least one chunk decision is required")
    decision_times = set()
    for decision in chunk_decisions:
        decision.validate_against_episode(episode)
        _require(decision.decision_t not in decision_times, "duplicate chunk decision_t")
        decision_times.add(decision.decision_t)

    intervention_ids = set()
    for intervention in interventions:
        intervention.validate_against_episode(episode)
        _require(
            intervention.intervention_id not in intervention_ids,
            "duplicate intervention_id",
        )
        intervention_ids.add(intervention.intervention_id)
