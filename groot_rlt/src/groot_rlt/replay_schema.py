# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical replay records for RL Token (RLT) actor-critic training.

The schema is intentionally stricter than a conventional ``(s, a, r, s')`` replay
buffer. RLT trains a chunked actor that is conditioned on a VLA reference action
chunk; during human intervention, the human command replaces that VLA reference
for replay. These records make the four paper-level requirements explicit:

* chunk-aligned transitions, including subsampled intermediate chunks;
* intervention replacement of the training reference;
* critical-phase-only replay segments;
* sparse terminal success/failure labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, TypeVar

import numpy as np


class CollectionStage(str, Enum):
    """Which RLT collection phase produced a transition."""

    VLA_WARMUP = "vla_warmup"
    ONLINE_RL = "online_rl"


class BehaviorSource(str, Enum):
    """Which controller produced the action that was physically executed."""

    VLA = "vla"
    ACTOR = "actor"
    HUMAN = "human"


class ReferenceSource(str, Enum):
    """Which command is the actor-conditioning and regularization reference."""

    VLA = "vla_reference"
    HUMAN = "human_intervention"


class TerminalOutcome(str, Enum):
    """Human sparse terminal label for the RLT critical segment."""

    SUCCESS = "success"
    FAILURE = "failure"

    @property
    def reward(self) -> float:
        return 1.0 if self is TerminalOutcome.SUCCESS else 0.0


EnumT = TypeVar("EnumT", bound=Enum)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float32_chunk(name: str, value: Any | None, *, chunk_len: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value, dtype=np.float32)
    _require(array.ndim == 2, f"{name} must have shape [chunk_len, action_dim], got {array.shape}")
    if chunk_len is not None:
        _require(
            array.shape[0] == chunk_len,
            f"{name} chunk length {array.shape[0]} != expected {chunk_len}",
        )
    _require(array.shape[1] > 0, f"{name} action_dim must be positive")
    return array


def _as_optional_float32_chunk(
    name: str,
    value: Any | None,
    *,
    chunk_len: int,
    action_dim: int,
) -> np.ndarray | None:
    if value is None:
        return None
    array = _as_float32_chunk(name, value, chunk_len=chunk_len)
    _require(
        array.shape[1] == action_dim,
        f"{name} action_dim {array.shape[1]} != expected {action_dim}",
    )
    return array


def _as_bool_mask(name: str, value: Any, *, chunk_len: int) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    _require(mask.shape == (chunk_len,), f"{name} must have shape ({chunk_len},), got {mask.shape}")
    return mask


def _as_float32_vector(name: str, value: Any, *, chunk_len: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    _require(
        vector.shape == (chunk_len,),
        f"{name} must have shape ({chunk_len},), got {vector.shape}",
    )
    return vector


def _normalize_enum_sequence(
    name: str,
    value: Sequence[EnumT | str] | EnumT | str,
    enum_cls: type[EnumT],
    *,
    chunk_len: int,
) -> tuple[EnumT, ...]:
    if isinstance(value, (enum_cls, str)):
        raw_values = [value] * chunk_len
    else:
        raw_values = list(value)
    _require(
        len(raw_values) == chunk_len,
        f"{name} must contain {chunk_len} entries, got {len(raw_values)}",
    )

    normalized: list[EnumT] = []
    for item in raw_values:
        if isinstance(item, enum_cls):
            normalized.append(item)
        else:
            try:
                normalized.append(enum_cls(item))
            except ValueError as exc:
                valid = ", ".join(member.value for member in enum_cls)
                raise ValueError(
                    f"{name} contains invalid value {item!r}; expected one of {valid}"
                ) from exc
    return tuple(normalized)


def _prefix_true_count(mask: np.ndarray, name: str) -> int:
    false_indices = np.flatnonzero(~mask)
    if false_indices.size == 0:
        return int(mask.shape[0])
    first_false = int(false_indices[0])
    _require(
        not bool(mask[first_false:].any()),
        f"{name} must be a prefix mask with no true values after the first false",
    )
    return first_false


def _allclose(a: np.ndarray, b: np.ndarray, *, atol: float = 1.0e-6) -> bool:
    return bool(np.allclose(a, b, rtol=1.0e-5, atol=atol))


def _require_chunk_equal(
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
    index: int,
) -> None:
    _require(
        _allclose(left[index], right[index]),
        f"{left_name}[{index}] must match {right_name}[{index}]",
    )


def compose_train_reference_chunk(
    *,
    vla_reference_chunk: np.ndarray,
    human_correction_chunk: np.ndarray | None,
    intervention_mask: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Build the reference chunk that should be stored in replay.

    Non-intervention timesteps keep the VLA reference. Intervention timesteps use
    the human command, matching the RLT rollout rule that human intervention
    replaces the VLA reference in the replay buffer.
    """

    vla = _as_float32_chunk("vla_reference_chunk", vla_reference_chunk)
    mask = _as_bool_mask("intervention_mask", intervention_mask, chunk_len=vla.shape[0])
    if not mask.any():
        return vla.copy()

    human = _as_optional_float32_chunk(
        "human_correction_chunk",
        human_correction_chunk,
        chunk_len=vla.shape[0],
        action_dim=vla.shape[1],
    )
    if human is None:
        raise ValueError(
            "human_correction_chunk is required when intervention_mask has true entries"
        )

    train_reference = vla.copy()
    train_reference[mask] = human[mask]
    return train_reference


@dataclass(frozen=True)
class CriticalPhaseSegment:
    """Episode section where RLT owns control and replay is valid."""

    segment_id: str
    episode_id: str
    task_id: str

    episode_start_t: int
    base_vla_prefix_end_t: int
    handoff_t: int
    terminal_t: int

    terminal_outcome: TerminalOutcome | str
    terminal_label_source: str = "human"
    handoff_source: str = "human"

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_outcome", TerminalOutcome(self.terminal_outcome))
        self.validate()

    def validate(self) -> None:
        for name in ("segment_id", "episode_id", "task_id"):
            _require(bool(getattr(self, name)), f"{name} must be non-empty")

        _require(self.episode_start_t >= 0, "episode_start_t must be non-negative")
        _require(
            self.episode_start_t <= self.base_vla_prefix_end_t <= self.handoff_t,
            "episode_start_t <= base_vla_prefix_end_t <= handoff_t is required",
        )
        _require(self.handoff_t < self.terminal_t, "handoff_t must be before terminal_t")
        _require(bool(self.terminal_label_source), "terminal_label_source must be non-empty")
        _require(bool(self.handoff_source), "handoff_source must be non-empty")

    @property
    def terminal_reward(self) -> float:
        return self.terminal_outcome.reward

    def contains_transition_window(self, sample_t: int, next_t: int) -> bool:
        return self.handoff_t <= sample_t < next_t <= self.terminal_t


@dataclass(frozen=True)
class RLTTransition:
    """One chunk-level RLT replay transition.

    ``train_reference_chunk`` is the canonical replay reference used for both actor
    conditioning and reference regularization. ``next_train_reference_chunk`` is
    the reference aligned to ``x_next`` for Eq. 3 bootstrapping. Keep
    ``vla_reference_chunk`` only as provenance when intervention replaces the
    training reference.
    """

    transition_id: str
    segment_id: str
    episode_id: str

    collection_stage: CollectionStage | str

    decision_t: int
    sample_t: int
    next_t: int
    chunk_offset: int
    chunk_stride: int
    chunk_len: int
    vla_horizon: int

    obs_key: str
    next_obs_key: str
    rl_token_key: str
    proprio_key: str
    next_rl_token_key: str
    next_proprio_key: str

    executed_action_chunk: Any
    train_reference_chunk: Any
    reward_chunk: Any
    valid_mask: Any
    reference_valid_mask: Any

    vla_reference_chunk: Any | None = None
    actor_proposed_chunk: Any | None = None
    human_correction_chunk: Any | None = None
    next_train_reference_chunk: Any | None = None
    next_reference_valid_mask: Any | None = None

    behavior_source: Sequence[BehaviorSource | str] | BehaviorSource | str = BehaviorSource.ACTOR
    reference_source: Sequence[ReferenceSource | str] | ReferenceSource | str = ReferenceSource.VLA
    intervention_mask: Any = None

    discounted_reward_sum: float = 0.0
    discount: float = 1.0
    done: bool = False
    terminal_outcome_if_done: TerminalOutcome | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collection_stage", CollectionStage(self.collection_stage))

        executed = _as_float32_chunk("executed_action_chunk", self.executed_action_chunk)
        _require(
            self.chunk_len == executed.shape[0],
            f"chunk_len {self.chunk_len} != executed_action_chunk length {executed.shape[0]}",
        )
        train_reference = _as_float32_chunk(
            "train_reference_chunk", self.train_reference_chunk, chunk_len=self.chunk_len
        )
        _require(
            train_reference.shape[1] == executed.shape[1],
            "train_reference_chunk action_dim must match executed_action_chunk",
        )

        valid_mask = _as_bool_mask("valid_mask", self.valid_mask, chunk_len=self.chunk_len)
        reference_valid_mask = _as_bool_mask(
            "reference_valid_mask", self.reference_valid_mask, chunk_len=self.chunk_len
        )
        reward_chunk = _as_float32_vector(
            "reward_chunk", self.reward_chunk, chunk_len=self.chunk_len
        )
        intervention_mask = _as_bool_mask(
            "intervention_mask", self.intervention_mask, chunk_len=self.chunk_len
        )

        behavior_source = _normalize_enum_sequence(
            "behavior_source", self.behavior_source, BehaviorSource, chunk_len=self.chunk_len
        )
        reference_source = _normalize_enum_sequence(
            "reference_source", self.reference_source, ReferenceSource, chunk_len=self.chunk_len
        )

        action_dim = executed.shape[1]
        vla_reference = _as_optional_float32_chunk(
            "vla_reference_chunk",
            self.vla_reference_chunk,
            chunk_len=self.chunk_len,
            action_dim=action_dim,
        )
        actor_proposed = _as_optional_float32_chunk(
            "actor_proposed_chunk",
            self.actor_proposed_chunk,
            chunk_len=self.chunk_len,
            action_dim=action_dim,
        )
        human_correction = _as_optional_float32_chunk(
            "human_correction_chunk",
            self.human_correction_chunk,
            chunk_len=self.chunk_len,
            action_dim=action_dim,
        )
        next_train_reference = _as_optional_float32_chunk(
            "next_train_reference_chunk",
            self.next_train_reference_chunk,
            chunk_len=self.chunk_len,
            action_dim=action_dim,
        )
        if next_train_reference is None:
            if self.done:
                next_train_reference = np.zeros_like(train_reference)
            else:
                raise ValueError(
                    "next_train_reference_chunk is required for non-terminal RLT transitions"
                )

        if self.next_reference_valid_mask is None:
            next_reference_valid_mask = np.zeros(self.chunk_len, dtype=np.bool_)
            if not self.done:
                next_reference_valid_mask[:] = True
        else:
            next_reference_valid_mask = _as_bool_mask(
                "next_reference_valid_mask",
                self.next_reference_valid_mask,
                chunk_len=self.chunk_len,
            )

        terminal_outcome = (
            None
            if self.terminal_outcome_if_done is None
            else TerminalOutcome(self.terminal_outcome_if_done)
        )

        object.__setattr__(self, "executed_action_chunk", executed)
        object.__setattr__(self, "train_reference_chunk", train_reference)
        object.__setattr__(self, "reward_chunk", reward_chunk)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "reference_valid_mask", reference_valid_mask)
        object.__setattr__(self, "intervention_mask", intervention_mask)
        object.__setattr__(self, "behavior_source", behavior_source)
        object.__setattr__(self, "reference_source", reference_source)
        object.__setattr__(self, "vla_reference_chunk", vla_reference)
        object.__setattr__(self, "actor_proposed_chunk", actor_proposed)
        object.__setattr__(self, "human_correction_chunk", human_correction)
        object.__setattr__(self, "next_train_reference_chunk", next_train_reference)
        object.__setattr__(self, "next_reference_valid_mask", next_reference_valid_mask)
        object.__setattr__(self, "terminal_outcome_if_done", terminal_outcome)

        self.validate_local()

    @property
    def action_dim(self) -> int:
        return int(self.executed_action_chunk.shape[1])

    @property
    def valid_steps(self) -> int:
        return _prefix_true_count(self.valid_mask, "valid_mask")

    def actor_input_reference_chunk(self, *, drop_reference: bool) -> np.ndarray:
        """Reference passed to the actor for one training sample.

        Reference-action dropout is a training-time transformation; this method
        returns a copy and never mutates the replay record.
        """

        if drop_reference:
            return np.zeros_like(self.train_reference_chunk)
        return self.train_reference_chunk.copy()

    def next_actor_input_reference_chunk(self) -> np.ndarray:
        """Reference passed to the target actor at ``x_next`` for Eq. 3."""

        return self.next_train_reference_chunk.copy()

    def as_training_item(self, *, drop_reference: bool = False) -> dict[str, Any]:
        """Return a learner-facing item with replay reference and actor input split."""

        return {
            "transition_id": self.transition_id,
            "obs_key": self.obs_key,
            "next_obs_key": self.next_obs_key,
            "rl_token_key": self.rl_token_key,
            "proprio_key": self.proprio_key,
            "next_rl_token_key": self.next_rl_token_key,
            "next_proprio_key": self.next_proprio_key,
            "executed_action_chunk": self.executed_action_chunk.copy(),
            "regularizer_reference_chunk": self.train_reference_chunk.copy(),
            "actor_input_reference_chunk": self.actor_input_reference_chunk(
                drop_reference=drop_reference
            ),
            "next_regularizer_reference_chunk": self.next_train_reference_chunk.copy(),
            "next_actor_input_reference_chunk": self.next_actor_input_reference_chunk(),
            "valid_mask": self.valid_mask.copy(),
            "reference_valid_mask": self.reference_valid_mask.copy(),
            "next_reference_valid_mask": self.next_reference_valid_mask.copy(),
            "intervention_mask": self.intervention_mask.copy(),
            "reward_chunk": self.reward_chunk.copy(),
            "discounted_reward_sum": float(self.discounted_reward_sum),
            "discount": float(self.discount),
            "done": bool(self.done),
        }

    def validate_local(self) -> None:
        for name in (
            "transition_id",
            "segment_id",
            "episode_id",
            "obs_key",
            "next_obs_key",
            "rl_token_key",
            "proprio_key",
            "next_rl_token_key",
            "next_proprio_key",
        ):
            _require(bool(getattr(self, name)), f"{name} must be non-empty")

        _require(self.chunk_len > 0, "chunk_len must be positive")
        _require(
            self.vla_horizon > self.chunk_len,
            "vla_horizon must be greater than chunk_len because RLT uses C < H",
        )
        _require(self.chunk_stride > 0, "chunk_stride must be positive")
        _require(self.decision_t >= 0, "decision_t must be non-negative")
        _require(self.sample_t >= self.decision_t, "sample_t must be >= decision_t")
        _require(self.chunk_offset == self.sample_t - self.decision_t, "chunk_offset is misaligned")
        _require(0 <= self.chunk_offset < self.vla_horizon, "chunk_offset must be in [0, H)")
        _require(
            self.chunk_offset % self.chunk_stride == 0,
            "chunk_offset must be an integer multiple of chunk_stride",
        )

        valid_steps = _prefix_true_count(self.valid_mask, "valid_mask")
        _require(valid_steps > 0, "valid_mask must contain at least one valid action")
        next_reference_valid_steps = _prefix_true_count(
            self.next_reference_valid_mask,
            "next_reference_valid_mask",
        )
        _require(
            np.array_equal(self.reference_valid_mask, self.valid_mask),
            "reference_valid_mask must match valid_mask for RLT replay records",
        )
        _require(
            not bool((self.intervention_mask & ~self.valid_mask).any()),
            "intervention_mask cannot be true on padded timesteps",
        )
        _require(
            not bool((self.reward_chunk[~self.valid_mask] != 0.0).any()),
            "reward_chunk must be zero on padded timesteps",
        )
        _require(
            self.next_t == self.sample_t + valid_steps,
            "next_t must equal sample_t plus the number of valid chunk steps",
        )
        _require(
            not bool(
                (self.next_train_reference_chunk[~self.next_reference_valid_mask] != 0.0).any()
            ),
            "next_train_reference_chunk must be zero on padded timesteps",
        )

        if self.collection_stage is CollectionStage.VLA_WARMUP:
            for idx in range(self.chunk_len):
                if self.valid_mask[idx]:
                    _require(
                        self.behavior_source[idx] is BehaviorSource.VLA,
                        "VLA warmup transitions must execute the VLA reference policy",
                    )
                    _require(
                        not bool(self.intervention_mask[idx]),
                        "VLA warmup transitions cannot contain intervention steps",
                    )

        for idx in range(self.chunk_len):
            if not self.valid_mask[idx]:
                continue

            if self.intervention_mask[idx]:
                _require(
                    self.behavior_source[idx] is BehaviorSource.HUMAN,
                    f"behavior_source[{idx}] must be HUMAN during intervention",
                )
                _require(
                    self.reference_source[idx] is ReferenceSource.HUMAN,
                    f"reference_source[{idx}] must be HUMAN during intervention",
                )
                _require(
                    self.human_correction_chunk is not None,
                    "human_correction_chunk is required for intervention steps",
                )
                _require_chunk_equal(
                    "executed_action_chunk",
                    self.executed_action_chunk,
                    "human_correction_chunk",
                    self.human_correction_chunk,
                    idx,
                )
                _require_chunk_equal(
                    "train_reference_chunk",
                    self.train_reference_chunk,
                    "human_correction_chunk",
                    self.human_correction_chunk,
                    idx,
                )
                continue

            _require(
                self.reference_source[idx] is ReferenceSource.VLA,
                f"reference_source[{idx}] must be VLA without intervention",
            )
            _require(self.vla_reference_chunk is not None, "vla_reference_chunk is required")
            _require_chunk_equal(
                "train_reference_chunk",
                self.train_reference_chunk,
                "vla_reference_chunk",
                self.vla_reference_chunk,
                idx,
            )

            if self.behavior_source[idx] is BehaviorSource.VLA:
                _require_chunk_equal(
                    "executed_action_chunk",
                    self.executed_action_chunk,
                    "vla_reference_chunk",
                    self.vla_reference_chunk,
                    idx,
                )
            elif self.behavior_source[idx] is BehaviorSource.ACTOR:
                _require(
                    self.actor_proposed_chunk is not None,
                    "actor_proposed_chunk is required for actor-executed steps",
                )
                _require_chunk_equal(
                    "executed_action_chunk",
                    self.executed_action_chunk,
                    "actor_proposed_chunk",
                    self.actor_proposed_chunk,
                    idx,
                )
            else:
                raise ValueError(f"behavior_source[{idx}] cannot be HUMAN without intervention")

        if self.done:
            _require(
                self.terminal_outcome_if_done is not None,
                "terminal_outcome_if_done is required when done=True",
            )
            _require(self.discount == 0.0, "terminal transitions must have discount=0")
            _require(
                next_reference_valid_steps == 0,
                "terminal transitions must not carry a valid next reference chunk",
            )
        else:
            _require(
                self.terminal_outcome_if_done is None,
                "terminal_outcome_if_done must be None when done=False",
            )
            _require(self.discount > 0.0, "non-terminal transitions must have positive discount")
            _require(
                next_reference_valid_steps > 0,
                "non-terminal transitions must carry a valid next reference chunk",
            )
            _require(
                self.discounted_reward_sum == 0.0,
                "non-terminal RLT discounted_reward_sum must be 0",
            )
            _require(
                not bool((self.reward_chunk[self.valid_mask] != 0.0).any()),
                "non-terminal RLT reward_chunk must be all zeros",
            )

    def validate_against_segment(self, segment: CriticalPhaseSegment, *, gamma: float) -> None:
        """Validate critical-phase membership, terminal reward, and C-step discount."""

        segment.validate()
        _require(self.segment_id == segment.segment_id, "segment_id does not match segment")
        _require(self.episode_id == segment.episode_id, "episode_id does not match segment")
        _require(
            segment.handoff_t <= self.decision_t <= self.sample_t,
            "decision_t must be inside the RLT critical phase",
        )
        _require(segment.contains_transition_window(self.sample_t, self.next_t), "not in segment")

        expected_done = self.next_t == segment.terminal_t
        _require(self.done == expected_done, "done must indicate whether next_t reaches terminal_t")

        if self.done:
            _require(
                self.terminal_outcome_if_done == segment.terminal_outcome,
                "terminal_outcome_if_done must match segment terminal_outcome",
            )
            expected_reward_chunk = np.zeros(self.chunk_len, dtype=np.float32)
            expected_reward_chunk[self.valid_steps - 1] = segment.terminal_reward
            _require(
                _allclose(self.reward_chunk, expected_reward_chunk),
                "terminal reward_chunk must place the sparse label on the final valid step",
            )
            expected_reward = float(
                sum((gamma**idx) * float(self.reward_chunk[idx]) for idx in range(self.valid_steps))
            )
            _require(
                math.isclose(
                    self.discounted_reward_sum, expected_reward, rel_tol=1.0e-6, abs_tol=1.0e-8
                ),
                "terminal discounted_reward_sum "
                f"{self.discounted_reward_sum} != discounted reward_chunk sum {expected_reward}",
            )
            _require(self.discount == 0.0, "terminal discount must be 0")
        else:
            _require(
                self.discounted_reward_sum == 0.0, "non-terminal discounted_reward_sum must be 0"
            )
            expected_discount = gamma**self.valid_steps
            _require(
                math.isclose(self.discount, expected_discount, rel_tol=1.0e-6, abs_tol=1.0e-8),
                f"discount {self.discount} != gamma**valid_steps {expected_discount}",
            )
