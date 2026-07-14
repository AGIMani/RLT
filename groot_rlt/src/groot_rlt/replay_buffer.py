# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay buffer for RL Token actor-critic training.

The RLT paper uses an off-policy replay buffer that aggregates VLA warmup
transitions, online RL rollouts, and optional human interventions. Each stored
transition already contains the executed action chunk and the training reference
chunk. Reference-action dropout is applied only when sampling a training batch;
the canonical replay record is never mutated.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from groot_rlt.collate import TensorResolver, collate_rlt_batch
from groot_rlt.replay_schema import CriticalPhaseSegment, RLTTransition


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RLTReplayBatch:
    """Sampled RLT replay batch.

    The batch stores only replay records and the per-sample reference dropout
    mask. ``collate_rlt_batch`` performs the learner-facing stacking.
    """

    transitions: tuple[RLTTransition, ...]
    reference_dropout_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.reference_dropout_mask, dtype=np.bool_)
        _require(mask.shape == (len(self.transitions),), "reference_dropout_mask shape mismatch")
        object.__setattr__(self, "reference_dropout_mask", mask)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def batch_size(self) -> int:
        return len(self.transitions)

    def as_training_batch(self, *, tensor_resolver: TensorResolver | None = None) -> dict[str, Any]:
        """Return a stacked dictionary suitable for actor-critic training."""

        return collate_rlt_batch(
            self.transitions,
            reference_dropout_mask=self.reference_dropout_mask,
            tensor_resolver=tensor_resolver,
        )


class RLTReplayBuffer:
    """Uniform off-policy replay buffer for validated ``RLTTransition`` records."""

    def __init__(
        self,
        *,
        capacity: int | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity is not None:
            _require(capacity > 0, "capacity must be positive when provided")
        self.capacity = capacity
        self._records: OrderedDict[str, RLTTransition] = OrderedDict()
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, transition_id: object) -> bool:
        return transition_id in self._records

    def __iter__(self):
        return iter(self._records.values())

    def clear(self) -> None:
        self._records.clear()

    def add(
        self,
        transition: RLTTransition,
        *,
        segment: CriticalPhaseSegment | None = None,
        gamma: float | None = None,
        replace: bool = False,
    ) -> RLTTransition | None:
        """Add one transition and return the evicted transition, if any.

        ``RLTTransition`` validates itself at construction. Passing ``segment``
        and ``gamma`` additionally checks critical-phase membership and Eq. 3
        discount/reward consistency before storage.
        """

        transition.validate_local()
        if segment is not None:
            _require(gamma is not None, "gamma is required when segment validation is requested")
            transition.validate_against_segment(segment, gamma=float(gamma))

        if transition.transition_id in self._records:
            _require(replace, f"duplicate transition_id: {transition.transition_id}")
            self._records[transition.transition_id] = transition
            self._records.move_to_end(transition.transition_id)
            return None

        self._records[transition.transition_id] = transition
        if self.capacity is not None and len(self._records) > self.capacity:
            _, evicted = self._records.popitem(last=False)
            return evicted
        return None

    def extend(
        self,
        transitions: Iterable[RLTTransition],
        *,
        segment: CriticalPhaseSegment | None = None,
        gamma: float | None = None,
        replace: bool = False,
    ) -> list[RLTTransition]:
        """Add multiple transitions and return all FIFO-evicted records."""

        evicted = []
        for transition in transitions:
            item = self.add(transition, segment=segment, gamma=gamma, replace=replace)
            if item is not None:
                evicted.append(item)
        return evicted

    def get(self, transition_id: str) -> RLTTransition:
        return self._records[transition_id]

    def sample(
        self,
        batch_size: int,
        *,
        reference_dropout_prob: float = 0.5,
        replace: bool = True,
    ) -> RLTReplayBatch:
        """Uniformly sample a learner batch from replay.

        ``reference_dropout_prob`` follows the paper's reference-action dropout:
        for a random subset of sampled transitions, the actor input reference is
        zeroed while the regularization target remains the stored reference.
        """

        _require(batch_size > 0, "batch_size must be positive")
        _require(len(self._records) > 0, "cannot sample from an empty replay buffer")
        _require(
            0.0 <= reference_dropout_prob <= 1.0,
            "reference_dropout_prob must be in [0, 1]",
        )
        if not replace:
            _require(
                batch_size <= len(self._records),
                "batch_size cannot exceed buffer length when replace=False",
            )

        records = tuple(self._records.values())
        indices = self._rng.choice(len(records), size=batch_size, replace=replace)
        dropout_mask = self._rng.random(batch_size) < float(reference_dropout_prob)
        transitions = tuple(records[int(index)] for index in indices)
        return RLTReplayBatch(
            transitions=transitions,
            reference_dropout_mask=dropout_mask,
        )

    def sample_training_batch(
        self,
        batch_size: int,
        *,
        reference_dropout_prob: float = 0.5,
        replace: bool = True,
        tensor_resolver: TensorResolver | None = None,
    ) -> dict[str, Any]:
        """Sample and immediately stack a learner-facing training batch."""

        return self.sample(
            batch_size,
            reference_dropout_prob=reference_dropout_prob,
            replace=replace,
        ).as_training_batch(tensor_resolver=tensor_resolver)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._records.keys())

    def by_collection_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for transition in self._records.values():
            key = transition.collection_stage.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_episode(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for transition in self._records.values():
            counts[transition.episode_id] = counts.get(transition.episode_id, 0) + 1
        return counts


def make_replay_buffer(
    transitions: Sequence[RLTTransition] = (),
    *,
    capacity: int | None = None,
    seed: int | None = None,
) -> RLTReplayBuffer:
    """Convenience constructor for tests and small offline replay builds."""

    buffer = RLTReplayBuffer(capacity=capacity, seed=seed)
    buffer.extend(transitions)
    return buffer
