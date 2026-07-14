# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build RLT replay transitions from episode-level collection records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from groot_rlt.episode_schema import (
    ChunkDecisionRecord,
    EpisodeSource,
    EpisodeStepRecord,
    InterventionRecord,
    RLTEpisodeRecord,
    StepBehaviorSource,
    StepReferenceSource,
    validate_episode_components,
)
from groot_rlt.replay_schema import (
    BehaviorSource,
    CollectionStage,
    ReferenceSource,
    RLTTransition,
    compose_train_reference_chunk,
)

TensorResolver = Mapping[str, Any] | Callable[[str], Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_tensor(resolver: TensorResolver, key: str) -> Any:
    if isinstance(resolver, Mapping):
        try:
            return resolver[key]
        except KeyError as exc:
            raise KeyError(f"tensor key not found: {key}") from exc
    return resolver(key)


def _as_action_vector(value: Any, *, key: str, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    _require(
        array.shape == (action_dim,),
        f"{key} must have shape ({action_dim},), got {array.shape}",
    )
    return array


def _as_action_horizon(value: Any, *, key: str, horizon: int, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    _require(
        array.shape == (horizon, action_dim),
        f"{key} must store one decision horizon with shape "
        f"({horizon}, {action_dim}), got {array.shape}",
    )
    return array


def _zero_chunk(*, chunk_len: int, action_dim: int) -> np.ndarray:
    return np.zeros((chunk_len, action_dim), dtype=np.float32)


def _prefix_mask(*, valid_steps: int, chunk_len: int) -> np.ndarray:
    mask = np.zeros(chunk_len, dtype=np.bool_)
    mask[:valid_steps] = True
    return mask


def _transition_id(episode_id: str, decision_t: int, sample_t: int) -> str:
    return f"{episode_id}:decision_{decision_t:06d}:sample_{sample_t:06d}"


def _infer_collection_stage(episode: RLTEpisodeRecord) -> CollectionStage:
    if episode.source is EpisodeSource.VLA_WARMUP:
        return CollectionStage.VLA_WARMUP
    if episode.source in {
        EpisodeSource.ONLINE_RL,
        EpisodeSource.HUMAN_INTERVENTION,
        EpisodeSource.MIXED,
    }:
        return CollectionStage.ONLINE_RL
    raise ValueError(
        "Cannot infer RLT collection_stage from demo episodes. "
        "Pass collection_stage explicitly after converting demo actions into VLA references."
    )


def _behavior_source_for_step(step: EpisodeStepRecord) -> BehaviorSource:
    if step.behavior_source is StepBehaviorSource.VLA:
        return BehaviorSource.VLA
    if step.behavior_source is StepBehaviorSource.ACTOR:
        return BehaviorSource.ACTOR
    if step.behavior_source is StepBehaviorSource.HUMAN_INTERVENTION:
        return BehaviorSource.HUMAN
    raise ValueError(
        f"step t={step.t} has behavior_source={step.behavior_source.value!r}, "
        "which cannot be represented as an RLT replay transition"
    )


def _validate_non_intervention_reference(step: EpisodeStepRecord) -> None:
    if step.reference_source is StepReferenceSource.VLA:
        return
    raise ValueError(
        f"step t={step.t} has reference_source={step.reference_source.value!r}; "
        "RLT replay requires VLA reference outside human intervention"
    )


def _check_decision_alignment(
    decision: ChunkDecisionRecord,
    decision_step: EpisodeStepRecord,
) -> None:
    _require(
        decision.obs_key == decision_step.obs_key,
        f"decision_t={decision.decision_t} obs_key does not match the step record",
    )
    _require(
        decision.rl_token_key == decision_step.rl_token_key,
        f"decision_t={decision.decision_t} rl_token_key does not match the step record",
    )
    _require(
        decision.proprio_key == decision_step.proprio_key,
        f"decision_t={decision.decision_t} proprio_key does not match the step record",
    )


def _find_covering_horizon(
    decision_horizons: Mapping[int, np.ndarray],
    *,
    start_t: int,
    valid_steps: int,
) -> tuple[int, np.ndarray] | None:
    if valid_steps <= 0:
        return None

    candidates: list[tuple[int, np.ndarray]] = []
    for decision_t, horizon in decision_horizons.items():
        offset = start_t - decision_t
        if 0 <= offset and offset + valid_steps <= horizon.shape[0]:
            candidates.append((decision_t, horizon))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


@dataclass(frozen=True)
class EpisodeTransitionBuilder:
    """Convert one collected RLT episode into validated replay transitions.

    VLA references are resolved per ``decision_t`` as full horizons with shape
    ``[vla_horizon, action_dim]``. A transition is uniquely identified by
    ``(decision_t, sample_t)`` so the same ``sample_t`` may appear under multiple
    VLA decision contexts. For Eq. 3 targets, the builder also stores the
    reference chunk aligned to ``x_next`` using the latest decision horizon that
    covers ``next_t``.
    """

    tensor_resolver: TensorResolver
    gamma: float
    allow_terminal_padding: bool = True
    collection_stage: CollectionStage | str | None = None
    validate_episode: bool = True

    def __post_init__(self) -> None:
        _require(0.0 <= self.gamma < 1.0, "gamma must be in [0, 1)")

    def build(
        self,
        *,
        episode: RLTEpisodeRecord,
        steps: Sequence[EpisodeStepRecord],
        chunk_decisions: Sequence[ChunkDecisionRecord],
        interventions: Sequence[InterventionRecord] = (),
    ) -> list[RLTTransition]:
        if self.validate_episode:
            validate_episode_components(episode, steps, chunk_decisions, interventions)

        collection_stage = (
            _infer_collection_stage(episode)
            if self.collection_stage is None
            else CollectionStage(self.collection_stage)
        )
        segment = episode.critical_segment()
        steps_by_t = {step.t: step for step in steps}
        transitions: list[RLTTransition] = []
        seen_transition_keys: set[tuple[int, int]] = set()
        sorted_decisions = sorted(chunk_decisions, key=lambda item: item.decision_t)
        decision_horizons: dict[int, np.ndarray] = {}

        for decision in sorted_decisions:
            decision_step = steps_by_t[decision.decision_t]
            _check_decision_alignment(decision, decision_step)
            decision_horizons[decision.decision_t] = _as_action_horizon(
                _resolve_tensor(self.tensor_resolver, decision.vla_reference_chunk_key),
                key=decision.vla_reference_chunk_key,
                horizon=decision.vla_horizon,
                action_dim=episode.action_dim,
            )

        for decision in sorted_decisions:
            vla_reference_horizon = decision_horizons[decision.decision_t]

            for sample_t in self._sample_times_for_decision(
                episode,
                decision,
                decision_horizons,
            ):
                transition_key = (decision.decision_t, sample_t)
                _require(
                    transition_key not in seen_transition_keys,
                    f"duplicate transition for decision_t={decision.decision_t}, sample_t={sample_t}",
                )
                seen_transition_keys.add(transition_key)

                transition = self._build_one_transition(
                    episode=episode,
                    steps_by_t=steps_by_t,
                    decision=decision,
                    vla_reference_horizon=vla_reference_horizon,
                    decision_horizons=decision_horizons,
                    collection_stage=collection_stage,
                    sample_t=sample_t,
                )
                transition.validate_against_segment(segment, gamma=self.gamma)
                transitions.append(transition)

        return transitions

    def _sample_times_for_decision(
        self,
        episode: RLTEpisodeRecord,
        decision: ChunkDecisionRecord,
        decision_horizons: Mapping[int, np.ndarray],
    ) -> list[int]:
        sample_times = []
        for chunk_offset in range(0, decision.vla_horizon, episode.chunk_stride):
            sample_t = decision.decision_t + chunk_offset
            if sample_t >= episode.terminal_t:
                break

            valid_steps = min(episode.chunk_len, episode.terminal_t - sample_t)
            done = sample_t + valid_steps == episode.terminal_t
            reference_fits = chunk_offset + valid_steps <= decision.vla_horizon
            if not reference_fits:
                continue

            needs_padding = valid_steps < episode.chunk_len
            if needs_padding and not done:
                continue
            if needs_padding and not self.allow_terminal_padding:
                raise ValueError(
                    f"terminal padding would be required for decision_t={decision.decision_t}, "
                    f"sample_t={sample_t}"
                )
            if not done and chunk_offset + episode.chunk_len > decision.vla_horizon:
                continue
            if not done:
                next_t = sample_t + valid_steps
                next_reference_valid_steps = min(episode.chunk_len, episode.terminal_t - next_t)
                if (
                    _find_covering_horizon(
                        decision_horizons,
                        start_t=next_t,
                        valid_steps=next_reference_valid_steps,
                    )
                    is None
                ):
                    continue

            sample_times.append(sample_t)
        return sample_times

    def _build_train_reference_for_window(
        self,
        *,
        episode: RLTEpisodeRecord,
        steps_by_t: Mapping[int, EpisodeStepRecord],
        vla_reference_horizon: np.ndarray,
        decision_t: int,
        start_t: int,
        valid_steps: int,
    ) -> np.ndarray:
        chunk_offset = start_t - decision_t
        vla_reference_chunk = _zero_chunk(
            chunk_len=episode.chunk_len,
            action_dim=episode.action_dim,
        )
        vla_reference_chunk[:valid_steps] = vla_reference_horizon[
            chunk_offset : chunk_offset + valid_steps
        ]

        intervention_mask = np.zeros(episode.chunk_len, dtype=np.bool_)
        human_correction_chunk = _zero_chunk(
            chunk_len=episode.chunk_len,
            action_dim=episode.action_dim,
        )
        has_intervention_steps = False
        for chunk_index in range(valid_steps):
            step = steps_by_t[start_t + chunk_index]
            if step.intervention:
                has_intervention_steps = True
                intervention_mask[chunk_index] = True
                human_correction_chunk[chunk_index] = _as_action_vector(
                    _resolve_tensor(self.tensor_resolver, step.human_correction_action_key or ""),
                    key=step.human_correction_action_key or "",
                    action_dim=episode.action_dim,
                )
                continue
            _validate_non_intervention_reference(step)

        return compose_train_reference_chunk(
            vla_reference_chunk=vla_reference_chunk,
            human_correction_chunk=human_correction_chunk if has_intervention_steps else None,
            intervention_mask=intervention_mask,
        )

    def _build_one_transition(
        self,
        *,
        episode: RLTEpisodeRecord,
        steps_by_t: Mapping[int, EpisodeStepRecord],
        decision: ChunkDecisionRecord,
        vla_reference_horizon: np.ndarray,
        decision_horizons: Mapping[int, np.ndarray],
        collection_stage: CollectionStage,
        sample_t: int,
    ) -> RLTTransition:
        chunk_offset = sample_t - decision.decision_t
        valid_steps = min(episode.chunk_len, episode.terminal_t - sample_t)
        next_t = sample_t + valid_steps
        done = next_t == episode.terminal_t
        needs_padding = valid_steps < episode.chunk_len
        _require(
            not needs_padding or done,
            "padding is allowed only for terminal tail transitions",
        )
        _require(
            not needs_padding or self.allow_terminal_padding,
            "terminal padding is disabled for this builder",
        )

        valid_mask = _prefix_mask(valid_steps=valid_steps, chunk_len=episode.chunk_len)
        vla_reference_chunk = _zero_chunk(
            chunk_len=episode.chunk_len, action_dim=episode.action_dim
        )
        vla_reference_chunk[:valid_steps] = vla_reference_horizon[
            chunk_offset : chunk_offset + valid_steps
        ]

        executed_action_chunk = _zero_chunk(
            chunk_len=episode.chunk_len, action_dim=episode.action_dim
        )
        actor_proposed_chunk = _zero_chunk(
            chunk_len=episode.chunk_len, action_dim=episode.action_dim
        )
        human_correction_chunk = _zero_chunk(
            chunk_len=episode.chunk_len, action_dim=episode.action_dim
        )
        has_actor_steps = False
        has_intervention_steps = False
        behavior_source: list[BehaviorSource] = [BehaviorSource.VLA] * episode.chunk_len
        reference_source: list[ReferenceSource] = [ReferenceSource.VLA] * episode.chunk_len
        intervention_mask = np.zeros(episode.chunk_len, dtype=np.bool_)

        for chunk_index in range(valid_steps):
            step = steps_by_t[sample_t + chunk_index]
            behavior = _behavior_source_for_step(step)
            behavior_source[chunk_index] = behavior

            executed_action_chunk[chunk_index] = _as_action_vector(
                _resolve_tensor(self.tensor_resolver, step.executed_action_key),
                key=step.executed_action_key,
                action_dim=episode.action_dim,
            )

            if step.intervention:
                has_intervention_steps = True
                intervention_mask[chunk_index] = True
                reference_source[chunk_index] = ReferenceSource.HUMAN
                human_correction_chunk[chunk_index] = _as_action_vector(
                    _resolve_tensor(self.tensor_resolver, step.human_correction_action_key or ""),
                    key=step.human_correction_action_key or "",
                    action_dim=episode.action_dim,
                )
                continue

            _validate_non_intervention_reference(step)
            reference_source[chunk_index] = ReferenceSource.VLA
            if behavior is BehaviorSource.ACTOR:
                has_actor_steps = True
                actor_proposed_chunk[chunk_index] = _as_action_vector(
                    _resolve_tensor(self.tensor_resolver, step.actor_proposed_action_key or ""),
                    key=step.actor_proposed_action_key or "",
                    action_dim=episode.action_dim,
                )

        train_reference_chunk = compose_train_reference_chunk(
            vla_reference_chunk=vla_reference_chunk,
            human_correction_chunk=human_correction_chunk if has_intervention_steps else None,
            intervention_mask=intervention_mask,
        )

        reward_chunk = np.zeros(episode.chunk_len, dtype=np.float32)
        if done:
            reward_chunk[valid_steps - 1] = episode.terminal_outcome.reward
        discounted_reward_sum = float(
            sum((self.gamma**idx) * float(reward_chunk[idx]) for idx in range(valid_steps))
        )
        next_reference_valid_steps = (
            0 if done else min(episode.chunk_len, episode.terminal_t - next_t)
        )
        next_reference_valid_mask = _prefix_mask(
            valid_steps=next_reference_valid_steps,
            chunk_len=episode.chunk_len,
        )
        next_train_reference_chunk = _zero_chunk(
            chunk_len=episode.chunk_len,
            action_dim=episode.action_dim,
        )
        if not done:
            next_covering_horizon = _find_covering_horizon(
                decision_horizons,
                start_t=next_t,
                valid_steps=next_reference_valid_steps,
            )
            _require(
                next_covering_horizon is not None,
                f"no VLA reference horizon covers next_t={next_t} for "
                f"decision_t={decision.decision_t}, sample_t={sample_t}",
            )
            next_decision_t, next_vla_reference_horizon = next_covering_horizon
            next_train_reference_chunk = self._build_train_reference_for_window(
                episode=episode,
                steps_by_t=steps_by_t,
                vla_reference_horizon=next_vla_reference_horizon,
                decision_t=next_decision_t,
                start_t=next_t,
                valid_steps=next_reference_valid_steps,
            )

        start_step = steps_by_t[sample_t]
        next_step = steps_by_t[next_t]
        return RLTTransition(
            transition_id=_transition_id(episode.episode_id, decision.decision_t, sample_t),
            segment_id=segment_id_for_episode(episode),
            episode_id=episode.episode_id,
            collection_stage=collection_stage,
            decision_t=decision.decision_t,
            sample_t=sample_t,
            next_t=next_t,
            chunk_offset=chunk_offset,
            chunk_stride=episode.chunk_stride,
            chunk_len=episode.chunk_len,
            vla_horizon=episode.vla_horizon,
            obs_key=start_step.obs_key,
            next_obs_key=next_step.obs_key,
            rl_token_key=start_step.rl_token_key,
            proprio_key=start_step.proprio_key,
            next_rl_token_key=next_step.rl_token_key,
            next_proprio_key=next_step.proprio_key,
            executed_action_chunk=executed_action_chunk,
            train_reference_chunk=train_reference_chunk,
            reward_chunk=reward_chunk,
            valid_mask=valid_mask,
            reference_valid_mask=valid_mask.copy(),
            vla_reference_chunk=vla_reference_chunk,
            actor_proposed_chunk=actor_proposed_chunk if has_actor_steps else None,
            human_correction_chunk=human_correction_chunk if has_intervention_steps else None,
            next_train_reference_chunk=next_train_reference_chunk,
            next_reference_valid_mask=next_reference_valid_mask,
            behavior_source=behavior_source,
            reference_source=reference_source,
            intervention_mask=intervention_mask,
            discounted_reward_sum=discounted_reward_sum,
            discount=0.0 if done else self.gamma**valid_steps,
            done=done,
            terminal_outcome_if_done=episode.terminal_outcome if done else None,
        )


def segment_id_for_episode(episode: RLTEpisodeRecord) -> str:
    """Return the segment id used by ``RLTEpisodeRecord.critical_segment``."""

    return f"{episode.episode_id}:critical"


def build_rlt_transitions_from_episode(
    *,
    episode: RLTEpisodeRecord,
    steps: Sequence[EpisodeStepRecord],
    chunk_decisions: Sequence[ChunkDecisionRecord],
    tensor_resolver: TensorResolver,
    gamma: float,
    interventions: Sequence[InterventionRecord] = (),
    allow_terminal_padding: bool = True,
    collection_stage: CollectionStage | str | None = None,
    validate_episode: bool = True,
) -> list[RLTTransition]:
    """Convenience wrapper around ``EpisodeTransitionBuilder``."""

    return EpisodeTransitionBuilder(
        tensor_resolver=tensor_resolver,
        gamma=gamma,
        allow_terminal_padding=allow_terminal_padding,
        collection_stage=collection_stage,
        validate_episode=validate_episode,
    ).build(
        episode=episode,
        steps=steps,
        chunk_decisions=chunk_decisions,
        interventions=interventions,
    )
