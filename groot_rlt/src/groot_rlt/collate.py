# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch collation utilities for RL Token actor-critic training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from groot_rlt.replay_schema import RLTTransition

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


def _stack_arrays(name: str, values: Sequence[Any]) -> np.ndarray:
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except ValueError as exc:
        shapes = [np.asarray(value).shape for value in values]
        raise ValueError(f"{name} entries must have matching shapes, got {shapes}") from exc


def _resolve_stack(
    *,
    name: str,
    resolver: TensorResolver,
    keys: Sequence[str],
) -> np.ndarray:
    return _stack_arrays(name, [_resolve_tensor(resolver, key) for key in keys])


def _dropout_mask(
    reference_dropout_mask: Sequence[bool] | np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if reference_dropout_mask is None:
        return np.zeros(batch_size, dtype=np.bool_)
    mask = np.asarray(reference_dropout_mask, dtype=np.bool_)
    _require(mask.shape == (batch_size,), "reference_dropout_mask shape mismatch")
    return mask


def _source_values(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [[item.value if hasattr(item, "value") else str(item) for item in row] for row in values],
        dtype=object,
    )


def collate_rlt_batch(
    transitions: Sequence[RLTTransition],
    *,
    reference_dropout_mask: Sequence[bool] | np.ndarray | None = None,
    tensor_resolver: TensorResolver | None = None,
) -> dict[str, Any]:
    """Collate replay transitions into the learner-facing RLT batch.

    The resulting batch mirrors the RLT update equations:

    * critic input: ``x=(z_rl, proprio)``, ``executed_action_chunk``, ``reward_chunk``,
      ``discount``, ``done``, ``x_next``, and the target actor's next reference chunk;
    * actor input: ``x`` plus ``actor_input_reference_chunk``;
    * actor regularization target: ``regularizer_reference_chunk``.

    Reference dropout is applied only to ``actor_input_reference_chunk``. The
    stored ``regularizer_reference_chunk`` remains unchanged.
    """

    transitions = tuple(transitions)
    _require(transitions, "cannot collate an empty RLT batch")

    batch_size = len(transitions)
    dropout_mask = _dropout_mask(reference_dropout_mask, batch_size=batch_size)
    items = tuple(
        transition.as_training_item(drop_reference=bool(drop_reference))
        for transition, drop_reference in zip(transitions, dropout_mask)
    )

    batch: dict[str, Any] = {
        "transition_id": tuple(item["transition_id"] for item in items),
        "episode_id": tuple(transition.episode_id for transition in transitions),
        "segment_id": tuple(transition.segment_id for transition in transitions),
        "collection_stage": tuple(transition.collection_stage.value for transition in transitions),
        "decision_t": np.asarray(
            [transition.decision_t for transition in transitions], dtype=np.int64
        ),
        "sample_t": np.asarray([transition.sample_t for transition in transitions], dtype=np.int64),
        "next_t": np.asarray([transition.next_t for transition in transitions], dtype=np.int64),
        "chunk_offset": np.asarray(
            [transition.chunk_offset for transition in transitions], dtype=np.int64
        ),
        "chunk_stride": np.asarray(
            [transition.chunk_stride for transition in transitions], dtype=np.int64
        ),
        "chunk_len": np.asarray(
            [transition.chunk_len for transition in transitions], dtype=np.int64
        ),
        "vla_horizon": np.asarray(
            [transition.vla_horizon for transition in transitions], dtype=np.int64
        ),
        "obs_key": tuple(item["obs_key"] for item in items),
        "next_obs_key": tuple(item["next_obs_key"] for item in items),
        "rl_token_key": tuple(item["rl_token_key"] for item in items),
        "proprio_key": tuple(item["proprio_key"] for item in items),
        "next_rl_token_key": tuple(item["next_rl_token_key"] for item in items),
        "next_proprio_key": tuple(item["next_proprio_key"] for item in items),
        "executed_action_chunk": _stack_arrays(
            "executed_action_chunk",
            [item["executed_action_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "critic_action_chunk": _stack_arrays(
            "critic_action_chunk",
            [item["executed_action_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "regularizer_reference_chunk": _stack_arrays(
            "regularizer_reference_chunk",
            [item["regularizer_reference_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "actor_input_reference_chunk": _stack_arrays(
            "actor_input_reference_chunk",
            [item["actor_input_reference_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "next_regularizer_reference_chunk": _stack_arrays(
            "next_regularizer_reference_chunk",
            [item["next_regularizer_reference_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "next_actor_input_reference_chunk": _stack_arrays(
            "next_actor_input_reference_chunk",
            [item["next_actor_input_reference_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "valid_mask": _stack_arrays("valid_mask", [item["valid_mask"] for item in items]).astype(
            np.bool_, copy=False
        ),
        "reference_valid_mask": _stack_arrays(
            "reference_valid_mask",
            [item["reference_valid_mask"] for item in items],
        ).astype(np.bool_, copy=False),
        "next_reference_valid_mask": _stack_arrays(
            "next_reference_valid_mask",
            [item["next_reference_valid_mask"] for item in items],
        ).astype(np.bool_, copy=False),
        "intervention_mask": _stack_arrays(
            "intervention_mask",
            [item["intervention_mask"] for item in items],
        ).astype(np.bool_, copy=False),
        "reward_chunk": _stack_arrays(
            "reward_chunk",
            [item["reward_chunk"] for item in items],
        ).astype(np.float32, copy=False),
        "discounted_reward_sum": np.asarray(
            [item["discounted_reward_sum"] for item in items], dtype=np.float32
        ),
        "discount": np.asarray([item["discount"] for item in items], dtype=np.float32),
        "done": np.asarray([item["done"] for item in items], dtype=np.bool_),
        "reference_dropout_mask": dropout_mask.copy(),
        "behavior_source": _source_values(
            [transition.behavior_source for transition in transitions]
        ),
        "reference_source": _source_values(
            [transition.reference_source for transition in transitions]
        ),
    }

    if tensor_resolver is not None:
        rl_token = _resolve_stack(
            name="rl_token",
            resolver=tensor_resolver,
            keys=batch["rl_token_key"],
        ).astype(np.float32, copy=False)
        proprio = _resolve_stack(
            name="proprio",
            resolver=tensor_resolver,
            keys=batch["proprio_key"],
        ).astype(np.float32, copy=False)
        next_rl_token = _resolve_stack(
            name="next_rl_token",
            resolver=tensor_resolver,
            keys=batch["next_rl_token_key"],
        ).astype(np.float32, copy=False)
        next_proprio = _resolve_stack(
            name="next_proprio",
            resolver=tensor_resolver,
            keys=batch["next_proprio_key"],
        ).astype(np.float32, copy=False)
        batch.update(
            {
                "rl_token": rl_token,
                "proprio": proprio,
                "next_rl_token": next_rl_token,
                "next_proprio": next_proprio,
                "x": {"rl_token": rl_token, "proprio": proprio},
                "x_next": {"rl_token": next_rl_token, "proprio": next_proprio},
            }
        )

    return batch
