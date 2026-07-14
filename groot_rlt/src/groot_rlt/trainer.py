# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level RLT training loop orchestration.

The paper's low-level update equations live in ``train.py``. This module wires
them into the full online RLT schedule: start from a warmup replay buffer,
alternate rollout collection with replay learning, use a high update-to-data
ratio, delay actor updates behind critic updates, soft-update the target critic,
and checkpoint the training state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from groot_rlt.collate import TensorResolver
from groot_rlt.legacy import legacy_module_aliases
from groot_rlt.replay_buffer import RLTReplayBuffer
from groot_rlt.replay_schema import CriticalPhaseSegment, RLTTransition
from groot_rlt.train import (
    RLTTrainConfig,
    soft_update_target_network,
    update_actor,
    update_critic,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RLTTrainerConfig:
    """System-level RLT loop defaults matching the paper's reported setup."""

    batch_size: int
    update_to_data_ratio: int = 5
    critic_updates_per_actor_update: int = 2
    target_update_tau: float = 0.005
    reference_dropout_prob: float = 0.5
    sample_with_replacement: bool = True
    min_replay_size: int = 1
    checkpoint_dir: str | Path | None = None
    checkpoint_every_rollouts: int | None = None
    checkpoint_every_learner_updates: int | None = None
    log_every_learner_updates: int = 1

    def validate(self) -> None:
        _require(self.batch_size > 0, "batch_size must be positive")
        _require(self.update_to_data_ratio > 0, "update_to_data_ratio must be positive")
        _require(
            self.critic_updates_per_actor_update > 0,
            "critic_updates_per_actor_update must be positive",
        )
        _require(0.0 <= self.target_update_tau <= 1.0, "target_update_tau must be in [0, 1]")
        _require(
            0.0 <= self.reference_dropout_prob <= 1.0,
            "reference_dropout_prob must be in [0, 1]",
        )
        _require(self.min_replay_size > 0, "min_replay_size must be positive")
        _require(self.log_every_learner_updates > 0, "log_every_learner_updates must be positive")
        if self.checkpoint_every_rollouts is not None:
            _require(
                self.checkpoint_every_rollouts > 0,
                "checkpoint_every_rollouts must be positive when provided",
            )
        if self.checkpoint_every_learner_updates is not None:
            _require(
                self.checkpoint_every_learner_updates > 0,
                "checkpoint_every_learner_updates must be positive when provided",
            )


@dataclass(frozen=True)
class RLTTrainerState:
    """Serializable counters for resume and logging."""

    rollout_iterations: int = 0
    learner_updates: int = 0
    critic_updates: int = 0
    actor_updates: int = 0
    target_updates: int = 0
    transitions_added: int = 0
    transitions_evicted: int = 0


@dataclass(frozen=True)
class RLTTrainerRollout:
    """Rollout result consumed by ``RLTTrainer.train_iteration``.

    ``update_units`` controls how many times the paper's update-to-data ratio is
    applied. If omitted, the number of new transitions is used.
    """

    transitions: tuple[RLTTransition, ...]
    update_units: int | None = None
    segment: CriticalPhaseSegment | None = None
    metrics: Mapping[str, float] | None = None

    @classmethod
    def from_transitions(
        cls,
        transitions: Iterable[RLTTransition],
        *,
        update_units: int | None = None,
        segment: CriticalPhaseSegment | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> "RLTTrainerRollout":
        return cls(
            transitions=tuple(transitions),
            update_units=update_units,
            segment=segment,
            metrics=metrics,
        )

    def __post_init__(self) -> None:
        _require(len(self.transitions) > 0, "rollout transitions must be non-empty")
        if self.update_units is not None:
            _require(self.update_units > 0, "update_units must be positive when provided")


class RolloutFn(Protocol):
    def __call__(self, trainer: "RLTTrainer") -> RLTTrainerRollout | Iterable[RLTTransition]: ...


class RLTTrainer:
    """Coordinate RLT rollout-learning alternation around the existing primitives."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: RLTReplayBuffer,
        train_config: RLTTrainConfig,
        trainer_config: RLTTrainerConfig,
        tensor_resolver: TensorResolver | None = None,
        device: torch.device | str | None = None,
        swanlab_run: Any | None = None,
        state: RLTTrainerState | None = None,
    ) -> None:
        train_config.validate()
        trainer_config.validate()
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.train_config = train_config
        self.trainer_config = trainer_config
        self.tensor_resolver = tensor_resolver
        self.device = None if device is None else torch.device(device)
        self.swanlab_run = swanlab_run
        self.state = state or RLTTrainerState()

    def add_transitions(
        self,
        transitions: Iterable[RLTTransition],
        *,
        segment: CriticalPhaseSegment | None = None,
        replace: bool = False,
    ) -> dict[str, float]:
        """Add rollout transitions to replay and update trainer counters."""

        transitions = tuple(transitions)
        _require(transitions, "cannot add an empty transition collection")
        evicted = self.replay_buffer.extend(
            transitions,
            segment=segment,
            gamma=self.train_config.gamma,
            replace=replace,
        )
        self.state = RLTTrainerState(
            rollout_iterations=self.state.rollout_iterations,
            learner_updates=self.state.learner_updates,
            critic_updates=self.state.critic_updates,
            actor_updates=self.state.actor_updates,
            target_updates=self.state.target_updates,
            transitions_added=self.state.transitions_added + len(transitions),
            transitions_evicted=self.state.transitions_evicted + len(evicted),
        )
        return {
            "rlt/trainer/transitions_added": float(len(transitions)),
            "rlt/trainer/transitions_evicted": float(len(evicted)),
            "rlt/trainer/replay_size": float(len(self.replay_buffer)),
        }

    def learn(self, num_learner_updates: int) -> list[dict[str, float]]:
        """Run replay updates using the configured critic-to-actor ratio."""

        self.trainer_config.validate()
        _require(num_learner_updates >= 0, "num_learner_updates must be non-negative")
        if num_learner_updates == 0:
            return []
        _require(
            len(self.replay_buffer) >= self.trainer_config.min_replay_size,
            "replay buffer does not yet contain enough transitions for learning",
        )
        if not self.trainer_config.sample_with_replacement:
            _require(
                self.trainer_config.batch_size <= len(self.replay_buffer),
                "batch_size cannot exceed replay length when sample_with_replacement=False",
            )

        metrics_history: list[dict[str, float]] = []
        for _ in range(num_learner_updates):
            step = self.state.learner_updates + 1
            batch = self.replay_buffer.sample_training_batch(
                batch_size=self.trainer_config.batch_size,
                reference_dropout_prob=self.trainer_config.reference_dropout_prob,
                replace=self.trainer_config.sample_with_replacement,
                tensor_resolver=self.tensor_resolver,
            )
            should_log = step % self.trainer_config.log_every_learner_updates == 0
            run = self.swanlab_run if should_log else None
            critic_metrics = update_critic(
                actor=self.actor,
                critic=self.critic,
                target_critic=self.target_critic,
                critic_optimizer=self.critic_optimizer,
                batch=batch,
                config=self.train_config,
                device=self.device,
                swanlab_run=run,
                step=step,
            )
            actor_metrics: dict[str, float] = {}
            target_updated = False
            next_critic_updates = self.state.critic_updates + 1
            if next_critic_updates % self.trainer_config.critic_updates_per_actor_update == 0:
                actor_metrics = update_actor(
                    actor=self.actor,
                    critic=self.critic,
                    actor_optimizer=self.actor_optimizer,
                    batch=batch,
                    config=self.train_config,
                    device=self.device,
                    swanlab_run=run,
                    step=step,
                )
                soft_update_target_network(
                    source=self.critic,
                    target=self.target_critic,
                    tau=self.trainer_config.target_update_tau,
                )
                target_updated = True

            self.state = RLTTrainerState(
                rollout_iterations=self.state.rollout_iterations,
                learner_updates=self.state.learner_updates + 1,
                critic_updates=next_critic_updates,
                actor_updates=self.state.actor_updates + (1 if actor_metrics else 0),
                target_updates=self.state.target_updates + (1 if target_updated else 0),
                transitions_added=self.state.transitions_added,
                transitions_evicted=self.state.transitions_evicted,
            )
            trainer_metrics = self._trainer_metrics(target_updated=target_updated)
            metrics = {**critic_metrics, **actor_metrics, **trainer_metrics}
            if run is not None:
                self._log_swanlab(trainer_metrics, step=step)
            metrics_history.append(metrics)
            self._maybe_checkpoint_after_learner_update()

        return metrics_history

    def train_iteration(
        self,
        rollout_fn: RolloutFn
        | Callable[["RLTTrainer"], RLTTrainerRollout | Iterable[RLTTransition]],
    ) -> dict[str, Any]:
        """Collect one rollout batch, add it to replay, then run ``G`` replay updates."""

        rollout = rollout_fn(self)
        if not isinstance(rollout, RLTTrainerRollout):
            rollout = RLTTrainerRollout.from_transitions(rollout)

        add_metrics = self.add_transitions(rollout.transitions, segment=rollout.segment)
        update_units = (
            rollout.update_units if rollout.update_units is not None else len(rollout.transitions)
        )
        num_updates = self.trainer_config.update_to_data_ratio * int(update_units)
        learner_metrics = self.learn(num_updates)
        self.state = RLTTrainerState(
            rollout_iterations=self.state.rollout_iterations + 1,
            learner_updates=self.state.learner_updates,
            critic_updates=self.state.critic_updates,
            actor_updates=self.state.actor_updates,
            target_updates=self.state.target_updates,
            transitions_added=self.state.transitions_added,
            transitions_evicted=self.state.transitions_evicted,
        )
        self._maybe_checkpoint_after_rollout()
        summary = {
            **add_metrics,
            "rlt/trainer/rollout_iterations": float(self.state.rollout_iterations),
            "rlt/trainer/requested_learner_updates": float(num_updates),
        }
        if rollout.metrics:
            summary.update(
                {f"rlt/rollout/{key}": float(value) for key, value in rollout.metrics.items()}
            )
        self._log_swanlab(summary, step=self.state.learner_updates)
        return {
            "rollout": rollout,
            "learner_metrics": learner_metrics,
            "summary": summary,
        }

    def train(self, rollout_fn: RolloutFn, *, num_iterations: int) -> list[dict[str, Any]]:
        """Run multiple rollout-learning iterations."""

        _require(num_iterations >= 0, "num_iterations must be non-negative")
        return [self.train_iteration(rollout_fn) for _ in range(num_iterations)]

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save networks, optimizers, replay contents, and trainer counters."""

        checkpoint_path = Path(path) if path is not None else self._default_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "gr00t_rlt_trainer_v1",
            "state": asdict(self.state),
            "trainer_config": asdict(self.trainer_config),
            "train_config": asdict(self.train_config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_capacity": self.replay_buffer.capacity,
            "replay_transitions": tuple(self.replay_buffer),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if hasattr(self.replay_buffer, "_rng"):
            payload["replay_rng_state"] = self.replay_buffer._rng.bit_generator.state
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizers: bool = True,
        load_replay: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint into the existing trainer object."""

        checkpoint = _torch_load_checkpoint(path, map_location=self.device)
        _require(
            checkpoint.get("format") == "gr00t_rlt_trainer_v1", "unsupported checkpoint format"
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if load_replay:
            self.replay_buffer.clear()
            self.replay_buffer.capacity = checkpoint.get("replay_capacity")
            self.replay_buffer.extend(checkpoint["replay_transitions"])
            if "replay_rng_state" in checkpoint and hasattr(self.replay_buffer, "_rng"):
                self.replay_buffer._rng.bit_generator.state = checkpoint["replay_rng_state"]
        if restore_rng:
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
        self.state = RLTTrainerState(**checkpoint["state"])
        return checkpoint

    def _trainer_metrics(self, *, target_updated: bool) -> dict[str, float]:
        return {
            "rlt/trainer/learner_updates": float(self.state.learner_updates),
            "rlt/trainer/critic_updates": float(self.state.critic_updates),
            "rlt/trainer/actor_updates": float(self.state.actor_updates),
            "rlt/trainer/target_updates": float(self.state.target_updates),
            "rlt/trainer/replay_size": float(len(self.replay_buffer)),
            "rlt/trainer/update_to_data_ratio": float(self.trainer_config.update_to_data_ratio),
            "rlt/trainer/critic_updates_per_actor_update": float(
                self.trainer_config.critic_updates_per_actor_update
            ),
            "rlt/trainer/target_updated": 1.0 if target_updated else 0.0,
        }

    def _default_checkpoint_path(self) -> Path:
        _require(
            self.trainer_config.checkpoint_dir is not None,
            "checkpoint_dir is required when no checkpoint path is provided",
        )
        return (
            Path(self.trainer_config.checkpoint_dir)
            / f"rlt_trainer_step_{self.state.learner_updates:08d}.pt"
        )

    def _maybe_checkpoint_after_rollout(self) -> None:
        every = self.trainer_config.checkpoint_every_rollouts
        if every is not None and self.state.rollout_iterations % every == 0:
            self.save_checkpoint()

    def _maybe_checkpoint_after_learner_update(self) -> None:
        every = self.trainer_config.checkpoint_every_learner_updates
        if every is not None and self.state.learner_updates % every == 0:
            self.save_checkpoint()

    def _log_swanlab(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        if self.swanlab_run is None:
            return
        log = getattr(self.swanlab_run, "log", None)
        if log is None:
            raise TypeError("swanlab_run must expose a log(metrics, step=...) method")
        if step is None:
            log(dict(metrics))
        else:
            log(dict(metrics), step=step)


def _torch_load_checkpoint(
    path: str | Path, *, map_location: torch.device | None
) -> dict[str, Any]:
    with legacy_module_aliases():
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
