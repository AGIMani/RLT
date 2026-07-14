# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for building the RLT VLA-warmup replay buffer.

The RLT paper pre-fills replay by rolling out the frozen VLA reference policy
before online actor-critic updates begin. This module keeps that warmup phase
separate from the Eq. 3/Eq. 5 update code: it consumes collected episode
sidecars, derives validated ``RLTTransition`` records, and fills replay with
``CollectionStage.VLA_WARMUP`` transitions.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from groot_rlt.episode_schema import (
    ChunkDecisionRecord,
    EpisodeSource,
    EpisodeStepRecord,
    InterventionRecord,
    RLTEpisodeRecord,
)
from groot_rlt.episode_transition_builder import (
    TensorResolver,
    build_rlt_transitions_from_episode,
)
from groot_rlt.legacy import load_pickle_with_legacy_aliases
from groot_rlt.replay_buffer import RLTReplayBuffer
from groot_rlt.replay_schema import CollectionStage, RLTTransition

WARMUP_REPLAY_FORMAT = "gr00t_rlt_warmup_replay_v1"
SUPPORTED_TENSOR_SUFFIXES = (".npy", ".npz", ".pt", ".pth")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RLTWarmupConfig:
    """Configuration for turning VLA warmup episodes into replay records."""

    gamma: float
    capacity: int | None = None
    seed: int | None = None
    allow_terminal_padding: bool = True
    require_vla_warmup_source: bool = True
    replace: bool = False

    def validate(self) -> None:
        _require(0.0 <= self.gamma < 1.0, "gamma must be in [0, 1)")
        if self.capacity is not None:
            _require(self.capacity > 0, "capacity must be positive when provided")


@dataclass(frozen=True)
class WarmupEpisodeComponents:
    """One collected warmup episode plus its tensor resolver."""

    episode: RLTEpisodeRecord
    steps: Sequence[EpisodeStepRecord]
    chunk_decisions: Sequence[ChunkDecisionRecord]
    interventions: Sequence[InterventionRecord]
    tensor_resolver: TensorResolver


@dataclass(frozen=True)
class RLTWarmupSummary:
    """Serializable summary of a warmup replay build."""

    episode_count: int
    built_transition_count: int
    stored_transition_count: int
    evicted_transition_count: int
    valid_action_step_count: int
    terminal_transition_count: int
    by_collection_stage: dict[str, int]
    by_episode: dict[str, int]


@dataclass(frozen=True)
class FileTensorResolver:
    """Resolve episode tensor keys to arrays stored on disk."""

    episode_dir: Path
    tensor_root: Path | None = None
    npz_array_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_dir", Path(self.episode_dir))
        if self.tensor_root is not None:
            object.__setattr__(self, "tensor_root", Path(self.tensor_root))

    def __call__(self, key: str) -> np.ndarray:
        return load_tensor_file(self.resolve_path(key), npz_array_key=self.npz_array_key)

    def resolve_path(self, key: str) -> Path:
        _require(bool(key), "tensor key must be non-empty")
        raw_path = Path(key).expanduser()
        candidates = list(self._candidate_paths(raw_path))
        for path in candidates:
            if path.exists():
                return path

        checked = ", ".join(path.as_posix() for path in candidates[:8])
        raise FileNotFoundError(f"tensor key {key!r} did not resolve to a file; checked: {checked}")

    def _candidate_paths(self, raw_path: Path):
        if raw_path.is_absolute():
            bases = [raw_path]
        else:
            bases = [self.episode_dir / raw_path, self.episode_dir / "tensors" / raw_path]
            if self.tensor_root is not None:
                bases.append(self.tensor_root / raw_path)

        for base in bases:
            yield base
            if base.suffix:
                continue
            for suffix in SUPPORTED_TENSOR_SUFFIXES:
                yield base.with_suffix(suffix)


def load_tensor_file(path: str | Path, *, npz_array_key: str | None = None) -> np.ndarray:
    """Load one tensor file used by RLT episode sidecars."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if npz_array_key is not None:
                return np.asarray(data[npz_array_key], dtype=np.float32)
            if "arr_0" in data:
                return np.asarray(data["arr_0"], dtype=np.float32)
            keys = list(data.keys())
            if len(keys) == 1:
                return np.asarray(data[keys[0]], dtype=np.float32)
            raise KeyError(f"{path} contains multiple arrays {keys}; pass npz_array_key explicitly")
    if suffix in {".pt", ".pth"}:
        import torch

        value = torch.load(path, map_location="cpu")
        if torch.is_tensor(value):
            return value.detach().cpu().to(dtype=torch.float32).numpy()
        return np.asarray(value, dtype=np.float32)

    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def build_warmup_transitions(
    components: WarmupEpisodeComponents,
    *,
    config: RLTWarmupConfig,
) -> list[RLTTransition]:
    """Build validated VLA-warmup replay transitions from one episode."""

    config.validate()
    if config.require_vla_warmup_source:
        _require(
            components.episode.source is EpisodeSource.VLA_WARMUP,
            "warmup replay requires episode.source == vla_warmup",
        )
        _require(
            len(components.interventions) == 0,
            "VLA warmup episodes cannot contain human interventions",
        )

    return build_rlt_transitions_from_episode(
        episode=components.episode,
        steps=components.steps,
        chunk_decisions=components.chunk_decisions,
        interventions=components.interventions,
        tensor_resolver=components.tensor_resolver,
        gamma=config.gamma,
        allow_terminal_padding=config.allow_terminal_padding,
        collection_stage=CollectionStage.VLA_WARMUP,
        validate_episode=True,
    )


def build_warmup_replay(
    episodes: Sequence[WarmupEpisodeComponents],
    *,
    config: RLTWarmupConfig,
) -> tuple[RLTReplayBuffer, RLTWarmupSummary]:
    """Fill a replay buffer with VLA warmup transitions."""

    config.validate()
    _require(len(episodes) > 0, "at least one warmup episode is required")
    replay = RLTReplayBuffer(capacity=config.capacity, seed=config.seed)
    built_transition_count = 0
    evicted_transition_count = 0

    for components in episodes:
        transitions = build_warmup_transitions(components, config=config)
        built_transition_count += len(transitions)
        evicted = replay.extend(
            transitions,
            segment=components.episode.critical_segment(),
            gamma=config.gamma,
            replace=config.replace,
        )
        evicted_transition_count += len(evicted)

    summary = summarize_warmup_replay(
        replay,
        episode_count=len(episodes),
        built_transition_count=built_transition_count,
        evicted_transition_count=evicted_transition_count,
    )
    return replay, summary


def summarize_warmup_replay(
    replay: RLTReplayBuffer,
    *,
    episode_count: int,
    built_transition_count: int,
    evicted_transition_count: int,
) -> RLTWarmupSummary:
    transitions = tuple(replay)
    return RLTWarmupSummary(
        episode_count=episode_count,
        built_transition_count=built_transition_count,
        stored_transition_count=len(transitions),
        evicted_transition_count=evicted_transition_count,
        valid_action_step_count=sum(transition.valid_steps for transition in transitions),
        terminal_transition_count=sum(1 for transition in transitions if transition.done),
        by_collection_stage=replay.by_collection_stage(),
        by_episode=replay.by_episode(),
    )


def save_warmup_replay(
    path: str | Path,
    replay: RLTReplayBuffer,
    *,
    summary: RLTWarmupSummary | None = None,
) -> None:
    """Serialize a warmup replay payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": WARMUP_REPLAY_FORMAT,
        "transitions": tuple(replay),
        "summary": None if summary is None else asdict(summary),
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_warmup_replay(path: str | Path) -> tuple[RLTReplayBuffer, dict[str, Any] | None]:
    """Load a replay payload written by ``save_warmup_replay``."""

    with Path(path).open("rb") as stream:
        payload = load_pickle_with_legacy_aliases(stream)
    _require(payload.get("format") == WARMUP_REPLAY_FORMAT, "unsupported warmup replay format")
    replay = RLTReplayBuffer()
    replay.extend(payload["transitions"])
    return replay, payload.get("summary")


def write_warmup_summary_json(
    path: str | Path,
    summary: RLTWarmupSummary,
    *,
    replay_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(summary)
    if replay_path is not None:
        payload["replay_path"] = Path(replay_path).as_posix()
    if extra is not None:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_warmup_episode_directory(
    episode_dir: str | Path,
    *,
    tensor_root: str | Path | None = None,
    npz_array_key: str | None = None,
) -> WarmupEpisodeComponents:
    """Load one episode directory that follows ``episode_schema.md``."""

    episode_dir = Path(episode_dir)
    episode = _load_dataclass_json(episode_dir / "episode.json", RLTEpisodeRecord)
    steps = _load_parquet_dataclasses(episode_dir / "steps.parquet", EpisodeStepRecord)
    chunk_decisions = _load_parquet_dataclasses(
        episode_dir / "chunk_decisions.parquet",
        ChunkDecisionRecord,
    )
    interventions = _load_jsonl_dataclasses(
        episode_dir / "interventions.jsonl",
        InterventionRecord,
        missing_ok=True,
    )
    return WarmupEpisodeComponents(
        episode=episode,
        steps=steps,
        chunk_decisions=chunk_decisions,
        interventions=interventions,
        tensor_resolver=FileTensorResolver(
            episode_dir=episode_dir,
            tensor_root=None if tensor_root is None else Path(tensor_root),
            npz_array_key=npz_array_key,
        ),
    )


def _load_dataclass_json(path: Path, cls):
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return cls(**payload)


def _load_parquet_dataclasses(path: Path, cls):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read RLT warmup parquet sidecars") from exc

    rows = pd.read_parquet(path).to_dict("records")
    return [cls(**_clean_record(row)) for row in rows]


def _load_jsonl_dataclasses(path: Path, cls, *, missing_ok: bool = False):
    if missing_ok and not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(cls(**json.loads(line)))
    return rows


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if _is_missing_scalar(value) else value) for key, value in row.items()}


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isscalar(value) and np.isnan(value))
    except TypeError:
        return False
