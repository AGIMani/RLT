from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import numpy as np

from rlt_online_rl.config import RLTOnlineRLConfig


@dataclasses.dataclass(frozen=True)
class QuantileStats:
    """Effective lower/upper normalization bounds plus their provenance.

    The historical field names are retained because trainer call sites already
    use ``q01``/``q99``.  For symmetric-minmax exports these fields contain the
    selected ``min``/``max`` bounds, not the raw quantiles.
    """

    q01: np.ndarray
    q99: np.ndarray
    representation: str | None = None
    mode: str = "quantile"
    layout_hash: str | None = None


def _load_quantile_stats(path: str) -> QuantileStats:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload["norm_stats"]["actions"]
    normalization = payload.get("normalization", {})
    lower_key = str(normalization.get("lower_key", "q01"))
    upper_key = str(normalization.get("upper_key", "q99"))
    if lower_key not in stats or upper_key not in stats:
        raise ValueError(f"normalization bounds {lower_key!r}/{upper_key!r} are missing from {path}")
    lower = np.asarray(stats[lower_key], dtype=np.float32)
    upper = np.asarray(stats[upper_key], dtype=np.float32)
    if lower.shape != upper.shape or lower.ndim not in (1, 2):
        raise ValueError(
            f"action normalization bounds must have matching [D] or [H,D] shapes, got {lower.shape} and {upper.shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("action normalization bounds must be finite")
    if np.any(upper < lower):
        raise ValueError("every action normalization upper bound must be >= its lower bound")
    return QuantileStats(
        q01=lower,
        q99=upper,
        representation=normalization.get("representation", normalization.get("action_representation")),
        mode=str(normalization.get("mode", "quantile")),
        layout_hash=payload.get("layout", {}).get("layout_hash"),
    )


def _quantile_normalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x - q01) / scale * 2.0 - 1.0


def _quantile_denormalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x + 1.0) * 0.5 * scale + q01


def _zero_row_mask(chunk: np.ndarray) -> np.ndarray:
    return np.all(np.isclose(chunk, 0.0), axis=-1, keepdims=True)


def _broadcast_state0(state0: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    state0 = np.asarray(state0, dtype=np.float32)
    while state0.ndim < chunk.ndim:
        state0 = np.expand_dims(state0, axis=-2)
    return state0


def _broadcast_state0_jax(state0: jnp.ndarray, chunk: jnp.ndarray) -> jnp.ndarray:
    state0 = jnp.asarray(state0, dtype=jnp.float32)
    while state0.ndim < chunk.ndim:
        state0 = jnp.expand_dims(state0, axis=-2)
    return state0


def resolve_delta_indices(
    action_dim: int,
    state_dim: int,
    delta_action_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    indices = (
        tuple(range(min(6, action_dim, state_dim)))
        if delta_action_indices is None
        else tuple(int(index) for index in delta_action_indices)
    )
    if len(set(indices)) != len(indices):
        raise ValueError(f"delta_action_indices contains duplicates: {indices}")
    if any(index < 0 or index >= action_dim or index >= state_dim for index in indices):
        raise ValueError(
            "delta_action_indices must be valid for both action and proprio dimensions; "
            f"got indices={indices}, action_dim={action_dim}, state_dim={state_dim}"
        )
    return indices


def jax_quantile_denormalize(x: jnp.ndarray, q01: jnp.ndarray, q99: jnp.ndarray) -> jnp.ndarray:
    q01 = jnp.asarray(q01, dtype=jnp.float32)
    q99 = jnp.asarray(q99, dtype=jnp.float32)
    scale = q99 - q01 + 1e-6
    return (jnp.asarray(x, dtype=jnp.float32) + 1.0) * 0.5 * scale + q01


def jax_delta_to_abs_chunk(
    chunk_delta: jnp.ndarray,
    state0: jnp.ndarray,
    delta_action_indices: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    chunk_delta = jnp.asarray(chunk_delta, dtype=jnp.float32)
    state0 = _broadcast_state0_jax(state0, chunk_delta)
    indices = resolve_delta_indices(int(chunk_delta.shape[-1]), int(state0.shape[-1]), delta_action_indices)
    if not indices:
        return chunk_delta
    index_array = jnp.asarray(indices, dtype=jnp.int32)
    return chunk_delta.at[..., index_array].add(state0[..., index_array])


def jax_denormalize_to_abs_chunk(
    chunk_norm: jnp.ndarray,
    state0: jnp.ndarray,
    q01: jnp.ndarray,
    q99: jnp.ndarray,
    *,
    action_representation: str,
    delta_action_indices: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    chunk_repr = jax_quantile_denormalize(chunk_norm, q01, q99)
    if action_representation == "abs":
        return chunk_repr
    return jax_delta_to_abs_chunk(chunk_repr, state0, delta_action_indices)


@dataclasses.dataclass(frozen=True)
class ActionRepresentationAdapter:
    rl_config: RLTOnlineRLConfig
    stats: QuantileStats

    @classmethod
    def from_config(cls, rl_config: RLTOnlineRLConfig) -> ActionRepresentationAdapter | None:
        if rl_config.action_norm_stats_path is None:
            return None
        stats = _load_quantile_stats(rl_config.action_norm_stats_path)
        if stats.representation is not None and stats.representation != rl_config.action_representation:
            raise ValueError(
                f"stats representation={stats.representation!r} does not match runtime "
                f"action_representation={rl_config.action_representation!r}"
            )
        if stats.q01.shape[-1] != rl_config.action_dim:
            raise ValueError(f"stats action dim {stats.q01.shape[-1]} != configured {rl_config.action_dim}")
        if stats.q01.ndim == 2:
            if stats.q01.shape[0] < rl_config.chunk_len:
                raise ValueError(f"stats horizon {stats.q01.shape[0]} < chunk_len {rl_config.chunk_len}")
            stats = dataclasses.replace(
                stats,
                q01=stats.q01[: rl_config.chunk_len],
                q99=stats.q99[: rl_config.chunk_len],
            )
        if rl_config.action_layout_hash is not None and stats.layout_hash != rl_config.action_layout_hash:
            raise ValueError(f"stats layout hash {stats.layout_hash!r} != configured {rl_config.action_layout_hash!r}")
        return cls(rl_config=rl_config, stats=stats)

    def _abs_to_delta_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_abs = np.asarray(chunk_abs, dtype=np.float32)
        state0 = _broadcast_state0(state0, chunk_abs)
        chunk_delta = chunk_abs.copy()
        zero_mask = _zero_row_mask(chunk_abs)
        indices = resolve_delta_indices(
            int(chunk_abs.shape[-1]),
            int(state0.shape[-1]),
            self.rl_config.delta_action_indices,
        )
        if indices:
            index_array = np.asarray(indices, dtype=np.int64)
            chunk_delta[..., index_array] = chunk_abs[..., index_array] - state0[..., index_array]
        return np.where(zero_mask, 0.0, chunk_delta)

    def _delta_to_abs_chunk(self, chunk_delta: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_delta = np.asarray(chunk_delta, dtype=np.float32)
        state0 = _broadcast_state0(state0, chunk_delta)
        chunk_abs = chunk_delta.copy()
        indices = resolve_delta_indices(
            int(chunk_delta.shape[-1]),
            int(state0.shape[-1]),
            self.rl_config.delta_action_indices,
        )
        if indices:
            index_array = np.asarray(indices, dtype=np.int64)
            chunk_abs[..., index_array] = chunk_delta[..., index_array] + state0[..., index_array]
        return chunk_abs

    def _to_representation(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_abs, dtype=np.float32)
        return self._abs_to_delta_chunk(chunk_abs, state0)

    def _from_representation(self, chunk_repr: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_repr, dtype=np.float32)
        return self._delta_to_abs_chunk(chunk_repr, state0)

    def normalize_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_repr = self._to_representation(chunk_abs, state0)
        normalized = _quantile_normalize(chunk_repr, self.stats)
        return np.where(_zero_row_mask(chunk_abs), 0.0, normalized).astype(np.float32, copy=False)

    def denormalize_to_abs_chunk(self, chunk_norm: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_repr = _quantile_denormalize(np.asarray(chunk_norm, dtype=np.float32), self.stats)
        return self._from_representation(chunk_repr, state0).astype(np.float32, copy=False)

    def normalize_ref_chunk(self, ref_chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        return self.normalize_chunk(ref_chunk_abs, state0)

    def prepare_training_batch(self, batch_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        proprio = np.asarray(batch_np["proprio"], dtype=np.float32)
        next_proprio = np.asarray(batch_np["next_proprio"], dtype=np.float32)
        transformed = dict(batch_np)
        transformed["ref_chunk"] = self.normalize_ref_chunk(batch_np["ref_chunk"], proprio)
        transformed["action_chunk"] = self.normalize_chunk(batch_np["action_chunk"], proprio)
        transformed["next_ref_chunk"] = self.normalize_ref_chunk(batch_np["next_ref_chunk"], next_proprio)
        transformed["proprio"] = proprio
        transformed["next_proprio"] = next_proprio
        return transformed
