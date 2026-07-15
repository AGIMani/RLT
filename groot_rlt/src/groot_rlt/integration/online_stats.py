#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Export statistics for Nero's real 19D EEF-and-hand command space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groot_rlt.integration.nero_action_contract import (
    EXECUTED_ACTION_CHANNEL_NAMES,
    EXECUTED_ACTION_DIM,
    ROT6D_CONVENTION,
    V3_ACTION_CHANNEL_NAMES,
    V3_POLICY_SPACE_SCHEMA,
    V3_STATE_CHANNEL_NAMES,
    semantic_layout_hash,
)

SCHEMA_NAME = "rlt_online_action_stats"
SCHEMA_VERSION = 3
LAYOUT_NAME = "nero_right_l10_executed_action"
LAYOUT_VERSION = 2
ACTION_DIM = EXECUTED_ACTION_DIM
CHANNEL_NAMES = EXECUTED_ACTION_CHANNEL_NAMES

STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")
NORMALIZATION_MODES = ("quantile", "symmetric_quantile")


@dataclass(frozen=True)
class _GroupSpec:
    name: str
    action_start: int
    action_end: int
    source_key: str
    source_start: int
    source_end: int

    @property
    def dim(self) -> int:
        return self.action_end - self.action_start


_GROUP_SPECS = (
    _GroupSpec("eef_9d", 0, 9, "action", 0, 9),
    _GroupSpec("hand_joint_target", 9, 19, "action", 9, 19),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def source_fingerprint(
    stats_payload: Mapping[str, Any], modality_payload: Mapping[str, Any]
) -> str:
    """Return a deterministic semantic fingerprint of both source documents."""

    return _sha256_bytes(
        _canonical_json_bytes({"modality.json": modality_payload, "stats.json": stats_payload})
    )


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object, got {type(value).__name__}")
    return value


def _require_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer, got {value!r}")
    return value


def _numeric_vector(value: Any, *, path: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{path} must be an array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{path}[{index}] must be numeric, got {item!r}")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}[{index}] must be numeric, got {item!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{path}[{index}] must be finite, got {item!r}")
        result.append(number)
    return result


def _validated_group(action_modalities: Mapping[str, Any], spec: _GroupSpec) -> dict[str, Any]:
    entry = _require_mapping(action_modalities.get(spec.name), path=f"modality.action.{spec.name}")
    source_start = _require_int(entry.get("start"), path=f"modality.action.{spec.name}.start")
    source_end = _require_int(entry.get("end"), path=f"modality.action.{spec.name}.end")
    source_key = entry.get("original_key", "action")
    if not isinstance(source_key, str):
        raise ValueError(f"modality.action.{spec.name}.original_key must be a string")

    actual = (source_key, source_start, source_end)
    expected = (spec.source_key, spec.source_start, spec.source_end)
    if actual != expected:
        raise ValueError(
            f"modality.action.{spec.name} must resolve to {expected[0]}"
            f"[{expected[1]}:{expected[2]}], got {actual[0]}[{actual[1]}:{actual[2]}]"
        )
    if source_end - source_start != spec.dim:
        raise ValueError(
            f"modality.action.{spec.name} has dimension {source_end - source_start}, "
            f"expected {spec.dim}"
        )
    return {
        "name": spec.name,
        "start": spec.action_start,
        "end": spec.action_end,
        "source": {
            "stats_key": source_key,
            "start": source_start,
            "end": source_end,
        },
    }


def _validated_v3_feature(
    info_payload: Mapping[str, Any],
    *,
    key: str,
    expected_dim: int,
    expected_names: Sequence[str] | None,
    dtype: str,
) -> None:
    features = _require_mapping(info_payload.get("features"), path="info.features")
    feature = _require_mapping(features.get(key), path=f"info.features.{key}")
    if feature.get("dtype") != dtype:
        raise ValueError(
            f"info.features.{key}.dtype must be {dtype!r}, got {feature.get('dtype')!r}"
        )
    shape = feature.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        raise ValueError(f"info.features.{key}.shape must be an array")
    if list(shape) != [expected_dim]:
        raise ValueError(f"info.features.{key}.shape must be [{expected_dim}], got {list(shape)!r}")
    names = feature.get("names")
    if expected_names is None:
        if names is not None:
            raise ValueError(f"info.features.{key}.names must be null, got {names!r}")
        return
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)):
        raise ValueError(f"info.features.{key}.names must be an array")
    if list(names) != list(expected_names):
        raise ValueError(
            f"info.features.{key}.names do not match the validated {ROT6D_CONVENTION} contract"
        )


def modality_from_lerobot_v3_metadata(
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the official v3 metadata and return the fixed 19D stats slices.

    The action and state names are checked one by one. The result only assigns
    already validated slices; it never reorders or transposes rot6d values.
    """

    info_payload = _require_mapping(info_payload, path="info")
    recap_payload = _require_mapping(recap_payload, path="teleop_stack_recap")
    if info_payload.get("codebase_version") != "v3.0":
        raise ValueError(
            f"info.codebase_version must be 'v3.0', got {info_payload.get('codebase_version')!r}"
        )
    if recap_payload.get("format_name") != "lerobot_v3_dagger":
        raise ValueError(
            "teleop_stack_recap.format_name must be 'lerobot_v3_dagger', got "
            f"{recap_payload.get('format_name')!r}"
        )
    if recap_payload.get("normalization_schema") != V3_POLICY_SPACE_SCHEMA:
        raise ValueError(
            "teleop_stack_recap.normalization_schema must be "
            f"{V3_POLICY_SPACE_SCHEMA!r}, got {recap_payload.get('normalization_schema')!r}"
        )
    policy_space = _require_mapping(
        recap_payload.get("policy_space"), path="teleop_stack_recap.policy_space"
    )
    expected_frames = {
        "observation_eef_frame": "policy_state",
        "model_eef_frame": "policy_state",
        "command_eef_frame": "genesis_world",
    }
    for key, expected in expected_frames.items():
        if policy_space.get(key) != expected:
            raise ValueError(
                f"teleop_stack_recap.policy_space.{key} must be {expected!r}, "
                f"got {policy_space.get(key)!r}"
            )
    transform = _require_mapping(
        policy_space.get("state_to_genesis_transform"),
        path="teleop_stack_recap.policy_space.state_to_genesis_transform",
    )
    expected_transform_values = {
        "source_frame": "policy_state",
        "target_frame": "genesis_world",
        "rot6d_convention": ROT6D_CONVENTION,
    }
    for key, expected in expected_transform_values.items():
        if transform.get(key) != expected:
            raise ValueError(
                "teleop_stack_recap.policy_space.state_to_genesis_transform."
                f"{key} must be {expected!r}, got {transform.get(key)!r}"
            )
    for key, expected_dim in (
        ("translation_xyz", 3),
        ("quaternion_xyzw", 4),
        ("eef_offset_translation_xyz", 3),
        ("eef_offset_quaternion_xyzw", 4),
    ):
        vector = _numeric_vector(
            transform.get(key),
            path=f"teleop_stack_recap.policy_space.state_to_genesis_transform.{key}",
        )
        if len(vector) != expected_dim:
            raise ValueError(
                "teleop_stack_recap.policy_space.state_to_genesis_transform."
                f"{key} must contain {expected_dim} values, got {len(vector)}"
            )

    _validated_v3_feature(
        info_payload,
        key="action",
        expected_dim=len(V3_ACTION_CHANNEL_NAMES),
        expected_names=V3_ACTION_CHANNEL_NAMES,
        dtype="float32",
    )
    _validated_v3_feature(
        info_payload,
        key="observation.state",
        expected_dim=len(V3_STATE_CHANNEL_NAMES),
        expected_names=V3_STATE_CHANNEL_NAMES,
        dtype="float32",
    )
    _validated_v3_feature(
        info_payload,
        key="intervention",
        expected_dim=1,
        expected_names=None,
        dtype="bool",
    )

    return {
        "rotation_convention": ROT6D_CONVENTION,
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "hand_joint_target": {"start": 9, "end": 19},
        },
    }


def _collect_stats(
    stats_payload: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]
) -> dict[str, list[float]]:
    source_vectors: dict[tuple[str, str], list[float]] = {}
    combined = {field: [] for field in STAT_FIELDS}

    for group in groups:
        source = _require_mapping(group["source"], path=f"layout.{group['name']}.source")
        source_key = str(source["stats_key"])
        source_start = int(source["start"])
        source_end = int(source["end"])
        source_stats = _require_mapping(stats_payload.get(source_key), path=f"stats.{source_key}")
        for field in STAT_FIELDS:
            cache_key = (source_key, field)
            if cache_key not in source_vectors:
                source_vectors[cache_key] = _numeric_vector(
                    source_stats.get(field), path=f"stats.{source_key}.{field}"
                )
            vector = source_vectors[cache_key]
            if len(vector) < source_end:
                raise ValueError(
                    f"stats.{source_key}.{field} has {len(vector)} channels, "
                    f"but {group['name']} requires [{source_start}:{source_end}]"
                )
            combined[field].extend(vector[source_start:source_end])

    for field, vector in combined.items():
        if len(vector) != ACTION_DIM:
            raise ValueError(f"assembled {field} has {len(vector)} channels, expected {ACTION_DIM}")

    for index, (minimum, maximum, q01, q99, std) in enumerate(
        zip(
            combined["min"],
            combined["max"],
            combined["q01"],
            combined["q99"],
            combined["std"],
            strict=True,
        )
    ):
        if minimum > maximum:
            raise ValueError(f"channel {index}: min {minimum} exceeds max {maximum}")
        if q01 > q99:
            raise ValueError(f"channel {index}: q01 {q01} exceeds q99 {q99}")
        if q01 < minimum or q99 > maximum:
            raise ValueError(
                f"channel {index}: quantiles [{q01}, {q99}] fall outside "
                f"observed range [{minimum}, {maximum}]"
            )
        if std < 0:
            raise ValueError(f"channel {index}: std must be non-negative, got {std}")
    return combined


def _layout(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    material = {
        "name": LAYOUT_NAME,
        "version": LAYOUT_VERSION,
        "action_dim": ACTION_DIM,
        "groups": list(groups),
        "channel_names": list(CHANNEL_NAMES),
        "rotation_convention": ROT6D_CONVENTION,
    }
    return {
        **material,
        # Runtime payloads can reproduce this without knowing dataset group
        # provenance; the full contract remains separately fingerprinted.
        "layout_hash": semantic_layout_hash(
            CHANNEL_NAMES,
            rotation_convention=ROT6D_CONVENTION,
        ),
        "contract_hash": _sha256_bytes(_canonical_json_bytes(material)),
    }


def build_online_stats(
    stats_payload: Mapping[str, Any],
    modality_payload: Mapping[str, Any],
    *,
    normalization_mode: str = "symmetric_quantile",
    eps: float = 1e-6,
    action_representation: str = "abs",
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert LeRobot metadata into the versioned online-RLT stats schema.

    This function performs no I/O. It validates the two actually executed
    action groups and deliberately ignores the checkpoint-only arm prediction.
    """

    stats_payload = _require_mapping(stats_payload, path="stats")
    modality_payload = _require_mapping(modality_payload, path="modality")
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"normalization_mode must be one of {NORMALIZATION_MODES}, got {normalization_mode!r}"
        )
    if action_representation != "abs":
        raise ValueError(
            "the Nero exporter supports only the absolute ('abs') action representation, "
            f"got {action_representation!r}"
        )
    if isinstance(eps, bool) or not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError(f"eps must be a finite positive number, got {eps!r}")
    eps = float(eps)

    if modality_payload.get("rotation_convention") != ROT6D_CONVENTION:
        raise ValueError(
            "modality.rotation_convention must explicitly match the validated inference "
            f"convention {ROT6D_CONVENTION!r}; got "
            f"{modality_payload.get('rotation_convention')!r}"
        )
    action_modalities = _require_mapping(modality_payload.get("action"), path="modality.action")
    groups = [_validated_group(action_modalities, spec) for spec in _GROUP_SPECS]
    collected = _collect_stats(stats_payload, groups)

    observed_min = list(collected["min"])
    observed_max = list(collected["max"])
    actions = {field: list(values) for field, values in collected.items()}
    actions["observed_min"] = observed_min
    actions["observed_max"] = observed_max
    if normalization_mode == "quantile":
        lower_key, upper_key = "q01", "q99"
    else:
        scale = [
            max(abs(low), abs(high), eps)
            for low, high in zip(collected["q01"], collected["q99"], strict=True)
        ]
        actions["min"] = [-value for value in scale]
        actions["max"] = scale
        lower_key, upper_key = "min", "max"

    upstream = _require_mapping(
        stats_payload.get("__fingerprints__", {}), path="stats.__fingerprints__"
    )
    selected_upstream = {key: upstream[key] for key in ("action",) if key in upstream}
    source = {
        "fingerprint": source_fingerprint(stats_payload, modality_payload),
        "upstream_stats_fingerprints": selected_upstream,
    }
    if source_metadata is not None:
        metadata = _require_mapping(source_metadata, path="source_metadata")
        conflicting = set(metadata).intersection(source)
        if conflicting:
            raise ValueError(
                f"source_metadata cannot override reserved keys: {sorted(conflicting)}"
            )
        source.update(metadata)

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "normalization": {
            "action_representation": action_representation,
            "mode": normalization_mode,
            "lower_key": lower_key,
            "upper_key": upper_key,
            "normalized_range": [-1.0, 1.0],
            "eps": eps,
        },
        "layout": _layout(groups),
        "source": source,
        "norm_stats": {"actions": actions},
    }


def _read_json_object(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return _require_mapping(value, path=str(path)), raw


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Prepared LeRobot dataset. Official v3 uses meta/info.json plus "
            "meta/teleop_stack_recap.json; legacy prepared datasets use meta/modality.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <dataset-dir>/meta/rlt_online_action_stats.json).",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=NORMALIZATION_MODES,
        default="symmetric_quantile",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    stats_path = dataset_dir / "meta" / "stats.json"
    info_path = dataset_dir / "meta" / "info.json"
    recap_path = dataset_dir / "meta" / "teleop_stack_recap.json"
    modality_path = dataset_dir / "meta" / "modality.json"
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else dataset_dir / "meta" / "rlt_online_action_stats.json"
    )

    stats_payload, stats_raw = _read_json_object(stats_path)
    source_files: dict[str, dict[str, str]] = {
        "stats.json": {"path": str(stats_path), "sha256": _sha256_bytes(stats_raw)}
    }
    source_format: str
    if info_path.is_file():
        info_payload, info_raw = _read_json_object(info_path)
    else:
        info_payload, info_raw = None, None
    if info_payload is not None and info_payload.get("codebase_version") == "v3.0":
        assert info_raw is not None
        recap_payload, recap_raw = _read_json_object(recap_path)
        modality_payload = modality_from_lerobot_v3_metadata(info_payload, recap_payload)
        source_files.update(
            {
                "info.json": {"path": str(info_path), "sha256": _sha256_bytes(info_raw)},
                "teleop_stack_recap.json": {
                    "path": str(recap_path),
                    "sha256": _sha256_bytes(recap_raw),
                },
            }
        )
        source_format = "lerobot_v3_dagger"
    else:
        modality_payload, modality_raw = _read_json_object(modality_path)
        source_files["modality.json"] = {
            "path": str(modality_path),
            "sha256": _sha256_bytes(modality_raw),
        }
        source_format = "legacy_modality_json"
    payload = build_online_stats(
        stats_payload,
        modality_payload,
        normalization_mode=args.normalization_mode,
        eps=args.eps,
        source_metadata={
            "dataset_dir": str(dataset_dir),
            "source_format": source_format,
            "rotation_convention": ROT6D_CONVENTION,
            "files": source_files,
        },
    )
    _write_json_atomic(output_path, payload, overwrite=args.overwrite)
    print(f"Wrote {ACTION_DIM}D {args.normalization_mode} absolute-action stats to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
