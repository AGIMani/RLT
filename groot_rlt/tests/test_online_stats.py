from __future__ import annotations

import copy
import json

import pytest

from groot_rlt.integration.nero_action_contract import (
    ROT6D_CONVENTION,
    V3_ACTION_CHANNEL_NAMES,
    V3_POLICY_SPACE_SCHEMA,
    V3_STATE_CHANNEL_NAMES,
)
from groot_rlt.integration.online_stats import (
    ACTION_DIM,
    CHANNEL_NAMES,
    build_online_stats,
    main,
    modality_from_lerobot_v3_metadata,
    source_fingerprint,
)
from groot_rlt.serving.groot_feature_policy import channel_layout_hash


def _stats() -> dict:
    action_q01 = [-(index + 1) * 0.1 for index in range(19)]
    action_q99 = [(index + 1) * 0.2 for index in range(19)]
    state_q01 = [-(index + 1) * 0.3 for index in range(26)]
    state_q99 = [(index + 1) * 0.4 for index in range(26)]

    def make_feature(q01: list[float], q99: list[float]) -> dict[str, list[float]]:
        size = len(q01)
        return {
            "mean": [float(index) for index in range(size)],
            "std": [0.5] * size,
            "min": [value - 1.0 for value in q01],
            "max": [value + 1.0 for value in q99],
            "q01": q01,
            "q99": q99,
        }

    return {
        "action": make_feature(action_q01, action_q99),
        "observation.state": make_feature(state_q01, state_q99),
        "__fingerprints__": {
            "action": "sha256:action",
            "observation.state": "sha256:state",
            "timestamp": "sha256:unused",
        },
    }


def _modality() -> dict:
    return {
        "rotation_convention": ROT6D_CONVENTION,
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "hand_joint_target": {"start": 9, "end": 19},
            "arm_joint_target": {
                "start": 0,
                "end": 7,
                "original_key": "observation.state",
            },
        },
    }


def _v3_info() -> dict:
    return {
        "codebase_version": "v3.0",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [19],
                "names": list(V3_ACTION_CHANNEL_NAMES),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [26],
                "names": list(V3_STATE_CHANNEL_NAMES),
            },
            "intervention": {"dtype": "bool", "shape": [1], "names": None},
        },
    }


def _v3_recap() -> dict:
    return {
        "format_name": "lerobot_v3_dagger",
        "normalization_schema": V3_POLICY_SPACE_SCHEMA,
        "policy_space": {
            "observation_eef_frame": "policy_state",
            "model_eef_frame": "policy_state",
            "command_eef_frame": "genesis_world",
            "state_to_genesis_transform": {
                "source_frame": "policy_state",
                "target_frame": "genesis_world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "eef_offset_translation_xyz": [0.0, 0.0, 0.0],
                "eef_offset_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "rot6d_convention": ROT6D_CONVENTION,
                "identity": False,
            },
        },
    }


def test_quantile_conversion_uses_only_real_19d_action() -> None:
    stats = _stats()
    modality = _modality()

    payload = build_online_stats(
        stats,
        modality,
        normalization_mode="quantile",
        source_metadata={"dataset_dir": "/dataset"},
    )

    actions = payload["norm_stats"]["actions"]
    expected_q01 = stats["action"]["q01"][:19]
    assert actions["q01"] == expected_q01
    assert actions["min"] == stats["action"]["min"][:19]
    assert payload["normalization"]["lower_key"] == "q01"
    assert payload["normalization"]["upper_key"] == "q99"
    assert payload["layout"]["action_dim"] == ACTION_DIM
    assert payload["layout"]["channel_names"] == list(CHANNEL_NAMES)
    assert len(payload["layout"]["groups"]) == 2
    assert payload["layout"]["rotation_convention"] == ROT6D_CONVENTION
    assert payload["layout"]["layout_hash"].startswith("sha256:")
    assert payload["layout"]["layout_hash"] == channel_layout_hash(
        CHANNEL_NAMES,
        rotation_convention=ROT6D_CONVENTION,
    )
    assert payload["source"]["fingerprint"] == source_fingerprint(stats, modality)
    assert payload["source"]["upstream_stats_fingerprints"] == {"action": "sha256:action"}
    assert payload["source"]["dataset_dir"] == "/dataset"


def test_symmetric_quantile_writes_effective_min_max_and_preserves_quantiles() -> None:
    stats = _stats()
    stats["action"]["q01"][0] = 0.0
    stats["action"]["q99"][0] = 0.0
    stats["action"]["min"][0] = -1.0
    stats["action"]["max"][0] = 1.0

    payload = build_online_stats(
        stats, _modality(), normalization_mode="symmetric_quantile", eps=0.25
    )

    actions = payload["norm_stats"]["actions"]
    expected_q01 = stats["action"]["q01"][:19]
    expected_q99 = stats["action"]["q99"][:19]
    expected_scale = [
        max(abs(low), abs(high), 0.25) for low, high in zip(expected_q01, expected_q99, strict=True)
    ]
    assert actions["q01"] == expected_q01
    assert actions["q99"] == expected_q99
    assert actions["min"] == [-value for value in expected_scale]
    assert actions["max"] == expected_scale
    assert actions["observed_min"][0] == -1.0
    assert actions["observed_max"][0] == 1.0
    assert payload["normalization"]["lower_key"] == "min"
    assert payload["normalization"]["upper_key"] == "max"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda modality: modality["action"]["hand_joint_target"].update(end=18),
            "hand_joint_target must resolve to action\\[9:19\\]",
        ),
    ],
)
def test_conversion_rejects_noncanonical_layout(mutation, match: str) -> None:
    modality = _modality()
    mutation(modality)
    with pytest.raises(ValueError, match=match):
        build_online_stats(_stats(), modality)


def test_source_fingerprint_changes_with_source_semantics() -> None:
    stats = _stats()
    changed = copy.deepcopy(stats)
    changed["action"]["mean"][0] += 1.0
    assert source_fingerprint(stats, _modality()) != source_fingerprint(changed, _modality())


def test_arm_state_statistics_do_not_affect_executed_action_bounds() -> None:
    stats = _stats()
    changed = copy.deepcopy(stats)
    changed["observation.state"]["q01"][0] -= 1000.0
    changed["observation.state"]["q99"][0] += 1000.0

    original_payload = build_online_stats(stats, _modality())
    changed_payload = build_online_stats(changed, _modality())

    assert original_payload["norm_stats"] == changed_payload["norm_stats"]


def test_v3_metadata_assigns_slices_without_reordering_row_first_action() -> None:
    modality = modality_from_lerobot_v3_metadata(_v3_info(), _v3_recap())

    assert modality == {
        "rotation_convention": ROT6D_CONVENTION,
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "hand_joint_target": {"start": 9, "end": 19},
        },
    }


def test_v3_metadata_rejects_changed_rot6d_order() -> None:
    info = _v3_info()
    names = info["features"]["action"]["names"]
    names[4], names[6] = names[6], names[4]

    with pytest.raises(ValueError, match="validated groot_row_major_first_two_rows contract"):
        modality_from_lerobot_v3_metadata(info, _v3_recap())


def test_stats_reject_legacy_modality_without_explicit_rot6d_contract() -> None:
    modality = _modality()
    modality.pop("rotation_convention")

    with pytest.raises(ValueError, match="must explicitly match"):
        build_online_stats(_stats(), modality)


def test_v3_metadata_requires_explicit_policy_space_transform() -> None:
    recap = _v3_recap()
    recap["policy_space"].pop("state_to_genesis_transform")

    with pytest.raises(ValueError, match="state_to_genesis_transform"):
        modality_from_lerobot_v3_metadata(_v3_info(), recap)


def test_v3_metadata_rejects_transform_with_non_inference_rot6d_order() -> None:
    recap = _v3_recap()
    recap["policy_space"]["state_to_genesis_transform"]["rot6d_convention"] = "mismatch"

    with pytest.raises(ValueError, match="rot6d_convention"):
        modality_from_lerobot_v3_metadata(_v3_info(), recap)


def test_cli_reads_official_v3_without_modality_json(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "stats.json").write_text(json.dumps(_stats()), encoding="utf-8")
    (meta_dir / "info.json").write_text(json.dumps(_v3_info()), encoding="utf-8")
    (meta_dir / "teleop_stack_recap.json").write_text(json.dumps(_v3_recap()), encoding="utf-8")

    assert main(["--dataset-dir", str(dataset_dir)]) == 0

    output = json.loads((meta_dir / "rlt_online_action_stats.json").read_text(encoding="utf-8"))
    assert output["layout"]["action_dim"] == 19
    assert output["source"]["source_format"] == "lerobot_v3_dagger"
    assert set(output["source"]["files"]) == {
        "stats.json",
        "info.json",
        "teleop_stack_recap.json",
    }
