from __future__ import annotations

import copy

import pytest

from groot_rlt.integration.online_stats import (
    ACTION_DIM,
    CHANNEL_NAMES,
    build_online_stats,
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
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "hand_joint_target": {"start": 9, "end": 19},
            "arm_joint_target": {
                "start": 0,
                "end": 7,
                "original_key": "observation.state",
            },
        }
    }


def test_quantile_conversion_assembles_strict_nero_order() -> None:
    stats = _stats()
    modality = _modality()

    payload = build_online_stats(
        stats,
        modality,
        normalization_mode="quantile",
        source_metadata={"dataset_dir": "/dataset"},
    )

    actions = payload["norm_stats"]["actions"]
    expected_q01 = stats["action"]["q01"][:19] + stats["observation.state"]["q01"][:7]
    assert actions["q01"] == expected_q01
    assert actions["min"] == stats["action"]["min"][:19] + stats["observation.state"]["min"][:7]
    assert payload["normalization"]["lower_key"] == "q01"
    assert payload["normalization"]["upper_key"] == "q99"
    assert payload["layout"]["action_dim"] == ACTION_DIM
    assert payload["layout"]["channel_names"] == list(CHANNEL_NAMES)
    assert payload["layout"]["groups"][2] == {
        "name": "arm_joint_target",
        "start": 19,
        "end": 26,
        "source": {"stats_key": "observation.state", "start": 0, "end": 7},
    }
    assert payload["layout"]["layout_hash"].startswith("sha256:")
    assert payload["layout"]["layout_hash"] == channel_layout_hash(CHANNEL_NAMES)
    assert payload["source"]["fingerprint"] == source_fingerprint(stats, modality)
    assert payload["source"]["upstream_stats_fingerprints"] == {
        "action": "sha256:action",
        "observation.state": "sha256:state",
    }
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
    expected_q01 = stats["action"]["q01"][:19] + stats["observation.state"]["q01"][:7]
    expected_q99 = stats["action"]["q99"][:19] + stats["observation.state"]["q99"][:7]
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
            lambda modality: modality["action"]["arm_joint_target"].update(original_key="action"),
            "arm_joint_target must resolve to observation.state",
        ),
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
