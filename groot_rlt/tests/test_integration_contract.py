from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.serving import msgpack_numpy
from groot_rlt.serving.groot_feature_policy import (
    FeatureContract,
    GrootN1d7FeatureBackend,
    MachineAFeaturePolicy,
    channel_layout_hash,
)
from groot_rlt.serving.groot_feature_server import _parse_key_map


class _FakeBackend:
    def infer_one(self, observation):
        offset = float(observation.get("offset", 0.0))
        action_dim = 27 if observation.get("extra_action_dim") else 26
        return {
            "z_rl": np.full((1, 2048), offset, dtype=np.float64),
            "ref_chunk": np.full((1, 12, action_dim), offset + 1.0, dtype=np.float64),
            "proprio": np.full((1, 26), offset + 2.0, dtype=np.float64),
        }


def test_machine_a_contract_supports_nero_26d_single_and_batch() -> None:
    policy = MachineAFeaturePolicy(
        _FakeBackend(), FeatureContract(z_dim=2048, chunk_len=10, action_dim=26, proprio_dim=26)
    )
    single = policy.infer({"offset": 3})
    assert single["z_rl"].shape == (2048,)
    assert single["ref_chunk"].shape == (10, 26)
    assert single["proprio"].shape == (26,)
    assert single["z_rl"].dtype == np.float32

    batch = policy.infer({"batch": [{"offset": 1}, {"offset": 2}]})
    assert len(batch["batch_results"]) == 2
    assert np.all(batch["batch_results"][1]["ref_chunk"] == 3.0)


def test_msgpack_numpy_wire_roundtrip() -> None:
    payload = {
        "z_rl": np.arange(8, dtype=np.float32),
        "ref_chunk": np.ones((10, 26), dtype=np.float32),
    }
    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    assert np.array_equal(restored["z_rl"], payload["z_rl"])
    assert np.array_equal(restored["ref_chunk"], payload["ref_chunk"])


def test_machine_a_rejects_extra_action_channels_instead_of_truncating() -> None:
    policy = MachineAFeaturePolicy(_FakeBackend(), FeatureContract())
    with pytest.raises(ValueError, match="exact action dimension"):
        policy.infer({"extra_action_dim": True})


def test_machine_a_rejects_serial_pseudo_batch_when_not_advertised() -> None:
    policy = MachineAFeaturePolicy(_FakeBackend(), FeatureContract(), supports_batch=False)
    with pytest.raises(ValueError, match="does not support batched"):
        policy.infer({"batch": [{"offset": 1}, {"offset": 2}]})


def test_nero_flat_observation_adapter_preserves_eef_hand_arm_order() -> None:
    backend = object.__new__(GrootN1d7FeatureBackend)
    backend.contract = FeatureContract()
    backend.image_key_map = {"head": "ego_view", "wrist": "wrist_view"}
    backend.flat_state_layout = None
    backend.default_instruction = "pick and place"
    backend.policy = SimpleNamespace(
        language_key="annotation.human.action.task_description",
        modality_configs={
            "video": SimpleNamespace(modality_keys=("ego_view", "wrist_view")),
            "state": SimpleNamespace(modality_keys=("eef_9d", "hand_joint_pos", "arm_joint_pos")),
        },
    )
    state = np.arange(26, dtype=np.float32)
    adapted = backend._adapt_observation(
        {
            "images": {
                "head": np.zeros((12, 16, 3), dtype=np.uint8),
                "wrist": np.ones((12, 16, 3), dtype=np.uint8),
            },
            "state": state,
            "prompt": "pick and place",
        }
    )
    assert adapted["video"]["ego_view"].shape == (1, 1, 12, 16, 3)
    assert np.array_equal(adapted["state"]["eef_9d"].reshape(-1), state[:9])
    assert np.array_equal(adapted["state"]["hand_joint_pos"].reshape(-1), state[9:19])
    assert np.array_equal(adapted["state"]["arm_joint_pos"].reshape(-1), state[19:26])
    assert adapted["language"]["annotation.human.action.task_description"] == [["pick and place"]]


def test_channel_layout_hash_is_order_sensitive() -> None:
    assert channel_layout_hash(["a[0]", "b[0]"]) != channel_layout_hash(["b[0]", "a[0]"])


def test_image_key_map_rejects_duplicate_sources_and_targets() -> None:
    assert _parse_key_map(["head=ego_view", "wrist=wrist_view"]) == {
        "head": "ego_view",
        "wrist": "wrist_view",
    }
    with pytest.raises(ValueError, match="duplicate key mapping"):
        _parse_key_map(["head=ego_view", "head=wrist_view"])
    with pytest.raises(ValueError, match="duplicate key mapping"):
        _parse_key_map(["head=ego_view", "wrist=ego_view"])


def test_backend_encodes_raw_tokens_before_action_head_mutates_them() -> None:
    class FakeActionHead:
        def _encode_features(self, backbone_output, _action_inputs):
            backbone_output["backbone_features"].add_(100.0)
            return SimpleNamespace(
                backbone_features=backbone_output["backbone_features"],
                state_features=torch.zeros((1, 1, 2)),
            )

        def get_action_with_features(self, **_kwargs):
            return {"action_pred": torch.zeros((1, 3, 2), dtype=torch.float32)}

    class FakeModel:
        action_head = FakeActionHead()

        def prepare_input(self, **_collated):
            return {}, SimpleNamespace(embodiment_id=torch.zeros((1,), dtype=torch.long))

        def backbone(self, _inputs):
            return {
                "backbone_features": torch.ones((1, 3, 2), dtype=torch.float32),
                "backbone_attention_mask": torch.ones((1, 3), dtype=torch.bool),
                "image_mask": torch.ones((1, 3), dtype=torch.bool),
            }

    class FakeEncoder:
        def encode_rl_token(self, packed, _mask):
            return packed[:, 0]

    backend = object.__new__(GrootN1d7FeatureBackend)
    backend.contract = FeatureContract(z_dim=2, chunk_len=2, action_dim=2, proprio_dim=2)
    backend.policy = SimpleNamespace(
        model=FakeModel(),
        processor=SimpleNamespace(
            decode_action=lambda _action, _tag, _states: {
                "joint": np.zeros((1, 3, 2), dtype=np.float32)
            }
        ),
        embodiment_tag="fake",
        modality_configs={
            "state": SimpleNamespace(modality_keys=("joint",)),
            "action": SimpleNamespace(modality_keys=("joint",)),
        },
    )
    backend.encoder = FakeEncoder()
    backend.token_scope = "all"
    backend.max_vl_tokens = 3
    backend.token_sampling = "uniform"
    backend.proprio_keys = None
    backend.num_inference_timesteps = 32
    backend._torch = SimpleNamespace(inference_mode=nullcontext)
    backend._pack_vl_tokens = lambda output, **_kwargs: (
        output["backbone_features"].clone(),
        torch.ones((1, 3), dtype=torch.bool),
        None,
        None,
        None,
    )
    backend._prepare_policy_inputs = lambda _observation: (
        {},
        [{"joint": np.asarray([[0.1, 0.2]], dtype=np.float32)}],
    )

    result = backend.infer_one({})
    assert np.array_equal(result["z_rl"], np.ones((2,), dtype=np.float32))
    assert result["num_inference_timesteps"] == 32


def test_explicit_invalid_groot_checkout_fails(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="checkout is invalid"):
        ensure_groot_repo(tmp_path)
