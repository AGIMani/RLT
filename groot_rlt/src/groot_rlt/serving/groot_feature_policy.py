"""GR00T N1.7 backend for the RLT Machine-A feature contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class FeatureBackend(Protocol):
    """Backend that produces one Machine-A payload from one observation."""

    def infer_one(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeatureContract:
    z_dim: int = 2048
    chunk_len: int = 10
    action_dim: int = 26
    proprio_dim: int = 26

    def validate(self) -> None:
        for name in ("z_dim", "chunk_len", "action_dim", "proprio_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


def channel_layout_hash(names: list[str] | tuple[str, ...]) -> str:
    """Return a stable fingerprint for an ordered per-dimension layout."""

    payload = json.dumps(list(names), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expand_layout(keys: tuple[str, ...], arrays: list[np.ndarray]) -> list[str]:
    return [
        f"{key}[{index}]" for key, array in zip(keys, arrays) for index in range(array.shape[-1])
    ]


class MachineAFeaturePolicy:
    """Validate GR00T output against the existing online-RL wire contract."""

    def __init__(
        self,
        backend: FeatureBackend,
        contract: FeatureContract,
        *,
        supports_batch: bool = True,
    ):
        contract.validate()
        self.backend = backend
        self.contract = contract
        self.supports_batch = supports_batch

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        z_rl = np.asarray(payload["z_rl"], dtype=np.float32)
        if z_rl.ndim == 2 and z_rl.shape[0] == 1:
            z_rl = z_rl[0]
        if z_rl.shape != (self.contract.z_dim,):
            raise ValueError(f"z_rl must have shape ({self.contract.z_dim},), got {z_rl.shape}")

        ref_chunk = np.asarray(payload["ref_chunk"], dtype=np.float32)
        if ref_chunk.ndim == 3 and ref_chunk.shape[0] == 1:
            ref_chunk = ref_chunk[0]
        if (
            ref_chunk.ndim != 2
            or ref_chunk.shape[0] < self.contract.chunk_len
            or ref_chunk.shape[1] != self.contract.action_dim
        ):
            raise ValueError(
                "ref_chunk must have exact action dimension and sufficient horizon: "
                f"expected [>={self.contract.chunk_len}, {self.contract.action_dim}], "
                f"got {ref_chunk.shape}"
            )
        ref_chunk = ref_chunk[: self.contract.chunk_len]

        proprio = np.asarray(payload["proprio"], dtype=np.float32)
        if proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        if proprio.shape != (self.contract.proprio_dim,):
            raise ValueError(
                f"proprio must have shape ({self.contract.proprio_dim},), got {proprio.shape}"
            )

        for name, expected_dim in (
            ("action_layout", self.contract.action_dim),
            ("proprio_layout", self.contract.proprio_dim),
        ):
            if name in payload and len(payload[name]) != expected_dim:
                raise ValueError(f"{name} must contain {expected_dim} names")

        return {**payload, "z_rl": z_rl, "ref_chunk": ref_chunk, "proprio": proprio}

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        if "batch" in request:
            if not self.supports_batch:
                raise ValueError("this GR00T feature backend does not support batched inference")
            observations = request["batch"]
            if not isinstance(observations, list):
                raise TypeError("batch request must contain a list of observations")
            return {
                "batch_results": [
                    self._normalize(self.backend.infer_one(observation))
                    for observation in observations
                ]
            }
        return self._normalize(self.backend.infer_one(request))


class GrootN1d7FeatureBackend:
    """Run GR00T action inference and RL-token encoding in one backbone pass.

    The returned ``ref_chunk`` is in processor-decoded physical action space.
    Normalization remains the responsibility of ``ActionRepresentationAdapter``
    in the online learner, matching the existing openpi Machine-A boundary.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        processor_path: str | Path,
        vlm_model_path: str | Path,
        rl_token_checkpoint: str | Path,
        embodiment_tag: str,
        device: str,
        contract: FeatureContract,
        strict: bool = True,
        token_scope: str | None = None,
        token_sampling: str | None = None,
        max_vl_tokens: int | None = None,
        proprio_keys: tuple[str, ...] | None = None,
        image_key_map: dict[str, str] | None = None,
        flat_state_layout: tuple[tuple[str, int], ...] | None = None,
        default_instruction: str | None = None,
        num_inference_timesteps: int | None = None,
    ) -> None:
        import torch

        from groot_rlt.representation.precompute_rl_tokens_and_vla_actions import (
            PrecomputeCheckpointRokaePolicy,
        )
        from groot_rlt.representation.train_vl_embedding_autoencoder import pack_vl_tokens
        from groot_rlt.representation.visualize_rl_token_umap import load_rl_token_encoder

        contract.validate()
        self.contract = contract
        self.device = torch.device(device)
        self.policy = PrecomputeCheckpointRokaePolicy(
            model_path=Path(model_path).expanduser().resolve(),
            processor_path=Path(processor_path).expanduser().resolve(),
            device=str(self.device),
            strict=strict,
            vlm_model_path=Path(vlm_model_path).expanduser().resolve(),
            embodiment_tag=embodiment_tag,
        )
        encoder, checkpoint_args, _, _ = load_rl_token_encoder(
            Path(rl_token_checkpoint).expanduser().resolve(), self.device
        )
        if encoder.config.rl_token_dim != contract.z_dim:
            raise ValueError(
                f"RL-token checkpoint dim {encoder.config.rl_token_dim} != z_dim {contract.z_dim}"
            )
        self.encoder = encoder
        trained_scope = str(checkpoint_args.get("token_scope", "all"))
        trained_sampling = str(checkpoint_args.get("token_sampling", "uniform"))
        trained_max_tokens = int(checkpoint_args.get("max_vl_tokens", encoder.config.max_vl_tokens))
        if token_scope is not None and token_scope != trained_scope:
            raise ValueError(
                f"serving token_scope={token_scope!r} differs from encoder training "
                f"token_scope={trained_scope!r}"
            )
        if token_sampling is not None and token_sampling != trained_sampling:
            raise ValueError(
                f"serving token_sampling={token_sampling!r} differs from encoder training "
                f"token_sampling={trained_sampling!r}"
            )
        if max_vl_tokens is not None and int(max_vl_tokens) != trained_max_tokens:
            raise ValueError(
                f"serving max_vl_tokens={max_vl_tokens} differs from encoder training "
                f"max_vl_tokens={trained_max_tokens}"
            )
        self.token_scope = trained_scope
        self.token_sampling = trained_sampling
        self.max_vl_tokens = trained_max_tokens
        if self.token_sampling == "random":
            raise ValueError(
                "token_sampling='random' is not allowed for online serving because it makes "
                "z_rl non-deterministic and perturbs the flow sampler RNG."
            )
        if self.max_vl_tokens > int(encoder.config.max_vl_tokens):
            raise ValueError(
                f"max_vl_tokens={self.max_vl_tokens} exceeds encoder capacity "
                f"{encoder.config.max_vl_tokens}"
            )
        backbone_dim = getattr(self.policy.model.config, "backbone_embedding_dim", None)
        if backbone_dim is not None and int(backbone_dim) != int(encoder.config.input_dim):
            raise ValueError(
                f"RL-token encoder input_dim={encoder.config.input_dim} does not match "
                f"GR00T backbone_embedding_dim={backbone_dim}"
            )
        trained_vlm_path = checkpoint_args.get("vlm_model_path")
        serving_vlm_path = Path(vlm_model_path).expanduser().resolve()
        if trained_vlm_path:
            trained_vlm = Path(str(trained_vlm_path)).expanduser()
            if trained_vlm.exists() and not serving_vlm_path.samefile(trained_vlm.resolve()):
                raise ValueError(
                    f"RL-token encoder was trained with VLM {trained_vlm.resolve()}, "
                    f"but serving uses {serving_vlm_path}"
                )
        self.rl_token_checkpoint_args = checkpoint_args
        self.proprio_keys = proprio_keys
        self.image_key_map = dict(image_key_map or {})
        self.flat_state_layout = flat_state_layout
        self.default_instruction = default_instruction
        action_keys = tuple(self.policy.modality_configs["action"].modality_keys)
        self.action_layout: list[str] | None = None
        if action_keys == ("eef_9d", "hand_joint_target", "arm_joint_target"):
            from groot_rlt.integration.online_stats import CHANNEL_NAMES

            self.action_layout = list(CHANNEL_NAMES)
        self.proprio_layout: list[str] | None = None
        if proprio_keys is None:
            try:
                state_layout = self._resolved_flat_state_layout()
            except ValueError:
                state_layout = None
            if state_layout is not None:
                self.proprio_layout = [
                    f"{key}[{index}]" for key, dim in state_layout for index in range(dim)
                ]
        self.action_layout_hash = (
            None if self.action_layout is None else channel_layout_hash(self.action_layout)
        )
        self.proprio_layout_hash = (
            None if self.proprio_layout is None else channel_layout_hash(self.proprio_layout)
        )
        self._pack_vl_tokens = pack_vl_tokens
        self._torch = torch
        if num_inference_timesteps is not None:
            if int(num_inference_timesteps) <= 0:
                raise ValueError("num_inference_timesteps must be positive")
            self.policy.model.action_head.num_inference_timesteps = int(num_inference_timesteps)
        self.num_inference_timesteps = int(self.policy.model.action_head.num_inference_timesteps)
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        model = self.policy.model
        missing = [
            name
            for name in ("prepare_input", "backbone", "action_head")
            if not hasattr(model, name)
        ]
        action_head = getattr(model, "action_head", None)
        for name in ("_encode_features", "get_action_with_features"):
            if action_head is None or not hasattr(action_head, name):
                missing.append(f"action_head.{name}")
        if missing:
            raise RuntimeError(
                "The selected GR00T checkout/checkpoint lacks required N1.7 feature APIs: "
                + ", ".join(missing)
            )

    @staticmethod
    def _batched_video(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.uint8)
        if array.ndim == 3:
            array = array[None, None]
        elif array.ndim == 4:
            array = array[None]
        if array.ndim != 5 or array.shape[0] != 1:
            raise ValueError(
                "Machine-A single observations require video HWC, THWC, or [1,T,H,W,C]; "
                f"got {array.shape}"
            )
        return array

    @staticmethod
    def _batched_state(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, None]
        elif array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(
                f"Machine-A single observations require state D, TD, or [1,T,D]; got {array.shape}"
            )
        return array

    def _resolved_flat_state_layout(self) -> tuple[tuple[str, int], ...]:
        if self.flat_state_layout is not None:
            return self.flat_state_layout
        keys = tuple(self.policy.modality_configs["state"].modality_keys)
        if keys == ("eef_9d", "hand_joint_pos", "arm_joint_pos"):
            return (("eef_9d", 9), ("hand_joint_pos", 10), ("arm_joint_pos", 7))
        if len(keys) == 1:
            return ((keys[0], self.contract.proprio_dim),)
        raise ValueError(
            "flat observation state requires --flat-state-field KEY=DIM for every GR00T "
            f"state key; checkpoint expects {keys}"
        )

    def _adapt_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Accept native GR00T observations or the legacy flat RLT wire shape."""

        expected_video = tuple(self.policy.modality_configs["video"].modality_keys)
        expected_state = tuple(self.policy.modality_configs["state"].modality_keys)
        language_key = self.policy.language_key

        if all(isinstance(observation.get(key), dict) for key in ("video", "state", "language")):
            videos = {key: self._batched_video(observation["video"][key]) for key in expected_video}
            states = {key: self._batched_state(observation["state"][key]) for key in expected_state}
            language = observation["language"][language_key]
            if isinstance(language, str):
                language = [[language]]
            elif language and isinstance(language[0], str):
                language = [list(language)]
            return {"video": videos, "state": states, "language": {language_key: language}}

        if "images" not in observation or "state" not in observation:
            raise ValueError(
                "observation must use native GR00T {video,state,language} or flat RLT "
                "{images,state,prompt} format"
            )
        source_images = observation["images"]
        if not isinstance(source_images, dict):
            raise TypeError("flat observation images must be a mapping")
        inverse_image_map = {target: source for source, target in self.image_key_map.items()}
        videos = {}
        for target in expected_video:
            source = inverse_image_map.get(target, target)
            if source not in source_images:
                raise KeyError(
                    f"missing image {source!r} for GR00T video key {target!r}; "
                    "use --image-key SOURCE=TARGET"
                )
            videos[target] = self._batched_video(source_images[source])

        flat_state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        layout = self._resolved_flat_state_layout()
        if tuple(key for key, _ in layout) != expected_state:
            raise ValueError(
                f"flat state layout keys {tuple(key for key, _ in layout)} do not match "
                f"checkpoint keys {expected_state}"
            )
        if sum(dim for _, dim in layout) != flat_state.size:
            raise ValueError(
                f"flat state layout totals {sum(dim for _, dim in layout)} values, "
                f"observation has {flat_state.size}"
            )
        states = {}
        offset = 0
        for key, dim in layout:
            states[key] = flat_state[offset : offset + dim][None, None]
            offset += dim
        instruction = observation.get("prompt", self.default_instruction)
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(
                "flat observation requires a non-empty prompt or --default-instruction"
            )
        return {
            "video": videos,
            "state": states,
            "language": {language_key: [[instruction]]},
        }

    def _prepare_policy_inputs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
        from gr00t.data.types import MessageType, VLAStepData

        observation = self._adapt_observation(observation)
        if self.policy.strict:
            self.policy.check_observation(observation)
        unbatched = self.policy._unbatch_observation(observation)
        processed = []
        states = []
        for item in unbatched:
            states.append(item["state"])
            step = VLAStepData(
                images=item["video"],
                states=item["state"],
                actions={},
                text=item["language"][self.policy.language_key][0],
                embodiment=self.policy.embodiment_tag,
            )
            processed.append(
                self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        collated = self.policy.collate_fn(processed)
        collated = self.policy._rec_to_dtype(collated, dtype=self._torch.bfloat16)
        return collated, states

    def _extract_proprio(self, states: dict[str, np.ndarray]) -> np.ndarray:
        keys = self.proprio_keys or tuple(self.policy.modality_configs["state"].modality_keys)
        missing = [key for key in keys if key not in states]
        if missing:
            raise KeyError(f"GR00T observation is missing configured proprio keys: {missing}")
        parts = []
        for key in keys:
            value = np.asarray(states[key], dtype=np.float32)
            if value.ndim == 1:
                parts.append(value)
            elif value.ndim == 2:
                parts.append(value[-1])
            else:
                raise ValueError(f"state[{key!r}] must be rank-1 or rank-2, got {value.shape}")
        proprio = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
        if proprio.size != self.contract.proprio_dim:
            raise ValueError(
                f"configured state keys provide {proprio.size} values, "
                f"but proprio_dim={self.contract.proprio_dim}"
            )
        return proprio

    def infer_one(self, observation: dict[str, Any]) -> dict[str, Any]:
        collated, states = self._prepare_policy_inputs(observation)
        if len(states) != 1:
            raise ValueError(
                "Machine-A single inference expects batch size 1; use the outer batch request "
                "for multiple observations"
            )

        model = self.policy.model
        with self._torch.inference_mode():
            backbone_inputs, action_inputs = model.prepare_input(**collated)
            backbone_output = model.backbone(backbone_inputs)
            # The RL-token encoder was trained on the raw backbone embeddings.
            # ``_encode_features`` applies VLLN/self-attention and mutates
            # ``backbone_output`` in place, so capture those raw tokens first.
            packed, packed_mask, _, _, _ = self._pack_vl_tokens(
                backbone_output,
                token_scope=self.token_scope,
                max_tokens=self.max_vl_tokens,
                token_sampling=self.token_sampling,
            )
            features = model.action_head._encode_features(backbone_output, action_inputs)
            model_pred = model.action_head.get_action_with_features(
                backbone_features=features.backbone_features,
                state_features=features.state_features,
                embodiment_id=action_inputs.embodiment_id,
                backbone_output=backbone_output,
                action_input=action_inputs,
                options=None,
            )
            z_rl = self.encoder.encode_rl_token(packed.float(), packed_mask)

        normalized_action = model_pred["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.policy.modality_configs["state"].modality_keys
        }
        decoded = self.policy.processor.decode_action(
            normalized_action,
            self.policy.embodiment_tag,
            batched_states,
        )
        action_keys = tuple(self.policy.modality_configs["action"].modality_keys)
        decoded_parts = [np.asarray(decoded[key][0], dtype=np.float32) for key in action_keys]
        ref_chunk = np.concatenate(decoded_parts, axis=-1)
        if ref_chunk.shape[0] < self.contract.chunk_len:
            raise ValueError(
                f"GR00T action horizon {ref_chunk.shape[0]} < chunk_len {self.contract.chunk_len}"
            )
        if ref_chunk.shape[1] != self.contract.action_dim:
            raise ValueError(
                f"GR00T action dim {ref_chunk.shape[1]} != action_dim {self.contract.action_dim}"
            )
        proprio_parts = [
            np.asarray(states[0][key], dtype=np.float32)
            for key in (self.proprio_keys or self.policy.modality_configs["state"].modality_keys)
        ]
        action_layout = getattr(self, "action_layout", None) or _expand_layout(
            action_keys, decoded_parts
        )
        proprio_keys = tuple(
            self.proprio_keys or self.policy.modality_configs["state"].modality_keys
        )
        if len(action_layout) != ref_chunk.shape[1]:
            raise RuntimeError("configured action layout does not match decoded action dimension")
        proprio_layout = getattr(self, "proprio_layout", None) or _expand_layout(
            proprio_keys, proprio_parts
        )
        if len(proprio_layout) != self.contract.proprio_dim:
            raise RuntimeError("configured proprio layout does not match proprio dimension")
        return {
            "z_rl": z_rl[0].detach().float().cpu().numpy(),
            "ref_chunk": ref_chunk,
            "proprio": self._extract_proprio(states[0]),
            "action_keys": action_keys,
            "action_layout": action_layout,
            "action_layout_hash": channel_layout_hash(action_layout),
            "proprio_layout": proprio_layout,
            "proprio_layout_hash": channel_layout_hash(proprio_layout),
            "action_space": "processor_decoded_physical",
            "num_inference_timesteps": self.num_inference_timesteps,
            "rtc_applied": False,
        }
