#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-loop GR00T action comparison on the Rokae xMate3 + L10 LeRobot dataset.

This evaluates the local ego+wrist Rokae xMate3 + L10 checkpoint on the
full-orientation multiview L10 dataset and writes explicit predicted-vs-GT
action files.

Example:

    .venv/bin/python examples/IsaacLab/compare_l10_gr00t_zero_shot_actions.py \
        --episode-indices 0 \
        --steps 128

Important:
    This script loads modality configs, statistics, image preprocessing, and
    relative-action settings from the finetuned checkpoint processor. It assumes
    an official-aligned checkpoint that contains processor_config.json and
    statistics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from groot_rlt.groot_repo import ensure_groot_repo

REPO_ROOT = ensure_groot_repo()

from groot_rlt.integration.checkpoint_policy_utils import (  # noqa: E402
    load_checkpoint_modality_config,
    strip_decode_only_options,
)
from groot_rlt.integration.defaults import (  # noqa: E402
    L10_MODEL_DIR,
    L10_PREPARED_DATASET_DIR,
    L10_VLM_MODEL_PATH,
)

DEFAULT_MODEL_PATH = L10_MODEL_DIR
DEFAULT_VLM_MODEL_PATH = L10_VLM_MODEL_PATH
DEFAULT_DATASET_DIR = L10_PREPARED_DATASET_DIR
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "l10_full_orientation_action_compare"
L10_POLICY_ACTION_NAMES = [
    "eef_9d.x",
    "eef_9d.y",
    "eef_9d.z",
    "eef_9d.rot6d_0",
    "eef_9d.rot6d_1",
    "eef_9d.rot6d_2",
    "eef_9d.rot6d_3",
    "eef_9d.rot6d_4",
    "eef_9d.rot6d_5",
    "hand_joint_target.thumb_cmc_pitch",
    "hand_joint_target.thumb_cmc_yaw",
    "hand_joint_target.index_mcp_pitch",
    "hand_joint_target.middle_mcp_pitch",
    "hand_joint_target.ring_mcp_pitch",
    "hand_joint_target.pinky_mcp_pitch",
    "hand_joint_target.index_mcp_roll",
    "hand_joint_target.ring_mcp_roll",
    "hand_joint_target.pinky_mcp_roll",
    "hand_joint_target.thumb_cmc_roll",
    "arm_joint_target.0",
    "arm_joint_target.1",
    "arm_joint_target.2",
    "arm_joint_target.3",
    "arm_joint_target.4",
    "arm_joint_target.5",
    "arm_joint_target.6",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vlm-model-path", type=str, default=str(DEFAULT_VLM_MODEL_PATH))
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=160, help="Max frames per episode to compare.")
    parser.add_argument(
        "--replan-horizon",
        type=int,
        default=8,
        help=(
            "Stride between model calls. The default RTC overlap is "
            "action_horizon - replan_horizon."
        ),
    )
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RTC action chunk overlap by feeding the previous predicted chunk back in.",
    )
    parser.add_argument(
        "--rtc-overlap-steps",
        type=int,
        default=None,
        help=(
            "Previous chunk steps to overlap into the next DiT sample. "
            "Defaults to action_horizon - replan_horizon."
        ),
    )
    parser.add_argument(
        "--rtc-frozen-steps",
        type=int,
        default=2,
        help="Initial overlap steps kept fixed from the previous action chunk.",
    )
    parser.add_argument(
        "--rtc-ramp-rate",
        type=float,
        default=6.0,
        help="Exponential ramp rate used by the GR00T RTC inpainting logic.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "torchcodec", "decord", "opencv"),
        default="ffmpeg",
        help="Backend used by gr00t.utils.video_utils.get_frames_by_indices.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'."
    )
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--strict-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Gr00tPolicy input/output validation.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_episode_indices(dataset_dir: Path) -> list[int]:
    episodes_file = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        return [int(row["episode_index"]) for row in _read_jsonl(episodes_file)]

    info = _read_json(dataset_dir / "meta" / "info.json")
    return list(range(int(info["total_episodes"])))


def _load_episode_parquet(dataset_dir: Path, episode_index: int) -> Any:
    import pandas as pd

    info = _read_json(dataset_dir / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    data_path = info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    parquet_path = dataset_dir / data_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _to_matrix(column: Any, *, expected_dim: int, name: str) -> np.ndarray:
    out = np.asarray([np.asarray(v, dtype=np.float32) for v in column], dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected {name} shape {out.shape}, expected (*, {expected_dim})")
    return out


def _feature_dim(dataset_dir: Path, feature_key: str) -> int:
    info = _read_json(dataset_dir / "meta" / "info.json")
    shape = info["features"][feature_key]["shape"]
    if len(shape) != 1:
        raise RuntimeError(f"Expected 1D feature {feature_key}, got shape={shape}")
    return int(shape[0])


def _resolve_processor_path(model_path: Path) -> Path:
    for candidate in (model_path / "processor", model_path.parent / "processor", model_path):
        if (candidate / "processor_config.json").exists():
            return candidate
    return model_path


class CheckpointRokaePolicy:
    """A Gr00tPolicy-compatible wrapper for official-aligned L10 checkpoints."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        strict: bool,
        vlm_model_path: Path | None = None,
    ) -> None:
        import gr00t.model  # noqa: F401
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
        from gr00t.policy.policy import BasePolicy
        from transformers import AutoConfig, AutoModel, AutoProcessor

        BasePolicy.__init__(self, strict=strict)
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        model_config = AutoConfig.from_pretrained(model_path)
        processor_overrides = {}
        if vlm_model_path is not None:
            local_vlm_model_path = Path(vlm_model_path).expanduser().resolve()
            model_config.model_name = str(local_vlm_model_path)
            processor_overrides["model_name"] = str(local_vlm_model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=model_config,
            transformers_loading_kwargs={"local_files_only": True},
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        processor_path = _resolve_processor_path(model_path)
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            transformers_loading_kwargs={"local_files_only": True},
            **processor_overrides,
        )
        self.processor.eval()

        all_modality_configs = self.processor.get_modality_configs()
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self._rec_to_dtype = _rec_to_dtype

        # Reuse the validated GR00T policy implementation after custom initialization.
        self._unbatch_observation = Gr00tPolicy._unbatch_observation.__get__(self)
        self.check_observation = Gr00tPolicy.check_observation.__get__(self)
        self.check_action = Gr00tPolicy.check_action.__get__(self)
        self.get_modality_config = Gr00tPolicy.get_modality_config.__get__(self)
        self.reset = Gr00tPolicy.reset.__get__(self)

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        previous_action: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch
        from gr00t.data.types import MessageType, VLAStepData

        unbatched_observations = self._unbatch_observation(observation)
        if previous_action is not None and len(unbatched_observations) != 1:
            raise ValueError("RTC previous_action currently supports batch size 1.")

        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            states.append(obs["state"])
            vla_step_data = VLAStepData(
                images=obs["video"],
                states=obs["state"],
                actions={} if previous_action is None else previous_action,
                text=obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = self._rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        with torch.inference_mode():
            model_pred = self.model.get_action(
                **collated_inputs,
                options=strip_decode_only_options(options),
            )
        normalized_action = model_pred["action_pred"].float()

        batched_states = {}
        for key in self.modality_configs["state"].modality_keys:
            batched_states[key] = np.stack([state[key] for state in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        return {key: value.astype(np.float32) for key, value in unnormalized_action.items()}, {}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        previous_action: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options, previous_action=previous_action)
        if self.strict:
            self.check_action(action)
        return action, info


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_video_frames(
    dataset_dir: Path,
    *,
    episode_index: int,
    video_key: str,
    frame_indices: np.ndarray,
    video_backend: str,
) -> np.ndarray:
    from gr00t.utils.video_utils import get_frames_by_indices

    info = _read_json(dataset_dir / "meta" / "info.json")
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    chunk_size = int(info.get("chunks_size", 1000))
    original_key = modality_meta["video"][video_key].get(
        "original_key", f"observation.images.{video_key}"
    )
    video_path = dataset_dir / info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        video_key=original_key,
        episode_index=episode_index,
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frames = get_frames_by_indices(
        str(video_path),
        np.asarray(frame_indices, dtype=np.int64),
        video_backend=video_backend,
        video_backend_kwargs={},
    )
    return np.asarray(frames, dtype=np.uint8)


def _extract_groups(
    matrix: np.ndarray,
    modality_meta: dict[str, Any],
    modality: str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    out = {}
    for key in keys:
        sl = modality_meta[modality][key]
        out[key] = matrix[:, int(sl["start"]) : int(sl["end"])].astype(np.float32, copy=False)
    return out


def _get_instruction(dataset_dir: Path, df: Any, *, override: str | None) -> str:
    if override:
        return override
    tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    task_map = {int(row["task_index"]): row["task"] for row in tasks}
    if "annotation.human.action.task_description" in df.columns:
        task_index = int(df["annotation.human.action.task_description"].iloc[0])
        return str(task_map.get(task_index, "teleop"))
    return "teleop"


def _build_observation(
    dataset_dir: Path,
    *,
    episode_index: int,
    step: int,
    states_by_key: dict[str, np.ndarray],
    modality_config: dict[str, Any],
    modality_meta: dict[str, Any],
    instruction: str,
    video_backend: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

    for key in modality_config["state"].modality_keys:
        delta = np.asarray(modality_config["state"].delta_indices, dtype=np.int64)
        indices = np.clip(step + delta, 0, len(states_by_key[key]) - 1)
        obs["state"][key] = states_by_key[key][indices][None, :].astype(np.float32, copy=False)

    dataset_video_keys = list(modality_meta.get("video", {}).keys())
    for key in modality_config["video"].modality_keys:
        delta = np.asarray(modality_config["video"].delta_indices, dtype=np.int64)
        indices = np.clip(step + delta, 0, len(states_by_key[next(iter(states_by_key))]) - 1)
        if key not in modality_meta.get("video", {}):
            raise KeyError(
                f"Video key '{key}' not in dataset modality.json. "
                f"All L10 policies require ego_view and wrist_view. Available: {dataset_video_keys}"
            )
        frames = _load_video_frames(
            dataset_dir,
            episode_index=episode_index,
            video_key=key,
            frame_indices=indices,
            video_backend=video_backend,
        )
        obs["video"][key] = frames[None, :].astype(np.uint8, copy=False)

    language_key = modality_config["language"].modality_keys[0]
    obs["language"][language_key] = [[instruction]]
    return obs


def _concat_action_dict(action: dict[str, np.ndarray], action_keys: list[str]) -> np.ndarray:
    chunks = []
    for key in action_keys:
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        chunks.append(value)
    return np.concatenate(chunks, axis=-1)


def _unbatch_action_dict(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    previous_action = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        previous_action[key] = arr.astype(np.float32, copy=True)
    return previous_action


def _rtc_options(
    *,
    args: argparse.Namespace,
    action_horizon: int,
    previous_action: dict[str, np.ndarray] | None,
) -> dict[str, Any] | None:
    if not bool(args.rtc) or previous_action is None:
        return None

    previous_horizon = min(int(v.shape[0]) for v in previous_action.values())
    overlap_steps = (
        int(args.rtc_overlap_steps)
        if args.rtc_overlap_steps is not None
        else action_horizon - int(args.replan_horizon)
    )
    overlap_steps = max(0, min(overlap_steps, previous_horizon, action_horizon))
    if overlap_steps <= 0:
        logging.warning(
            "RTC requested but overlap is 0. Use --replan-horizon smaller than action horizon %d.",
            action_horizon,
        )
        return None

    frozen_steps = max(0, min(int(args.rtc_frozen_steps), overlap_steps))
    return {
        "action_horizon": int(previous_horizon),
        "rtc_overlap_steps": int(overlap_steps),
        "rtc_frozen_steps": int(frozen_steps),
        "rtc_ramp_rate": float(args.rtc_ramp_rate),
    }


def _plot_actions(
    *,
    output_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    frame_indices: np.ndarray,
    action_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable; skipping plot: %s", exc)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_dim = gt.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(12, max(3, 2.2 * n_dim)), sharex=True)
    if n_dim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        label = action_names[i] if i < len(action_names) else f"action_{i}"
        ax.plot(frame_indices, gt[:, i], label="ground truth", linewidth=1.4)
        ax.plot(frame_indices, pred[:, i], label="prediction", linewidth=1.1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("dataset frame index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _summarize_errors(gt: np.ndarray, pred: np.ndarray, action_names: list[str]) -> dict[str, Any]:
    err = pred - gt
    per_dim_mae = np.mean(np.abs(err), axis=0)
    per_dim_rmse = np.sqrt(np.mean(err * err, axis=0))
    return {
        "num_compared_steps": int(gt.shape[0]),
        "action_dim": int(gt.shape[1]),
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "arm_xyz_mae": float(np.mean(np.abs(err[:, :3]))) if gt.shape[1] >= 3 else None,
        "wrist_rot6d_mae": float(np.mean(np.abs(err[:, 3:9]))) if gt.shape[1] >= 9 else None,
        "hand_mae": float(np.mean(np.abs(err[:, 9:19]))) if gt.shape[1] >= 19 else None,
        "arm_joint_mae": float(np.mean(np.abs(err[:, 19:26]))) if gt.shape[1] >= 26 else None,
        "per_dim": [
            {
                "index": int(i),
                "name": action_names[i] if i < len(action_names) else f"action_{i}",
                "mae": float(per_dim_mae[i]),
                "rmse": float(per_dim_rmse[i]),
            }
            for i in range(gt.shape[1])
        ],
    }


def _action_names(dataset_dir: Path) -> list[str]:
    info = _read_json(dataset_dir / "meta" / "info.json")
    return list(info["features"]["action"].get("names") or [])


def _policy_action_names(raw_action_names: list[str], action_dim: int) -> list[str]:
    if action_dim == len(L10_POLICY_ACTION_NAMES):
        return list(L10_POLICY_ACTION_NAMES)
    names = list(raw_action_names)
    while len(names) < action_dim:
        idx = len(names)
        if idx < len(L10_POLICY_ACTION_NAMES):
            names.append(L10_POLICY_ACTION_NAMES[idx])
        else:
            names.append(f"action_{idx}")
    return names[:action_dim]


def _run_episode(
    *,
    policy: CheckpointRokaePolicy,
    dataset_dir: Path,
    episode_index: int,
    modality_config: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    df = _load_episode_parquet(dataset_dir, episode_index)
    states = _to_matrix(
        df["observation.state"],
        expected_dim=_feature_dim(dataset_dir, "observation.state"),
        name="observation.state",
    )
    actions = _to_matrix(
        df["action"],
        expected_dim=_feature_dim(dataset_dir, "action"),
        name="action",
    )

    states_by_key = _extract_groups(
        states,
        modality_meta,
        "state",
        list(modality_config["state"].modality_keys),
    )
    actions_by_key = {}
    for key in modality_config["action"].modality_keys:
        sl = modality_meta["action"][key]
        source = states if sl.get("original_key") == "observation.state" else actions
        actions_by_key[key] = source[:, int(sl["start"]) : int(sl["end"])].astype(
            np.float32, copy=False
        )
    gt_action_full = np.concatenate(
        [actions_by_key[key] for key in modality_config["action"].modality_keys],
        axis=-1,
    )
    action_keys = list(modality_config["action"].modality_keys)
    action_delta_indices = np.asarray(modality_config["action"].delta_indices, dtype=np.int64)
    action_names = _policy_action_names(_action_names(dataset_dir), int(gt_action_full.shape[1]))
    instruction = _get_instruction(dataset_dir, df, override=args.instruction)

    actual_steps = min(int(args.steps), len(df))
    action_horizon = len(modality_config["action"].delta_indices)
    pred_rows: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    frame_rows: list[int] = []
    inference_times: list[float] = []
    previous_action: dict[str, np.ndarray] | None = None

    for step in range(0, actual_steps, int(args.replan_horizon)):
        obs = _build_observation(
            dataset_dir,
            episode_index=episode_index,
            step=step,
            states_by_key=states_by_key,
            modality_config=modality_config,
            modality_meta=modality_meta,
            instruction=instruction,
            video_backend=str(args.video_backend),
        )
        rtc_options = _rtc_options(
            args=args,
            action_horizon=action_horizon,
            previous_action=previous_action,
        )
        tic = time.perf_counter()
        pred_action, _ = policy.get_action(
            obs,
            options=rtc_options,
            previous_action=previous_action if rtc_options is not None else None,
        )
        inference_times.append(time.perf_counter() - tic)
        pred_chunk = _concat_action_dict(pred_action, action_keys)
        previous_action = _unbatch_action_dict(pred_action)

        valid_gt_indices = step + action_delta_indices[: len(pred_chunk)]
        valid_gt_indices = valid_gt_indices[valid_gt_indices < actual_steps]
        horizon = min(int(args.replan_horizon), len(pred_chunk), len(valid_gt_indices))
        if horizon <= 0:
            break
        pred_rows.append(pred_chunk[:horizon])
        gt_rows.append(gt_action_full[valid_gt_indices[:horizon]])
        frame_rows.extend(valid_gt_indices[:horizon].tolist())
        logging.info(
            "episode=%d step=%d horizon=%d rtc=%s inference=%.3fs",
            episode_index,
            step,
            horizon,
            "off" if rtc_options is None else rtc_options,
            inference_times[-1],
        )

    pred = np.concatenate(pred_rows, axis=0).astype(np.float32)
    gt = np.concatenate(gt_rows, axis=0).astype(np.float32)
    frames = np.asarray(frame_rows, dtype=np.int64)
    metrics = _summarize_errors(gt, pred, action_names)
    metrics.update(
        {
            "episode_index": int(episode_index),
            "instruction": instruction,
            "avg_inference_s": float(np.mean(inference_times)) if inference_times else 0.0,
            "max_inference_s": float(np.max(inference_times)) if inference_times else 0.0,
            "replan_horizon": int(args.replan_horizon),
            "rtc": bool(args.rtc),
            "rtc_overlap_steps": (
                int(args.rtc_overlap_steps)
                if args.rtc_overlap_steps is not None
                else max(0, action_horizon - int(args.replan_horizon))
            ),
            "rtc_frozen_steps": int(args.rtc_frozen_steps),
            "rtc_ramp_rate": float(args.rtc_ramp_rate),
        }
    )

    episode_prefix = output_dir / f"episode_{episode_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_prefix.with_suffix(".npz"),
        frame_index=frames,
        pred_action=pred,
        gt_action=gt,
        action_names=np.asarray(action_names, dtype=str),
    )
    with episode_prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if not bool(args.no_plot):
        _plot_actions(
            output_path=episode_prefix.with_suffix(".png"),
            gt=gt,
            pred=pred,
            frame_indices=frames,
            action_names=action_names,
        )
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    vlm_model_path = Path(args.vlm_model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not vlm_model_path.exists():
        raise FileNotFoundError(f"VLM model path not found: {vlm_model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    modality_config = load_checkpoint_modality_config(model_path)
    device = _resolve_device(str(args.device))

    logging.info("model_path=%s", model_path)
    logging.info("vlm_model_path=%s", vlm_model_path)
    logging.info("dataset_dir=%s", dataset_dir)
    logging.info("video_keys=%s", list(modality_config["video"].modality_keys))
    logging.info("device=%s", device)
    if device == "cpu":
        logging.warning(
            "CUDA is not visible; loading the 3B base model on CPU will be very slow and memory heavy."
        )
    logging.info("Loading official-aligned checkpoint processor config and statistics.")
    policy = CheckpointRokaePolicy(
        model_path=model_path,
        device=device,
        strict=bool(args.strict_policy),
        vlm_model_path=vlm_model_path,
    )

    all_metrics = []
    for episode_index in args.episode_indices:
        all_metrics.append(
            _run_episode(
                policy=policy,
                dataset_dir=dataset_dir,
                episode_index=int(episode_index),
                modality_config=modality_config,
                output_dir=output_dir,
                args=args,
            )
        )

    summary = {
        "model_path": str(model_path),
        "vlm_model_path": str(vlm_model_path),
        "dataset_dir": str(dataset_dir),
        "video_keys": list(modality_config["video"].modality_keys),
        "episodes": all_metrics,
        "average_mae": float(np.mean([m["mae"] for m in all_metrics])) if all_metrics else None,
        "average_mse": float(np.mean([m["mse"] for m in all_metrics])) if all_metrics else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("summary saved to %s", output_dir / "summary.json")
    logging.info("average_mae=%s average_mse=%s", summary["average_mae"], summary["average_mse"])


if __name__ == "__main__":
    main()
