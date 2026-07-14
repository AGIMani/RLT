#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Precompute RL tokens and VLA reference action chunks for actor/critic training.

The script writes two independent cache folders:

1. RL token cache: one compact token per selected LeRobot timestep, produced by
   the trained RLTokenEncoder. It can reuse an existing VL embedding cache.
2. VLA reference action cache: one GR00T action chunk per selected timestep,
   produced by the frozen/finetuned VLA policy.

Examples:

    # Fast path for the existing trimmed dataset and VL embedding cache.
    ./.venv/bin/python groot-rlt-precompute \
        --dataset-dir outputs/IsaacLab/trimmed \
        --rl-token-source cache \
        --skip-vla-action

    # Exact path for a dataset without a matching embedding cache, e.g. interrupt.
    ./.venv/bin/python groot-rlt-precompute \
        --dataset-dir outputs/IsaacLab/interrupt \
        --rl-token-source dataset \
        --model-path checkpoints/rokae_xmate3_l10_full_orientation_overfit/checkpoint-48000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.paths import VL_EMBEDDING_CACHE_DIR

REPO_ROOT = ensure_groot_repo()
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from groot_rlt.integration.checkpoint_policy_utils import (  # noqa: E402
    load_checkpoint_modality_config,
)
from groot_rlt.integration.defaults import (  # noqa: E402
    L10_BASE_MODEL_PATH,
    L10_INSTRUCTION,
    L10_MODEL_DIR,
    L10_PREPARED_DATASET_DIR,
    L10_VLM_MODEL_PATH,
)
from groot_rlt.integration.lerobot_policy_helpers import (  # noqa: E402
    CheckpointRokaePolicy,
    _action_names,
    _all_episode_indices,
    _build_observation,
    _extract_groups,
    _feature_dim,
    _get_instruction,
    _load_episode_parquet,
    _policy_action_names,
    _resolve_device,
    _rtc_options,
    _to_matrix,
    _unbatch_action_dict,
)
from groot_rlt.representation.visualize_rl_token_umap import (  # noqa: E402
    extract_from_cache,
    extract_from_dataset,
    load_rl_token_encoder,
)
from groot_rlt.representation.visualize_rl_token_umap import (  # noqa: E402
    latest_checkpoint as latest_rl_token_checkpoint,
)

DEFAULT_DATASET_DIR = L10_PREPARED_DATASET_DIR
DEFAULT_RL_CHECKPOINT_DIR = (
    REPO_ROOT / "outputs" / "IsaacLab" / "vl_embedding_autoencoder_pi_cached"
)
DEFAULT_EMBEDDING_CACHE_DIR = VL_EMBEDDING_CACHE_DIR
DEFAULT_RL_TOKEN_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "precomputed_rl_tokens"
DEFAULT_VLA_ACTION_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "IsaacLab" / "precomputed_vla_reference_actions"
)


@dataclass(frozen=True)
class EpisodeSelection:
    episode_indices: list[int]
    max_steps_per_episode: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groot-repo-path",
        type=str,
        default=str(REPO_ROOT),
        help="Isaac-GR00T checkout used for models, examples, data, and default paths.",
    )
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--embodiment-tag", type=str, default=EmbodimentTag.NEW_EMBODIMENT.value)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--episode-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)

    parser.add_argument("--skip-rl-token", action="store_true")
    parser.add_argument("--rl-token-source", choices=("cache", "dataset"), default="cache")
    parser.add_argument("--rl-token-output-dir", type=str, default=str(DEFAULT_RL_TOKEN_OUTPUT_DIR))
    parser.add_argument("--rl-token-checkpoint", type=str, default=None)
    parser.add_argument(
        "--rl-token-checkpoint-dir", type=str, default=str(DEFAULT_RL_CHECKPOINT_DIR)
    )
    parser.add_argument("--embedding-cache-dir", type=str, default=str(DEFAULT_EMBEDDING_CACHE_DIR))
    parser.add_argument("--rl-batch-size", type=int, default=8)
    parser.add_argument("--max-rl-samples", type=int, default=None)
    parser.add_argument("--cache-seed", type=int, default=None)
    parser.add_argument("--cache-episode-sampling-rate", type=float, default=None)
    parser.add_argument(
        "--cache-progress-metadata", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-cache-dataset-mismatch",
        action="store_true",
        help=(
            "Allow --rl-token-source cache when the cache manifest dataset_dir differs from "
            "--dataset-dir. This is usually unsafe because tokens are tied to source frames."
        ),
    )
    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default=None)
    parser.add_argument(
        "--token-sampling",
        choices=("head", "tail", "uniform", "random"),
        default=None,
    )
    parser.add_argument("--max-vl-tokens", type=int, default=None)
    parser.add_argument("--autoencoder-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--base-model-path", type=str, default=str(L10_BASE_MODEL_PATH))
    parser.add_argument("--vlm-model-path", type=str, default=str(L10_VLM_MODEL_PATH))
    parser.add_argument("--load-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use-flash-attention", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-vla-action", action="store_true")
    parser.add_argument(
        "--vla-action-output-dir", type=str, default=str(DEFAULT_VLA_ACTION_OUTPUT_DIR)
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="GR00T model checkpoint. Defaults to latest checkpoint-* under --model-dir.",
    )
    parser.add_argument("--model-dir", type=str, default=str(L10_MODEL_DIR))
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help=(
            "Processor directory. Defaults to <model-path>/processor, then "
            "<model-path>/../processor, then --model-dir/processor."
        ),
    )
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "torchcodec", "decord", "opencv"),
        default="ffmpeg",
    )
    parser.add_argument("--action-batch-size", type=int, default=1)
    parser.add_argument("--action-frame-stride", type=int, default=None)
    parser.add_argument("--strict-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-frozen-steps", type=int, default=2)
    parser.add_argument("--rtc-ramp-rate", type=float, default=6.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_model_checkpoint(model_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is None:
            continue
        if (child / "model.safetensors.index.json").exists() or (
            child / "model.safetensors"
        ).exists():
            candidates.append((step, child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-<step> model found under {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_processor_path(model_path: Path, model_dir: Path, override: str | None) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Processor path not found: {path}")
        return path
    candidates = [
        model_path / "processor",
        model_path.parent / "processor",
        model_dir / "processor",
        model_path,
    ]
    for candidate in candidates:
        if (candidate / "processor_config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find processor_config.json. Pass --processor-path explicitly."
    )


def selected_episodes(dataset_dir: Path, args: argparse.Namespace) -> EpisodeSelection:
    episodes = (
        [int(ep) for ep in args.episode_indices]
        if args.episode_indices is not None and len(args.episode_indices) > 0
        else _all_episode_indices(dataset_dir)
    )
    if args.max_episodes is not None:
        episodes = episodes[: max(0, int(args.max_episodes))]
    return EpisodeSelection(
        episode_indices=episodes,
        max_steps_per_episode=args.max_steps_per_episode,
    )


def scalar_metadata_from_row(row: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"observation.state", "action"}:
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (list, tuple, dict)):
            continue
        metadata[key] = to_jsonable(value)
    return metadata


class ParquetRowLookup:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.tables: dict[int, pd.DataFrame] = {}

    def get(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        if episode_index not in self.tables:
            self.tables[episode_index] = _load_episode_parquet(self.dataset_dir, episode_index)
        table = self.tables[episode_index]
        if "frame_index" in table.columns:
            matches = table.index[table["frame_index"].astype(int) == int(frame_index)]
            if len(matches) > 0:
                return scalar_metadata_from_row(table.iloc[int(matches[0])])
        if 0 <= int(frame_index) < len(table):
            return scalar_metadata_from_row(table.iloc[int(frame_index)])
        return {}


def numeric_column(rows: list[dict[str, Any]], key: str) -> np.ndarray | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            values.append(np.nan)
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return np.asarray(values, dtype=np.float32)


def save_episode_npz(
    *,
    output_path: Path,
    arrays: dict[str, np.ndarray],
    metadata_rows: list[dict[str, Any]],
) -> None:
    for key in sorted({key for row in metadata_rows for key in row}):
        if key in arrays:
            continue
        values = numeric_column(metadata_rows, key)
        if values is not None:
            arrays[key.replace(".", "__")] = values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def filter_and_sort_token_rows(
    *,
    tokens: np.ndarray,
    metadata: list[dict[str, Any]],
    dataset_dir: Path,
    selection: EpisodeSelection,
    frame_stride: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = ParquetRowLookup(dataset_dir)
    selected_episode_set = set(selection.episode_indices)
    enriched: list[tuple[np.ndarray, dict[str, Any]]] = []
    per_episode_counts: dict[int, int] = {}

    for sample_index, (token, row) in enumerate(zip(tokens, metadata)):
        try:
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if episode_index not in selected_episode_set:
            continue
        if frame_index % max(1, int(frame_stride)) != 0:
            continue
        count = per_episode_counts.get(episode_index, 0)
        if selection.max_steps_per_episode is not None and count >= selection.max_steps_per_episode:
            continue
        per_episode_counts[episode_index] = count + 1

        merged = {
            **lookup.get(episode_index, frame_index),
            **row,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "rl_token_source_index": int(sample_index),
        }
        enriched.append((token, merged))

    enriched.sort(key=lambda item: (int(item[1]["episode_index"]), int(item[1]["frame_index"])))
    if not enriched:
        raise RuntimeError("No RL token rows matched the requested episode/frame selection.")
    sorted_tokens = np.stack([item[0] for item in enriched], axis=0).astype(np.float32)
    sorted_metadata = [item[1] for item in enriched]
    return sorted_tokens, sorted_metadata


def precompute_rl_tokens(
    args: argparse.Namespace, dataset_dir: Path, selection: EpisodeSelection
) -> None:
    output_dir = Path(args.rl_token_output_dir).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    device = torch.device(_resolve_device(str(args.device)))
    checkpoint = (
        Path(args.rl_token_checkpoint).expanduser().resolve()
        if args.rl_token_checkpoint is not None
        else latest_rl_token_checkpoint(Path(args.rl_token_checkpoint_dir).expanduser().resolve())
    )
    autoencoder, ckpt_args, checkpoint_step_value, checkpoint_loss = load_rl_token_encoder(
        checkpoint, device
    )
    if args.rl_token_source == "cache":
        cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()
        cache_manifest = read_json(cache_dir / "manifest.json")
        cache_dataset = Path(str(cache_manifest.get("dataset_dir", ""))).expanduser()
        if not cache_dataset.is_absolute():
            cache_dataset = (REPO_ROOT / cache_dataset).resolve()
        else:
            cache_dataset = cache_dataset.resolve()
        if cache_dataset != dataset_dir.resolve() and not bool(args.allow_cache_dataset_mismatch):
            raise RuntimeError(
                "--rl-token-source cache requires the cache dataset to match --dataset-dir.\n"
                f"cache manifest dataset_dir: {cache_dataset}\n"
                f"requested dataset_dir:       {dataset_dir.resolve()}\n"
                "Use --rl-token-source dataset for this dataset, or pass "
                "--allow-cache-dataset-mismatch only if you know the frame order is identical."
            )

    rl_args = argparse.Namespace(**vars(args))
    rl_args.source = args.rl_token_source
    rl_args.checkpoint = str(checkpoint)
    rl_args.checkpoint_dir = str(args.rl_token_checkpoint_dir)
    rl_args.output_dir = str(output_dir)
    rl_args.dataset_dir = str(dataset_dir)
    rl_args.batch_size = int(args.rl_batch_size)
    rl_args.max_samples = args.max_rl_samples
    rl_args.episode_indices = (
        selection.episode_indices if args.rl_token_source == "dataset" else None
    )
    rl_args.max_episodes = args.max_episodes
    rl_args.max_frames_per_episode = args.max_steps_per_episode
    rl_args.keyframe_column = None
    rl_args.keyframe_values = None
    rl_args.video_backend = args.video_backend
    rl_args.cache_progress_metadata = bool(args.cache_progress_metadata)
    rl_args.cache_seed = (
        args.cache_seed if args.cache_seed is not None else int(ckpt_args.get("seed", args.seed))
    )
    rl_args.cache_episode_sampling_rate = args.cache_episode_sampling_rate
    rl_args.token_scope = args.token_scope or ckpt_args.get("token_scope", "all")
    rl_args.token_sampling = args.token_sampling or ckpt_args.get("token_sampling", "uniform")
    rl_args.max_vl_tokens = int(args.max_vl_tokens or ckpt_args.get("max_vl_tokens", 512))
    rl_args.autoencoder_bf16 = (
        bool(ckpt_args.get("autoencoder_bf16", device.type == "cuda"))
        if args.autoencoder_bf16 is None
        else bool(args.autoencoder_bf16)
    )
    rl_args.autoencoder_bf16 = bool(rl_args.autoencoder_bf16 and device.type == "cuda")

    tic = time.time()
    if args.rl_token_source == "cache":
        tokens, metadata, extra = extract_from_cache(
            autoencoder=autoencoder,
            args=rl_args,
            device=device,
            autoencoder_bf16=bool(rl_args.autoencoder_bf16),
        )
    else:
        tokens, metadata, extra = extract_from_dataset(
            autoencoder=autoencoder,
            args=rl_args,
            device=device,
            autoencoder_bf16=bool(rl_args.autoencoder_bf16),
        )

    tokens, metadata = filter_and_sort_token_rows(
        tokens=tokens,
        metadata=metadata,
        dataset_dir=dataset_dir,
        selection=selection,
        frame_stride=int(args.frame_stride),
    )
    np.save(output_dir / "rl_tokens.npy", tokens)
    write_jsonl(output_dir / "metadata.jsonl", metadata)

    rows_by_episode: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, row in enumerate(metadata):
        rows_by_episode.setdefault(int(row["episode_index"]), []).append((row_index, row))
    for episode_index, items in sorted(rows_by_episode.items()):
        indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
        rows = [row for _, row in items]
        save_episode_npz(
            output_path=output_dir / f"episode_{episode_index:06d}.npz",
            arrays={
                "rl_token": tokens[indices],
                "frame_index": np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
                "episode_index": np.full(len(rows), episode_index, dtype=np.int64),
                "rl_token_source_index": np.asarray(
                    [row["rl_token_source_index"] for row in rows], dtype=np.int64
                ),
            },
            metadata_rows=rows,
        )

    manifest = {
        "kind": "rl_token_cache",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "source": args.rl_token_source,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step_value,
        "checkpoint_last_loss": checkpoint_loss,
        "embedding_cache_dir": str(Path(args.embedding_cache_dir).expanduser().resolve()),
        "num_samples": int(tokens.shape[0]),
        "token_dim": int(tokens.shape[1]),
        "episode_indices": selection.episode_indices,
        "frame_stride": int(args.frame_stride),
        "max_steps_per_episode": selection.max_steps_per_episode,
        "elapsed_seconds": float(time.time() - tic),
        "args": vars(args),
        **extra,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[rl-token] wrote {tokens.shape} -> {output_dir}")


class PrecomputeCheckpointRokaePolicy(CheckpointRokaePolicy):
    """GR00T policy wrapper that keeps the precompute script's explicit processor path."""

    def __init__(
        self,
        *,
        model_path: Path,
        processor_path: Path,
        device: str,
        strict: bool,
        vlm_model_path: Path,
        embodiment_tag: EmbodimentTag | str,
    ) -> None:
        import gr00t.model  # noqa: F401
        from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
        from gr00t.policy.policy import BasePolicy
        from transformers import AutoConfig, AutoModel, AutoProcessor

        BasePolicy.__init__(self, strict=strict)
        self.embodiment_tag = (
            embodiment_tag
            if isinstance(embodiment_tag, EmbodimentTag)
            else EmbodimentTag.resolve(embodiment_tag)
        )
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        model_config.model_name = str(vlm_model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=model_config,
            transformers_loading_kwargs={"local_files_only": True},
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            transformers_loading_kwargs={"local_files_only": True},
            model_name=str(vlm_model_path),
        )
        self.processor.eval()

        all_modality_configs = self.processor.get_modality_configs()
        self.modality_configs = {
            key: value
            for key, value in all_modality_configs[self.embodiment_tag.value].items()
            if key != "rl_info"
        }
        self.collate_fn = self.processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self._rec_to_dtype = _rec_to_dtype
        self._unbatch_observation = Gr00tPolicy._unbatch_observation.__get__(self)
        self.check_observation = Gr00tPolicy.check_observation.__get__(self)
        self.check_action = Gr00tPolicy.check_action.__get__(self)
        self.get_modality_config = Gr00tPolicy.get_modality_config.__get__(self)
        self.reset = Gr00tPolicy.reset.__get__(self)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        previous_action: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        from gr00t.data.types import MessageType, VLAStepData

        if self.strict:
            self.check_observation(observation)
        unbatched_observations = self._unbatch_observation(observation)
        if previous_action is not None and len(unbatched_observations) != 1:
            raise ValueError("RTC previous_action requires batch size 1.")

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
            model_pred = self.model.get_action(**collated_inputs, options=options)
        normalized_action = model_pred["action_pred"].float()

        batched_states = {}
        for key in self.modality_configs["state"].modality_keys:
            batched_states[key] = np.stack([state[key] for state in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        action = {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
        if self.strict:
            self.check_action(action)
        return action, {}


def merge_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {"video": {}, "state": {}, "language": {}}
    for key in observations[0]["video"]:
        merged["video"][key] = np.concatenate([obs["video"][key] for obs in observations], axis=0)
    for key in observations[0]["state"]:
        merged["state"][key] = np.concatenate([obs["state"][key] for obs in observations], axis=0)
    for key in observations[0]["language"]:
        values = []
        for obs in observations:
            values.extend(obs["language"][key])
        merged["language"][key] = values
    return merged


def concat_batched_action_dict(action: dict[str, np.ndarray], action_keys: list[str]) -> np.ndarray:
    chunks = []
    batch_size = None
    for key in action_keys:
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 2:
            value = value[None, ...]
        if value.ndim != 3:
            raise RuntimeError(f"Unexpected action[{key}] shape {value.shape}")
        batch_size = value.shape[0] if batch_size is None else batch_size
        if value.shape[0] != batch_size:
            raise RuntimeError(f"Action batch mismatch for {key}: {value.shape[0]} != {batch_size}")
        chunks.append(value)
    return np.concatenate(chunks, axis=-1).astype(np.float32)


def ground_truth_action_chunk(
    *,
    step: int,
    action_delta_indices: np.ndarray,
    actions_by_key: dict[str, np.ndarray],
    action_keys: list[str],
) -> np.ndarray:
    length = len(next(iter(actions_by_key.values())))
    indices = np.clip(step + action_delta_indices, 0, length - 1)
    return np.concatenate([actions_by_key[key][indices] for key in action_keys], axis=-1).astype(
        np.float32
    )


def episode_steps(length: int, *, stride: int, max_steps: int | None) -> list[int]:
    steps = list(range(0, int(length), max(1, int(stride))))
    if max_steps is not None:
        steps = steps[: max(0, int(max_steps))]
    return steps


def precompute_vla_actions(
    args: argparse.Namespace, dataset_dir: Path, selection: EpisodeSelection
) -> None:
    output_dir = Path(args.vla_action_output_dir).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path is not None
        else latest_model_checkpoint(model_dir)
    )
    processor_path = resolve_processor_path(model_path, model_dir, args.processor_path)
    vlm_model_path = Path(args.vlm_model_path).expanduser().resolve()
    device = _resolve_device(str(args.device))

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    modality_config = load_checkpoint_modality_config(model_path, embodiment_tag)
    policy = PrecomputeCheckpointRokaePolicy(
        model_path=model_path,
        processor_path=processor_path,
        device=device,
        strict=bool(args.strict_policy),
        vlm_model_path=vlm_model_path,
        embodiment_tag=embodiment_tag,
    )

    action_stride = int(args.action_frame_stride or args.frame_stride)
    action_batch_size = 1 if bool(args.rtc) else max(1, int(args.action_batch_size))
    action_keys = list(modality_config["action"].modality_keys)
    action_delta_indices = np.asarray(modality_config["action"].delta_indices, dtype=np.int64)
    action_names: list[str] | None = None
    summary_rows = []
    tic_all = time.time()

    for episode_index in selection.episode_indices:
        episode_tic = time.time()
        modality_meta = read_json(dataset_dir / "meta" / "modality.json")
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
        for key in action_keys:
            sl = modality_meta["action"][key]
            source = states if sl.get("original_key") == "observation.state" else actions
            actions_by_key[key] = source[:, int(sl["start"]) : int(sl["end"])].astype(
                np.float32, copy=False
            )
        gt_action_full = np.concatenate([actions_by_key[key] for key in action_keys], axis=-1)
        if action_names is None:
            action_names = _policy_action_names(
                _action_names(dataset_dir),
                int(gt_action_full.shape[1]),
            )
        instruction = _get_instruction(
            dataset_dir, df, override=args.instruction or L10_INSTRUCTION
        )
        steps = episode_steps(
            len(df),
            stride=action_stride,
            max_steps=selection.max_steps_per_episode,
        )

        reference_chunks: list[np.ndarray] = []
        gt_chunks: list[np.ndarray] = []
        inference_seconds: list[float] = []
        metadata_rows: list[dict[str, Any]] = []
        previous_action: dict[str, np.ndarray] | None = None

        for start in range(0, len(steps), action_batch_size):
            batch_steps = steps[start : start + action_batch_size]
            observations = [
                _build_observation(
                    dataset_dir,
                    episode_index=episode_index,
                    step=step,
                    states_by_key=states_by_key,
                    modality_config=modality_config,
                    modality_meta=modality_meta,
                    instruction=instruction,
                    video_backend=str(args.video_backend),
                )
                for step in batch_steps
            ]
            observation = merge_observations(observations)
            options = (
                _rtc_options(
                    args=args,
                    action_horizon=len(action_delta_indices),
                    previous_action=previous_action,
                )
                if bool(args.rtc)
                else None
            )
            tic = time.time()
            pred_action, _ = policy.get_action(
                observation,
                options=options,
                previous_action=previous_action if options is not None else None,
            )
            elapsed = time.time() - tic
            pred_chunk = concat_batched_action_dict(pred_action, action_keys)
            if bool(args.rtc):
                previous_action = _unbatch_action_dict(pred_action)
            for batch_offset, step in enumerate(batch_steps):
                reference_chunks.append(pred_chunk[batch_offset])
                gt_chunks.append(
                    ground_truth_action_chunk(
                        step=step,
                        action_delta_indices=action_delta_indices,
                        actions_by_key=actions_by_key,
                        action_keys=action_keys,
                    )
                )
                inference_seconds.append(float(elapsed / max(1, len(batch_steps))))
                row_meta = scalar_metadata_from_row(df.iloc[int(step)])
                row_meta.update(
                    {
                        "episode_index": int(episode_index),
                        "frame_index": int(df["frame_index"].iloc[step])
                        if "frame_index" in df.columns
                        else int(step),
                        "step_index": int(step),
                        "instruction": instruction,
                    }
                )
                metadata_rows.append(row_meta)
            print(
                f"[vla-action] episode={episode_index:06d} "
                f"{min(start + action_batch_size, len(steps))}/{len(steps)}",
                flush=True,
            )

        reference = np.stack(reference_chunks, axis=0).astype(np.float32)
        gt = np.stack(gt_chunks, axis=0).astype(np.float32)
        frame_index = np.asarray([row["frame_index"] for row in metadata_rows], dtype=np.int64)
        step_index = np.asarray([row["step_index"] for row in metadata_rows], dtype=np.int64)
        inference = np.asarray(inference_seconds, dtype=np.float32)

        arrays = {
            "frame_index": frame_index,
            "step_index": step_index,
            "reference_action_chunk": reference,
            "ground_truth_action_chunk": gt,
            "ground_truth_action": gt_action_full[step_index],
            "inference_seconds": inference,
            "action_names": np.asarray(action_names, dtype=str),
            "action_keys": np.asarray(action_keys, dtype=str),
        }
        for key in action_keys:
            sl = modality_meta["action"][key]
            start_idx = sum(
                actions_by_key[k].shape[1] for k in action_keys[: action_keys.index(key)]
            )
            end_idx = start_idx + actions_by_key[key].shape[1]
            arrays[f"reference_action_{key}"] = reference[:, :, start_idx:end_idx]
            arrays[f"ground_truth_action_{key}"] = gt[:, :, start_idx:end_idx]
            arrays[f"action_slice_{key}"] = np.asarray([start_idx, end_idx], dtype=np.int64)

        save_episode_npz(
            output_path=output_dir / f"episode_{episode_index:06d}.npz",
            arrays=arrays,
            metadata_rows=metadata_rows,
        )
        write_jsonl(output_dir / f"episode_{episode_index:06d}.metadata.jsonl", metadata_rows)
        episode_summary = {
            "episode_index": int(episode_index),
            "num_steps": int(reference.shape[0]),
            "action_horizon": int(reference.shape[1]),
            "action_dim": int(reference.shape[2]),
            "avg_inference_seconds": float(np.mean(inference)) if len(inference) else 0.0,
            "max_inference_seconds": float(np.max(inference)) if len(inference) else 0.0,
            "elapsed_seconds": float(time.time() - episode_tic),
        }
        summary_rows.append(episode_summary)
        print(
            f"[vla-action] episode={episode_index:06d} wrote {reference.shape} "
            f"avg_inference={episode_summary['avg_inference_seconds']:.3f}s"
        )

    manifest = {
        "kind": "vla_reference_action_cache",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "model_step": checkpoint_step(model_path),
        "model_dir": str(model_dir),
        "processor_path": str(processor_path),
        "vlm_model_path": str(vlm_model_path),
        "episode_indices": selection.episode_indices,
        "frame_stride": int(action_stride),
        "max_steps_per_episode": selection.max_steps_per_episode,
        "action_batch_size": int(action_batch_size),
        "rtc": bool(args.rtc),
        "action_keys": action_keys,
        "action_names": action_names or [],
        "episodes": summary_rows,
        "elapsed_seconds": float(time.time() - tic_all),
        "args": vars(args),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[vla-action] wrote {len(summary_rows)} episode files -> {output_dir}")


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    selection = selected_episodes(dataset_dir, args)
    if not selection.episode_indices:
        raise RuntimeError("No episodes selected.")

    print(f"dataset_dir={dataset_dir}")
    print(f"episodes={selection.episode_indices}")
    if not args.skip_rl_token:
        precompute_rl_tokens(args, dataset_dir, selection)
    if not args.skip_vla_action:
        precompute_vla_actions(args, dataset_dir, selection)
    print("[done] precompute finished")


if __name__ == "__main__":
    main()
