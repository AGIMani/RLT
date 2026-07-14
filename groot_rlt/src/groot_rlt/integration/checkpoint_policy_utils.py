#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading official-aligned GR00T finetune checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groot_rlt.groot_repo import ensure_groot_repo

ensure_groot_repo()

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.utils import parse_modality_configs  # noqa: E402


def resolve_processor_path(model_path: Path) -> Path:
    """Return the processor directory saved with a finetune checkpoint."""

    model_path = Path(model_path).expanduser().resolve()
    for candidate in (model_path, model_path / "processor", model_path.parent / "processor"):
        if (candidate / "processor_config.json").exists():
            return candidate
    raise FileNotFoundError(
        f"No processor_config.json found for checkpoint {model_path}. "
        "Official-aligned finetune checkpoints must include processor files."
    )


def load_checkpoint_modality_config(
    model_path: Path,
    embodiment_tag: EmbodimentTag | str = EmbodimentTag.NEW_EMBODIMENT,
) -> dict[str, Any]:
    """Load the embodiment modality config serialized inside the checkpoint processor."""

    if isinstance(embodiment_tag, str):
        embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
    processor_path = resolve_processor_path(model_path)
    config = json.loads((processor_path / "processor_config.json").read_text(encoding="utf-8"))
    modality_configs = config.get("processor_kwargs", {}).get("modality_configs", {})
    if embodiment_tag.value not in modality_configs:
        available = sorted(modality_configs.keys())
        raise ValueError(
            f"Checkpoint processor at {processor_path} does not contain embodiment "
            f"{embodiment_tag.value!r}. Available: {available}"
        )
    return parse_modality_configs(modality_configs)[embodiment_tag.value]


def strip_decode_only_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove legacy decode-only options before calling the model action head."""

    if not options:
        return None
    cleaned = {key: value for key, value in options.items() if key != "reference_action"}
    return cleaned or None
