"""Filesystem locations owned by the Groot-RLT project."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the Groot-RLT checkout root used for generated artifacts.

    ``GROOT_RLT_PROJECT_ROOT`` is authoritative in containers and installed
    environments.  Editable/source checkouts are discovered by walking up to
    the nearest directory containing both ``.git`` and ``groot_rlt``.
    """

    configured_root = os.environ.get("GROOT_RLT_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "groot_rlt").is_dir():
            return candidate

    return (Path.home() / "groot-rlt").resolve()


PROJECT_ROOT = project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = OUTPUT_ROOT / "cache"
VL_EMBEDDING_CACHE_DIR = CACHE_ROOT / "vl_embeddings"
