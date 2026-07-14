"""Compatibility helpers for artifacts written before the package migration."""

from __future__ import annotations

import importlib
import pickle
import sys
from contextlib import contextmanager
from typing import Any, BinaryIO, Iterator

_LEGACY_PREFIX = "gr00t.data.rlt"
_MODULES = (
    "collate",
    "episode_schema",
    "episode_transition_builder",
    "networks",
    "replay_buffer",
    "replay_schema",
    "train",
    "trainer",
    "warmup",
)


@contextmanager
def legacy_module_aliases() -> Iterator[None]:
    """Temporarily map old pickle module paths to ``groot_rlt`` modules."""

    aliases = {_LEGACY_PREFIX: importlib.import_module("groot_rlt")}
    aliases.update(
        {
            f"{_LEGACY_PREFIX}.{name}": importlib.import_module(f"groot_rlt.{name}")
            for name in _MODULES
        }
    )
    inserted: list[str] = []
    for old_name, module in aliases.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = module
            inserted.append(old_name)
    try:
        yield
    finally:
        for old_name in reversed(inserted):
            sys.modules.pop(old_name, None)


def load_pickle_with_legacy_aliases(stream: BinaryIO) -> Any:
    """Load a pickle containing either new or pre-migration class paths."""

    with legacy_module_aliases():
        return pickle.load(stream)
