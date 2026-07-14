"""Locate an Isaac-GR00T checkout for optional GR00T integration tools."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ENV_VAR = "GROOT_REPO_PATH"


def _cli_value(argv: list[str]) -> str | None:
    for index, item in enumerate(argv):
        if item == "--groot-repo-path" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--groot-repo-path="):
            return item.split("=", 1)[1]
    return None


def _is_checkout(path: Path) -> bool:
    return (path / "gr00t" / "__init__.py").is_file() and (path / "pyproject.toml").is_file()


def _candidate_paths(explicit: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (explicit, _cli_value(sys.argv), os.environ.get(ENV_VAR)):
        if value:
            candidates.append(Path(value).expanduser())

    cwd = Path.cwd()
    candidates.extend((cwd, cwd / "Isaac-GR00T", cwd.parent / "Isaac-GR00T"))
    for parent in Path(__file__).resolve().parents:
        candidates.extend((parent / "Isaac-GR00T", parent.parent / "Isaac-GR00T"))

    spec = importlib.util.find_spec("gr00t")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent.parent)
    return candidates


def ensure_groot_repo(explicit: str | Path | None = None) -> Path:
    """Return a GR00T checkout and prepend it to ``sys.path``.

    Resolution order starts with ``--groot-repo-path``/``GROOT_REPO_PATH`` and
    then checks common sibling layouts. The helper intentionally does not make
    GR00T a package dependency because GR00T and openpi use incompatible Python
    and Transformers versions.
    """

    requested = explicit or _cli_value(sys.argv) or os.environ.get(ENV_VAR)
    if requested:
        requested_path = Path(requested).expanduser().resolve()
        if not _is_checkout(requested_path):
            raise RuntimeError(
                f"Requested Isaac-GR00T checkout is invalid: {requested_path}. "
                "Expected gr00t/__init__.py and pyproject.toml."
            )
        path_text = str(requested_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        return requested_path

    seen: set[Path] = set()
    for candidate in _candidate_paths(explicit):
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not _is_checkout(path):
            continue
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        return path
    checked = "\n  - ".join(str(path) for path in seen)
    raise RuntimeError(
        "Unable to locate an Isaac-GR00T checkout. Pass --groot-repo-path, set "
        f"{ENV_VAR}, or install/run from the GR00T Python 3.10 environment. Checked:\n  - "
        f"{checked}"
    )
