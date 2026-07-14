from __future__ import annotations

import importlib
import sys

from groot_rlt.legacy import legacy_module_aliases


def test_legacy_module_aliases_resolve_old_pickle_paths() -> None:
    old_name = "gr00t.data.rlt.replay_schema"
    previous = sys.modules.pop(old_name, None)
    try:
        with legacy_module_aliases():
            old_module = importlib.import_module(old_name)
            new_module = importlib.import_module("groot_rlt.replay_schema")
            assert old_module is new_module
            assert old_module.RLTTransition is new_module.RLTTransition
        assert old_name not in sys.modules
    finally:
        if previous is not None:
            sys.modules[old_name] = previous
