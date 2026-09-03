"""Smoke tests: verify all modules are importable.

This is the test that Phase 0 must satisfy on its own — before any
real logic exists. If a module fails to import, this test goes red
and points at the exact file.
"""

from __future__ import annotations

import importlib


def test_src_package_imports():
    importlib.import_module("src")


def test_config_imports():
    mod = importlib.import_module("src.config")
    assert hasattr(mod, "TARGET_COL")
    assert hasattr(mod, "RANDOM_SEED")
    assert mod.TARGET_COL == "Will_Buy_EV"
    assert mod.RANDOM_SEED == 42


def test_utils_imports():
    mod = importlib.import_module("src.utils")
    assert hasattr(mod, "seed_everything")
    assert callable(mod.seed_everything)
