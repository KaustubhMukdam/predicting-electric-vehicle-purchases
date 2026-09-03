"""Conftest: shared pytest fixtures.

This file is auto-loaded by pytest. Fixtures defined here are available
to every test file without explicit import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable as a top-level package from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence LightGBM's "categorical_feature in Dataset has different
# cardinality" warning during tests — we deliberately train on small
# slices and the warning is noise.
pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning",
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Path to the data/ directory. Tests that need the CSVs use this."""
    return project_root / "data"
