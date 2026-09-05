"""Tests for the Jupyter notebooks under notebooks/.

We treat the notebooks as deliverable artifacts (they're the
Kaggle-ready runner + the EDA reproduction), so we test them like
any other module: they must exist, parse as valid JSON, and have
the right structural shape.

These tests do NOT execute the notebooks. Execution happens
either locally (the user runs `jupyter nbconvert --execute`) or on
Kaggle (the user uploads and runs the notebook). What we test
here is that the notebooks are *runnable* in principle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _load_notebook(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# EDA.ipynb
# ---------------------------------------------------------------------------
def test_eda_notebook_exists():
    p = NOTEBOOKS_DIR / "EDA.ipynb"
    assert p.exists(), f"Missing: {p}"


def test_eda_notebook_is_valid_json():
    p = NOTEBOOKS_DIR / "EDA.ipynb"
    nb = _load_notebook(p)
    assert "cells" in nb
    assert "metadata" in nb
    assert nb.get("nbformat") == 4


def test_eda_notebook_has_at_least_one_code_cell():
    p = NOTEBOOKS_DIR / "EDA.ipynb"
    nb = _load_notebook(p)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 1, "EDA.ipynb has no code cells"


def test_eda_notebook_uses_the_real_data_paths():
    """The EDA notebook should reference `data/train.csv` (or the Kaggle
    equivalent) — i.e., it must actually load the data, not a stub."""
    p = NOTEBOOKS_DIR / "EDA.ipynb"
    nb = _load_notebook(p)
    all_source = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "train.csv" in all_source
    assert "read_csv" in all_source


def test_eda_notebook_imports_required_libraries():
    nb = _load_notebook(NOTEBOOKS_DIR / "EDA.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    for lib in ("pandas", "numpy"):
        assert f"import {lib}" in src or f"import {lib}\n" in src, (
            f"EDA.ipynb is missing `import {lib}`"
        )


# ---------------------------------------------------------------------------
# train.ipynb
# ---------------------------------------------------------------------------
def test_train_notebook_exists():
    p = NOTEBOOKS_DIR / "train.ipynb"
    assert p.exists(), f"Missing: {p}"


def test_train_notebook_is_valid_json():
    p = NOTEBOOKS_DIR / "train.ipynb"
    nb = _load_notebook(p)
    assert "cells" in nb
    assert nb.get("nbformat") == 4


def test_train_notebook_calls_all_required_pipeline_functions():
    """The training notebook must import and call every module in src/
    that the pipeline depends on. If any module is missing from the
    notebook, the E2E contract is broken."""
    nb = _load_notebook(NOTEBOOKS_DIR / "train.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    required_calls = [
        "load_data",         # src.data
        "build_features",    # src.features
        "make_folds",        # src.cv
        "train_lgbm",        # src.train_lgbm
        "make_submission",   # src.predict
    ]
    for fn in required_calls:
        assert fn in src, f"train.ipynb does not call {fn}()"


def test_train_notebook_writes_submission_to_kaggle_working():
    """The notebook must write its output to /kaggle/working/submission.csv
    (or a clearly-equivalent path) so Kaggle picks it up as a notebook output."""
    nb = _load_notebook(NOTEBOOKS_DIR / "train.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "/kaggle/working/submission.csv" in src


def test_train_notebook_zips_mlruns_for_download():
    """The notebook must zip /kaggle/working/mlruns so the run history
    can be downloaded as a notebook output."""
    nb = _load_notebook(NOTEBOOKS_DIR / "train.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "mlruns" in src
    assert ".zip" in src


def test_train_notebook_uses_full_train_data():
    """The notebook should NOT slice train (unlike the E2E test). For
    the real Kaggle run, we want all 668,665 rows."""
    nb = _load_notebook(NOTEBOOKS_DIR / "train.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    # We don't slice train — we pass it directly to train_lgbm.
    # The call site is `train_lgbm(train=train_feat, ...)` where
    # `train_feat` is built from the full `train_raw`.
    assert "train_lgbm(" in src
    # There should be no .sample(frac=...) call on the train set.
    # (We allow it on test for some workflows, but the current notebook
    # uses the full test set too.)
    assert "train_raw.sample" not in src


def test_train_notebook_all_code_cells_parse_as_valid_python():
    """Each code cell in the notebook must be syntactically valid Python
    (or contain only Jupyter shell-magic, which is not Python).
    Catches typos and missing imports before the user hits Run All.
    """
    nb = _load_notebook(NOTEBOOKS_DIR / "train.ipynb")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Strip lines that are pure Jupyter shell-magic; they aren't
        # valid Python and are perfectly fine in a notebook.
        python_src = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%", "?"))
        )
        try:
            compile(python_src, f"<train.ipynb#cell-{i}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"train.ipynb cell {i} has a syntax error:\n{e}")


def test_eda_notebook_all_code_cells_parse_as_valid_python():
    nb = _load_notebook(NOTEBOOKS_DIR / "EDA.ipynb")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        python_src = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%", "?"))
        )
        try:
            compile(python_src, f"<EDA.ipynb#cell-{i}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"EDA.ipynb cell {i} has a syntax error:\n{e}")


def test_tune_optuna_notebook_exists():
    p = NOTEBOOKS_DIR / "tune_optuna.ipynb"
    assert p.exists(), f"Missing: {p}"


def test_tune_optuna_notebook_is_valid_json():
    nb = _load_notebook(NOTEBOOKS_DIR / "tune_optuna.ipynb")
    assert "cells" in nb
    assert nb.get("nbformat") == 4


def test_tune_optuna_notebook_calls_run_optuna_sweep():
    nb = _load_notebook(NOTEBOOKS_DIR / "tune_optuna.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "run_optuna_sweep(" in src


def test_tune_optuna_notebook_writes_v2_submission():
    nb = _load_notebook(NOTEBOOKS_DIR / "tune_optuna.ipynb")
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "submission_lgbm_v2.csv" in src


def test_tune_optuna_notebook_all_code_cells_parse_as_valid_python():
    nb = _load_notebook(NOTEBOOKS_DIR / "tune_optuna.ipynb")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        python_src = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%", "?"))
        )
        try:
            compile(python_src, f"<tune_optuna.ipynb#cell-{i}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"tune_optuna.ipynb cell {i} has a syntax error:\n{e}")
