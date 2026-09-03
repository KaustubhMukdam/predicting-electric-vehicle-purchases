"""Thin wrapper around MLflow.

Goals
-----
- Same API locally and on Kaggle (file backend; no remote server).
- One context manager for runs so callers never forget to close a run.
- Convenience helpers for the two things we log a lot:
  numpy arrays (OOF / test predictions) and arbitrary files (submissions).

Everything goes through the standard `mlflow` module. The wrapper exists
to centralize the experiment name, the file-backend default, and the
"how do we serialize a numpy array" decision.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow
import numpy as np

from src.config import MLRUNS_DIR, MLFLOW_EXPERIMENT_NAME


def set_tracking_uri(uri: str | os.PathLike) -> None:
    """Set the MLflow tracking URI. Accepts a `file:` URI or a plain
    directory path (which we promote to a `file:` URI)."""
    uri = str(uri)
    if not uri.startswith(("file:", "http:", "https:", "databricks:")):
        uri = f"file:{uri}"
    mlflow.set_tracking_uri(uri)


def _ensure_local_tracking_uri() -> None:
    """If the tracking URI has not been set explicitly, point MLflow at
    the local `mlruns/` directory inside the project. This is the
    default behavior on both local and Kaggle runs unless the caller
    overrides it (e.g. via `set_tracking_uri` in the notebook)."""
    if mlflow.get_tracking_uri() in ("", None):
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        set_tracking_uri(MLRUNS_DIR)


def get_or_create_experiment(name: str = MLFLOW_EXPERIMENT_NAME) -> str:
    """Return the experiment id (str) for `name`, creating it if needed.

    Always sets the experiment as the active one before returning, so
    subsequent `start_run()` calls land in the right place.
    """
    _ensure_local_tracking_uri()
    experiment = mlflow.set_experiment(name)
    return experiment.experiment_id


@contextmanager
def start_run(
    run_name: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> Iterator[mlflow.entities.Run]:
    """Context manager wrapping `mlflow.start_run`. Ensures the
    experiment exists, sets the run name if given, and always ends
    the run on exit (even on exceptions)."""
    get_or_create_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict) -> None:
    """Log a flat dict of params. Values are coerced to str (MLflow
    requirement)."""
    if not params:
        return
    safe = {k: str(v) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a flat dict of metrics. Optionally with a step counter."""
    if not metrics:
        return
    safe = {k: float(v) for k, v in metrics.items()}
    mlflow.log_metrics(safe, step=step)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    mlflow.log_metric(key, float(value), step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    mlflow.log_artifact(local_path, artifact_path)


def log_numpy_array(
    arr: np.ndarray,
    name: str,
    artifact_path: str | None = None,
) -> None:
    """Serialize a numpy array to a temp `.npy` file and log it as an
    MLflow artifact. Cleans up the temp file after upload."""
    if not name.endswith(".npy"):
        name = f"{name}.npy"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / name
        np.save(path, arr)
        mlflow.log_artifact(str(path), artifact_path)
