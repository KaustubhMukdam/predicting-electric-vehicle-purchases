"""Tests for src/tracking.py.

These tests redirect MLflow to a temp directory via monkey-patched
tracking URI. They verify that our wrapper produces real MLflow runs
that can be queried via the MLflow client API.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pytest


@pytest.fixture
def temp_mlruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point MLflow at a temp dir for the duration of the test."""
    mlflow_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    yield mlflow_dir
    # Reset so the next test isn't poisoned.
    mlflow.set_tracking_uri("")
    # Best-effort cleanup of the global active experiment.
    try:
        mlflow.set_experiment(None)  # type: ignore[arg-type]
    except Exception:
        pass


def test_start_run_creates_an_active_run(temp_mlruns):
    from src.tracking import start_run

    with start_run(run_name="test_run") as run:
        assert run is not None
        assert mlflow.active_run() is not None
        run_id = run.info.run_id

    # Run should be queryable after exiting the context.
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    fetched = client.get_run(run_id)
    assert fetched.info.run_name == "test_run"


def test_log_params_records_all_params(temp_mlruns):
    from src.tracking import log_params, start_run

    with start_run(run_name="log_params_test"):
        log_params({"learning_rate": 0.05, "num_leaves": 31})

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params.get("learning_rate") == "0.05"
    assert runs[0].data.params.get("num_leaves") == "31"


def test_log_metrics_records_all_metrics(temp_mlruns):
    from src.tracking import log_metrics, start_run

    with start_run(run_name="log_metrics_test"):
        log_metrics({"auc": 0.94, "log_loss": 0.31})

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    metrics = runs[0].data.metrics
    assert metrics["auc"] == pytest.approx(0.94)
    assert metrics["log_loss"] == pytest.approx(0.31)


def test_log_metric_supports_step(temp_mlruns):
    from src.tracking import log_metric, start_run

    with start_run(run_name="log_metric_step_test"):
        for step in range(3):
            log_metric("train_loss", 0.5 - step * 0.1, step=step)

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    run_id = client.search_runs(experiment_ids=[exp.experiment_id])[0].info.run_id
    history = client.get_metric_history(run_id, "train_loss")
    steps = sorted(h.step for h in history)
    assert steps == [0, 1, 2]


def test_log_artifact_writes_file(temp_mlruns, tmp_path):
    from src.tracking import log_artifact, start_run

    f = tmp_path / "submission.csv"
    f.write_text("id,Will_Buy_EV\n1,0.5\n")

    with start_run(run_name="log_artifact_test"):
        log_artifact(str(f), artifact_path="submissions")

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    run_id = client.search_runs(experiment_ids=[exp.experiment_id])[0].info.run_id
    artifacts = client.list_artifacts(run_id, path="submissions")
    assert any(a.path == "submissions/submission.csv" for a in artifacts)


def test_log_numpy_array_as_artifact(temp_mlruns, tmp_path):
    from src.tracking import log_numpy_array, start_run

    arr = np.linspace(0, 1, 100)

    with start_run(run_name="log_array_test"):
        log_numpy_array(arr, name="oof_preds", artifact_path="oof")

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{temp_mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    run_id = client.search_runs(experiment_ids=[exp.experiment_id])[0].info.run_id
    local_path = client.download_artifacts(run_id, "oof/oof_preds.npy")
    loaded = np.load(local_path)
    np.testing.assert_array_equal(loaded, arr)


def test_set_tracking_uri_helper(temp_mlruns):
    """Our set_tracking_uri should redirect to the given directory."""
    from src.tracking import set_tracking_uri

    new_dir = temp_mlruns / "subdir"
    set_tracking_uri(str(new_dir))
    assert mlflow.get_tracking_uri().endswith("subdir")


def test_start_run_without_explicit_run_name(temp_mlruns):
    from src.tracking import start_run

    with start_run():
        run_id = mlflow.active_run().info.run_id
    assert run_id is not None


def test_get_or_create_experiment_creates_named_experiment(temp_mlruns):
    from src.tracking import get_or_create_experiment

    exp_id = get_or_create_experiment("my-test-experiment")
    assert isinstance(exp_id, str) and exp_id
    # Calling twice returns the same id.
    exp_id_2 = get_or_create_experiment("my-test-experiment")
    assert exp_id == exp_id_2


def test_ensure_local_tracking_uri_sets_file_store_opt_in(monkeypatch, tmp_path):
    """On Kaggle (MLflow 2.22+), the file backend is in maintenance mode
    and refuses to run unless `MLFLOW_ALLOW_FILE_STORE=true` is set.
    `_ensure_local_tracking_uri` must set this env var so the experiment
    creation that follows doesn't blow up.
    """
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    from src.tracking import _ensure_local_tracking_uri

    _ensure_local_tracking_uri()
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE") == "true"
