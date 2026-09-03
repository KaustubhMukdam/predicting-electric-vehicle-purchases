"""Tests for src/predict.py.

`make_submission()` is the only public function. It takes:
    - test_ids:        the id column from the test DataFrame (or any
                       iterable aligned to test_pred)
    - test_pred:       predicted probabilities, in [0, 1]
    - template_path:   path to data/sample_submission.csv
    - out_path:        where to write the submission file

It produces a CSV whose column layout exactly matches the template
(id, Will_Buy_EV), with the target column replaced by `test_pred`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import SAMPLE_SUBMISSION_CSV


pytestmark = pytest.mark.skipif(
    not SAMPLE_SUBMISSION_CSV.exists(),
    reason="data/sample_submission.csv not present",
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_make_submission_writes_a_csv_file(tmp_path: Path):
    from src.predict import make_submission

    # Build a tiny 3-row template so the test doesn't depend on the
    # real 286k-row sample_submission.csv.
    tiny_template = tmp_path / "tiny_template.csv"
    tiny_template.write_text("id,Will_Buy_EV\n1,0.0\n2,0.0\n3,0.0\n")

    test_ids = pd.Series([1, 2, 3])
    test_pred = np.array([0.1, 0.5, 0.9])
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, tiny_template, out)
    assert out.exists()


def test_make_submission_csv_has_same_shape_as_template(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)

    # Use ids that match the real template for a realistic test.
    test_ids = template["id"]
    test_pred = np.linspace(0, 1, n)
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out)

    out_df = pd.read_csv(out)
    assert out_df.shape == template.shape
    assert list(out_df.columns) == list(template.columns)


def test_make_submission_preserves_id_column_exactly(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    test_ids = template["id"]
    test_pred = np.linspace(0, 1, len(template))
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out)

    out_df = pd.read_csv(out)
    np.testing.assert_array_equal(out_df["id"].to_numpy(), test_ids.to_numpy())


def test_make_submission_writes_predictions_in_target_column(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    test_ids = template["id"]
    test_pred = np.linspace(0, 1, len(template))
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out)

    out_df = pd.read_csv(out)
    # The Will_Buy_EV column should be exactly the predicted probs.
    np.testing.assert_array_almost_equal(
        out_df["Will_Buy_EV"].to_numpy(), test_pred
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_make_submission_rejects_length_mismatch(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    with pytest.raises(ValueError):
        make_submission(
            template["id"].iloc[: n - 1],   # one short
            np.linspace(0, 1, n),
            SAMPLE_SUBMISSION_CSV,
            tmp_path / "submission.csv",
        )


def test_make_submission_rejects_predictions_outside_unit_interval(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    bad = np.linspace(0, 1, n)
    bad[0] = 1.5  # out of range
    with pytest.raises(ValueError):
        make_submission(
            template["id"], bad, SAMPLE_SUBMISSION_CSV, tmp_path / "submission.csv"
        )


def test_make_submission_rejects_predictions_with_nan(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    bad = np.linspace(0, 1, n)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        make_submission(
            template["id"], bad, SAMPLE_SUBMISSION_CSV, tmp_path / "submission.csv"
        )


def test_make_submission_rejects_nan_ids(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    bad_ids = template["id"].copy()
    bad_ids.iloc[5] = -1  # technically valid; we instead test NaN below
    # Use a Series with a real NaN.
    nan_ids = pd.Series([float("nan")] * n)
    with pytest.raises(ValueError):
        make_submission(
            nan_ids,
            np.linspace(0, 1, n),
            SAMPLE_SUBMISSION_CSV,
            tmp_path / "submission.csv",
        )


def test_make_submission_raises_if_template_missing(tmp_path: Path):
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    with pytest.raises((FileNotFoundError, OSError)):
        make_submission(
            template["id"],
            np.linspace(0, 1, n),
            tmp_path / "does_not_exist.csv",
            tmp_path / "submission.csv",
        )


def test_make_submission_raises_if_target_column_missing(tmp_path: Path):
    """Template missing the target column must error, not silently produce
    a broken file."""
    from src.predict import make_submission

    bad_template = tmp_path / "bad_template.csv"
    bad_template.write_text("id,foo\n1,0.5\n2,0.7\n")
    n = 2
    with pytest.raises((KeyError, ValueError)):
        make_submission(
            pd.Series([1, 2]),
            np.array([0.1, 0.9]),
            bad_template,
            tmp_path / "submission.csv",
        )


# ---------------------------------------------------------------------------
# Idempotence and integration
# ---------------------------------------------------------------------------
def test_make_submission_is_idempotent(tmp_path: Path):
    """Calling make_submission twice with the same inputs produces identical files."""
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    test_ids = template["id"]
    test_pred = np.linspace(0, 1, n)
    out1 = tmp_path / "sub1.csv"
    out2 = tmp_path / "sub2.csv"
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out1)
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_make_submission_creates_parent_directory(tmp_path: Path):
    """If the out_path's parent doesn't exist, it should be created."""
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    out = tmp_path / "nested" / "dir" / "submission.csv"
    make_submission(
        template["id"], np.linspace(0, 1, n), SAMPLE_SUBMISSION_CSV, out
    )
    assert out.exists()


def test_make_submission_round_trip_with_real_template(tmp_path: Path):
    """Use the actual sample_submission template and verify the produced
    file has the same row count, same id range, and a Will_Buy_EV column
    that is a valid probability in [0, 1]."""
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    test_pred = np.full(n, 0.1746450016076809)  # base rate
    out = tmp_path / "submission.csv"
    make_submission(template["id"], test_pred, SAMPLE_SUBMISSION_CSV, out)

    out_df = pd.read_csv(out)
    assert len(out_df) == 286_571
    assert out_df["Will_Buy_EV"].between(0, 1).all()
    assert out_df["Will_Buy_EV"].notna().all()
    # First and last ids should match the template's range.
    assert out_df["id"].iloc[0] == template["id"].iloc[0]
    assert out_df["id"].iloc[-1] == template["id"].iloc[-1]


def test_make_submission_full_real_template_with_unique_predictions(tmp_path: Path):
    """Production-scale test: every prediction is unique and ordered by id.
    Confirms no row shuffling, no truncation, no off-by-one at scale."""
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    test_ids = template["id"]
    test_pred = np.linspace(0.0, 1.0, n)
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, SAMPLE_SUBMISSION_CSV, out)

    out_df = pd.read_csv(out)
    assert len(out_df) == n
    # Ids must be in the exact original order.
    np.testing.assert_array_equal(out_df["id"].to_numpy(), test_ids.to_numpy())
    # Predictions must be in the exact order we passed in.
    np.testing.assert_array_almost_equal(
        out_df["Will_Buy_EV"].to_numpy(), test_pred
    )
    # Boundary values 0.0 and 1.0 must both appear.
    assert out_df["Will_Buy_EV"].iloc[0] == pytest.approx(0.0)
    assert out_df["Will_Buy_EV"].iloc[-1] == pytest.approx(1.0)


def test_make_submission_preserves_extra_template_columns(tmp_path: Path):
    """If the template has columns beyond {id, target}, they must be
    preserved unchanged. (Some Kaggle templates include extra metadata.)"""
    from src.predict import make_submission

    # Build a multi-column template (mimics what some competitions ship).
    big_template = tmp_path / "big_template.csv"
    n = 286_571
    extra_col_values = np.arange(n, dtype=np.int64) * 7
    big_template.write_text(
        "id,Will_Buy_EV,extra_meta\n"
        + "\n".join(f"{i},0.0,{v}" for i, v in zip(range(n), extra_col_values))
        + "\n"
    )

    test_ids = pd.Series(range(n))
    test_pred = np.linspace(0.0, 1.0, n)
    out = tmp_path / "submission.csv"
    make_submission(test_ids, test_pred, big_template, out)

    out_df = pd.read_csv(out)
    assert list(out_df.columns) == ["id", "Will_Buy_EV", "extra_meta"]
    np.testing.assert_array_equal(
        out_df["extra_meta"].to_numpy(), extra_col_values
    )
    # Target column should be exactly what we passed in.
    np.testing.assert_array_almost_equal(
        out_df["Will_Buy_EV"].to_numpy(), test_pred
    )


def test_make_submission_preserves_id_dtype_at_scale(tmp_path: Path):
    """I ids are int64 in the real template; the output must also be int64
    (not coerced to float, which would break some downstream readers)."""
    from src.predict import make_submission

    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n = len(template)
    out = tmp_path / "submission.csv"
    make_submission(
        template["id"],
        np.full(n, 0.5),
        SAMPLE_SUBMISSION_CSV,
        out,
    )
    out_df = pd.read_csv(out)
    # After a round-trip through CSV, pandas may upgrade to int64.
    # What we must NOT see is float-with-trailing-zeros.
    assert pd.api.types.is_integer_dtype(out_df["id"]), (
        f"id dtype after CSV round-trip is {out_df['id'].dtype}, "
        "expected integer"
    )
