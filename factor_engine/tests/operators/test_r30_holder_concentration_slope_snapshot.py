from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


ensure_cleaned_loaded()


def _inputs():
    # Every report date precedes the observation window; future-prefix tests
    # concern later visible revisions, not impossible future-dated reports.
    idx = pd.date_range("2024-08-01", periods=9, freq="B")
    snapshots = [
        "2023-12-31",
        "2024-01-31",
        None,
        "2024-03-31",
        None,
        "2024-01-31",  # later revision of an already-visible snapshot
        None,
        "2024-07-15",  # deliberately irregular spacing
        None,
    ]
    values = [1.0, 2.0, 2.0, 4.0, 4.0, 2.5, 2.5, 7.0, 7.0]
    conc = pd.DataFrame({"A": values, "B": [3.0, np.nan, np.nan, 5.0, 5.0, np.nan, np.nan, 9.0, 9.0]}, index=idx)
    sd = pd.DataFrame({"A": pd.to_datetime(snapshots), "B": pd.to_datetime(snapshots)}, index=idx)
    return conc, sd


def _pl_frame(frame: pd.DataFrame, time_col="date") -> pl.DataFrame:
    data = {time_col: frame.index.to_pydatetime().tolist()}
    for c in frame.columns:
        data[c] = [None if pd.isna(v) else v for v in frame[c].to_list()]
    return pl.DataFrame(data)


def _oracle(values, snapshots, window):
    visible = {}
    current = np.nan
    out = []
    for value, sd in zip(values, snapshots):
        if pd.notna(sd):
            visible[pd.Timestamp(sd).normalize()] = float(value)
            ordered = sorted(visible.items())[-window:]
            x = np.array([d.value / 8.64e13 for d, _ in ordered], dtype=float)
            y = np.array([v for _, v in ordered], dtype=float)
            finite = np.isfinite(y)
            if finite.sum() >= 3:
                xc = x[finite] - x[finite].mean()
                current = np.dot(xc, y[finite] - y[finite].mean()) / np.dot(xc, xc)
            else:
                current = np.nan
        out.append(current)
    return np.asarray(out)


def _acceleration_oracle(values, snapshots, window):
    visible = {}
    current = np.nan
    out = []
    for value, sd in zip(values, snapshots):
        if pd.notna(sd):
            visible[pd.Timestamp(sd).normalize()] = float(value)
            ordered = sorted(visible.items())
            slopes = []
            for i in range(len(ordered)):
                segment = ordered[max(0, i - window + 1) : i + 1]
                x = np.array([d.value / 8.64e13 for d, _ in segment])
                y = np.array([v for _, v in segment])
                finite = np.isfinite(y)
                if finite.sum() < 3:
                    slopes.append(np.nan)
                else:
                    xc = x[finite] - x[finite].mean()
                    slopes.append(np.dot(xc, y[finite] - y[finite].mean()) / np.dot(xc, xc))
            current = slopes[-1] - slopes[-2] if len(slopes) >= 2 else np.nan
        out.append(current)
    return np.asarray(out)


def test_snapshot_slope_parity_and_independent_calendar_date_oracle():
    conc, sd = _inputs()
    pandas_op = OperatorRegistry.get("holder_concentration_slope", "pandas_numpy")
    polars_op = OperatorRegistry.get("holder_concentration_slope", "polars")
    got_pd = pandas_op.calculate(conc, 4, sd)
    got_pl = polars_op.calculate(_pl_frame(conc), 4, _pl_frame(sd)).to_pandas().set_index("date")

    expected = _oracle(conc["A"], sd["A"], 4)
    np.testing.assert_allclose(got_pd["A"], expected, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got_pl["A"], expected, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        got_pl[["A", "B"]].to_numpy(), got_pd.to_numpy(),
        rtol=0, atol=1e-12, equal_nan=True,
    )
    assert got_pd["A"].iloc[2] != got_pd["A"].iloc[3]  # no daily-row rolling
    assert got_pd["A"].iloc[3] == got_pd["A"].iloc[4]  # missing ID carries forward

    # Independent OLS oracle: explicit design matrix with an intercept, using
    # centred calendar days to avoid epoch-scale conditioning artifacts.
    x = pd.to_datetime(["2023-12-31", "2024-01-31", "2024-03-31", "2024-07-15"])
    x = x.to_numpy().astype("datetime64[D]").astype(float)
    y = np.array([1.0, 2.5, 4.0, 7.0])
    design = np.column_stack([x - x.mean(), np.ones(len(x))])
    expected_lstsq = np.linalg.lstsq(design, y, rcond=None)[0][0]
    assert np.isclose(got_pd["A"].iloc[-1], expected_lstsq, atol=1e-12)


def test_future_revision_cannot_rewrite_prefix_on_either_backend():
    conc, sd = _inputs()
    cut = 5
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("holder_concentration_slope", backend)
        if backend == "pandas_numpy":
            full = op.calculate(conc, 4, sd)
            prefix = op.calculate(conc.iloc[:cut], 4, sd.iloc[:cut])
            np.testing.assert_allclose(full.iloc[:cut], prefix, equal_nan=True)
        else:
            full = op.calculate(_pl_frame(conc), 4, _pl_frame(sd))
            prefix = op.calculate(_pl_frame(conc.iloc[:cut]), 4, _pl_frame(sd.iloc[:cut]))
            np.testing.assert_allclose(
                full["A"].to_numpy()[:cut], prefix["A"].to_numpy(), equal_nan=True
            )


def test_missing_snapshot_value_fails_closed_until_three_finite_snapshots():
    conc, sd = _inputs()
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("holder_concentration_slope", backend)
        out = op.calculate(conc, 4, sd) if backend == "pandas_numpy" else op.calculate(_pl_frame(conc), 4, _pl_frame(sd))
        values = out["B"].to_numpy()
        assert np.isnan(values[:7]).all()
        assert np.isfinite(values[7:]).all()


def test_no_snapshot_argument_retains_legacy_daily_path():
    conc, _ = _inputs()
    pd_out = OperatorRegistry.get("holder_concentration_slope", "pandas_numpy").calculate(conc, 4, None)
    pl_out = OperatorRegistry.get("holder_concentration_slope", "polars").calculate(_pl_frame(conc), 4, None)
    np.testing.assert_allclose(
        pd_out["A"].to_numpy(), pl_out["A"].to_numpy(), atol=1e-12, rtol=0, equal_nan=True
    )


def test_snapshot_acceleration_parity_oracle_and_future_prefix():
    conc, sd = _inputs()
    expected = _acceleration_oracle(conc["A"], sd["A"], 4)
    pandas_op = OperatorRegistry.get("holder_concentration_acceleration", "pandas_numpy")
    polars_op = OperatorRegistry.get("holder_concentration_acceleration", "polars")
    got_pd = pandas_op.calculate(conc, 4, sd)
    got_pl = polars_op.calculate(_pl_frame(conc), 4, _pl_frame(sd))
    np.testing.assert_allclose(got_pd["A"], expected, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(got_pl["A"], expected, atol=1e-12, rtol=0, equal_nan=True)
    cut = 5
    prefix_pd = pandas_op.calculate(conc.iloc[:cut], 4, sd.iloc[:cut])
    prefix_pl = polars_op.calculate(_pl_frame(conc.iloc[:cut]), 4, _pl_frame(sd.iloc[:cut]))
    np.testing.assert_allclose(got_pd["A"].iloc[:cut], prefix_pd["A"], equal_nan=True)
    np.testing.assert_allclose(got_pl["A"].to_numpy()[:cut], prefix_pl["A"].to_numpy(), equal_nan=True)


def test_polars_timestamp_identity_is_preserved_and_axes_fail_closed():
    conc, sd = _inputs()
    conc_pl = _pl_frame(conc, time_col="timestamp")
    sd_pl = _pl_frame(sd, time_col="timestamp")
    for canonical in ("holder_concentration_slope", "holder_concentration_acceleration"):
        op = OperatorRegistry.get(canonical, "polars")
        for snapshot in (sd_pl, None):
            out = op.calculate(conc_pl, 4, snapshot)
            assert out["timestamp"].to_list() == conc_pl["timestamp"].to_list()
            assert out.columns == conc_pl.columns

        with np.testing.assert_raises(ValueError):
            op.calculate(conc_pl, 4, sd_pl.drop("B"))
        with np.testing.assert_raises(ValueError):
            op.calculate(conc_pl, 4, sd_pl.select("timestamp", "B", "A"))
        with np.testing.assert_raises(ValueError):
            op.calculate(conc_pl, 4, sd_pl.reverse())


def test_invalid_nonmissing_snapshot_date_fails_in_both_backends():
    conc, sd = _inputs()
    bad_pd = sd.astype(object)
    bad_pd.iloc[4, 0] = "not-a-date"
    bad_pl = _pl_frame(sd).with_columns(
        pl.when(pl.arange(0, pl.len()) == 4)
        .then(pl.lit("not-a-date"))
        .otherwise(pl.col("A").cast(pl.String))
        .alias("A")
    )
    for canonical in ("holder_concentration_slope", "holder_concentration_acceleration"):
        with np.testing.assert_raises((TypeError, ValueError)):
            OperatorRegistry.get(canonical, "pandas_numpy").calculate(conc, 4, bad_pd)
        with np.testing.assert_raises(Exception):
            OperatorRegistry.get(canonical, "polars").calculate(_pl_frame(conc), 4, bad_pl)
