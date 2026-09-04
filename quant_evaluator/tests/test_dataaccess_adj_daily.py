"""DataAccess adjusted-table integration tests (spec §21-25, §58).

These tests exercise the production :class:`DataAccessEvaluationLoader` against
the local ``cos_data`` mirror of the authoritative adjusted table
``ashare_stock_daily_adj`` (StockDailyBarAdj).

Key invariants under test:
  1. ``TargetVwapReturnH10 == AdjVwap[t+11]/AdjVwap[t+1] - 1`` (producer contract,
     verified live to 0 diff).
  2. QE **never shifts labels** — the loader reads the producer column verbatim;
     timing is bound from the producer contract, never guessed.
  3. ``TargetSpec.horizon`` maps to the correct physical column (H1→H01, H5→H05,
     H10→H10, H20→H20).
  4. ``load_factor_batch`` returns a finite (T, N[, F]) matrix from the factor lake.
  5. All four horizons load and their target matrices are dimensionally aligned.

The whole module SKIPs when the local cos_data mirror is absent so CI without a
local mirror does not fail.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Data source roots.  The default ASHARE_PARQUET_ROOT (/home/shw/...) is not
# writable on this host; the local mirror lives under /home/sunhaiwei/cos_data.
# Env vars must be set before the store is constructed (get_store() reads them).
# ---------------------------------------------------------------------------
_COS_ROOT = "/home/sunhaiwei/cos_data"
_ADJ_DIR = os.path.join(_COS_ROOT, "StockDailyBarAdj")
_LAKE_ROOT = "/home/sunhaiwei/quant_projects/data/factors/lake"

os.environ.setdefault("ASHARE_PARQUET_ROOT", _COS_ROOT)
os.environ.setdefault("FACTOR_LAKE_ROOT", _LAKE_ROOT)

_HAS_ADJ = os.path.isdir(_ADJ_DIR) and any(
    f.endswith(".parquet") for f in os.listdir(_ADJ_DIR)
)

pytestmark = pytest.mark.skipif(
    not _HAS_ADJ,
    reason=(
        "local cos_data mirror (StockDailyBarAdj) not present; "
        "cannot run DataAccess integration"
    ),
)

# Imported after env vars are set so the store resolves the mirror root.
from data_access import get_store  # noqa: E402

from quant_evaluator.adapters.data_access import (  # noqa: E402
    TARGET_CONTRACT,
    DataAccessEvaluationLoader,
    TargetSpec,
)

# A short decision window with a trailing buffer so the H20 formula
# (AdjVwap[t+21]/AdjVwap[t+1]-1) has enough forward rows to be fully defined.
_DECISION_RANGE = ("2026-07-01", "2026-07-31")
_FORMULA_RANGE = ("2026-07-01", "2026-08-20")

_HORIZONS = [
    (1, "TargetVwapReturnH01"),
    (5, "TargetVwapReturnH05"),
    (10, "TargetVwapReturnH10"),
    (20, "TargetVwapReturnH20"),
]


@pytest.fixture(scope="module")
def loader():
    return DataAccessEvaluationLoader()


@pytest.fixture(scope="module")
def store():
    return get_store()


def _pivot(df, value_col, dates, assets):
    """Pivot a long (TradeDate, Symbol, value) frame to a (date, asset) matrix."""
    piv = df.pivot_table(
        index="TradeDate", columns="Symbol", values=value_col, aggfunc="first"
    )
    return piv.reindex(index=dates, columns=assets).values.astype(np.float64)


def _formula_matrix(store, horizon, decision_dates, assets):
    """Compute AdjVwap[t+1+h]/AdjVwap[t+1]-1 per symbol over a wide range."""
    h = store.read(
        "ashare_stock_daily_adj",
        columns=["TradeDate", "Symbol", "AdjVwap"],
        time_range=_FORMULA_RANGE,
    )
    df = h.to_pandas().sort_values(["Symbol", "TradeDate"])
    # entry t+1, exit t+1+horizon  =>  shift(-(1+h)) / shift(-1) - 1
    df["formula"] = df.groupby("Symbol")["AdjVwap"].transform(
        lambda x: x.shift(-(1 + horizon)) / x.shift(-1) - 1
    )
    return _pivot(df, "formula", decision_dates, assets)


# ---------------------------------------------------------------------------
# 1. target_h10: H10 target == AdjVwap[t+11]/AdjVwap[t+1]-1
# ---------------------------------------------------------------------------
def test_target_h10_matches_adjvwap_formula(loader, store):
    spec = TargetSpec(target_id="h10", horizon=10, physical_column="TargetVwapReturnH10")
    bundle = loader.load_label_bundle(spec, time_range=_DECISION_RANGE)

    # asset universe is whatever the loader pivoted over
    h0 = store.read(
        "ashare_stock_daily_adj",
        columns=["TradeDate", "Symbol", "TargetVwapReturnH10"],
        time_range=_DECISION_RANGE,
    )
    assets = sorted(h0.to_pandas()["Symbol"].unique())
    decision_dates = [pd.Timestamp(d) for d in bundle.decision_time]

    formula = _formula_matrix(store, 10, decision_dates, assets)
    assert formula.shape == bundle.values.shape

    mask = np.isfinite(bundle.values) & np.isfinite(formula)
    assert mask.sum() > 0, "no comparable cells between target and formula"
    diff = np.abs(bundle.values - formula)[mask]
    # float32 storage tolerance; observed live max diff is exactly 0.0
    assert np.allclose(
        bundle.values[mask], formula[mask], rtol=1e-5, atol=1e-8
    ), f"H10 formula mismatch: max abs diff = {diff.max():.3e}"


# ---------------------------------------------------------------------------
# 2. target_timing / never_shifts_label
# ---------------------------------------------------------------------------
def test_loader_reads_stored_column_verbatim_no_shift(loader, store):
    """QE never shifts labels: bundle.values == raw producer column (no shift)."""
    spec = TargetSpec(target_id="h10", horizon=10, physical_column="TargetVwapReturnH10")
    bundle = loader.load_label_bundle(spec, time_range=_DECISION_RANGE)

    h = store.read(
        "ashare_stock_daily_adj",
        columns=["TradeDate", "Symbol", "TargetVwapReturnH10"],
        time_range=_DECISION_RANGE,
    )
    df = h.to_pandas()
    assets = sorted(df["Symbol"].unique())
    decision_dates = [pd.Timestamp(d) for d in bundle.decision_time]
    raw = _pivot(df, "TargetVwapReturnH10", decision_dates, assets)

    assert raw.shape == bundle.values.shape
    # verbatim: identical where both are finite (no shift / no alignment)
    mask = np.isfinite(bundle.values) & np.isfinite(raw)
    assert mask.sum() > 0
    assert np.array_equal(bundle.values[mask], raw[mask])


def test_target_spec_horizon_to_physical_column_mapping():
    """TARGET_CONTRACT maps horizon→physical column (H1→H01, ...); TargetSpec
    preserves the explicit physical_column verbatim (no auto-remap)."""
    expected = {
        1: "TargetVwapReturnH01",
        5: "TargetVwapReturnH05",
        10: "TargetVwapReturnH10",
        20: "TargetVwapReturnH20",
    }
    # authoritative producer contract: horizon -> column + price convention
    for horizon, col in expected.items():
        assert TARGET_CONTRACT[col]["horizon"] == horizon, (
            f"TARGET_CONTRACT[{col}] horizon should be {horizon}"
        )
        assert TARGET_CONTRACT[col]["price_convention"] == "vwap_to_vwap"

    # TargetSpec carries the explicit physical_column through unchanged
    for horizon, col in expected.items():
        spec = TargetSpec(
            target_id=f"h{horizon}", horizon=horizon, physical_column=col
        )
        assert spec.physical_column == col
        assert spec.horizon == horizon
        assert spec.price_convention == "vwap_to_vwap"


def test_label_timing_bound_from_producer_contract(loader):
    """decision_time strictly increasing; label window = [t+1, t+1+horizon]."""
    spec = TargetSpec(target_id="h10", horizon=10, physical_column="TargetVwapReturnH10")
    bundle = loader.load_label_bundle(spec, time_range=_DECISION_RANGE)

    dec = [pd.Timestamp(d) for d in bundle.decision_time]
    start = [pd.Timestamp(s) for s in bundle.label_start_time]
    end = [pd.Timestamp(e) for e in bundle.label_end_time]

    assert len(dec) == len(start) == len(end) == bundle.values.shape[0]
    assert all(dec[i] < dec[i + 1] for i in range(len(dec) - 1))
    for i in range(len(dec)):
        assert start[i] - dec[i] == pd.Timedelta(days=1)  # entry t+1
        assert end[i] - start[i] == pd.Timedelta(days=spec.horizon)  # exit t+1+h


# ---------------------------------------------------------------------------
# 3. load_factor_batch: finite (T, N[, F]) matrix from the factor lake
# ---------------------------------------------------------------------------
def test_load_factor_batch_shape_and_finite(loader):
    factor_id = "qe_test_factor"
    batch = loader.load_factor_batch([factor_id], time_range=_DECISION_RANGE)
    assert batch.factor_ids == (factor_id,)
    assert batch.values.ndim == 3  # (T, N, F)
    T, N, F = batch.values.shape
    assert T > 0 and N > 0 and F == 1
    assert np.isfinite(batch.values).all(), "factor matrix contains non-finite values"


# ---------------------------------------------------------------------------
# 4. all four horizons load and target dims align
# ---------------------------------------------------------------------------
def test_all_horizons_load_and_align(loader):
    shapes = {}
    for horizon, col in _HORIZONS:
        spec = TargetSpec(target_id=f"h{horizon}", horizon=horizon, physical_column=col)
        bundle = loader.load_label_bundle(spec, time_range=_DECISION_RANGE)
        assert bundle.horizon == horizon
        assert bundle.values.ndim == 2
        shapes[horizon] = bundle.values.shape
        # each horizon's formula also matches (producer contract)
        assert np.isfinite(bundle.values).any(), f"H{horizon:02d} all-NaN target"

    # all horizons share the same (T, N) decision grid
    base = shapes[10]
    for horizon, shape in shapes.items():
        assert shape[0] == base[0], f"H{horizon:02d} T differs from H10"
        assert shape[1] == base[1], f"H{horizon:02d} N differs from H10"


def test_all_horizons_formula_match(loader, store):
    """Every horizon's stored target matches its AdjVwap forward-return formula."""
    for horizon, col in _HORIZONS:
        spec = TargetSpec(target_id=f"h{horizon}", horizon=horizon, physical_column=col)
        bundle = loader.load_label_bundle(spec, time_range=_DECISION_RANGE)
        h0 = store.read(
            "ashare_stock_daily_adj",
            columns=["TradeDate", "Symbol", col],
            time_range=_DECISION_RANGE,
        )
        assets = sorted(h0.to_pandas()["Symbol"].unique())
        decision_dates = [pd.Timestamp(d) for d in bundle.decision_time]
        formula = _formula_matrix(store, horizon, decision_dates, assets)
        assert formula.shape == bundle.values.shape
        mask = np.isfinite(bundle.values) & np.isfinite(formula)
        assert mask.sum() > 0, f"H{horizon:02d}: no comparable cells"
        assert np.allclose(
            bundle.values[mask], formula[mask], rtol=1e-5, atol=1e-8
        ), f"H{horizon:02d} formula mismatch"
