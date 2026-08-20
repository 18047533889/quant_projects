# -*- coding: utf-8 -*-
"""Review-10 config / market / coverage / source-ref fixes (R10-P0-024..027).

* R10-P0-024 — the strict (production) DSL surface rejects an implicit
  DailyBar->StockMinuteBar source swap; the compat surface keeps it.
* R10-P0-025 — ``market`` (MarketID) and ``calendar_id`` (calendar identity)
  are resolved separately, never conflated into one variable.
* R10-P0-026 — an unknown / misspelt market raises ValueError, never a silent
  US fallback.
* R10-P0-027 — coverage is split into universe_breadth / field_coverage /
  mask_retention (a sparse field must not report 100% breadth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.columns import col
from api.mining_integration import export_dsl_allowlist_json
from market.universe import apply_universe_mask, coverage_metrics
from runtime.config import FactorEngineConfig, RunConfig
from runtime.config_runtime import _resolve_market, resolve_run_kwargs


# ---------------------------------------------------------------------------
# R10-P0-025: market / calendar split
# ---------------------------------------------------------------------------

def test_resolve_market_splits_calendar_from_market():
    # only a calendar configured -> market inferred, calendar kept distinct
    assert _resolve_market(_config(calendar="SSE")) == ("ashare", "SSE")
    assert _resolve_market(_config(calendar="NASDAQ")) == ("us", "NASDAQ")
    # explicit market wins; calendar stays the calendar
    assert _resolve_market(_config(market="us", calendar="NASDAQ")) == ("us", "NASDAQ")
    assert _resolve_market(_config(market="ashare", calendar="SSE")) == ("ashare", "SSE")
    # neither set
    assert _resolve_market(_config()) == (None, None)


def test_resolved_run_kwargs_carries_both_ids():
    opts = resolve_run_kwargs(_config(calendar="SSE"))
    assert opts.market == "ashare"
    assert opts.calendar_id == "SSE"


def _config(market: str | None = None, calendar: str | None = None) -> FactorEngineConfig:
    return FactorEngineConfig(
        factor=None,  # type: ignore[arg-type]
        data_source=None,  # type: ignore[arg-type]
        run=RunConfig(mode="research", market=market, calendar=calendar),
    )


# ---------------------------------------------------------------------------
# R10-P0-026: unknown market raises
# ---------------------------------------------------------------------------

def test_unknown_market_raises_no_us_fallback():
    with pytest.raises(ValueError, match="unknown market"):
        export_dsl_allowlist_json(market="ahsare")  # typo must not become US
    assert export_dsl_allowlist_json(market="us")["operator_policy"] == "afv_us_pv_daily"
    assert export_dsl_allowlist_json(market="ashare")["operator_policy"] == "lqtp_pv_daily"


# ---------------------------------------------------------------------------
# R10-P0-027: three-metric coverage
# ---------------------------------------------------------------------------

def test_coverage_metrics_split_three_dimensions():
    idx = pd.date_range("2026-01-01", periods=5)
    panel = pd.DataFrame(np.arange(25.0).reshape(5, 5), index=idx, columns=list("ABCDE"))
    panel.iloc[0, 0] = np.nan  # one missing pre-mask cell
    mask = pd.DataFrame(
        np.ones((5, 5)), index=idx, columns=list("ABCDE")
    )
    mask["A"] = 0.0  # column A is OUT of universe entirely
    masked = apply_universe_mask(panel, mask)
    m = coverage_metrics(panel, masked, mask)

    # eligible = 20 of 25 cells; factor finite on all eligible -> breadth 0.8
    assert m.universe_breadth == pytest.approx(0.8)
    assert m.field_coverage == pytest.approx(1.0)
    # retained finite / pre-mask finite = 20 / 24
    assert m.mask_retention == pytest.approx(20.0 / 24.0)
    # backward-compatible alias
    assert m.coverage_ratio == pytest.approx(m.mask_retention)


def test_sparse_field_does_not_report_high_breadth():
    idx = pd.date_range("2026-01-01", periods=5)
    # a sparse field: only the first 2 instruments carry values (of 5)
    panel = pd.DataFrame(
        np.full((5, 5), np.nan), index=idx, columns=list("ABCDE")
    )
    panel.iloc[:, :2] = 1.0
    mask = pd.DataFrame(np.ones((5, 5)), index=idx, columns=list("ABCDE"))
    masked = apply_universe_mask(panel, mask)
    m = coverage_metrics(panel, masked, mask)
    # universe is full (breadth 1.0) but the FIELD only covers 40% of it
    assert m.universe_breadth == pytest.approx(1.0)
    assert m.field_coverage == pytest.approx(0.4)
    assert m.mask_retention == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# R10-P0-024: strict minute source swap
# ---------------------------------------------------------------------------

def test_strict_surface_rejects_implicit_minute_source():
    from api.source_ref import transform_source_col

    with pytest.raises(ValueError, match="strict"):
        transform_source_col(col("close"), "minute_bar", strict=True, period=5, index=0)
    # the compat (non-strict) surface keeps the implicit StockMinuteBar inference
    out = transform_source_col(col("close"), "minute_bar", period=5, index=0)
    assert out is not None
    assert "__fe_source_ref_v1__" in str(out.name)
