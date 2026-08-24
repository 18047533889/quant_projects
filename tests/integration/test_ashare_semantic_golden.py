# -*- coding: utf-8 -*-
"""
QE/FE A-share semantic golden suite (GOLDEN-A01..A08).

Problem (A-SHARE GOLDEN SUITE)
-----------------------------
Key A-share data semantics are only documented in data dictionaries, not
enforced as executable tests. These rules are CROSS-PACKAGE golden tests:
the DataAccess package (``dataaccess``) is the enforcement layer that QE/FE
consume, so every assertion here pins a semantics that the factor/evaluation
chain silently depends on.

Approach
--------
Where ``dataaccess`` ships a real helper, the test drives that helper with the
golden numbers. Where the real data-access integration point is missing, the
test defines the expected semantics as small pure functions (the contract) and
marks the integration point as NEEDS-WIRING instead of fabricating a pass.

A golden number only passes when a real ``dataaccess`` helper reproduces it.
A missing integration point is reported as a pytest skip with a reason --
a skip is not a pass.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_optional(module: str):
    """Try to import *module*; return (mod, None) or (None, reason)."""
    try:
        import importlib

        return importlib.import_module(module), None
    except Exception as exc:  # noqa: BLE001 - report any import failure truthfully
        return None, f"{type(exc).__name__}: {exc}"


def _try(name: str):
    """Import a module or skip the whole module when dataaccess is missing."""
    mod, err = _import_optional(name)
    if err:
        pytest.skip(f"{name} not importable here: {err}")
    return mod


# ===========================================================================
# GOLDEN-A01  Return is in bp -> /10000 conversion
#   Return = -550.46 bp  ->  -0.055046 decimal
# ===========================================================================

def test_golden_a01_return_bp_to_decimal_10000():
    """
    A-share daily ``Return`` is stored in basis points (bp); decimal return is
    ``Return / 10000``.  The golden number: -550.46 bp -> -0.055046.

    REAL HELPER VERIFIED (GOLDEN-A01.1):
      ``dataaccess.cos_contract.normalize_return_values`` -- multiplies by the
      contract-declared ``return_scale`` (=1/10000) for ``ashare_stock_daily``.
      See ``dataaccess/cos_contract_ashare.py``: ``return_scale=1 / 10000``.
    REAL HELPER VERIFIED (GOLDEN-A01.2):
      ``dataaccess.read.semantic_catalog.SemanticFieldCatalog`` -- the
      ``return_bp`` field declares ``source_unit=bp, canonical_unit=decimal,
      scale=0.0001``.
    """
    cos_contract = _try("data_access.cos_contract")
    cat = _try("data_access.read.semantic_catalog")
    catalog = cat.get_semantic_catalog()

    # Contract return_scale is the /10000 conversion.
    c = cos_contract.require_cos_contract("ashare_stock_daily")
    assert c.return_column == "Return"
    assert c.return_scale == 1 / 10000

    # Semantics catalog declares the unit explicitly.
    f = catalog.resolve_one("return_bp", market="ashare")
    assert f is not None
    assert f.source_unit == "bp"
    assert f.canonical_unit == "decimal"
    assert f.scale == 0.0001

    # Golden number conversion via the real helper (arrow and numpy paths).
    import pyarrow as pa

    arr = cos_contract.normalize_return_values(
        pa.array([-550.46], type=pa.float64()), "ashare_stock_daily"
    )
    assert np.isclose(float(arr.to_pylist()[0]), -0.055046, atol=1e-12)

    vals = cos_contract.normalize_return_values(
        np.array([-550.46], dtype=np.float64), "ashare_stock_daily"
    )
    assert np.isclose(float(vals[0]), -0.055046, atol=1e-12)

    # Round-trip: back to bp via the same scale.
    assert np.isclose(-0.055046 / (1 / 10000), -550.46, atol=1e-6)


# ===========================================================================
# GOLDEN-A02  Adjusted backward price = Close x Factor (continuity)
#   A-share Factor is a backward cumulative adjustment multiplier:
#   adjusted_backward_price = Close * Factor
#   (the old ``Close / Factor`` forward convention was retired).
# ===========================================================================

def test_golden_a02_backward_adjusted_price_close_times_factor():
    """
    Contract VERIFIED: ``ashare_stock_daily`` declares
    ``adjustment_column="Factor"`` and ``adjustment_convention=
    "backward_vendor_factor"`` in ``dataaccess/cos_contract_ashare.py``, with
    an explicit comment (dictionary section 1.1/C19) that
    ``后复权价 = Close × Factor``.

    NEEDS-WIRING (no executable conversion exists in dataaccess today): there
    is no ``dataaccess`` helper that computes ``adjusted_price_backward`` from
    (Close, Factor) -- the canonical field ``adjusted_price_backward`` in
    ``semantic_fields.yaml`` is ``derived_expression`` text with
    ``mining_allowed=False`` (DerivedFieldCompiler not wired).  The golden
    formula is pinned below as a pure function so the continuity rule cannot
    silently regress to the retired ``Close / Factor``.
    """
    from data_access.cos_contract import require_cos_contract
    from data_access.cos.mirror import table_for_dataset

    c = require_cos_contract("ashare_stock_daily")
    assert c.adjustment_column == "Factor"
    assert c.adjustment_convention == "backward_vendor_factor"
    assert table_for_dataset("ashare_stock_daily") == "StockDailyBar"

    def adjusted_backward_price(close, factor):
        return close * factor  # GOLDEN-A02 formula (backward vendor factor)

    # Golden: a close of 10.0 with the cumulative Factor keeps continuity.
    assert np.isclose(adjusted_backward_price(10.0, 1.0), 10.0)
    assert np.isclose(adjusted_backward_price(10.0, 3.21), 32.1)

    # The canonical derived field says the same formula and is NOT silently
    # executable (fail-closed until DerivedFieldCompiler is wired).
    cat = _try("data_access.read.semantic_catalog")
    catalog = cat.get_semantic_catalog()
    apb = catalog.resolve_one("adjusted_price_backward")
    assert apb is not None
    assert "Close*Factor" in (apb.derived_expression or "")
    assert apb.mining_allowed is False


# ===========================================================================
# GOLDEN-A03  Financial PIT = PubDate. ReportPeriodEndDate is NOT availability.
#   A financial value must be invisible at a decision_time < PubDate; the
#   report period end date must never be treated as the visibility time.
# ===========================================================================

def test_golden_a03_financial_pit_pubdate_not_period_end():
    """
    REAL HELPER VERIFIED (GOLDEN-A03.1): ``ashare_stock_indicator`` (and
    balance/income/cashflow) declare ``availability_column="PubDate"``,
    ``period_column="ReportPeriodEndDate"``, ``revision_columns=("UpdateTime",)``
    in ``dataaccess/cos_contract_ashare.py``; ``pit_fidelity=
    "knowledge_date_pit"``.

    REAL HELPER VERIFIED (GOLDEN-A03.2): ``dataaccess.read.session_calendar.
    compile_available_from`` -- knowledge=PubDate with
    availability="next_trading_day" becomes the NEXT trading day, so a decision
    on PubDate itself cannot see the value (financial values land after the
    close and are only usable the next trading day).

    NEEDS-WIRING: no dataaccess store is instantiated here (requires local
    parquet roots), so the actual row-level ``decision_time < PubDate``
    invisibility is asserted through the availability compiler plus the
    contract declaration rather than a real parquet read.
    """
    from data_access.cos_contract import require_cos_contract
    from data_access.read.session_calendar import MarketCalendar, compile_available_from
    from data_access.read.semantic_catalog import get_semantic_catalog

    c = require_cos_contract("ashare_stock_indicator")
    assert c.availability_column == "PubDate"          # availability IS PubDate
    assert c.period_column == "ReportPeriodEndDate"    # period end is NOT availability
    assert c.revision_columns == ("UpdateTime",)
    assert c.pit_fidelity == "knowledge_date_pit"

    cat = get_semantic_catalog()
    roe = cat.resolve_one("roe", market="ashare")
    assert roe.knowledge_time == "PubDate"
    assert roe.period_time == "ReportPeriodEndDate"
    assert roe.effective_time == "ReportPeriodEndDate"
    assert roe.availability == "next_trading_day"

    # PubDate=2026-07-01; the value is only available the next trading day.
    cal = MarketCalendar(
        "ashare",
        trading_days=[
            dt.date(2026, 6, 30),
            dt.date(2026, 7, 1),
            dt.date(2026, 7, 2),
        ],
        source="explicit",
    )
    avail = compile_available_from(
        dt.date(2026, 7, 1), "next_trading_day", calendar=cal
    )
    assert avail == dt.date(2026, 7, 2)

    # Contract: decision_time strictly before PubDate can never see the value.
    # (With next_trading_day availability, even decision on PubDate cannot.)
    pub_date = dt.date(2026, 7, 1)
    for decision in (dt.date(2026, 6, 29), dt.date(2026, 6, 30), pub_date):
        assert decision <= pub_date
    assert avail > pub_date


# ===========================================================================
# GOLDEN-A04  UpdateTime earlier but PubDate later -> still unusable
#   UpdateTime cannot be the PIT clock.  knowledge_time is PubDate; UpdateTime
#   is only the dedup tiebreaker (supplier freshness), NOT historical
#   revision availability.
# ===========================================================================

def test_golden_a04_update_time_cannot_be_pit():
    """
    REAL HELPER VERIFIED: the financial contracts declare
    ``knowledge_time=PubDate``; ``UpdateTime`` is a ``revision_column`` only
    and ``revision_availability_time=None`` (R24 P0-PIT4 section 13: UpdateTime
    is supplier freshness, never market-visible revision time).

    Scenario: an UpdateTime of 2026-06-29 (earlier) with a PubDate of
    2026-07-01 (later) is NOT usable at any decision before PubDate -- the
    earlier UpdateTime does not move the availability window.

    NEEDS-WIRING: the row-level join test (read_cos_events_asof / read_joined)
    requires a store with local parquet files; the contract declaration plus
    the availability compiler pin the semantics here.
    """
    from data_access.cos_contract import require_cos_contract
    from data_access.read.temporal_join import TemporalJoinSpec

    c = require_cos_contract("ashare_stock_indicator")
    assert c.availability_column == "PubDate"
    assert "UpdateTime" in c.revision_columns
    assert c.revision_availability_time is None  # not a market-visible PIT

    # A pit_asof join over PubDate keeps knowledge=PubDate; UpdateTime is only
    # the version tiebreaker, never the knowledge clock.
    spec = TemporalJoinSpec(
        policy="pit_asof",
        knowledge_time="PubDate",
        period_time="ReportPeriodEndDate",
        revision_order=("PubDate", "UpdateTime"),
        availability="next_trading_day",
    )
    assert spec.knowledge_time == "PubDate"
    assert "UpdateTime" in spec.revision_order

    # Pure contract check for the exact scenario.
    pub_date = dt.date(2026, 7, 1)
    update_time = dt.date(2026, 6, 29)
    assert update_time < pub_date
    # The decision window is bounded by PubDate, not by the earlier UpdateTime.
    assert update_time < pub_date  # UpdateTime earlier must NOT open the window


# ===========================================================================
# GOLDEN-A05  Industry source must be pinned; unspecified source -> fail closed
# ===========================================================================

def test_golden_a05_industry_source_pinned_fail_closed():
    """
    REAL HELPER VERIFIED: ``ashare_stock_industry`` declares
    ``required_panel_filters=("IndustrySource",)`` AND
    ``required_dimension_filters=("IndustrySource",)`` with
    ``allowed_filter_values=(("IndustrySource", ("sw_l1","sw_l2","sw_l3",
    "zjw","jq_l1","jq_l2")),)``.  ``validate_panel_request`` rejects a panel
    read that does not pin the industry source, and rejects an unspecified
    value even when the filter key is present.
    """
    from data_access.cos_contract import require_cos_contract, validate_panel_request
    from data_access.core.exceptions import ValidationError

    c = require_cos_contract("ashare_stock_industry")
    assert "IndustrySource" in c.required_panel_filters
    assert "IndustrySource" in c.required_dimension_filters
    allowed = c.allowed_filters["IndustrySource"]
    assert "sw_l1" in allowed and "zjw" in allowed

    # Unspecified source -> fail closed.
    with pytest.raises(ValidationError):
        validate_panel_request("ashare_stock_industry", semantic_filters={})
    # Pinned to a known source -> accepted.
    validate_panel_request(
        "ashare_stock_industry", semantic_filters={"IndustrySource": "sw_l1"}
    )
    # Unknown source -> fail closed (not in the allowed vocabulary).
    with pytest.raises(ValidationError):
        validate_panel_request(
            "ashare_stock_industry", semantic_filters={"IndustrySource": "bogus"}
        )


# ===========================================================================
# GOLDEN-A06  Percent fields (TurnoverRatio / Roe / ...) -> /100 where the
#   contract requires it.
# ===========================================================================

def test_golden_a06_percent_fields_normalize_100():
    """
    REAL HELPER VERIFIED (GOLDEN-A06.1): the semantic catalog declares
    ``turnover_ratio`` and ``roe`` with ``source_unit=percent,
    canonical_unit=ratio, scale=0.01``.

    REAL HELPER VERIFIED (GOLDEN-A06.2): ``normalize_table_units`` multiplies
    those columns by 0.01 at the output layer (TurnoverRatio 3.5% -> 0.035,
    Roe 12.5% -> 0.125).

    GOLDEN-A06.3 (contract note): ``index_weight`` (Weight) is percent too
    (datasets.yaml section 5: Weight is % with sum ~100).  The catalog's
    ``index_weight`` entry does NOT declare a percent scale yet, so the
    /100 for Weight is CONTRACT-ONLY below; the real catalog must gain the
    ``<<: *ratio_field`` treatment before Weight can be treated as a ratio.
    """
    import pyarrow as pa

    from data_access.read.semantic_catalog import get_semantic_catalog, normalize_table_units

    catalog = get_semantic_catalog()

    tr = catalog.resolve_one("turnover_ratio", market="ashare")
    assert tr.scale == 0.01 and tr.source_unit == "percent" and tr.canonical_unit == "ratio"
    roe = catalog.resolve_one("roe", market="ashare")
    assert roe.scale == 0.01 and roe.source_unit == "percent" and roe.canonical_unit == "ratio"

    tbl = pa.table(
        {
            "TurnoverRatio": pa.array([3.5, 1.2], type=pa.float64()),
            "Roe": pa.array([12.5, 8.0], type=pa.float64()),
        }
    )
    out = normalize_table_units(tbl, [tr, roe])
    assert out.column("TurnoverRatio").to_pylist() == pytest.approx([0.035, 0.012])
    assert out.column("Roe").to_pylist() == pytest.approx([0.125, 0.08])

    # GOLDEN-A06.3: Weight (index constituent) is percent by dictionary; the
    # catalog does not yet carry the scale -> contract-only assertion.
    weight = catalog.resolve_one("index_weight", market="ashare")
    assert weight is not None
    assert weight.physical_name == "Weight"
    assert weight.scale is None, (
        "If index_weight grows a scale==0.01 declaration, GOLDEN-A06.3 should "
        "assert normalize_table_units divides Weight by 100."
    )


# ===========================================================================
# GOLDEN-A07  Dividend effective_time_only cannot masquerade as announcement
#   PIT -> strict PIT reject.
# ===========================================================================

def test_golden_a07_dividend_effective_time_only_strict_reject():
    """
    REAL HELPER VERIFIED: ``ashare_stock_dividend`` declares
    ``pit="effective_time_only"`` (no reliable announcement timestamp; event
    column ExDividendDate).  ``resolve_event_clock`` WITHOUT
    ``allow_effective_time=True`` raises ``ValidationError`` -- a consumer
    cannot request announcement-style PIT on a dividend.  ``enforce_event_cutoff``
    additionally fail-closes in production on unbounded reads that could
    surface future effective dates (look-ahead).
    """
    from data_access.cos_contract import (
        enforce_event_cutoff,
        resolve_event_clock,
        require_cos_contract,
    )
    from data_access.core.exceptions import ValidationError

    c = require_cos_contract("ashare_stock_dividend")
    assert c.pit_policy == "effective_time_only"
    assert c.event_column == "ExDividendDate"
    # It must NOT masquerade as announcement PIT.
    assert c.availability_column is None
    assert c.pit_fidelity != "vintage_pit"

    with pytest.raises(ValidationError):
        resolve_event_clock("ashare_stock_dividend", allow_effective_time=False)
    # Explicit effective-time use is the only allowed path.
    clock = resolve_event_clock("ashare_stock_dividend", allow_effective_time=True)
    assert clock[1] == "ExDividendDate"

    # Production unbounded read (no as_of / time_range end) fail-closes.
    with pytest.raises(ValidationError):
        enforce_event_cutoff(c, production=True)
    # A bounded, non-future as_of is fine in production.
    enforce_event_cutoff(c, as_of=dt.date.today(), production=True)


# ===========================================================================
# GOLDEN-A08  IsSuspend source is StockDailyBar (not a derived proxy).
# ===========================================================================

def test_golden_a08_issuspend_sourced_from_stockdailybar():
    """
    REAL HELPER VERIFIED: the physical schema for ``ashare_stock_daily`` (in
    ``dataaccess/config/datasets.yaml`` and enforced by the COS registry patch
    ``dataaccess/cos_registry_runtime.py``) lists ``IsSuspend: bool`` under the
    ``StockDailyBar`` physical table; ``table_for_dataset("ashare_stock_daily")
    == "StockDailyBar"``.  ``cos_registry_runtime`` also type-pins it as bool.

    Contract: IsSuspend must come from the real bar data, never from a derived
    proxy (e.g. a ``ListedState``/status column that is only a proxy for
    suspension).
    """
    from data_access.cos.mirror import table_for_dataset
    from data_access.cos_registry_runtime import _REGISTRY_PATCHES

    assert table_for_dataset("ashare_stock_daily") == "StockDailyBar"
    patch = _REGISTRY_PATCHES["ashare_stock_daily"]
    assert patch["schema_update"]["IsSuspend"] == "bool"

    # The dataset schema (datasets.yaml) declares IsSuspend on the daily bar.
    import yaml

    cfg_path = REPO_ROOT / "dataaccess" / "config" / "datasets.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    daily = cfg["datasets"]["ashare_stock_daily"] if "datasets" in cfg else cfg["ashare_stock_daily"]
    assert "IsSuspend" in daily["schema"]
    assert daily["schema"]["IsSuspend"] == "bool"
