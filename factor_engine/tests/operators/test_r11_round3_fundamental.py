# -*- coding: utf-8 -*-
"""Round-3 operator audit items 27-30 (fundamental / holder ownership).

* Item 27 — fundamental ledger must support post-release restatements: a
  ``daily x`` + ``daily period_id`` table is not enough; the ledger records a
  release row and every supersede/revision row per period and exposes a PIT
  ``value_as_of`` lookup so operators pick the value valid on the decision date
  and never blend the original and revised values.
* Item 28 — holder/ownership operators expose effective coverage (how many
  holders are observed / how much equity they cover) so a sparse-holder report
  is not treated as a strong signal.
* Item 29 — consecutive-period semantics: a consecutive pair must be two
  DISTINCT periods of the SAME report type with a known prior value, never two
  revisions of the same period.
* Item 30 — threshold-relative score operators declare input semantics
  (Return/Rate/flow/stock vs raw Price/Volume) as ``input_units`` / tags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry

# Import the OWNED operator modules directly (round-3 file-disjoint rule).  This
# keeps the suite green even while a concurrent session is mid-edit on the
# shared layer-governance surface (the full ``load_all`` finalize can be
# transiently blocked by another session's in-progress surface changes).
import factor_engine.cleaned_operators.fundamental.transforms_v2  # noqa: F401  (fin_* + ledger)
import factor_engine.cleaned_operators.fundamental.ledger  # noqa: F401  (fin_period_restated/…)
import factor_engine.cleaned_operators.fundamental.quality_v2  # noqa: F401
import factor_engine.cleaned_operators.fundamental.accruals_scores  # noqa: F401  (piotroski/altman/strength)
import factor_engine.cleaned_operators.fundamental.ops  # noqa: F401  (operating_margin/current_ratio/…)
import factor_engine.cleaned_operators.fundamental.polars_fundamental  # noqa: F401  (polars twins)
import factor_engine.cleaned_operators.shareholder.churn_network  # noqa: F401  (holder coverage)


def _op(canonical: str, backend: str = "pandas_numpy"):
    return OperatorRegistry.get(canonical, backend=backend)


def _panel(periods, values, columns=("S0",)):
    """Daily as-of panel + row-aligned report-period panel."""
    idx = pd.date_range("2025-01-01", periods=len(periods), freq="B")
    x = pd.DataFrame(np.asarray(values, dtype=float)[:, None], index=idx, columns=list(columns))
    pid = pd.DataFrame(np.asarray(periods, dtype=object)[:, None], index=idx, columns=list(columns))
    return x, pid


# ---------------------------------------------------------------------------
# Item 27 — fundamental revision ledger (release + supersede timestamps)
# ---------------------------------------------------------------------------

def _restatement_panel():
    # 2025Q1 released at 100 (rows 0-1), 2025Q2 released at 110 (rows 2-3),
    # 2025Q1 RESTATED to 150 at row 4, then 2025Q2 resumes at 110.
    periods = ["2025Q1", "2025Q1", "2025Q2", "2025Q2", "2025Q1", "2025Q2", "2025Q2"]
    values = [100.0, 100.0, 110.0, 110.0, 150.0, 110.0, 110.0]
    return _panel(periods, values)


def test_period_ledger_records_release_and_revision_rows():
    from factor_engine.cleaned_operators.fundamental.ledger import (
        PeriodLedgerEntry,
        _period_key,
        scan_period_ledger,
    )

    x, pid = _restatement_panel()
    entries = scan_period_ledger(x["S0"], pid["S0"])
    q1 = entries[_period_key("2025Q1")]
    q2 = entries[_period_key("2025Q2")]
    assert isinstance(q1, PeriodLedgerEntry)
    # release timestamps
    assert q1.first_seen_at == 0
    assert q2.first_seen_at == 2
    # revision timestamps
    assert q1.revision_count == 1
    assert q1.last_revised_at == 4
    assert q1.is_restated
    assert not q2.is_restated
    # observation history (original value retained, never blended)
    assert tuple(obs.value for obs in q1.observations) == (100.0, 150.0)
    assert tuple(obs.observed_at for obs in q1.observations) == (0, 4)


def test_period_ledger_value_as_of_picks_revision_valid_on_decision_date():
    from factor_engine.cleaned_operators.fundamental.ledger import _period_key, scan_period_ledger

    x, pid = _restatement_panel()
    entries = scan_period_ledger(x["S0"], pid["S0"])
    q1 = entries[_period_key("2025Q1")]
    # Before the revision row (decision rows 0..3) -> original value.
    assert q1.value_as_of(0) == pytest.approx(100.0)
    assert q1.value_as_of(3) == pytest.approx(100.0)
    # At/after the revision row (decision row 4) -> revised value.
    assert q1.value_as_of(4) == pytest.approx(150.0)
    assert q1.value_as_of(6) == pytest.approx(150.0)
    # first/latest accessors never blend.
    assert q1.first_value == pytest.approx(100.0)
    assert q1.latest_value == pytest.approx(150.0)


def test_period_restated_operator_is_as_of_flag():
    x, pid = _restatement_panel()
    out = _op("fin_period_restated").calculate(x, pid)
    vals = out["S0"].tolist()
    # 2025Q1: released 0,0; 2025Q2 released 0,0; Q1 restated at row4 -> 1; Q2 0,0.
    assert vals[:4] == [0.0, 0.0, 0.0, 0.0]
    assert vals[4] == 1.0
    assert vals[5:] == [0.0, 0.0]


def test_period_revision_count_operator():
    x, pid = _restatement_panel()
    out = _op("fin_period_revision_count").calculate(x, pid)
    vals = out["S0"].tolist()
    assert vals[0] == 0.0
    assert vals[4] == 1.0  # one supersede event for 2025Q1
    assert vals[5] == 0.0  # 2025Q2 never revised


def test_period_revision_age_operator():
    x, pid = _restatement_panel()
    out = _op("fin_period_revision_age").calculate(x, pid, max_days=504)
    vals = out["S0"].tolist()
    # 2025Q1 never revised at rows 0-1 -> NaN; revised at row 4 -> 0.
    assert np.isnan(vals[0])
    assert vals[4] == 0.0
    # 2025Q2 never revised -> NaN.
    assert np.isnan(vals[5])


def test_restatement_does_not_silently_blend_into_period_change():
    # ``fin_diff`` at the revision row must NOT pair the revised value against
    # the same period's original value (it uses the prior DISTINCT ordinal).
    x, pid = _restatement_panel()
    out = _op("fin_diff").calculate(x, pid, periods=1)
    vals = out["S0"].tolist()
    # row 4 current=2025Q1(150), prior ordinal 2024Q4 is not visible -> NaN.
    assert np.isnan(vals[4])
    # row 5 current=2025Q2(110) vs prior 2025Q1 (REVISED 150, visible since row4).
    assert vals[5] == pytest.approx(110.0 - 150.0)
    # row 3 (before the restatement was disclosed) used the ORIGINAL 100.
    assert vals[3] == pytest.approx(110.0 - 100.0)


def test_revision_ledger_fails_closed_on_unparseable_period():
    from factor_engine.cleaned_operators.fundamental.ledger import scan_period_ledger

    x, pid = _panel(["UNPARSABLE", "UNPARSABLE"], [1.0, 2.0])
    entries = scan_period_ledger(x["S0"], pid["S0"], require_parseable=True)
    assert entries == {}


# ---------------------------------------------------------------------------
# Item 28 — holder coverage
# ---------------------------------------------------------------------------

def _holder_args(cur_r, cur_id, n_rows=1):
    """Build the 20 rank-slot panels (s1..s10, sid1..sid10)."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    out = []
    for group, empty in ((cur_r, np.nan), (cur_id, None)):
        for slot in range(1, 11):
            values = group.get(slot)
            if values is None:
                values = [empty] * n_rows
            out.append(pd.DataFrame({"A": values}, index=dates))
    return out


def test_holder_disclosure_count_coverage_share_sum():
    args = _holder_args({1: [0.5], 2: [0.3]}, {1: ["h1"], 2: ["h2"]})
    count = _op("holder_disclosure_count").calculate(*args)
    coverage = _op("holder_disclosure_coverage").calculate(*args)
    share = _op("holder_topk_share_sum").calculate(*args)
    assert count.iloc[0, 0] == pytest.approx(2.0)
    assert coverage.iloc[0, 0] == pytest.approx(0.2)  # 2 / 10 top-K slots
    assert share.iloc[0, 0] == pytest.approx(0.8)


def test_holder_coverage_sparse_disclosure_is_visible():
    # Only 1 of the 10 slots populated -> coverage 0.1, count 1.
    args = _holder_args({1: [0.5]}, {1: ["h1"]})
    count = _op("holder_disclosure_count").calculate(*args)
    coverage = _op("holder_disclosure_coverage").calculate(*args)
    assert count.iloc[0, 0] == pytest.approx(1.0)
    assert coverage.iloc[0, 0] == pytest.approx(0.1)


def test_holder_coverage_empty_disclosure_is_zero_not_strong_signal():
    args = _holder_args({}, {})
    count = _op("holder_disclosure_count").calculate(*args)
    coverage = _op("holder_disclosure_coverage").calculate(*args)
    share = _op("holder_topk_share_sum").calculate(*args)
    assert count.iloc[0, 0] == pytest.approx(0.0)
    assert coverage.iloc[0, 0] == pytest.approx(0.0)
    assert share.iloc[0, 0] == pytest.approx(0.0)


def test_holder_coverage_unknown_ratio_fails_closed():
    # h2 has an ID but no ShareRatio -> unknown holding, not a confirmed 0%.
    args = _holder_args({1: [0.5]}, {1: ["h1"], 2: ["h2"]})
    for canon in ("holder_disclosure_count", "holder_disclosure_coverage", "holder_topk_share_sum"):
        out = _op(canon).calculate(*args)
        assert np.isnan(out.iloc[0, 0]), canon


def test_holder_coverage_conflicting_duplicate_fails_closed():
    # Same ID repeated with conflicting ratios -> ambiguous snapshot.
    args = _holder_args({1: [0.5], 2: [0.3]}, {1: ["h1"], 2: ["h1"]})
    out = _op("holder_disclosure_count").calculate(*args)
    assert np.isnan(out.iloc[0, 0])


def test_holder_coverage_duplicate_consistent_id_counts_once():
    # Same ID repeated with the SAME ratio is a plain duplicate -> count once.
    args = _holder_args({1: [0.5], 2: [0.5]}, {1: ["h1"], 2: ["h1"]})
    count = _op("holder_disclosure_count").calculate(*args)
    coverage = _op("holder_disclosure_coverage").calculate(*args)
    assert count.iloc[0, 0] == pytest.approx(1.0)
    assert coverage.iloc[0, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Item 29 — consecutive-period semantics (distinct periods, never revisions)
# ---------------------------------------------------------------------------

def test_consecutive_pair_is_distinct_period_never_two_revisions():
    # Two revisions of the SAME period must never form a "consecutive pair".
    x, pid = _restatement_panel()
    out = _op("fin_pct_change").calculate(x, pid, periods=1)
    vals = out["S0"].tolist()
    # row 4 (2025Q1 restated) has no distinct prior period visible -> NaN.
    assert np.isnan(vals[4])
    # row 5 (2025Q2) uses the distinct prior period 2025Q1 (revised 150).
    assert vals[5] == pytest.approx(110.0 / 150.0 - 1.0)


@pytest.mark.parametrize("canonical", ["fin_ttm", "fin_average_balance", "fin_monotonicity", "fin_sign_change_count"])
def test_consecutive_window_fails_closed_on_skipped_period(canonical):
    # 2025Q3 is skipped: 2025Q1 -> 2025Q2 -> 2025Q4.  A consecutive window
    # spanning the gap must fail closed to NaN rather than bridge Q2 and Q4.
    periods = ["2025Q1", "2025Q1", "2025Q2", "2025Q2", "2025Q4", "2025Q4"]
    values = [10.0, 10.0, 20.0, 20.0, 40.0, 40.0]
    x, pid = _panel(periods, values)
    kwargs = {"periods_per_year": 4} if canonical == "fin_ttm" else {"periods": 4 if canonical != "fin_sign_change_count" else 3}
    out = _op(canonical).calculate(x, pid, **kwargs)
    # At the last 2025Q4 row the ordinal window [Q1,Q2,Q4] is not contiguous.
    assert np.isnan(out["S0"].iloc[-1]), canonical


def test_consecutive_semantics_pandas_polars_parity_on_gap():
    pl = pytest.importorskip("polars")
    periods = ["2025Q1", "2025Q1", "2025Q2", "2025Q2", "2025Q4", "2025Q4"]
    values = [10.0, 10.0, 20.0, 20.0, 40.0, 40.0]
    x, pid = _panel(periods, values)

    def to_polars(frame):
        data = {}
        for c in frame.columns:
            if frame[c].dtype == object:
                data[c] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in frame[c].tolist()]
            else:
                data[c] = frame[c].to_numpy()
        return pl.DataFrame(data)

    for canonical, kwargs in (
        ("fin_ttm", {"periods_per_year": 4}),
        ("fin_average_balance", {"periods": 4}),
        ("fin_monotonicity", {"periods": 4}),
    ):
        pd_out = _op(canonical, "pandas_numpy").calculate(x, pid, **kwargs)
        pl_out = _op(canonical, "polars").calculate(to_polars(x), to_polars(pid), **kwargs)
        np.testing.assert_allclose(
            pd_out["S0"].to_numpy(),
            pl_out["S0"].to_numpy(),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )
    # The final (gap-spanning) row is NaN in BOTH backends.
    pd_out = _op("fin_ttm", "pandas_numpy").calculate(x, pid, periods_per_year=4)
    assert np.isnan(pd_out["S0"].iloc[-1])


# ---------------------------------------------------------------------------
# Item 30 — input-semantic restriction on threshold-relative score operators
# ---------------------------------------------------------------------------

def test_threshold_score_ops_declare_rate_input_units():
    units = _op("piotroski_f_score").metadata.input_units
    assert units["roa"] == "rate"
    assert units["ocf"] == "rate"
    assert units["total_capital"] == "stock"
    assert "input_semantics:rate" in set(_op("piotroski_f_score").metadata.tags)


def test_growth_ops_declare_rate_not_price_volume():
    for canon in ("fin_pct_change", "fin_qoq", "fin_yoy", "fin_growth", "fin_cagr"):
        meta = _op(canon).metadata
        assert meta.input_units.get("x") == "rate", canon
        assert "input_semantics:rate" in set(meta.tags), canon
    # fin_ttm takes single-period FLOW (a sum), not a raw price/volume.
    assert _op("fin_ttm").metadata.input_units.get("x") == "flow"
    # fin_average_balance takes a balance-sheet STOCK.
    assert _op("fin_average_balance").metadata.input_units.get("x") == "stock"


def test_ratio_ops_declare_flow_stock_input_units():
    # ``operating_margin``/``current_ratio`` are registered from the legacy
    # fundamental ``ops.py`` but are NOT surfaced as extended-only canonicals,
    # so post-``load_all`` governance prunes them from the registry.  Assert the
    # metadata on the operator CLASS directly.
    from factor_engine.cleaned_operators.fundamental.ops import (
        CurrentRatioOp,
        DebtToEquityOp,
        OperatingMarginOp,
        QuickRatioOp,
    )

    assert OperatingMarginOp.metadata.input_units == {
        "operating_income": "flow", "revenue": "flow",
    }
    assert CurrentRatioOp.metadata.input_units == {
        "current_assets": "stock", "current_liabilities": "stock",
    }
    assert QuickRatioOp.metadata.input_units["inventory"] == "stock"
    assert DebtToEquityOp.metadata.input_units["total_equity"] == "stock"
    # ``holder_pledge_ratio`` IS surfaced (churn_network ``_mk``) so it stays
    # reachable through the post-governance registry.
    assert _op("holder_pledge_ratio").metadata.input_units == {
        "pledge_shares": "shares", "total_capital": "shares",
    }


def test_field_contract_semantic_type_declared():
    from factor_engine.cleaned_operators.fundamental.field_contract import (
        FUNDAMENTAL_RATIO_FIELD_CONTRACTS,
    )

    opm = FUNDAMENTAL_RATIO_FIELD_CONTRACTS["operating_margin"]
    assert opm["operating_income"].semantic_type == "flow"
    cr = FUNDAMENTAL_RATIO_FIELD_CONTRACTS["current_ratio"]
    assert cr["current_assets"].semantic_type == "stock"
    pf = FUNDAMENTAL_RATIO_FIELD_CONTRACTS["piotroski_f_score"]
    assert pf["roa"].semantic_type == "rate"
    assert pf["total_capital"].semantic_type == "stock"
