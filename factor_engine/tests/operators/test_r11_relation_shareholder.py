# -*- coding: utf-8 -*-
"""R11 long-tail audit closure: relation/index/distribution semantics,
shareholder churn network, stateful rotation, coverage/imputation.

Covers findings #118-#130, #153-#157, #175-#179 that landed in this batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(canonical: str):
    return OperatorRegistry.get(canonical, backend="pandas_numpy")


def _holder_args(cur_r, cur_id, prev_r, prev_id, n_rows=1):
    """Build the 40 rank-slot panels (s1..s10, sid1..sid10, p1..p10, psid1..psid10).

    Each ``*_r`` / ``*_id`` is a 10-slot dict {slot_index: [row values]}; empty
    slots default to NaN ratios / None ids.
    """
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    out = []
    for group, empty in ((cur_r, np.nan), (cur_id, None), (prev_r, np.nan), (prev_id, None)):
        for slot in range(1, 11):
            values = group.get(slot)
            if values is None:
                values = [empty] * n_rows
            out.append(pd.DataFrame({"A": values}, index=dates))
    return out


# ---------------------------------------------------------------------------
# #118 relation weighted outputs carry correct units
# ---------------------------------------------------------------------------
def test_relation_weighted_output_units():
    # weighted mean -> same_as:value ; plain sum -> same_as:value
    for canon in ("relation_rank_weighted_sum", "relation_peer_weighted_mean_ex_self", "relation_topk_sum"):
        meta = _op(canon).metadata
        assert meta.output_unit == "same_as:value", canon
        assert any("unit:same_as:value" == t for t in (meta.tags or [])), canon
    # (Δvalue) * weight -> unit(value)*unit(weight)
    meta = _op("relation_weighted_change").metadata
    assert meta.output_unit == "unit(value)*unit(weight)"
    assert any("unit:unit(value)*unit(weight)" == t for t in (meta.tags or []))


# ---------------------------------------------------------------------------
# #119 index_entry_exit_event is signed state, not boolean
# ---------------------------------------------------------------------------
def test_index_entry_exit_event_signed_state():
    idx = pd.date_range("2024-01-01", periods=4)
    member = pd.DataFrame([[0.0], [1.0], [1.0], [0.0]], index=idx, columns=["A"])
    out = _op("index_entry_exit_event").calculate(member)
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[2, 0] == 0.0
    assert out.iloc[3, 0] == -1.0
    meta = _op("index_entry_exit_event").metadata
    assert meta.output_unit == "state_signed"
    assert any("unit:state_signed" == t for t in (meta.tags or []))
    assert any("signed_state" == t for t in (meta.tags or []))


# ---------------------------------------------------------------------------
# #120 missing_policy="false" is not a production membership interpretation
# ---------------------------------------------------------------------------
def test_relation_transition_missing_policy_false_rejected_at_contract():
    member = pd.DataFrame({"A": [1.0, np.nan, 1.0, 0.0]}, index=pd.date_range("2024-01-01", periods=4))
    # The legacy "false" interpretation is explicitly non-production: the call
    # contract rejects it (choices=("break",)).  The kernel still honours it for
    # the legacy direct-call path, but the public calculate() gate fails closed.
    with pytest.raises(Exception):
        _op("relation_entry_count").calculate(member, window=4, missing_policy="false")
    spec = _op("relation_entry_count").metadata.param_specs.get("missing_policy")
    assert spec is not None and spec.choices == ("break",)


# ---------------------------------------------------------------------------
# #121 complete-graph PageRank/diffusion resolved analytically (closed form)
# ---------------------------------------------------------------------------
def test_group_signal_attraction_share_is_analytic():
    # On the group's complete graph the PageRank iteration collapses to
    # pr[i] = (1-d)/m + d*sig[i]/S — verified against the closed form.
    idx = pd.date_range("2024-01-01", periods=1)
    x = pd.DataFrame([[1.0, 3.0, 2.0]], index=idx, columns=["A", "B", "C"])
    g = pd.DataFrame([["g", "g", "g"]], index=idx, columns=["A", "B", "C"])
    out = _op("group_signal_attraction_share").calculate(x, g, damping=0.85)
    S, d, m = 6.0, 0.85, 3
    expected = [0.15 / m + d * v / S for v in (1.0, 3.0, 2.0)]
    np.testing.assert_allclose(out.to_numpy()[0], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# #122 _stack_panels must not silently reindex
# ---------------------------------------------------------------------------
def test_stack_panels_mismatched_axes_raise():
    from cleaned_operators.relation import distribution as rdist

    base = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    mis_col = pd.DataFrame({"B": [1.0, 2.0]}, index=base.index)
    with pytest.raises(ValueError):
        rdist._stack_panels(base, mis_col)
    mis_idx = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-03", periods=3))
    with pytest.raises(ValueError):
        rdist._stack_panels(base, mis_idx)
    # The operator-level path fails closed too.
    with pytest.raises(ValueError):
        _op("relation_topk_concentration").calculate(base, mis_col, k=1)


# ---------------------------------------------------------------------------
# #123 relation distribution skew/kurtosis unit: dimensionless
# ---------------------------------------------------------------------------
def test_relation_skew_kurtosis_dimensionless():
    for canon in ("relation_distribution_skew", "relation_distribution_kurtosis",
                  "group_skewness", "group_kurtosis"):
        meta = _op(canon).metadata
        assert meta.output_unit == "dimensionless", canon


# ---------------------------------------------------------------------------
# #124 rank mobility slot-identity problem: slot vs entity mobility differ
# ---------------------------------------------------------------------------
def test_rank_mobility_slot_vs_entity_differ():
    idx2 = pd.date_range("2024-01-01", periods=2)
    # Slot panels: rank1 = [0.5, 0.5], rank2 = [0.3, 0.3] over two rows.
    r1 = pd.DataFrame({"inst": [0.5, 0.5]}, index=idx2)
    r2 = pd.DataFrame({"inst": [0.3, 0.3]}, index=idx2)
    slot = _op("relation_rank_mobility").calculate(r1, r2, window=1)
    assert slot.iloc[1, 0] == pytest.approx(0.0)  # slot values unchanged

    # Same holdings, pure rank swap (A/B swap slots) — entity values unchanged.
    idx = pd.date_range("2024-01-01", periods=1)
    cur_s = [pd.DataFrame({"inst": [v]}, index=idx) for v in (0.5, 0.3)]
    prev_s = [pd.DataFrame({"inst": [v]}, index=idx) for v in (0.5, 0.3)]
    cur_id = [pd.DataFrame({"inst": [v]}, index=idx) for v in ("A", "B")]
    prev_id = [pd.DataFrame({"inst": [v]}, index=idx) for v in ("B", "A")]
    args = (
        cur_s + [pd.DataFrame({"inst": [np.nan]}, index=idx)] * 8
        + cur_id + [pd.DataFrame({"inst": [None]}, index=idx)] * 8
        + prev_s + [pd.DataFrame({"inst": [np.nan]}, index=idx)] * 8
        + prev_id + [pd.DataFrame({"inst": [None]}, index=idx)] * 8
    )
    entity = _op("relation_rank_entity_mobility").calculate(*args)
    assert entity.iloc[0, 0] == pytest.approx(0.2)  # A: |0.5-0.3|, B: |0.3-0.5|
    # The two MUST differ (slot=0, entity=0.2) — that is the slot-identity point.
    assert abs(slot.iloc[1, 0] - entity.iloc[0, 0]) > 1e-9
    # Documented semantics: slot mobility is explicitly tagged slot_identity.
    assert any("slot_identity" == t for t in (_op("relation_rank_mobility").metadata.tags or []))


# ---------------------------------------------------------------------------
# #125 concentration acceleration history = 2 * window
# ---------------------------------------------------------------------------
def test_concentration_acceleration_history_formula():
    from runtime.execution_contract import history_requirement

    hr = history_requirement("relation_concentration_acceleration", {"window": 5})
    assert hr.kind == "finite"
    assert hr.rows == 10  # 2 * window


# ---------------------------------------------------------------------------
# #126 churn network entity ID must not be str(pd.NA)
# ---------------------------------------------------------------------------
def test_churn_network_na_id_is_not_a_node():
    op = _op("holder_id_matched_churn")
    # cur rank2 carries a pd.NA id — it must not become an "<NA>" node.
    args = _holder_args(
        {1: [0.5], 2: [0.3]},
        {1: ["h1"], 2: [pd.NA]},
        {1: [0.4], 2: [0.2]},
        {1: ["h1"], 2: ["h2"]},
    )
    out = op.calculate(*args)
    val = float(out.iloc[0, 0])
    # With the NA id excluded: cur={h1:0.5}, prev={h1:0.4,h2:0.2} -> 0.5*(0.1+0.2)=0.15
    assert val == pytest.approx(0.15)
    # If "<NA>" were a real node, cur={h1:0.5,"<NA>":0.3} -> 0.5*(0.1+0.3+0.2)=0.30
    assert val != pytest.approx(0.30)


# ---------------------------------------------------------------------------
# #127 structured entity key (HolderID is the node identity)
# ---------------------------------------------------------------------------
def test_churn_network_entity_key_is_holder_id():
    from cleaned_operators.shareholder.churn_network import _id_key

    # A holder row with a missing id is not a node; a real id is kept verbatim.
    assert _id_key(pd.NA) is None
    assert _id_key(pd.NaT) is None
    assert _id_key(np.nan) is None
    assert _id_key(None) is None
    assert _id_key("h1") == "h1"
    # Same id repeated with conflicting ratios fails closed (ambiguous share
    # natures / duplicate record) — holder id is the sole node key.
    conflict = _holder_args(
        {1: [0.5], 2: [0.2]},
        {1: ["h1"], 2: ["h1"]},
        {1: [0.5], 2: [0.3], 3: [0.2]},
        {1: ["h1"], 2: ["h2"], 3: ["h3"]},
    )
    out = _op("holder_id_matched_churn").calculate(*conflict)
    assert np.isnan(out.iloc[0, 0])


# ---------------------------------------------------------------------------
# #128 / #129 top-10 absence is a disclosure exit, not total shareholder exit
# ---------------------------------------------------------------------------
def test_top10_absence_is_disclosure_exit_naming():
    for canon in ("holder_exit_share", "holder_id_matched_exit_share"):
        entry = OperatorRegistry._catalog[canon]
        desc = entry.get("description", "")
        assert "披露" in desc or "top-K" in desc or "top K" in desc, (canon, desc)
        assert "非全体股东" in desc, (canon, desc)
    for canon in ("holder_entry_share", "holder_id_matched_entry_share"):
        desc = OperatorRegistry._catalog[canon].get("description", "")
        assert "披露" in desc or "top-K" in desc, (canon, desc)


# ---------------------------------------------------------------------------
# #130 pledge/freeze ratio domain enforcement
# ---------------------------------------------------------------------------
def test_pledge_freeze_ratio_domain():
    idx = pd.date_range("2024-01-01", periods=1)
    for canon in ("holder_pledge_ratio", "holder_freeze_ratio", "holder_locked_share_ratio"):
        op = _op(canon)
        # valid: 30 / 100 = 0.3
        valid = op.calculate(
            pd.DataFrame({"A": [30.0]}, index=idx), pd.DataFrame({"A": [100.0]}, index=idx)
        )
        assert valid.iloc[0, 0] == pytest.approx(0.3)
        # negative numerator -> domain violation -> NaN
        neg = op.calculate(
            pd.DataFrame({"A": [-5.0]}, index=idx), pd.DataFrame({"A": [100.0]}, index=idx)
        )
        assert np.isnan(neg.iloc[0, 0])
        # non-positive denominator -> NaN
        zero_den = op.calculate(
            pd.DataFrame({"A": [30.0]}, index=idx), pd.DataFrame({"A": [0.0]}, index=idx)
        )
        assert np.isnan(zero_den.iloc[0, 0])
        # ratio > 1 -> NaN
        over = op.calculate(
            pd.DataFrame({"A": [150.0]}, index=idx), pd.DataFrame({"A": [100.0]}, index=idx)
        )
        assert np.isnan(over.iloc[0, 0])


# ---------------------------------------------------------------------------
# #153 rank churn decomposition (intersection / composition / combined)
# ---------------------------------------------------------------------------
def test_rank_churn_decomposition_provided():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(0, 1, (30, 6)), columns=list("ABCDEF"))
    for canon in ("cs_rank_churn", "cs_rank_composition_churn", "cs_rank_combined_churn"):
        assert _op(canon) is not None
        out = _op(canon).calculate(x, lag=3)
        assert out.shape == x.shape
    # Composition churn of a FIXED pool with values changing is zero.
    idx = pd.date_range("2024-01-01", periods=4)
    fixed = pd.DataFrame(
        np.stack([np.arange(6.0) for _ in range(4)]), index=idx, columns=list("ABCDEF")
    )
    comp = _op("cs_rank_composition_churn").calculate(fixed, lag=1)
    assert comp.iloc[-1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# #154 / #155 / #156 tail retention: cohort explicit, real tail, exact top-k
# ---------------------------------------------------------------------------
def test_tail_retention_quantile_must_be_real_tail():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 1, (40, 6)), columns=list("ABCDEF"))
    with pytest.raises(ValueError):
        _op("cs_tail_retention").calculate(x, lag=3, quantile=0.8, side="top")
    with pytest.raises(ValueError):
        _op("cs_tail_retention").calculate(x, lag=3, quantile=0.0, side="top")
    # A real tail (q=0.2) runs and the effective breadth is ceil(q*N)=2.
    out = _op("cs_tail_retention").calculate(x, lag=3, quantile=0.2, side="top")
    assert out.notna().to_numpy().sum() > 0
    breadth = _op("cs_tail_breadth").calculate(x, quantile=0.2, side="top")
    arr = breadth.to_numpy(dtype=float)
    finite_rows = arr[np.isfinite(arr).any(axis=1)]
    if finite_rows.size:
        assert float(np.unique(finite_rows)[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# #157 GroupState / GlobalState typing
# ---------------------------------------------------------------------------
def test_group_state_global_state_typed():
    assert _op("cs_hartigan_dip").metadata.role == "global_state"
    assert any("global_state" == t for t in (_op("cs_hartigan_dip").metadata.tags or []))
    for canon in ("cs_rank_churn", "cs_tail_retention",
                  "cs_rank_composition_churn", "cs_rank_combined_churn",
                  "cs_tail_breadth", "cs_physical_panel_coverage",
                  "cs_universe_coverage"):
        meta = _op(canon).metadata
        assert meta.role in ("group_state", "global_state"), canon


# ---------------------------------------------------------------------------
# #175-177 cs_impute min_finite must actually gate imputation
# ---------------------------------------------------------------------------
def test_cs_impute_min_finite_gates():
    idx = pd.date_range("2024-01-01", periods=2)
    x = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [np.nan, np.nan], "C": [np.nan, 5.0], "D": [np.nan, np.nan]},
        index=idx,
    )
    op = _op("cs_impute_median")
    strict = op.calculate(x, min_finite=3)
    # row0 has 1 finite (<3) and row1 has 1 finite (<3) -> nothing imputed.
    assert np.isnan(strict.iloc[0, 1]) and np.isnan(strict.iloc[1, 1])
    loose = op.calculate(x, min_finite=1)
    assert loose.iloc[0, 1] == pytest.approx(1.0)
    assert loose.iloc[1, 1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# #178 coverage denominator: physical panel vs declared universe
# ---------------------------------------------------------------------------
def test_coverage_physical_vs_universe_split():
    idx = pd.date_range("2024-01-01", periods=2)
    x = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [2.0, 3.0], "C": [np.nan, 4.0]}, index=idx
    )
    # Universe excludes C (delisted / out-of-universe column).
    universe = pd.DataFrame({"A": [1, 1], "B": [1, 0], "C": [0, 1]}, index=idx, dtype=float)

    phys = _op("cs_physical_panel_coverage").calculate(x)
    univ = _op("cs_universe_coverage").calculate(x, universe)

    # Physical panel coverage divides by ALL 3 columns present.
    assert phys.iloc[0, 0] == pytest.approx(2.0 / 3.0)
    assert phys.iloc[1, 0] == pytest.approx(2.0 / 3.0)
    # Universe coverage divides by in-universe names only:
    # row0 universe={A,B} both finite -> 2/2 ; row1 universe={A,C}, A missing -> 1/2.
    assert univ.iloc[0, 0] == pytest.approx(1.0)
    assert univ.iloc[1, 0] == pytest.approx(0.5)
    # They must differ on at least one row — that is the split.
    assert phys.iloc[0, 0] != pytest.approx(univ.iloc[0, 0])
    assert phys.iloc[1, 0] != pytest.approx(univ.iloc[1, 0])


# ---------------------------------------------------------------------------
# #179 generic ffill is gated by the field's forward-fill allowance
# ---------------------------------------------------------------------------
def test_ffill_forward_fill_allowance_gate():
    # ``ffill`` is a DSL-level operator that layer_governance unregisters from
    # the final OperatorRegistry (unlimited forward fill is not operational-
    # production), so the contract gate is asserted on the operator class.
    from cleaned_operators.common.data_cleaning import FillForward

    op = FillForward()
    idx = pd.date_range("2024-01-01", periods=4)
    x = pd.DataFrame({"A": [1.0, np.nan, np.nan, 4.0]}, index=idx)
    assert op._calculate_series(x)["A"].tolist() == [1.0, 1.0, 1.0, 4.0]
    # A field that does not allow forward fill (returns / events / revisions)
    # stays missing — fail closed, never a stale value.
    blocked = op._calculate_series(x, forward_fill_allowed=False)
    assert np.isnan(blocked["A"].iloc[1]) and np.isnan(blocked["A"].iloc[2])
    # A bounded gap limit prevents long-stale carries.
    bounded = op._calculate_series(x, max_ffill_gap=1)
    bvals = bounded["A"].tolist()
    assert bvals[0] == 1.0 and bvals[1] == 1.0 and bvals[3] == 4.0
    assert np.isnan(bvals[2])
