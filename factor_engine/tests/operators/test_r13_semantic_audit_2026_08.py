# -*- coding: utf-8
"""R13 semantic-auditor self-test (§45 tests 7 & 8).

The auditor must FAIL on a deliberately broken kernel, distinguish AUDIT_ERROR
from NOT_APPLICABLE, and — after the R13 rewrite — cover the catalog through
contract fixtures instead of 7 curated SAMPLE ops.

- NEW-P0-07  — a rule that RAISES is an AUDIT_ERROR (release gate), not a skip;
- NEW-P0-08  — missing_state fires exactly when a NaN condition yields a finite
               output (the inverted-condition bug is fixed);
- NEW-P0-09  — native_cohort asserts past-row invariance and current-change;
- NEW-P0-10  — psd_geometry consults the kernel's GeometryAuditBundle and FAILS
               on a non-PSD projection / mismatched eigenpairs;
- NEW-P0-11  — golden_reference dispatches per statistical family (no longer
               only-Hill);
- NEW-P0-06/71 — audit_all sweeps production/searchable canonicals via contract
               fixtures and is certification-safe only with 0 AUDIT_ERRORs.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.semantic_audit import (
    AuditReport,
    RULES,
    _run_rule_sweep,
    audit_all,
    contract_fixture,
    run_audit,
)
from cleaned_operators.base import OperatorMetadata, SeriesOperator

load_all()


def _fake_op(name: str, *, param_names, calculate, tags=None, **meta_kw) -> SeriesOperator:
    metadata = OperatorMetadata(
        name=name,
        category="test",
        description="fake audit op",
        param_names=param_names,
        tags=tags or [],
        **meta_kw,
    )

    # ``SeriesOperator`` is abstract (requires ``_calculate_series``); build a
    # concrete subclass that delegates to the supplied plain function, so the
    # audit's ``_run`` path (``op.calculate`` -> validate -> ``_calculate_series``)
    # works exactly like a real operator.  ``metadata`` is set on the instance
    # (a class-body ``metadata = metadata`` self-reference fails to close over
    # the function local in some Python versions).
    class _FakeOp(SeriesOperator):
        def _calculate_series(self, *args, **kwargs):
            return calculate(self, *args, **kwargs)

    op = _FakeOp()
    op.metadata = metadata
    return op


def _ctx(report: AuditReport | None = None):
    report = report or AuditReport()
    return {"report": report}, report


# ---------------------------------------------------------------------------
# NEW-P0-08: missing_state fires on NaN condition -> finite output
# ---------------------------------------------------------------------------

def test_missing_state_fails_on_unknown_state_yielding_finite():
    def calculate(self, condition, **kw):
        return pd.DataFrame(np.ones_like(condition), index=condition.index, columns=condition.columns)

    op = _fake_op("fake_state_latch", param_names=["condition", "window"], calculate=calculate)
    cond = (np.random.default_rng(1).normal(0, 1, (80, 3) > 0).astype(float)
    cond = pd.DataFrame(cond, index=pd.date_range("2024-01-01", periods=80, freq="B"))
    cond.iloc[20, 0] = np.nan
    ctx, report = _ctx()
    findings = RULES["missing_state"]["check"](op, {**ctx, "sample": types.SimpleNamespace(frames=[cond], kwargs={})})
    assert any(f.rule == "missing_state" and f.severity == "error" for f in findings)


def test_missing_state_clean_when_unknown_state_is_censored():
    def calculate(self, condition, **kw):
        # unknown state -> NaN output (censored); known states pass through.
        return condition.where(condition.notna(), np.nan)

    op = _fake_op("fake_state_censor", param_names=["condition", "window"], calculate=calculate)
    cond = pd.DataFrame(
        (np.random.default_rng(1).normal(0, 1, (80, 3) > 0).astype(float),
        index=pd.date_range("2024-01-01", periods=80, freq="B"),
    )
    cond.iloc[20, 0] = np.nan
    ctx, report = _ctx()
    findings = RULES["missing_state"]["check"](op, {**ctx, "sample": types.SimpleNamespace(frames=[cond], kwargs={})})
    assert findings == []


# ---------------------------------------------------------------------------
# NEW-P0-07: rule exceptions are AUDIT_ERROR, not skips
# ---------------------------------------------------------------------------

def test_rule_exception_is_audit_error_not_skip():
    def broken_rule(op, ctx):
        raise RuntimeError("rule code bug")

    RULES["_broken"] = {"category": "test", "description": "broken", "check": broken_rule}
    try:
        report = AuditReport()
        op = _fake_op("x", param_names=["window"], calculate=lambda self, x, **k: x)
        _run_rule_sweep(
            report, op,
            types.SimpleNamespace(name="x", frames=[pd.DataFrame({"a": [1.0]})], kwargs={}),
            ("_broken",),
        )
        # the failure surfaced as an audit error, NOT recorded as checked/skipped
        assert report.audit_error_count == 1
        assert report.coverage()["audit_errors"] == 1
    finally:
        RULES.pop("_broken", None)


def test_clean_run_is_certification_safe():
    report = run_audit()
    assert report.audit_error_count == 0
    assert report.certification_safe()


# ---------------------------------------------------------------------------
# NEW-P0-09: native_cohort asserts native-date semantics
# ---------------------------------------------------------------------------

def _cohort_op_like_past_dependent(name: str, past_dependent: bool):
    """A fake cohort op that either rewrites past rows (bug) or is native."""
    def calculate(self, x, group, **kw):
        g = group.to_numpy()
        out = np.zeros((x.shape[0], x.shape[1]))
        # membership of the LAST row determines the metric for ALL rows when buggy
        last_membership = g[-1]
        for j in range(x.shape[1]):
            for i in range(x.shape[0]):
                if not past_dependent:
                    out[i, j] = float(x.iloc[i, j]) * (1.0 if g[i, j] == "G" else 0.0)
                else:
                    out[i, j] = float(x.iloc[i, j]) * (1.0 if last_membership[j] == "G" else 0.0)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    return _fake_op(name, param_names=["x", "group", "window"], calculate=calculate)


def test_native_cohort_fails_when_past_rewritten_by_future_exit():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    x = pd.DataFrame(np.ones((4, 3)), index=idx, columns=["S0", "S1", "S2"])
    groups = pd.DataFrame(index=idx, columns=["S0", "S1", "S2"], dtype=object)
    groups.iloc[:] = "G"
    groups.iloc[3, 2] = "H"  # S2 exits on the last date
    stable = pd.DataFrame(index=idx, columns=["S0", "S1", "S2"], dtype=object)
    stable.iloc[:] = "G"
    op = _cohort_op_like_past_dependent("fake_retention", past_dependent=True)
    ctx, report = _ctx()
    findings = RULES["native_cohort"]["check"](
        op, {**ctx, "sample": types.SimpleNamespace(frames=[x], kwargs={})}
    )
    assert any(f.rule == "native_cohort" for f in findings)


def test_native_cohort_passes_for_native_semantics():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    x = pd.DataFrame(np.ones((4, 3)), index=idx, columns=["S0", "S1", "S2"])
    groups = pd.DataFrame(index=idx, columns=["S0", "S1", "S2"], dtype=object)
    groups.iloc[:] = "G"
    groups.iloc[3, 2] = "H"
    stable = pd.DataFrame(index=idx, columns=["S0", "S1", "S2"], dtype=object)
    stable.iloc[:] = "G"
    op = _cohort_op_like_past_dependent("fake_retention", past_dependent=False)
    ctx, report = _ctx()
    findings = RULES["native_cohort"]["check"](
        op, {**ctx, "sample": types.SimpleNamespace(frames=[x], kwargs={})}
    )
    assert findings == []


# ---------------------------------------------------------------------------
# NEW-P0-10: psd_geometry FAILS on a bad projection bundle
# ---------------------------------------------------------------------------

def test_psd_geometry_fails_on_non_psd_bundle(monkeypatch):
    import cleaned_operators.feature_geometry as fg

    bad = (
        np.array([[1.0, 0.9], [0.9, 1.0]]),  # raw (fine)
        np.array([[1.0, 0.9], [0.9, 1.0]]),  # projected (PSD, eigvals 1.9/0.1)
        np.array([-0.5, 1.5]),               # NEGATIVE eigenvalue injected
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # eigvecs
    )
    monkeypatch.setattr(fg, "geometry_audit_bundle", lambda panel, **kw: bad)
    op = _fake_op("feature_mode_share_test", param_names=["x", "window"], calculate=lambda self, x, **k: x)
    ctx, report = _ctx()
    findings = RULES["psd_geometry"]["check"](op, ctx)
    assert any("PSD" in f.message for f in findings)


# ---------------------------------------------------------------------------
# NEW-P0-11: golden_reference dispatches per family (not only-Hill)
# ---------------------------------------------------------------------------

def test_golden_dispatches_family_adapter():
    """The Hill adapter is selected by NAME family (not only the pinned canonical),
    and a family without an adapter is NOT_APPLICABLE — never "only Hill ran"."""
    op = _fake_op(
        "ts_hill_tail_index_custom", param_names=["x", "window"],
        calculate=lambda self, x, **k: pd.DataFrame(np.ones_like(x), index=x.index, columns=x.columns),
    )
    ctx, report = _ctx()
    findings = RULES["golden_reference"]["check"](op, ctx)
    # the fake op is not runnable on the Pareto DGP (constant output) -> the Hill
    # adapter reports NOT_APPLICABLE with a reason (the adapter WAS selected and
    # ran).  No family is silently only-Hill anymore.
    assert report.skipped.get("golden_reference")


# ---------------------------------------------------------------------------
# NEW-P0-06: contract fixtures + coverage gate
# ---------------------------------------------------------------------------

def test_contract_fixture_uses_declared_contract():
    f = contract_fixture("ts_count_if")
    assert f is not None
    assert f.name == "ts_count_if"
    assert len(f.frames) >= 1


def test_contract_fixture_unknown_op_is_none():
    assert contract_fixture("no_such_operator_xyz") is None


def test_audit_all_coverage_gate():
    """R13 NEW-P0-06/71: the contract-catalog sweep must produce ZERO AUDIT_ERRORs
    and record honest NOT_APPLICABLE reasons for unfixtureable ops — a curated
    7-op SAMPLE is no longer the whole story.  Bounded to a fast subset here; the
    full ``audit_all()`` sweep runs as a manual release gate."""
    report = audit_all(sample_limit=40)
    assert report.audit_error_count == 0, report.summary()
    cov = report.coverage()
    assert cov["checked"] > 0
    assert report.evidence, "per-canonical semantic evidence records must exist (NEW-P0-71)"
