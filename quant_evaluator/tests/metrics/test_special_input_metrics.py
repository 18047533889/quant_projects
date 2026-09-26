"""Special-input metrics: default-suite fixtures for the 12 metrics the
comprehensive audit could not previously measure (102/140 -> 114/140).

Each family pins the exact fixture shape that unblocks the metrics, so a
future contract tightening fails here instead of silently dropping coverage:

1. Generalization evidence (8 metrics, plan §13.6): a single-factor
   ``TrainVsValidationArtifact`` whose factor ids/versions are aligned with
   the evaluation request (``("prof",)`` + authoritative
   ``context_refs.factor_versions``).  Mismatched evidence axes (e.g.
   ("f0","f1") against a ("prof",) request) are rejected by the runtime.
2. worst_calendar_month/quarter/year: a session-aligned datetime fixture —
   the probe time axis is tz-aware UTC instants (04:00Z == 12:00 local in the
   snapshot's Asia/Shanghai timezone) whose local dates all sit inside the
   ``CalendarSnapshot`` trading days, which are buffered into 2023-12 /
   2025-01 so full calendar periods are coverage-bracketed.  An integer time
   axis is rejected ("timezone-aware datetime instants"); a tz-aware axis not
   matching the label/decision coordinates is rejected ("Portfolio time
   coordinates do not match").
3. turnover_cost: benchmark/capital metrics require an explicitly tagged
   execution trajectory leg — a typed ``ExecutablePortfolioArtifact`` with
   ``provenance["leg"] == "cost_drag"``, execution certification, and
   NET_EXECUTABLE cost scope, whose values are non-negative cost-rate
   magnitudes (not PnL).  Formula (docs/METRIC_REFERENCE.md,
   ``## turnover_cost``): Cost_bp = 1e4 * mean(c_t over observed).

Every test asserts (a) the metric evaluates through the public ``evaluate``
entry point, (b) determinism — two calls produce byte-identical value
payloads, and (c) a hand-computed oracle from the documented formula.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from data_access.r30.calendar_snapshot import CalendarSnapshot
from quant_evaluator.api.requests import MetricValue
from quant_evaluator.contracts.artifact_types import (
    ExecutablePortfolioArtifact,
    ProbePortfolioArtifact,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.metrics.calendar_returns import (
    compute_worst_calendar_month,
    compute_worst_calendar_quarter,
    compute_worst_calendar_year,
)
from quant_evaluator.metrics.generalization_evidence import (
    build_train_validation_artifact,
)
from quant_evaluator.metrics.turnover_cost import compute_turnover_cost

FACTOR_IDS = ("prof",)
FACTOR_VERSION = "v1"

# Generalization oracles from docs/METRIC_REFERENCE.md (plan §13.6):
# deltas are signed validation - train; retention is validation / train.
GEN_ORACLES = {
    "parameter_generalization": 1.0 / 1.2,
    "train_predictive_dimension": 1.2,
    "train_validation_icir_delta": 0.4 - 0.5,
    "train_validation_rankic_delta": 0.02 - 0.03,
    "train_validation_sharpe_delta": 1.2 - 1.5,
    "train_validation_shape_delta": 0.9 - 1.0,
    "validation_predictive_dimension": 1.0,
    "validation_retention": 1.0 / 1.2,
}


def _value_digest(bundle) -> str:
    """Byte-level determinism proxy: every metric value payload, sorted."""
    payload = {
        metric_id: {
            "value": mv.value,
            "valid": mv.valid,
            "observation_count": mv.observation_count,
            "metric_version": mv.metric_version,
        }
        for metric_id, mv in bundle.metric_values.items()
        if isinstance(mv, MetricValue)
    }
    return json.dumps(payload, sort_keys=True, allow_nan=True)


def _int_inputs(T=40, N=3, seed=7):
    rng = np.random.default_rng(seed)
    times = AxisRef("t", "int", T, np.arange(T))
    assets = AxisRef("a", "str", N, tuple(f"s{i}" for i in range(N)))
    fb = FactorBatch(
        FACTOR_IDS, times, assets,
        np.ascontiguousarray(rng.normal(size=(T, N, 1))),
        context_refs={"factor_versions": {FACTOR_IDS[0]: FACTOR_VERSION}},
    )
    lb = LabelBundle(
        target_id="r", values=np.ascontiguousarray(rng.normal(size=(T, N))),
        horizon=1, decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)), asset_axis=assets,
    )
    first = evaluate(fb, lb, metrics=("long_short_returns",))
    series = first.artifacts["long_short_returns"]
    probe = ProbePortfolioArtifact(
        np.ascontiguousarray(np.asarray(series.values, dtype=np.float64)),
        time_index=tuple(series.time_axis.time_index), factor_ids=FACTOR_IDS,
    )
    gen = build_train_validation_artifact(
        np.array([1.2]), np.array([1.0]),
        train_rankic=np.array([0.03]), validation_rankic=np.array([0.02]),
        train_icir=np.array([0.5]), validation_icir=np.array([0.4]),
        train_sharpe=np.array([1.5]), validation_sharpe=np.array([1.2]),
        train_shape=np.array([1.0]), validation_shape=np.array([0.9]),
        train_factor_ids=FACTOR_IDS, validation_factor_ids=FACTOR_IDS,
        train_factor_versions=(FACTOR_VERSION,),
        validation_factor_versions=(FACTOR_VERSION,),
        metric_instance="rank_ic:h1:daily",
        metric_instance_refs={
            "rankic": "audit:rankic", "icir": "audit:icir",
            "sharpe": "audit:sharpe", "shape": "audit:shape",
        },
    )
    cost = ExecutablePortfolioArtifact(
        np.ascontiguousarray(np.full((T, 1), 0.0005)),
        time_index=tuple(range(T)), factor_ids=FACTOR_IDS,
        provenance={
            "leg": "cost_drag", "execution_certified": True,
            "cost_scope": "NET_EXECUTABLE",
            "execution_ref": "special-inputs:execution-ledger",
        },
    )
    return fb, lb, probe, gen, cost


def _session_days():
    start, end = dt.date(2023, 12, 28), dt.date(2025, 1, 3)
    return [
        start + dt.timedelta(days=i)
        for i in range((end - start).days + 1)
        if np.is_busday(start + dt.timedelta(days=i))
    ]


def _calendar_inputs(seed=11, N=2):
    rng = np.random.default_rng(seed)
    days_all = _session_days()
    observed = [d for d in days_all if d.year == 2024]
    instants = tuple(
        dt.datetime.combine(d, dt.time(4, 0), tzinfo=dt.timezone.utc)
        for d in observed
    )
    Tc = len(observed)
    assets = AxisRef("a", "str", N, tuple(f"s{i}" for i in range(N)))
    fb = FactorBatch(
        FACTOR_IDS, AxisRef("t", "datetime", Tc, np.asarray(instants, dtype=object)),
        assets, np.ascontiguousarray(rng.normal(size=(Tc, N, 1))),
        context_refs={"factor_versions": {FACTOR_IDS[0]: FACTOR_VERSION}},
    )
    lb = LabelBundle(
        target_id="r", values=np.ascontiguousarray(rng.normal(size=(Tc, N))),
        horizon=1, decision_time=instants, label_start_time=instants,
        label_end_time=tuple(
            dt.datetime.combine(d + dt.timedelta(days=1), dt.time(4, 0),
                                tzinfo=dt.timezone.utc)
            for d in observed
        ),
        asset_axis=assets,
    )
    first = evaluate(fb, lb, metrics=("long_short_returns",))
    series = first.artifacts["long_short_returns"]
    cal_vals = np.asarray(series.values, dtype=np.float64)
    scale = min(1.0, 0.01 / max(np.nanmax(np.abs(cal_vals)), 1e-12))
    cal_vals = np.ascontiguousarray(cal_vals * scale)
    assert np.nanmin(cal_vals) >= -1.0, "capital returns must stay >= -1"
    probe = ProbePortfolioArtifact(
        cal_vals, time_index=tuple(series.time_axis.time_index),
        factor_ids=FACTOR_IDS,
    )
    snap = CalendarSnapshot(
        market="fixture", source_version="special-inputs-fixture-v1",
        timezone="Asia/Shanghai",
        trading_days=tuple(d.isoformat() for d in days_all),
        sessions=tuple((d.isoformat(), "09:30:00", "15:00:00", 1) for d in days_all),
        early_close=(), snapshot_id="special-inputs-calendar-sha256",
    )
    return fb, lb, probe, snap


# --------------------------------------------------------------------------
# 1. Generalization evidence fixture (8 previously unmeasurable metrics)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("metric_id", "expected"), sorted(GEN_ORACLES.items()))
def test_generalization_evidence_fixture_binds_metric_to_request_axis(metric_id, expected):
    fb, lb, probe, gen, _ = _int_inputs()
    out = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe,
                   generalization_evidence=gen)
    mv = out.metric_values[metric_id]
    assert isinstance(mv, MetricValue)
    assert mv.valid is True
    assert mv.value == pytest.approx(expected)
    repeat = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe,
                      generalization_evidence=gen)
    assert _value_digest(repeat) == _value_digest(out)


def test_generalization_evidence_axis_mismatch_still_rejected():
    """The fixture works because the axis matches — prove the guard is the
    reason the old ("f0","f1") fixture failed, i.e. the contract is intact."""
    from quant_evaluator.runtime.evaluator import InvalidContractError
    fb, lb, probe, _, _ = _int_inputs()
    mismatched = build_train_validation_artifact(
        np.array([1.2]), np.array([1.0]),
        metric_instance="rank_ic:h1:daily",
        train_factor_ids=("f0",), validation_factor_ids=("f0",),
        train_factor_versions=("v1",), validation_factor_versions=("v1",),
    )
    with pytest.raises(InvalidContractError, match="factor axis must match"):
        evaluate(fb, lb, metrics=("parameter_generalization",),
                 portfolio_returns=probe, generalization_evidence=mismatched)


# --------------------------------------------------------------------------
# 2. Session-aligned calendar fixture (3 previously unmeasurable metrics)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "expected_periods"),
    [(compute_worst_calendar_month, 12), (compute_worst_calendar_quarter, 4),
     (compute_worst_calendar_year, 1)],
)
def test_session_aligned_fixture_measures_calendar_metrics(function, expected_periods):
    fb, lb, probe, snap = _calendar_inputs()
    mid = function.__name__.replace("compute_", "")
    out = evaluate(fb, lb, metrics=(mid,), portfolio_returns=probe,
                   calendar_snapshot=snap)
    mv = out.metric_values[mid]
    assert isinstance(mv, MetricValue)
    assert mv.valid is True
    assert np.isfinite(mv.value)
    assert mv.observation_count == expected_periods
    repeat = evaluate(fb, lb, metrics=(mid,), portfolio_returns=probe,
                      calendar_snapshot=snap)
    assert _value_digest(repeat) == _value_digest(out)


@pytest.mark.parametrize(
    ("function", "days", "observed", "expected"),
    [
        # Feb 2025: three sessions at -5% each -> 0.95^3 - 1 (doc formula:
        # product of (1+r) over the snapshot's local-month sessions, worst).
        (
            compute_worst_calendar_month,
            ("2025-01-31", "2025-02-03", "2025-02-04", "2025-02-05", "2025-03-03"),
            ("2025-02-03", "2025-02-04", "2025-02-05"),
            0.95**3 - 1.0,
        ),
        # Q1 2025: three sessions at -10% -> 0.9^3 - 1.
        (
            compute_worst_calendar_quarter,
            ("2024-12-31", "2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01"),
            ("2025-01-02", "2025-02-03", "2025-03-03"),
            0.9**3 - 1.0,
        ),
        # Calendar year 2025: three sessions at -10% -> 0.9^3 - 1.
        (
            compute_worst_calendar_year,
            ("2024-12-31", "2025-01-02", "2025-06-02", "2025-12-31", "2026-01-02"),
            ("2025-01-02", "2025-06-02", "2025-12-31"),
            0.9**3 - 1.0,
        ),
    ],
)
def test_calendar_metric_oracle_matches_documented_formula(function, days, observed, expected):
    instants = tuple(
        dt.datetime.fromisoformat(f"{day}T04:00:00+00:00") for day in observed
    )
    snap = CalendarSnapshot(
        market="fixture", source_version="oracle-fixture-v1",
        timezone="Asia/Shanghai", trading_days=tuple(days), sessions=(),
        early_close=(), snapshot_id="oracle-calendar-sha256",
    )
    daily = -0.05 if function is compute_worst_calendar_month else -0.1
    returns = np.full((len(observed), 1), daily)
    artifact = function(returns, instants, ("f",), snap)
    assert artifact.values[0] == pytest.approx(expected)


# --------------------------------------------------------------------------
# 3. cost_drag leg fixture (1 previously unmeasurable metric)
# --------------------------------------------------------------------------


def test_turnover_cost_measured_through_cost_drag_leg():
    fb, lb, _, _, cost = _int_inputs()
    out = evaluate(fb, lb, metrics=("turnover_cost",), portfolio_returns=cost)
    mv = out.metric_values["turnover_cost"]
    assert isinstance(mv, MetricValue)
    assert mv.valid is True
    # Oracle: constant 0.0005 cost-rate per period -> 0.0005 * 1e4 = 5 bps.
    assert mv.value == pytest.approx(5.0)
    repeat = evaluate(fb, lb, metrics=("turnover_cost",), portfolio_returns=cost)
    assert _value_digest(repeat) == _value_digest(out)


def test_turnover_cost_oracle_matches_documented_formula():
    # Cost_bp = 1e4 * mean(c_t over observed): mean(0.001, 0.002) = 0.0015
    # -> 15 bps.  NaN is unobserved, negative magnitudes are invalid.
    assert compute_turnover_cost(np.array([0.001, np.nan, 0.002])) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_turnover_cost(np.array([-0.001]))
