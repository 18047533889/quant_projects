# -*- coding: utf-8 -*-
"""Model-audit remediation M-070..M-074 + P0/P1 checkpoint, q/r-scale and
warm-up audits for the Kalman state-space family.

M-070 — all six Kalman canonicals are stateful (causal one-pass filter).  The
module exposes :data:`KALMAN_STATEFUL_CANONICALS` +
:func:`kalman_stateful_contract` and surfaces the contract as metadata tags.

P0 (2026-08-11) — honest degradation: the Kalman kernels have NO checkpoint
restore authority in ``StatefulCheckpointRegistry`` / ``stateful_runtime``, so
the contract declares ``checkpointable=False`` and ``restore_strategy="full_replay"``
instead of the old false-green ``checkpointable:true`` claim.  The machine
contract "claimed checkpointable == registry registered" is asserted in
``test_model_audit_kalman_stateful.py``.

M-071 — :func:`_scale_qr` + the public ``scale_mode`` parameter implement a
DIMENSIONLESS q/r mode: ``q_eff = cq * var_x``, ``r_eff = cr * var_x``
(``var_x`` = input variance), so relative parameters transfer across input
scales.  Default ``scale_mode="absolute"`` keeps legacy behaviour unchanged.

M-072 — ``ts_kalman_beta`` is THROUGH-ORIGIN (``y = b*x``, no intercept);
tagged ``through_origin:true`` on every beta variant, no ``ts_kalman_alpha_beta``.

M-073 — the beta operators declare ``input_units`` (return-vs-return /
excess-return-vs-market).

M-074 — the beta warmup is FINITE-PAIR (accumulates the first ``min_warmup``
finite ``(y, x)`` pairs across gaps; default 20); ``"contiguous"`` is NOT
implemented.  ``min_warmup`` is a real gate — fewer finite pairs yields NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators.ts_model.state_space import (  # noqa: E402
    KALMAN_STATEFUL_CANONICALS,
    _scale_qr,
    kalman_stateful_contract,
    kalman_stateful_contracts,
    kalman_warmup_default,
    kalman_warmup_policy,
)


def _load():
    # Register the Kalman operators directly (the module's decorators register on
    # import).  ``load_all()`` is deliberately NOT used: the full-registry
    # registration audit is broken in this tree by a concurrent WIP on an
    # unrelated operator (ts_first_passage_bias/polars arity), which would make
    # every load-dependent test fail for reasons unrelated to Kalman.
    import factor_engine.cleaned_operators.ts_model.state_space  # noqa: F401


def _get(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    assert op is not None, name
    return op


def _panel(vals: np.ndarray, instruments: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=vals.shape[0], freq="D")
    cols = [f"C{k}" for k in range(vals.shape[1] if vals.ndim == 2 else instruments)]
    if vals.ndim == 1:
        vals = vals[:, None]
    return pd.DataFrame(vals, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# M-070 — stateful contract machinery
# ---------------------------------------------------------------------------

def test_stateful_canonical_set_is_exact():
    assert KALMAN_STATEFUL_CANONICALS == frozenset({
        "ts_kalman_level",
        "ts_kalman_trend",
        "ts_kalman_beta",
        "ts_kalman_beta_change",
        "ts_kalman_beta_uncertainty",
        "ts_kalman_innovation_z",
    })


def test_stateful_contract_has_all_required_fields():
    for canonical in KALMAN_STATEFUL_CANONICALS:
        c = kalman_stateful_contract(canonical)
        assert c["stateful"] is True, canonical
        assert isinstance(c["state_schema_version"], str), canonical
        assert c["state_schema_version"] == f"{canonical}.v2", canonical
        # P0 (2026-08-11): honest degradation — no Kalman checkpoint authority
        # in StatefulCheckpointRegistry / stateful_runtime, so the contract must
        # NOT claim checkpointable; execution model is full-history replay.
        assert c["checkpointable"] is False, canonical
        assert c["restore_strategy"] == "full_replay", canonical
        assert c["time_shard_safe"] is False, canonical  # state spans the whole prefix
        assert c["reset_semantics"] == "reset_on_first_finite_observation", canonical
        assert c["missing_update_semantics"] == "predict_only_covariance_growth", canonical
        assert c["revision_replay_semantics"] == "deterministic_replay", canonical


def test_stateful_contracts_covers_all_and_stable():
    all_contracts = kalman_stateful_contracts()
    assert set(all_contracts) == set(KALMAN_STATEFUL_CANONICALS)
    assert list(all_contracts) == sorted(KALMAN_STATEFUL_CANONICALS)


def test_all_kalman_canonicals_carry_stateful_metadata_tags():
    """M-070: every registered Kalman canonical surfaces the stateful contract
    as machine-readable metadata tags.  P0: the tags must mirror the contract's
    honest ``checkpointable:false`` / ``restore_strategy:full_replay``."""
    _load()
    for name in sorted(KALMAN_STATEFUL_CANONICALS):
        op = _get(name)
        tags = set(op.metadata.tags)
        assert "stateful:true" in tags, (name, sorted(tags))
        assert any(t.startswith("state_schema_version:") for t in tags), name
        assert any(t.startswith("checkpointable:") for t in tags), name
        assert "checkpointable:false" in tags, (name, sorted(tags))
        assert "restore_strategy:full_replay" in tags, (name, sorted(tags))
        assert any(t.startswith("time_shard_safe:") for t in tags), name
        assert any(t.startswith("reset_semantics:") for t in tags), name
        assert any(t.startswith("missing_update_semantics:") for t in tags), name
        assert any(t.startswith("revision_replay_semantics:") for t in tags), name


# ---------------------------------------------------------------------------
# M-071 — dimensionless q/r helper (conservative: helper + docs, no surface param)
# ---------------------------------------------------------------------------

def test_scale_qr_default_is_absolute_unchanged():
    assert _scale_qr(1e-3, 1.0) == (1e-3, 1.0)
    assert _scale_qr(0.5, 2.0) == (0.5, 2.0)


def test_scale_qr_dimensionless_scales_by_scale_squared():
    q, r = _scale_qr(0.01, 1.0, scale_mode="dimensionless", scale=0.02)
    assert q == pytest.approx(0.01 * 0.02 ** 2)
    assert r == pytest.approx(1.0 * 0.02 ** 2)


def test_scale_qr_fails_closed():
    with pytest.raises(ValueError):
        _scale_qr(1e-3, 1.0, scale_mode="banana", scale=1.0)
    with pytest.raises(ValueError):
        _scale_qr(1e-3, 1.0, scale_mode="dimensionless", scale=None)
    with pytest.raises(ValueError):
        _scale_qr(1e-3, 1.0, scale_mode="dimensionless", scale=-1.0)
    with pytest.raises(ValueError):
        _scale_qr(1e-3, 1.0, scale_mode="dimensionless", scale=np.nan)
    with pytest.raises(ValueError):
        _scale_qr(-1.0, 1.0, scale_mode="dimensionless", scale=1.0)
    with pytest.raises(ValueError):
        _scale_qr(1e-3, 0.0, scale_mode="dimensionless", scale=1.0)


def test_kalman_public_param_surface():
    """M-071/M-074: the public surface declares ``scale_mode`` (default
    ``"absolute"``, legacy variance semantics unchanged) and, on the beta
    variants, the configurable ``min_warmup`` (default 20)."""
    _load()
    assert list(_get("ts_kalman_level").metadata.param_names) == ["x", "q", "r", "scale_mode"]
    assert list(_get("ts_kalman_beta").metadata.param_names) == ["y", "x", "q", "r", "scale_mode", "min_warmup"]
    assert list(_get("ts_kalman_trend").metadata.param_names) == ["x", "q_level", "q_trend", "r", "scale_mode"]


def test_scale_mode_defaults_absolute_no_behavior_change():
    """M-071: default ``scale_mode="absolute"`` must reproduce the legacy
    absolute-variance output byte-for-byte (the P1 fix is opt-in)."""
    _load()
    rng = np.random.default_rng(5)
    x = rng.standard_normal(60) * 0.02
    level = _get("ts_kalman_level")
    explicit_abs = level.calculate(_panel(x), q=1e-4, r=1.0, scale_mode="absolute")["C0"].to_numpy()
    default = level.calculate(_panel(x), q=1e-4, r=1.0)["C0"].to_numpy()
    assert np.allclose(explicit_abs, default, equal_nan=True)
    assert np.array_equal(np.isnan(explicit_abs), np.isnan(default))


def test_min_warmup_param_default_is_twenty():
    """M-074: default beta warm-up floor is 20 (configurable, non-searchable)."""
    _load()
    beta = _get("ts_kalman_beta")
    spec = beta.metadata.param_specs["min_warmup"]
    assert spec.dtype is int and spec.min >= 1 and spec.searchable is False
    assert kalman_warmup_default() == 20


# ---------------------------------------------------------------------------
# M-072 — through-origin beta (no intercept)
# ---------------------------------------------------------------------------

def test_beta_metadata_tags_through_origin():
    _load()
    for name in ("ts_kalman_beta", "ts_kalman_beta_change", "ts_kalman_beta_uncertainty"):
        tags = set(_get(name).metadata.tags)
        assert "through_origin:true" in tags, (name, sorted(tags))
    # the through-origin constraint must be documented in the operator description
    desc = _get("ts_kalman_beta").metadata.description
    assert "无截距" in desc and "通过原点" in desc, desc


def test_beta_recovers_through_origin_slope():
    """y = 1.5*x exactly: the warmup OLS through the origin must recover 1.5."""
    _load()
    rng = np.random.default_rng(21)
    n = 40
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(
        _panel(y), _panel(x), q=1e-3, r=1.0
    )["C0"].to_numpy()
    # default min_warmup=20 -> the first 19 rows are inside the warmup window
    assert np.isnan(out[:19]).all(), "rows before the 20th finite pair must be NaN"
    tail = out[20:]
    assert np.isfinite(tail).all(), "beta tail must be finite"
    assert np.median(tail) == pytest.approx(1.5, abs=0.05)


# ---------------------------------------------------------------------------
# M-073 — typed input units for the beta family
# ---------------------------------------------------------------------------

def test_beta_input_units_declared():
    _load()
    expected = {"y": "return", "x": "market_return"}
    for name in ("ts_kalman_beta", "ts_kalman_beta_change", "ts_kalman_beta_uncertainty"):
        assert _get(name).metadata.input_units == expected, name


# ---------------------------------------------------------------------------
# M-074 — finite-pair warmup policy
# ---------------------------------------------------------------------------

def test_warmup_policy_is_finite_pair():
    assert kalman_warmup_policy() == "finite_pair"
    desc = _get("ts_kalman_beta").metadata.description
    assert "finite-pair" in desc, desc


def test_beta_warmup_accumulates_finite_pairs_across_gaps():
    """M-074: warmup takes the first ``min_warmup`` FINITE pairs even across
    gaps.  With ``min_warmup=5`` and gaps at rows 2/5/8 the 5th finite pair sits
    at row 6 (past two gaps); a finite-pair policy must initialise beta there,
    whereas a contiguous policy would stay NaN until 5 consecutive finite rows
    after the last gap."""
    _load()
    rng = np.random.default_rng(22)
    n = 30
    x = rng.standard_normal(n)
    y = 1.5 * x
    y[[2, 5, 8]] = np.nan
    x[[2, 5, 8]] = np.nan
    out = _get("ts_kalman_beta").calculate(
        _panel(y), _panel(x), q=1e-3, r=1.0, min_warmup=5
    )["C0"].to_numpy()
    # rows 0..5 are still inside the warmup window -> NaN
    assert np.isnan(out[:6]).all(), f"warmup rows should be NaN, got {out[:6]}"
    # the 5th finite pair at row 6 initialises the filter
    assert np.isfinite(out[6]), (
        "beta should initialise at the 5th finite pair (row 6) under finite-pair "
        f"warmup, got {out[6]}"
    )
    assert out[6] == pytest.approx(1.5, abs=0.01)


def test_beta_warmup_gate_default_short_series_all_nan():
    """M-074/P1: the default warm-up floor (20 finite pairs) is a REAL gate —
    a short series with fewer finite pairs yields all-NaN (never a fabricated
    value and never an error)."""
    _load()
    rng = np.random.default_rng(23)
    n = 12  # < default min_warmup=20 finite pairs
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(
        _panel(y), _panel(x), q=1e-3, r=1.0
    )["C0"].to_numpy()
    assert np.isnan(out).all(), f"short series must be all-NaN, got {out}"


def test_beta_warmup_gate_explicit_min_warmup():
    """M-074/P1: ``min_warmup`` is configurable — fewer finite pairs than the
    requested floor yields NaN; at the floor the filter initialises."""
    _load()
    rng = np.random.default_rng(24)
    n = 40
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(
        _panel(y), _panel(x), q=1e-3, r=1.0, min_warmup=10
    )["C0"].to_numpy()
    assert np.isnan(out[:9]).all(), "rows before the 10th finite pair must be NaN"
    assert np.isfinite(out[9:]).all(), "beta must initialise at the 10th finite pair"
