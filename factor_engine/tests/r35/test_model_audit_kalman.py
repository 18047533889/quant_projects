# -*- coding: utf-8 -*-
"""Model-audit remediation M-070..M-074 for the Kalman state-space family.

M-070 — all six Kalman canonicals are stateful (causal one-pass filter).  The
module exposes :data:`KALMAN_STATEFUL_CANONICALS` +
:func:`kalman_stateful_contract` and surfaces the contract as metadata tags;
``ts_kalman_level`` is the only one declared ``stateful=True`` in the shared
``model_contract.py`` (reconciler: mirror the other five there).

M-071 — :func:`_scale_qr` implements a DIMENSIONLESS q/r mode
(``q_eff = cq * scale**2``, ``r_eff = cr * scale**2``) as a helper +
documentation ONLY — the public parameter surface is unchanged (default stays
absolute-variance).

M-072 — ``ts_kalman_beta`` is THROUGH-ORIGIN (``y = b*x``, no intercept);
tagged ``through_origin:true`` on every beta variant, no ``ts_kalman_alpha_beta``.

M-073 — the beta operators declare ``input_units`` (return-vs-return /
excess-return-vs-market).

M-074 — the beta warmup is FINITE-PAIR (accumulates the first ``_BETA_WARMUP``
finite ``(y, x)`` pairs across gaps); ``"contiguous"`` is NOT implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators.ts_model.state_space import (  # noqa: E402
    KALMAN_STATEFUL_CANONICALS,
    _scale_qr,
    kalman_stateful_contract,
    kalman_stateful_contracts,
    kalman_warmup_policy,
)


def _load():
    from cleaned_operators import load_all

    load_all()


def _get(name: str):
    from cleaned_operators.registry import OperatorRegistry

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
        assert c["state_schema_version"] == f"{canonical}.v1", canonical
        assert c["checkpointable"] is True, canonical
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
    as machine-readable metadata tags."""
    _load()
    for name in sorted(KALMAN_STATEFUL_CANONICALS):
        op = _get(name)
        tags = set(op.metadata.tags)
        assert "stateful:true" in tags, (name, sorted(tags))
        assert any(t.startswith("state_schema_version:") for t in tags), name
        assert any(t.startswith("checkpointable:") for t in tags), name
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


def test_kalman_public_param_surface_unchanged():
    """M-071: the dimensionless helper must NOT add a ``scale_mode`` parameter
    to the public surface (default behaviour untouched)."""
    _load()
    assert list(_get("ts_kalman_level").metadata.param_names) == ["x", "q", "r"]
    assert list(_get("ts_kalman_beta").metadata.param_names) == ["y", "x", "q", "r"]
    assert list(_get("ts_kalman_trend").metadata.param_names) == ["x", "q_level", "q_trend", "r"]


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
    tail = out[15:]
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
    """M-074: warmup takes the first ``_BETA_WARMUP`` FINITE pairs even across
    gaps.  With gaps at rows 2/5/8 the 5th finite pair sits at row 6 (past two
    gaps); a finite-pair policy must initialise beta there, whereas a contiguous
    policy would stay NaN until 5 consecutive finite rows after the last gap."""
    _load()
    rng = np.random.default_rng(22)
    n = 30
    x = rng.standard_normal(n)
    y = 1.5 * x
    y[[2, 5, 8]] = np.nan
    x[[2, 5, 8]] = np.nan
    out = _get("ts_kalman_beta").calculate(
        _panel(y), _panel(x), q=1e-3, r=1.0
    )["C0"].to_numpy()
    # rows 0..5 are still inside the warmup window -> NaN
    assert np.isnan(out[:6]).all(), f"warmup rows should be NaN, got {out[:6]}"
    # the 5th finite pair at row 6 initialises the filter
    assert np.isfinite(out[6]), (
        "beta should initialise at the 5th finite pair (row 6) under finite-pair "
        f"warmup, got {out[6]}"
    )
    assert out[6] == pytest.approx(1.5, abs=0.01)
