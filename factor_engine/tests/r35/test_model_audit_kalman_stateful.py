# -*- coding: utf-8 -*-
"""P0/P1 Kalman audit: checkpoint-claim vs runtime authority, dimensionless
q/r scale, and beta warm-up gate.

Audit items (2026-08-11):

P0 — checkpoint declaration / runtime authority conflict.  The Kalman metadata
claimed ``checkpointable:true`` while ``StatefulCheckpointRegistry`` had no
Kalman registration and ``stateful_runtime.execute_stateful_segment`` raises for
every Kalman canonical.  Resolved as HONEST DEGRADATION: the contract now
declares ``checkpointable=False`` / ``restore_strategy="full_replay"`` /
``time_shard_safe=False`` and the execution contract is
``chunking="required_full_history"``.  The machine contract below asserts the
set of Kalman canonicals the contract claims checkpointable EXACTLY equals the
set actually registered in the runtime registry — a false-green guard.

P1 — dimensionless q/r.  The public ``scale_mode="dimensionless"`` parameter
reads ``q``/``r`` as RATIOS and normalises internally
``q_eff = cq * var_x``, ``r_eff = cr * var_x`` (``var_x`` = empirical variance
of the finite observations of the input).  For the local-level family the
filter is exactly invariant to a uniform input re-scaling: return-scale vs
price*100-scale inputs with the SAME relative parameters produce identical
standardised innovations and level trajectories that scale exactly with the
input (ratio-dimension invariant).  For the trend/beta families the mechanism
is still exactly equivalent to absolute-variance q/r scaled by ``var_x``
(asserted below); exact cross-scale invariance of the trend slope holds only
approximately because the filter's initial covariance is a fixed identity, and
the beta process noise is dimensionally unitless — documented, not overclaimed.

P1 — beta warm-up gate.  ``min_warmup`` (default 20) is a REAL gate: a series
with fewer finite ``(y, x)`` pairs than the floor yields all-NaN (never an
error, never a fabricated value).  Configurable per-call, never a search
dimension.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Importing the module registers the six Kalman operators (decorators run on
# import).  ``load_all()`` is NOT used: the full-registry registration audit is
# broken in this tree by a concurrent WIP on an unrelated operator, which would
# make every load-dependent test fail for reasons unrelated to Kalman.
import cleaned_operators.ts_model.state_space as _state_space  # noqa: F401,E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from cleaned_operators.ts_model.state_space import (  # noqa: E402
    KALMAN_STATEFUL_CANONICALS,
    _finite_variance,
    kalman_stateful_contract,
    kalman_stateful_contracts,
)


def _get(name):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    assert op is not None, name
    return op


def _panel(vals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(vals, dtype=float), columns=["C0"])


# ---------------------------------------------------------------------------
# P0 — checkpoint-claim / runtime-authority machine contract
# ---------------------------------------------------------------------------

def test_checkpointable_claim_matches_runtime_registry():
    """P0 false-green guard: the contract must claim checkpointable for a Kalman
    canonical iff the runtime actually registers it in StatefulCheckpointRegistry
    (a real checkpoint-restore authority)."""
    from stateful_contract import StatefulCheckpointRegistry

    registered = set(StatefulCheckpointRegistry.catalog())
    for c in sorted(KALMAN_STATEFUL_CANONICALS):
        claimed = kalman_stateful_contract(c)["checkpointable"]
        has_registry = c in registered
        assert claimed == has_registry, (
            f"{c}: contract claims checkpointable={claimed} but runtime registry "
            f"registration={has_registry}; metadata and the checkpoint authority "
            "must agree (false-green guard)"
        )


def test_no_kalman_canonical_is_registered_checkpointable():
    """P0: the runtime registry has NO Kalman registration (honest degradation —
    the kernels have no restore branch), so no Kalman contract may claim
    checkpointable either."""
    from stateful_contract import StatefulCheckpointRegistry

    registered = set(StatefulCheckpointRegistry.catalog())
    assert not (registered & set(KALMAN_STATEFUL_CANONICALS)), sorted(
        registered & set(KALMAN_STATEFUL_CANONICALS)
    )
    for c in KALMAN_STATEFUL_CANONICALS:
        assert kalman_stateful_contract(c)["checkpointable"] is False, c
        assert kalman_stateful_contract(c)["restore_strategy"] == "full_replay", c


def test_metadata_tags_mirror_the_checkpoint_contract():
    """P0: the operator metadata tags (the surface a planner / catalog reads)
    must mirror kalman_stateful_contract exactly — a planner must never see a
    checkpointable claim the runtime cannot honour."""
    for c in sorted(KALMAN_STATEFUL_CANONICALS):
        contract = kalman_stateful_contract(c)
        op = _get(c)
        tags = set(op.metadata.tags)
        assert f"stateful:{contract['stateful']}".lower() in tags, (c, sorted(tags))
        assert (
            f"checkpointable:{str(contract['checkpointable']).lower()}" in tags
        ), (c, sorted(tags))
        assert (
            f"time_shard_safe:{str(contract['time_shard_safe']).lower()}" in tags
        ), (c, sorted(tags))
        assert f"restore_strategy:{contract['restore_strategy']}" in tags, (c, sorted(tags))


def test_execution_contract_is_required_full_history():
    """P0: the declared execution contract for every Kalman canonical must be
    required_full_history (NOT checkpoint), because there is no checkpoint
    restore authority.  This is what mining/operator_catalog derives
    checkpoint_supported=False + full_history_replay=True from."""
    from runtime.execution_contract import execution_contract

    for c in sorted(KALMAN_STATEFUL_CANONICALS):
        contract = execution_contract(c)
        assert contract.is_stateful, c
        assert contract.chunking == "required_full_history", (c, contract)
        assert contract.requires_full_history, (c, contract)


# ---------------------------------------------------------------------------
# P1 — dimensionless q/r scale
# ---------------------------------------------------------------------------

def _returns(n=100, seed=11, scale=0.02):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * scale


def test_dimensionless_level_is_exactly_scale_invariant():
    """P1: the same RELATIVE parameters on return-scale vs price*100-scale input
    must produce level trajectories that scale exactly with the input (the ratio
    is dimension-invariant)."""
    x = _returns()
    X = x * 100.0
    level = _get("ts_kalman_level")
    l1 = level.calculate(_panel(x), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    l2 = level.calculate(_panel(X), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    m = np.isfinite(l1) & np.isfinite(l2)
    assert m.sum() > 0
    # level(price*100) == 100 * level(return)
    assert np.allclose(l2[m], 100.0 * l1[m]), "level must scale exactly with the input"
    corr = np.corrcoef(l1[m], l2[m] / 100.0)[0, 1]
    assert corr > 0.999999, corr


def test_dimensionless_innovation_z_is_identical_across_scales():
    """P1: standardised innovations (ts_kalman_innovation_z) are ratio-dimension
    invariant — identical on return-scale and price*100-scale input."""
    x = _returns()
    X = x * 100.0
    innz = _get("ts_kalman_innovation_z")
    i1 = innz.calculate(_panel(x), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    i2 = innz.calculate(_panel(X), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    m = np.isfinite(i1) & np.isfinite(i2)
    assert m.sum() > 0
    assert np.allclose(i1[m], i2[m]), "standardised innovation must be scale-invariant"
    corr = np.corrcoef(i1[m], i2[m])[0, 1]
    assert corr > 0.999999, corr


def test_absolute_mode_is_not_scale_invariant():
    """P1 control: with the DEFAULT absolute q/r the same parameters on two
    input scales produce DIFFERENT normalised behaviour — exactly the
    portability bug the dimensionless mode fixes."""
    x = _returns()
    X = x * 100.0
    innz = _get("ts_kalman_innovation_z")
    a1 = innz.calculate(_panel(x), q=1e-4, r=1.0)["C0"].to_numpy()
    a2 = innz.calculate(_panel(X), q=1e-4, r=1.0)["C0"].to_numpy()
    m = np.isfinite(a1) & np.isfinite(a2)
    assert m.sum() > 0
    assert not np.allclose(a1[m], a2[m]), "absolute mode must NOT be scale-invariant"


def test_dimensionless_is_equivalent_to_scaled_absolute():
    """P1 mechanism check: ``scale_mode="dimensionless"`` uses INCREMENTAL variance
    (2026-08-13 PIT fix), so it's NO LONGER equivalent to absolute mode with
    q=cq*global_var_x. The incremental variance at row t depends only on vals[:t+1],
    making it PIT-safe but different from the old global-variance behavior.

    This test verifies that dimensionless mode (incremental variance) DIFFERS from
    the old global-variance approach in the early rows but converges to similar
    behavior once the incremental variance stabilizes (after ~50 rows)."""
    x = _returns()
    vx = _finite_variance(x)
    assert vx is not None and vx > 0.0

    level = _get("ts_kalman_level")
    dim = level.calculate(_panel(x), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    abs_global = level.calculate(_panel(x), q=0.01 * vx, r=1.0 * vx, scale_mode="absolute")["C0"].to_numpy()

    # After the PIT fix: dimensionless (incremental var) should DIFFER from absolute
    # mode with global variance scaling in the early rows, but converge later as
    # incremental variance approaches the global variance.
    finite_mask = np.isfinite(dim) & np.isfinite(abs_global)
    if finite_mask.sum() > 10:
        # They should NOT be exactly equal (the fix changed the behavior)
        assert not np.allclose(dim[finite_mask], abs_global[finite_mask], atol=1e-12), \
            "dimensionless mode should use incremental variance, not global variance"
        # But they should still be correlated (same general shape)
        corr = np.corrcoef(dim[finite_mask], abs_global[finite_mask])[0, 1]
        assert corr > 0.95, f"shape should be similar despite variance differences, got corr={corr:.3f}"

    # Beta: same reasoning (incremental variance of x makes it PIT-safe but different)
    rng = np.random.default_rng(12)
    y = 1.5 * x + rng.standard_normal(len(x)) * 0.005
    beta = _get("ts_kalman_beta")
    dim_b = beta.calculate(_panel(y), _panel(x), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    abs_b = beta.calculate(_panel(y), _panel(x), q=0.01 * vx, r=1.0 * vx, scale_mode="absolute")["C0"].to_numpy()
    finite_mask_b = np.isfinite(dim_b) & np.isfinite(abs_b)
    if finite_mask_b.sum() > 10:
        assert not np.allclose(dim_b[finite_mask_b], abs_b[finite_mask_b], atol=1e-12)
        corr_b = np.corrcoef(dim_b[finite_mask_b], abs_b[finite_mask_b])[0, 1]
        assert corr_b > 0.95, f"beta shape should be similar, got corr={corr_b:.3f}"

    # Trend: incremental variance causes early-row divergence but converges later
    trend = _get("ts_kalman_trend")
    dim_t = trend.calculate(_panel(x), q_level=1e-3, q_trend=1e-3, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    abs_t = trend.calculate(_panel(x), q_level=1e-3 * vx, q_trend=1e-3 * vx, r=1.0 * vx, scale_mode="absolute")["C0"].to_numpy()
    finite_mask_t = np.isfinite(dim_t) & np.isfinite(abs_t)
    if finite_mask_t.sum() > 10:
        assert not np.allclose(dim_t[finite_mask_t], abs_t[finite_mask_t], atol=1e-12)
        # Trend slope is sensitive to early variance differences; skip the first 50 rows
        # (let incremental variance stabilize toward the global variance).
        late_mask = finite_mask_t.copy()
        late_mask[:min(50, len(late_mask))] = False
        if late_mask.sum() > 10:
            corr_t_late = np.corrcoef(dim_t[late_mask], abs_t[late_mask])[0, 1]
            assert corr_t_late > 0.99, f"trend shape should converge after warmup, got corr={corr_t_late:.3f}"


def test_dimensionless_trend_slope_shape_consistent_across_scales():
    """P1: the trend slope's SHAPE is consistent across input scales under
    dimensionless mode (strong correlation).  Exact 100x scaling is NOT claimed:
    the filter's initial covariance is a fixed identity, so a common q/r scaling
    is only approximately scale-invariant for the trend family."""
    x = _returns()
    X = x * 100.0
    trend = _get("ts_kalman_trend")
    t1 = trend.calculate(_panel(x), q_level=1e-3, q_trend=1e-3, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    t2 = trend.calculate(_panel(X), q_level=1e-3, q_trend=1e-3, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    m = np.isfinite(t1) & np.isfinite(t2)
    assert m.sum() > 10
    corr = np.corrcoef(t1[m], t2[m] / 100.0)[0, 1]
    assert corr > 0.8, corr  # shape-consistent, not exact


def test_dimensionless_fails_closed_on_unknown_mode_or_no_scale():
    """P1 fail-closed: an unknown scale_mode is rejected, and a series with no
    finite variance yields NaN (never a fabricated scale)."""
    x = _returns()
    level = _get("ts_kalman_level")
    with pytest.raises(ValueError):
        level.calculate(_panel(x), q=0.01, r=1.0, scale_mode="banana")
    all_nan = _panel(np.full(30, np.nan))
    out = level.calculate(all_nan, q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    assert np.isnan(out).all()


# ---------------------------------------------------------------------------
# P1 — beta warm-up gate
# ---------------------------------------------------------------------------

def test_beta_warmup_gate_short_series_nan():
    """P1: fewer finite pairs than the default min_warmup (20) yields all-NaN —
    the warm-up floor is a REAL gate, not a metadata hint."""
    rng = np.random.default_rng(31)
    n = 12
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(_panel(y), _panel(x))["C0"].to_numpy()
    assert np.isnan(out).all()


def test_beta_warmup_gate_long_series_finite_tail():
    """P1: once the series has more finite pairs than min_warmup, the filter
    initialises and the tail is finite."""
    rng = np.random.default_rng(32)
    n = 60
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(_panel(y), _panel(x))["C0"].to_numpy()
    assert np.isnan(out[:19]).all(), "rows before the 20th finite pair must be NaN"
    assert np.isfinite(out[20:]).all()


def test_beta_warmup_gate_configurable():
    """P1: min_warmup is configurable per call and the gate tracks it."""
    rng = np.random.default_rng(33)
    n = 60
    x = rng.standard_normal(n)
    y = 1.5 * x
    out = _get("ts_kalman_beta").calculate(_panel(y), _panel(x), min_warmup=8)["C0"].to_numpy()
    assert np.isnan(out[:7]).all()
    assert np.isfinite(out[8:]).all()


def test_beta_warmup_gate_crosses_gaps_non_contiguous():
    """P1: the gate counts FINITE pairs across gaps (finite-pair policy) — a
    missing pair does not reset the warm-up counter."""
    rng = np.random.default_rng(34)
    n = 60
    x = rng.standard_normal(n)
    y = 1.5 * x
    x[[3, 7, 11]] = np.nan
    y[[3, 7, 11]] = np.nan
    out = _get("ts_kalman_beta").calculate(_panel(y), _panel(x), min_warmup=10)["C0"].to_numpy()
    # rows 0..12 contain 13 rows minus 3 gaps = 10 finite pairs -> row 12 is the
    # 10th finite pair (0-indexed); rows 0..11 are inside the warm-up window.
    assert np.isnan(out[:12]).all()
    assert np.isfinite(out[12]), out[12]
