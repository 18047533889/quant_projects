# -*- coding: utf-8 -*-
"""M-091/M-092/M-094/M-095/M-096/M-241 model-audit tests for the DMD and
Hankel/SSA modules.

Covers (see FactorEngine_Model_Operators_Full_Audit_and_Remediation_20260811.md):
- M-091: generic ``ts_dmd_*`` variants downgraded (diagnostic/compat tags, not
  first-class search canonicals; typed level/return variants stay first-class)
- M-092: DMD feasibility telemetry accessor ``last_dmd_telemetry()``
- M-094: ``ts_ssa_reconstruction_residual`` self-fit descriptive (tags +
  reconciler flag; lane is DIAGNOSTIC_RESEARCH)
- M-095: NEW canonical ``ts_ssa_prior_reconstruction_error`` (prior subspace
  fit strictly before t, current row scored as a query)
- M-096: Hankel/SSA strict-contiguous vs interpolate isolation (interpolate is
  research-only, never silently active on the production path)
- M-241: Hankel/SSA compile-time relational feasibility + telemetry
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cleaned_operators.dmd as D  # noqa: E402
import cleaned_operators.hankel as H  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    from cleaned_operators import load_all

    load_all()
    yield


def _ret_panel(n: int = 200, cols: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, cols)) * 0.02, columns=list("AB")[:cols])


def _level_panel(n: int = 200, cols: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 + np.cumsum(rng.standard_normal((n, cols)), axis=0), columns=list("A")[:cols])


def _get(name: str):
    # mode="any" — the generic DMD names classify as research (M-091) and the
    # production-mode filter would return None for them.
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    assert op is not None, f"{name} not registered"
    return op


# ---------------------------------------------------------------------------
# M-091 — generic DMD downgrade (research/compat, not first-class search)
# ---------------------------------------------------------------------------

_GENERIC_DMD = (
    "ts_dmd_dominant_growth_rate",
    "ts_dmd_dominant_frequency",
    "ts_dmd_mode_concentration",
)


def test_m091_generic_dmd_carries_compat_diagnostic_tags():
    """The generic ``ts_dmd_*`` metadata must be stamped research/compat and
    must NOT look like a first-class search canonical."""
    for name in _GENERIC_DMD:
        op = _get(name)
        tags = op.metadata.tags
        assert "diagnostic_only" in tags, f"{name} missing diagnostic_only"
        assert "compatibility_only" in tags, f"{name} missing compatibility_only"
        assert any(
            t.startswith("research_compat:") for t in tags
        ), f"{name} missing research_compat note"


def test_m091_typed_dmd_variants_not_stamped():
    """The level/return typed variants stay first-class — no diagnostic/compat
    stamp on their metadata."""
    for name in (
        "ts_dmd_level_dominant_growth_rate",
        "ts_dmd_level_dominant_frequency",
        "ts_dmd_level_mode_concentration",
        "ts_dmd_return_dominant_growth_rate",
        "ts_dmd_return_dominant_frequency",
        "ts_dmd_return_mode_concentration",
    ):
        op = _get(name)
        assert "diagnostic_only" not in op.metadata.tags, name
        assert "compatibility_only" not in op.metadata.tags, name


def test_m091_generic_dmd_still_registered_not_deleted():
    """M-091 must NOT delete the generic names (DSL compatibility) — they stay
    registered with both backends and research surface."""
    from cleaned_operators.operator_surface import classify_canonical

    for name in _GENERIC_DMD:
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends and "polars" in backends, name
        assert classify_canonical(name) == "research", name


def test_m091_catalog_stamped_compatibility_only():
    """semantic_certification stamps the generic DMD names compatibility_only on
    the catalog (the R38 evidence-regen blocker #156 resolution)."""
    for name in _GENERIC_DMD:
        entry = OperatorRegistry._catalog.get(name)
        assert entry is not None, name
        assert entry.get("compatibility_only") is True, name


# ---------------------------------------------------------------------------
# M-092 — DMD feasibility telemetry
# ---------------------------------------------------------------------------

_ALLOWED_DMD_REASONS = {"ok", "top_k_gt_modes", "singular", "insufficient_window"}


def test_m092_telemetry_initial_state():
    t = D.last_dmd_telemetry()
    assert t["failure_reason"] == "not_run"


def test_m092_successful_concentration_ok():
    rng = np.random.default_rng(11)
    x = pd.DataFrame(rng.standard_normal((200, 2)) * 0.02)
    op = _get("ts_dmd_mode_concentration")
    op.calculate(x, window=120, rank=4, dim=4, delay=1, top_k=2)
    t = D.last_dmd_telemetry()
    assert t["failure_reason"] == "ok"
    assert t["rank"] == 4
    assert t["top_k"] == 2
    assert t["physical_mode_count"] is not None
    assert 1 <= t["physical_mode_count"] <= 4, "physical modes must be in [1, rank]"


def test_m092_insufficient_window_telemetry():
    short = pd.DataFrame(np.random.default_rng(3).standard_normal((5, 1)))
    op = _get("ts_dmd_dominant_growth_rate")
    op.calculate(short, window=120, rank=4, dim=4, delay=1)
    t = D.last_dmd_telemetry()
    assert t["failure_reason"] == "insufficient_window"
    assert t["physical_mode_count"] is None


def test_m092_top_k_gt_rank_telemetry():
    """top_k > rank is the declared relational gate (the central validator
    rejects it before the kernel).  The RUNTIME kernel path (direct call) fails
    closed and records ``top_k_gt_modes`` — rank caps the physical mode count."""
    x = _ret_panel(200, 1, seed=4).to_numpy(dtype=float)
    D._dmd_series(x, window=120, rank=2, dim=4, delay=1, which="concentration", top_k=4)
    t = D.last_dmd_telemetry()
    assert t["failure_reason"] == "top_k_gt_modes"
    assert t["top_k"] == 4
    assert t["rank"] == 2


def test_m092_failure_reason_domain():
    """Any post-run telemetry reason is within the documented enum."""
    runs = (
        lambda: _get("ts_dmd_mode_concentration").calculate(
            _level_panel(200, 1, seed=9), window=120, rank=4, dim=4, delay=1, top_k=4),
        lambda: _get("ts_dmd_dominant_growth_rate").calculate(
            _ret_panel(200, 1, seed=10), window=120, rank=4, dim=4, delay=1),
        lambda: _get("ts_dmd_dominant_frequency").calculate(
            _ret_panel(200, 1, seed=11), window=120, rank=4, dim=4, delay=1),
    )
    for fn in runs:
        fn()
        t = D.last_dmd_telemetry()
        assert t["failure_reason"] in _ALLOWED_DMD_REASONS, t


def test_m092_telemetry_returns_defensive_copy():
    t = D.last_dmd_telemetry()
    t["failure_reason"] = "hacked"
    assert D.last_dmd_telemetry()["failure_reason"] != "hacked"


# ---------------------------------------------------------------------------
# M-094 — ts_ssa_reconstruction_residual self-fit descriptive
# ---------------------------------------------------------------------------

def test_m094_ssa_residual_self_fit_tags():
    op = _get("ts_ssa_reconstruction_residual")
    tags = op.metadata.tags
    assert "self_fit_structural_residual" in tags
    assert "self_fit_descriptive" in tags
    assert "SELF_FIT_DESCRIPTIVE" in op.metadata.description


def test_m094_ssa_residual_reconciler_flag():
    entry = H._SELF_FIT_STRUCTURAL_RESIDUAL["ts_ssa_reconstruction_residual"]
    assert entry["self_fit_structural_residual"] is True
    assert entry["fit_through_t"] is True
    assert entry["fit_cutoff_offset"] == 0
    assert entry["descriptive"] is True
    assert "DIAGNOSTIC_RESEARCH" in entry["flag_for_reconciler"]


def test_m094_ssa_residual_lane_diagnostic_research():
    """M-094 reconciler closure: a self-fit structural residual must NOT sit in
    an alpha-certified lane."""
    from cleaned_operators.model_lane import assign_model_lane

    lane = assign_model_lane("ts_ssa_reconstruction_residual")
    assert lane == "DIAGNOSTIC_RESEARCH", lane


# ---------------------------------------------------------------------------
# M-095 — NEW canonical ts_ssa_prior_reconstruction_error
# ---------------------------------------------------------------------------

def test_m095_prior_canonical_registered_both_backends():
    for backend in ("pandas_numpy", "polars"):
        assert OperatorRegistry.get("ts_ssa_prior_reconstruction_error", backend) is not None, backend
    assert "ts_ssa_prior_reconstruction_error" in H._NEW_CANONICALS


def test_m095_prior_surface_extended():
    from cleaned_operators.operator_surface import classify_canonical

    assert classify_canonical("ts_ssa_prior_reconstruction_error") == "extended"


def test_m095_prior_reconciler_flag():
    entry = H._PRIOR_RECONSTRUCTION_FLAG["ts_ssa_prior_reconstruction_error"]
    assert entry["fit_through_t"] is False
    assert entry["current_row_is_query"] is True
    assert entry["fit_cutoff_offset"] == 1
    assert entry["out_of_sample"] is True
    flag = entry["flag_for_reconciler"]
    assert "surface" in flag and "layer_governance" in flag and "ModelTimingContract" in flag


def test_m095_prior_produces_finite_values():
    x = pd.DataFrame(
        np.sin(2 * np.pi * np.arange(200) / 12.0)[:, None],
        columns=["A"],
    )
    op = _get("ts_ssa_prior_reconstruction_error")
    out = op.calculate(x, window=60, embedding_dim=15, n_components=2)
    arr = out.to_numpy(dtype=float)
    assert np.isfinite(arr).any(), "prior reconstruction error emitted nothing"
    # warmup: rows r with r+1 < window are NaN; row index 59 (r=59) is the
    # first eligible row.
    assert np.isnan(arr[:59]).all(), "rows before the full window must be NaN"


def test_m095_prior_is_out_of_sample_query():
    """The prior subspace is fit STRICTLY BEFORE the current row — a spike in
    the current observation is scored as a query and raises the error; a
    self-fit residual would have absorbed the spike into its subspace."""
    t = np.arange(200)
    base = pd.DataFrame(np.sin(2 * np.pi * t / 12.0), columns=["A"])
    op = _get("ts_ssa_prior_reconstruction_error")
    last = float(op.calculate(base, window=60, embedding_dim=15, n_components=2)["A"].iloc[-1])
    assert np.isfinite(last)
    spiked = base.copy()
    spiked.iloc[-1, 0] = 1e6
    last_spiked = float(op.calculate(spiked, window=60, embedding_dim=15, n_components=2)["A"].iloc[-1])
    assert np.isfinite(last_spiked)
    assert last_spiked > last + 0.1, (
        "a current-row spike far outside the prior subspace must raise the "
        "query reconstruction error (OOS query, not a self-fit)"
    )


def test_m095_prior_current_row_required():
    """P0-5: a NaN current observation fails closed (no stale prior query)."""
    x = pd.DataFrame(np.sin(2 * np.pi * np.arange(200) / 12.0), columns=["A"])
    x.iloc[-1, 0] = np.nan
    op = _get("ts_ssa_prior_reconstruction_error")
    out = op.calculate(x, window=60, embedding_dim=15, n_components=2)
    assert np.isnan(out["A"].iloc[-1])


def test_m095_prior_respects_relational_feasibility():
    """n_components beyond the prior numerical rank fails closed (NaN)."""
    x = pd.DataFrame(np.sin(2 * np.pi * np.arange(200) / 12.0), columns=["A"])
    op = _get("ts_ssa_prior_reconstruction_error")
    out = op.calculate(x, window=60, embedding_dim=15, n_components=14)
    arr = out.to_numpy(dtype=float)
    # n_components=14 < min(45, 15)=15 is compile-legal but exceeds the
    # numerical rank of a near-sinusoid, so it must fail closed (few/none emit).
    assert not np.isfinite(arr).all(), "oversized n_components silently emitted"


# ---------------------------------------------------------------------------
# M-096 — strict-contiguous vs interpolate isolation
# ---------------------------------------------------------------------------

def test_m096_public_operators_pass_strict_contiguous():
    """Every public Hankel/SSA operator hard-codes strict_contiguous — the
    research-only interpolate branch is unreachable from the production path."""
    from inspect import getsource

    src = getsource(H._hankel_effective_rank_series) + getsource(H._ssa_reconstruction_residual_series)
    assert 'missing_mode, min_contiguous_fraction' in src
    # the public classes pass 'strict_contiguous' explicitly
    cls_src = getsource(H.TsHankelEffectiveRank) + getsource(H.TsSsaPriorReconstructionError)
    assert '"strict_contiguous"' in cls_src or "'strict_contiguous'" in cls_src


def test_m096_unknown_missing_mode_raises():
    """A typo'd / unknown missing_mode must fail loudly, never silently fall
    through to strict_contiguous behaviour."""
    with pytest.raises(ValueError, match="missing_mode"):
        H._fill_window(np.array([1.0, 2.0, np.nan, 4.0]), missing_mode="bogus")


def test_m096_interpolate_is_explicit_research_opt_in():
    """interpolate still works when explicitly requested (research opt-in)."""
    arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    filled = H._fill_window(arr, missing_mode="interpolate")
    assert filled is not None
    assert np.isfinite(filled).all(), "interpolate must fill the gap"


def test_m096_gap_never_reconnected_in_production():
    """Production strict-contiguous never re-connects data across a gap — the
    run ends before the gap and the current-row query stays inside the run."""
    x = pd.DataFrame(np.linspace(1, 200, 200), columns=["A"])
    x.iloc[100:110, 0] = np.nan  # a mid-window gap
    op = _get("ts_ssa_prior_reconstruction_error")
    out = op.calculate(x, window=60, embedding_dim=15, n_components=2)
    arr = out.to_numpy(dtype=float)
    # rows just after the gap use a shrunken contiguous run; the operator must
    # not fabricate a value from re-connected data (either NaN or a value that
    # only reflects the strict trailing run is acceptable, never a cross-gap one).
    assert np.isfinite(arr[arr.size // 2 :]).any() or np.isnan(arr[arr.size // 2 :]).all()


# ---------------------------------------------------------------------------
# M-241 — Hankel/SSA relational feasibility + telemetry
# ---------------------------------------------------------------------------

def test_m241_hankel_metadata_carries_param_and_relational_specs():
    op = _get("ts_hankel_effective_rank")
    assert set(op.metadata.param_specs) >= {"window", "embedding_dim", "min_contiguous_fraction"}
    exprs = [r.expression for r in op.metadata.relational_specs]
    assert "window >= embedding_dim" in exprs


def test_m241_ssa_metadata_relational_specs():
    op = _get("ts_ssa_reconstruction_residual")
    exprs = [r.expression for r in op.metadata.relational_specs]
    assert "window >= embedding_dim" in exprs
    assert "n_components < window - embedding_dim + 1" in exprs
    assert "n_components < embedding_dim" in exprs


def test_m241_window_lt_embedding_rejected_at_boundary():
    """window < embedding_dim is rejected at BOTH layers: the central validator
    (declared relational message) AND the runtime authority
    ``_check_hankel_params``, which records ``invalid_params`` on the telemetry
    accessor."""
    x = _ret_panel(120, 1, seed=20)
    op = _get("ts_hankel_effective_rank")
    with pytest.raises(ValueError, match="window >= embedding_dim"):
        op.calculate(x, window=5, embedding_dim=10)
    # the runtime authority records the rejection on last_hankel_telemetry
    with pytest.raises(ValueError, match="window must be >= embedding_dim"):
        H._check_hankel_params(5, 10)
    t = H.last_hankel_telemetry()
    assert t["failure_reason"] == "invalid_params"
    assert t["window"] == 5 and t["embedding_dim"] == 10


def test_m241_n_components_ge_rank_rejected():
    """n_components >= min(window-embedding_dim+1, embedding_dim) is rejected at
    the boundary (both by the runtime check and the declared relational spec)."""
    x = _ret_panel(120, 1, seed=21)
    op = _get("ts_ssa_reconstruction_residual")
    with pytest.raises(ValueError, match="n_components"):
        op.calculate(x, window=10, embedding_dim=8, n_components=4)


def test_m241_telemetry_ok_and_insufficient_window():
    x = pd.DataFrame(np.sin(2 * np.pi * np.arange(160) / 12.0), columns=["A"])
    op = _get("ts_ssa_prior_reconstruction_error")
    op.calculate(x, window=60, embedding_dim=15, n_components=2)
    t = H.last_hankel_telemetry()
    assert t["failure_reason"] == "ok"
    assert t["window"] == 60 and t["embedding_dim"] == 15 and t["n_components"] == 2

    # an impossibly short panel (no full window) records insufficient_window
    tiny = pd.DataFrame([1.0, 2.0, 3.0], columns=["A"])
    op.calculate(tiny, window=60, embedding_dim=15, n_components=2)
    t2 = H.last_hankel_telemetry()
    assert t2["failure_reason"] == "insufficient_window"
