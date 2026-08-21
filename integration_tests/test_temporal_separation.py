# -*- coding: utf-8 -*-
"""
End-to-end factor -> label temporal-separation tests (INT-T1..T3) -- R21 Q6 gate.

The R21-LEAKAGE-PIT-AUDIT (Q6) found that no integration test proves a factor
computed for evaluation at date ``t`` uses ONLY data available at or before
``t``.  The existing suites assert artifact REFS (identity/snapshot/split/
order/lineage) and source-level PIT semantics, but never perturb future rows
and assert the value at ``t`` is unchanged.

These tests close that gap with SMALL SYNTHETIC data and the REAL packages:

  - ``test_factor_computed_at_t_uses_only_data_le_t`` (INT-T1)
    Perturb every input row strictly AFTER ``t`` by a huge, asset-dependent
    amount; recompute the factor through the real FactorEngine; the value AT
    ``t`` must be bit-identical.  The same factor rows then flow into the real
    QE evaluator; the per-day IC at ``t`` must also be bit-identical
    (factor -> evaluation end-to-end), while IC rows after ``t`` DO move.

  - ``test_factor_prefix_invariance`` (INT-T2)
    Factor value at ``t`` computed from the FULL panel must equal the value at
    ``t`` computed from the PREFIX [0..t] only -- CausalPrefixInvariance fed
    end-to-end across FE (FactorEngine ts_mean) -> FP
    (factor_preprocess rolling_mean) -> QE (evaluation).

  - ``test_label_horizon_beyond_t_is_rejected`` (INT-T3)
    A factor at ``t`` combined with a label bundle whose window reaches past
    ``t`` must be rejected by the FO LabelBundle/split-plan validation.  The
    causal-chain boundary that DOES exist is exercised for real (FO
    ``validate_split_plan`` forward-label-overlap rejection; QE LabelBundle
    causal-chain rejection of backward-reaching labels); the missing wiring
    (QE has no guard tying ``label_end_time`` to the evaluation boundary) is
    pinned and documented rather than fabricated into a pass.

Import layout: factor_engine is installed as FLAT editable modules (no
top-level ``factor_engine`` package) and factor_preprocess / factor_optimizer
are only importable from their repo subdirectories, so this module prepends
the three package roots to ``sys.path`` -- the same approach the platform
tests use.  A package that cannot be imported produces a SKIP with a reason
(a skip is not a pass).

Performance: ``FactorEngine(...)`` construction spends ~80s once on a cold
operator-registry bootstrap (backend certification / polars capability
classification).  The ``_fe_run_factor`` module-scoped fixture constructs ONE
engine with a swappable in-memory panel and reuses it for every run in this
module, keeping the whole module well under the 120s budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# factor_engine -> flat modules (runtime./ir./api./backend./storage.);
# factor_preprocess / factor_optimizer -> their own top-level packages.
for _pkg_root in (
    REPO_ROOT / "factor_engine",
    REPO_ROOT / "factor_preprocess",
    REPO_ROOT / "factor_optimizer",
):
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))


def _import_optional(module: str):
    """Try to import *module*; return (mod, None) or (None, reason)."""
    try:
        import importlib

        return importlib.import_module(module), None
    except Exception as exc:  # noqa: BLE001 - report any import failure truthfully
        return None, f"{type(exc).__name__}: {exc}"


# ===========================================================================
# Shared synthetic panel
# ===========================================================================

N_TIMES = 16
# >= 10 assets so QE's per-day Pearson IC (min_obs=10, metrics/ic.py) is
# defined on the cross-section; 3 assets would make every daily IC NaN.
N_ASSETS = 12
ASSETS = tuple(f"INV{i:02d}" for i in range(N_ASSETS))
# The leakage-probe target date: row position 8 (index 8 of the panel).
T_LEAK = pd.Timestamp("2026-01-09")
TIDX = pd.date_range("2026-01-01", periods=N_TIMES, freq="D")


def _make_panel(seed: int = 0) -> pd.Series:
    """Return a MultiIndex Series (timestamp, instrument) named 'close'."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([TIDX, ASSETS], names=["timestamp", "instrument"])
    values = rng.normal(size=(N_TIMES, N_ASSETS)).ravel()
    return pd.Series(values, index=idx, name="close")


def _perturbed_after_t(close: pd.Series, offset: float = 1.0e6) -> pd.Series:
    """Copy of *close* with every row strictly AFTER T_LEAK shifted by
    ``offset * (asset_index + 1)``.

    The offset is ASSET-DEPENDENT so the cross-sectional factor profile (and
    therefore the per-day IC) after ``t`` genuinely moves -- a constant shift
    would be invisible to a cross-sectional correlation and make the IC
    contrast assertion vacuous.  Rows at or before ``t`` are untouched.
    """
    future = np.asarray(close.index.get_level_values("timestamp")) > T_LEAK
    asset_scale = np.asarray(
        close.index.get_level_values("instrument").map(
            {asset: i + 1 for i, asset in enumerate(ASSETS)}
        )
    )
    new_values = close.to_numpy().copy()
    new_values[future] += offset * asset_scale[future]
    return pd.Series(new_values, index=close.index, name="close")


def _mem_source(close: pd.Series):
    """Build a DataSource that serves an in-memory panel (plain panel).

    The panel is SWAPPABLE (``set_panel``) so one FactorEngine instance can be
    reused across runs with different synthetic panels.
    """
    from storage.sources.datasource import DataSource, TemporalContract

    class MemSource(DataSource):
        def __init__(self, series):
            self._series = series

        def set_panel(self, series):
            self._series = series

        def load_column(self, name):
            if name != "close":
                raise KeyError(f"unexpected column requested: {name}")
            return self._series

        def temporal_contract(self):
            return TemporalContract(
                temporal_sensitivity="none",
                snapshot_capability="none",
                join_capability="generic_asof",
            )

    return MemSource(close)


def _factor():
    from api import ts_mean
    from api.columns import col
    from api.factor import Factor

    return Factor(
        name="ts_mean_close_5",
        expr=ts_mean(col("close"), 5),
        freq="1d",
        universe="u1",
    )


@pytest.fixture(scope="module")
def _fe_run_factor():
    """Run ``ts_mean(close, 5)`` through ONE shared FactorEngine (pandas).

    Constructing a FactorEngine costs ~80s ONCE (cold operator-registry
    bootstrap); reusing one engine with a swappable panel makes every
    subsequent run <0.5s.  Returns ``run(close) -> MultiIndex Series``.
    """
    mod, err = _import_optional("runtime.engine")
    if err:
        pytest.skip(f"factor_engine not importable here: {err}")
    from backend.factory import build_backend
    from runtime.engine import FactorEngine

    source = _mem_source(_make_panel(seed=1234))
    engine = FactorEngine(
        backend=build_backend("pandas"),
        data_source=source,
        run_mode="research",
    )
    factor = _factor()

    def run(close: pd.Series) -> pd.Series:
        source.set_panel(close)
        return engine.run(factor)["result"]

    return run


def _exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-identical comparison; NaN == NaN (structural equality)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    return bool(np.array_equal(a[~nan_a], b[~nan_b]))


def _to_factor_batch(factor_series: pd.Series) -> "FactorBatch":
    """Pivot a (timestamp, instrument) factor Series into a QE FactorBatch."""
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch

    wide = factor_series.unstack("instrument").reindex(
        index=TIDX, columns=list(ASSETS)
    )
    return FactorBatch(
        factor_ids=("f",),
        time_axis=AxisRef(
            "t",
            "datetime64[ns]",
            N_TIMES,
            values=np.array(TIDX, dtype="datetime64[ns]"),
        ),
        asset_axis=AxisRef("a", "str", N_ASSETS, values=np.array(ASSETS)),
        values=wide.to_numpy()[:, :, None],
    )


def _label_bundle() -> "LabelBundle":
    """Causal forward-return label bundle aligned with the panel."""
    from quant_evaluator.contracts.label_bundle import LabelBundle

    labels = np.random.default_rng(1).normal(size=(N_TIMES, N_ASSETS))
    return LabelBundle(
        target_id="fwd_ret",
        values=labels,
        horizon=1,
        decision_time=tuple(TIDX),
        label_start_time=tuple(TIDX),
        label_end_time=tuple(t + pd.Timedelta(days=1) for t in TIDX),
    )


def _evaluate_pearson_ic_series(factor_series: pd.Series) -> np.ndarray:
    """Run the REAL QE Evaluator on (factor, causal label bundle); return IC."""
    from quant_evaluator.runtime.evaluator import Evaluator

    evaluator = Evaluator(enable_cache=False)
    result = evaluator.evaluate(
        _to_factor_batch(factor_series),
        _label_bundle(),
        [{"metric_id": "pearson_ic_series", "metric_kind": "custom"}],
        use_chunking=False,
    )
    return np.asarray(result.metrics["pearson_ic_series"])  # (T, 1)


def _t_leak_position() -> int:
    return int(np.flatnonzero(np.asarray(TIDX) == np.datetime64(T_LEAK))[0])


# ===========================================================================
# INT-T1: factor at t uses only data <= t
# ===========================================================================

def test_factor_computed_at_t_uses_only_data_le_t(_fe_run_factor):
    """
    Compute ``ts_mean(close, 5)`` on the full panel; then perturb every row
    strictly AFTER t by a huge, asset-dependent amount and recompute.  The
    value AT t must be bit-identical (no look-ahead), and the QE per-day IC
    AT t evaluated on those factor rows must be bit-identical too
    (factor -> evaluation).  IC rows strictly AFTER t must move -- otherwise
    the test could not distinguish a real causal factor from a no-op.
    """
    qe_ok, err_qe = _import_optional("quant_evaluator.runtime.evaluator")
    if err_qe:
        pytest.skip("import failure: " + err_qe)

    pos = _t_leak_position()

    panel = _make_panel(seed=0)
    base = _fe_run_factor(panel)
    perturbed = _fe_run_factor(_perturbed_after_t(panel))

    # The perturbed panel must actually differ AFTER t (guard against a no-op).
    future = np.asarray(panel.index.get_level_values("timestamp")) > T_LEAK
    assert future.any()
    base_future = base.to_numpy()[future]
    pert_future = perturbed.to_numpy()[future]
    assert not np.allclose(base_future, pert_future, equal_nan=True), (
        "perturbation had no effect after t -- the test would be vacuous"
    )

    # The value AT t is bit-identical across the two runs.
    base_at_t = base.xs(T_LEAK, level="timestamp")
    pert_at_t = perturbed.xs(T_LEAK, level="timestamp")
    assert base_at_t.index.equals(pert_at_t.index)
    assert _exact_equal(base_at_t.to_numpy(), pert_at_t.to_numpy()), (
        "factor value at t CHANGED when rows after t were perturbed: "
        "look-ahead in FE factor computation"
    )

    # End-to-end: the QE per-day IC at t on those factor rows is bit-identical.
    base_ic = _evaluate_pearson_ic_series(base)
    pert_ic = _evaluate_pearson_ic_series(perturbed)
    assert _exact_equal(base_ic[pos, :], pert_ic[pos, :]), (
        "QE per-day IC at t CHANGED when factor rows after t were perturbed: "
        "look-ahead into the evaluation"
    )
    # IC rows strictly after t legitimately consume future rows -> they move.
    assert not _exact_equal(base_ic[pos + 1 :, :], pert_ic[pos + 1 :, :]), (
        "per-day IC after t did NOT move under an asset-dependent future "
        "perturbation -- either the perturbation is being ignored or the "
        "evaluation is not consuming the perturbed rows"
    )


# ===========================================================================
# INT-T2: factor prefix invariance across FE -> FP -> QE
# ===========================================================================

def test_factor_prefix_invariance(_fe_run_factor):
    """
    CausalPrefixInvariance fed end-to-end across FE -> FP -> QE:

    - FE: ``ts_mean(close, 5)`` at t computed from the full panel equals the
      same factor computed from the prefix [0..t] only (real FactorEngine).
    - FP: ``rolling_mean`` (factor_preprocess) at t on the full long panel
      equals the transform computed on the prefix only.
    - QE: the per-day IC at t evaluated on the full-panel factor rows equals
      the IC evaluated on the prefix-only factor rows.
    """
    fp_ok, err_fp = _import_optional("factor_preprocess.transforms.rolling")
    qe_ok, err_qe = _import_optional("quant_evaluator.runtime.evaluator")
    missing = [e for e in (err_fp, err_qe) if e]
    if missing:
        pytest.skip("import failure: " + "; ".join(missing))

    pos = _t_leak_position()
    close = _make_panel(seed=3)
    prefix_close = close[close.index.get_level_values("timestamp") <= T_LEAK]

    # --- FE leg: FactorEngine ts_mean on full panel vs prefix. ---
    full = _fe_run_factor(close)
    pref = _fe_run_factor(prefix_close)
    full_at_t = full.xs(T_LEAK, level="timestamp")
    pref_at_t = pref.xs(T_LEAK, level="timestamp")
    assert full_at_t.index.equals(pref_at_t.index)
    assert _exact_equal(full_at_t.to_numpy(), pref_at_t.to_numpy()), (
        "FE factor at t computed on the prefix [0..t] differs from the full "
        "panel -- CausalPrefixInvariance violated in FactorEngine"
    )

    # --- FP leg: factor_preprocess rolling_mean on the long panel. ---
    from factor_preprocess.transforms.rolling import rolling_mean

    def _to_long(series: pd.Series) -> pd.DataFrame:
        df = series.reset_index()  # columns: timestamp, instrument, close
        df = df.rename(
            columns={"timestamp": "date", "instrument": "asset_id", "close": "value"}
        )
        return df.sort_values(["asset_id", "date"]).reset_index(drop=True)

    long_full = _to_long(close)
    long_pref = _to_long(prefix_close)
    fp_full = rolling_mean(long_full, window=5, min_periods=5).to_numpy()
    fp_pref = rolling_mean(long_pref, window=5, min_periods=5).to_numpy()
    full_at_t_fp = fp_full[long_full["date"].to_numpy() == T_LEAK]
    pref_at_t_fp = fp_pref[long_pref["date"].to_numpy() == T_LEAK]
    assert _exact_equal(full_at_t_fp, pref_at_t_fp), (
        "FP rolling_mean at t computed on the prefix differs from the full "
        "panel -- CausalPrefixInvariance violated in factor_preprocess"
    )

    # --- QE leg: evaluation on full-panel rows vs prefix-only rows. ---
    full_ic = _evaluate_pearson_ic_series(full)
    pref_padded = pref.reindex(full.index)  # prefix run has no rows > t
    pref_ic = _evaluate_pearson_ic_series(pref_padded)
    assert _exact_equal(full_ic[pos, :], pref_ic[pos, :]), (
        "QE per-day IC at t evaluated on prefix-only factor rows differs from "
        "the full-panel evaluation -- CausalPrefixInvariance violated in QE"
    )


# ===========================================================================
# INT-T3: label horizon beyond t is rejected at the boundary that exists
# ===========================================================================

def test_label_horizon_beyond_t_is_rejected(_fe_run_factor):
    """
    A factor at ``t`` combined with a label window that reaches past ``t`` must
    be rejected.  Asserted at the REAL fail-closed boundaries that exist:

    (a) FO ``validate_split_plan``: a train segment ending at ``t`` with a
        label horizon reaching INTO the test segment is rejected
        (forward-label overlap) -- via ``SplitPlan.label_horizon`` and via a
        FO ``LabelBundle`` with ``label_horizon`` (see the early-return note).

    (b) QE ``LabelBundle`` causal-chain validation: a label window that starts
        BEFORE its decision time (the label is knowable before the signal) is
        rejected; a non-monotonic decision axis is rejected.

    (c) DOCUMENTED WIRING GAP (R21-Q1/Q6): QE has no evaluation-boundary guard
        tying ``label_end_time`` to the factor evaluation window.  A
        causal-chain-valid bundle whose label window extends PAST the last
        factor row constructs and evaluates today.  The leak vector is pinned
        below as a contract assertion: the day a QE-side boundary guard
        exists, this assertion flips and the gap is closed.
    """
    fo_ok, err_fo = _import_optional("factor_optimizer.contracts.splits")
    qe_ok, err_qe = _import_optional("quant_evaluator.contracts.label_bundle")
    qe_ev_ok, err_ev = _import_optional("quant_evaluator.runtime.evaluator")
    missing = [e for e in (err_fo, err_qe, err_ev) if e]
    if missing:
        pytest.skip("import failure: " + "; ".join(missing))

    from factor_optimizer.contracts.splits import (
        LabelBundle as FOLabelBundle,
        SplitPlan,
        validate_split_plan,
    )
    from quant_evaluator.contracts.label_bundle import LabelBundle as QELabelBundle
    from quant_evaluator.runtime.evaluator import Evaluator

    pos = _t_leak_position()

    # ---- (a) FO split-plan forward-label-overlap rejection. ----
    # Train occupies rows 0..t (the factor through t); test starts at t+1;
    # a 3-day label horizon from the last train row reaches into test -> leak.
    n = pos + 4
    train_mask = [True] * (pos + 1) + [False] * (n - pos - 1)
    val_mask = [False] * n
    test_mask = [False] * (pos + 1) + [True] * (n - pos - 1)
    time_index = tuple(TIDX[:n].to_pydatetime())

    with pytest.raises(ValueError, match="forward-label overlap"):
        validate_split_plan(
            SplitPlan(
                split_id="s1",
                train_mask=train_mask,
                validation_mask=val_mask,
                test_mask=test_mask,
                metadata={},
                time_index=time_index,
                label_horizon=3,
            )
        )
    # Same rejection when the horizon arrives via a FO LabelBundle.  NOTE the
    # real boundary today: ``_validate_temporal_leakage`` early-returns when
    # ``SplitPlan.label_horizon == purge == embargo == validation_embargo == 0``
    # BEFORE it consults the label_bundle (splits.py:210-211), so a bundle
    # alone with a zero plan-level horizon silently skips the arithmetic.  The
    # bundle's ``label_horizon`` only takes effect when the plan-level fields
    # are non-zero.  We assert the enforced path (bundle + plan-level
    # label_horizon=3) and document the early-return gap in the evidence file.
    fo_label_bundle = FOLabelBundle(
        label_start_time=time_index,
        label_end_time=tuple(
            pd.Timestamp(x) + pd.Timedelta(days=3) for x in time_index
        ),
        label_horizon=3,
    )
    with pytest.raises(ValueError, match="forward-label overlap"):
        validate_split_plan(
            SplitPlan(
                split_id="s1",
                train_mask=train_mask,
                validation_mask=val_mask,
                test_mask=test_mask,
                metadata={},
                time_index=time_index,
                label_bundle=fo_label_bundle,
                label_horizon=3,
            )
        )
    # A purged plan (train 0..t, purge=2, test at t+3) is accepted: the gap
    # fully absorbs the 3-day label window.
    n2 = pos + 5
    train2 = [True] * (pos + 1) + [False] * (n2 - pos - 1)
    val2 = [False] * n2
    test2 = [False] * (pos + 3) + [True] * (n2 - pos - 3)
    validate_split_plan(
        SplitPlan(
            split_id="s1",
            train_mask=train2,
            validation_mask=val2,
            test_mask=test2,
            metadata={},
            time_index=tuple(TIDX[:n2].to_pydatetime()),
            purge=2,
        )
    )

    # ---- (b) QE LabelBundle causal-chain rejection (real boundary). ----
    times = tuple(TIDX[: pos + 2])
    with pytest.raises(ValueError):
        # label window STARTS before decision_time -> knowable before signal.
        QELabelBundle(
            target_id="leaky",
            values=np.zeros(len(times)),
            horizon=5,
            decision_time=times,
            label_start_time=tuple(t - pd.Timedelta(days=3) for t in times),
            label_end_time=tuple(t + pd.Timedelta(days=2) for t in times),
        )
    with pytest.raises(ValueError):
        # non-monotonic decision axis -> causal chain unexpressible.
        QELabelBundle(
            target_id="leaky",
            values=np.zeros(len(times)),
            horizon=1,
            decision_time=tuple(reversed(times)),
            label_start_time=tuple(times),
            label_end_time=tuple(t + pd.Timedelta(days=1) for t in times),
        )

    # ---- (c) Pinned wiring gap: no QE evaluation-boundary guard. ----
    # Build a factor through the real FE chain, then a causal-chain-valid label
    # bundle whose window extends PAST the last evaluated factor row.
    factor_series = _fe_run_factor(_make_panel(seed=0))
    eval_times = tuple(TIDX)
    horizon_past_boundary = 10
    bundle_past = QELabelBundle(
        target_id="fwd_ret_10",
        values=np.zeros((N_TIMES, N_ASSETS)),
        horizon=horizon_past_boundary,
        decision_time=eval_times,
        label_start_time=eval_times,
        label_end_time=tuple(t + pd.Timedelta(days=horizon_past_boundary) for t in eval_times),
    )
    last_factor_time = pd.Timestamp(TIDX[-1])
    assert bundle_past.label_end_time[-1] > last_factor_time, (
        "the label window must reach past the evaluation boundary for this "
        "gap-pin to be meaningful"
    )
    # It constructs (causal-chain-valid) and evaluates today -- this is the
    # missing QE guard, documented in evidence/r2/R21-INTEGRATION-TEMPORAL-SEPARATION.yaml.
    evaluator = Evaluator(enable_cache=False)
    result = evaluator.evaluate(
        _to_factor_batch(factor_series),
        bundle_past,
        [{"metric_id": "pearson_ic_series", "metric_kind": "custom"}],
        use_chunking=False,
    )
    assert "pearson_ic_series" in result.metrics, (
        "if a QE-side label-boundary guard is added, this gap-pin must be "
        "updated to assert the rejection instead of the current acceptance"
    )
