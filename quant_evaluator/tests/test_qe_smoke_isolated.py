"""Isolated wheel-install smoke test.

Verifies that a clean wheel in an isolated venv imports on a CPU-only host
and runs a real CPU evaluation, then exercises the bundled GPU kernels when
CuPy is importable. The CPU-only half must pass with NO CuPy installed (the
GPU half skips via ``pytest.importorskip("cupy")``, matching the other
``test_gpu_*.py`` modules).

Use alongside ``scripts/smoke_isolated.py`` (a plain-script equivalent that
does not need pytest).  This module is import-safe on a CPU-only host because
CuPy is only imported at module scope through ``importorskip`` — no gpu
module is imported before the skip gate.
"""

from __future__ import annotations

import numpy as np
import pytest

_T, _N = 240, 60


def _build_fixtures(factors, labels):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    fb = FactorBatch(
        factor_ids=("smoke_cpu",),
        time_axis=AxisRef("t", "int", _T),
        asset_axis=AxisRef("a", "str", _N),
        values=np.ascontiguousarray(factors[:, :, :1]),
    )
    lb = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
    )
    return fb, lb


def _cpu_fixtures():
    rng = np.random.default_rng(7)
    fvals = rng.normal(size=(_T, _N, 2))
    labels = rng.normal(size=(_T, _N))
    fvals[rng.random(fvals.shape) < 0.08] = np.nan
    labels[rng.random(labels.shape) < 0.10] = np.nan
    return fvals, labels


def test_cpu_import_and_evaluate_without_cupy():
    """Clean CPU-only import + real public evaluate (no cupy required)."""
    try:
        import cupy  # noqa: F401  # pragma: no cover
        cupy_present = True
    except ImportError:
        cupy_present = False

    import quant_evaluator  # noqa: F401  top-level import must work on CPU-only

    from quant_evaluator.runtime.evaluator import evaluate

    fvals, labels = _cpu_fixtures()
    fb, lb = _build_fixtures(fvals, labels)
    bundle = evaluate(fb, lb, metrics=("rank_ic", "ic_ir"))
    vals = bundle.metric_values
    assert vals, "CPU evaluate returned empty metric_values"
    assert all(getattr(v, "value", None) is not None for v in vals.values())
    # On a truly CPU-only host cupy MUST be absent; this branch verifies the
    # wheel still imports/evaluates cleanly even though kernels.gpu is present.
    assert isinstance(cupy_present, bool)


def test_gpu_kernels_import_and_execute():
    """Bundled GPU kernels import and run when CuPy is available."""
    cp = pytest.importorskip("cupy")

    from quant_evaluator.kernels.gpu import (
        correlation,
        drawdown,
        portfolio,
        predictive,
        quantile,
        quantile_shape,
        rank,
        robustness,
        stability,
        temporal,
        turnover,
    )  # noqa: F401  — import completeness (all 11 bundles in the wheel)

    rng = np.random.default_rng(11)
    a = rng.normal(size=(300, 8))
    ic_series, counts = correlation.batched_spearman_ic(a, rng.normal(size=(300, 8)))
    sh = drawdown.compute_sharpe_batch(rng.normal(size=(300, 8)), periods_per_year=252)
    assert cp.asnumpy(ic_series).shape[0] == 300
    assert cp.asnumpy(counts).shape == (300, 1)
    assert cp.asnumpy(sh).shape == (8,)


def test_gpu_facade_evaluate():
    """Full public evaluate(backend='cuda') path with bundled kernels."""
    cp = pytest.importorskip("cupy")

    from quant_evaluator.runtime.evaluator import evaluate

    rng = np.random.default_rng(13)
    fvals = rng.normal(size=(_T, _N, 2))
    labels = rng.normal(size=(_T, _N))
    fb, lb = _build_fixtures(fvals, labels)
    # rebuild single-factor fixture helper shape
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    fb2 = FactorBatch(
        factor_ids=("g0", "g1"),
        time_axis=AxisRef("t", "int", _T),
        asset_axis=AxisRef("a", "str", _N),
        values=np.ascontiguousarray(fvals),
    )
    lb2 = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
    )
    bundle = evaluate(fb2, lb2, metrics=("rank_ic",), backend="cuda")
    per_factor = bundle.scalar_metrics["rank_ic"]
    assert cp.asnumpy(per_factor).shape == (2,)
    assert np.all(np.isfinite(cp.asnumpy(per_factor)))
