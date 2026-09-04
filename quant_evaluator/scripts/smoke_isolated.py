#!/usr/bin/env python3
"""Isolated-env smoke for quant_evaluator (wheel install verification).

Checks, in order:
  1. CPU-only import of the top-level package plus the public `evaluate`
     facade with only the core deps (numpy/scipy/pandas/psutil) installed.
  2. A real single-factor CPU evaluation returning non-empty metric_values.
  3. If CuPy is importable (GPU host), imports every module under
     ``quant_evaluator.kernels.gpu`` and executes one real batched GPU kernel
     (batched_spearman_ic + compute_sharpe_batch), then runs the full
     ``evaluate(..., backend="cuda")`` GPU facade path.

Run inside an isolated venv that has the clean wheel installed:
    python scripts/smoke_isolated.py
"""

from __future__ import annotations

import sys

_T, _N, _F = 240, 60, 2


def _run_cpu() -> bool:
    import numpy as np
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    rng = np.random.default_rng(0)
    fvals = rng.normal(size=(_T, _N, _F))
    labels = rng.normal(size=(_T, _N))
    fvals[rng.random(fvals.shape) < 0.08] = np.nan
    labels[rng.random(labels.shape) < 0.10] = np.nan

    fb = FactorBatch(
        factor_ids=("smoke_cpu",),
        time_axis=AxisRef("t", "int", _T),
        asset_axis=AxisRef("a", "str", _N),
        values=np.ascontiguousarray(fvals[:, :, :1]),
    )
    lb = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
    )
    bundle = evaluate(fb, lb, metrics=("rank_ic", "ic_ir"))
    vals = bundle.metric_values
    if not vals or any(getattr(v, "value", None) is None for v in vals.values()):
        raise RuntimeError(f"CPU evaluate returned empty/None values: {vals}")
    print(f"  CPU evaluate ok -> { {k: float(v.value) for k, v in vals.items()} }")
    return True


def _run_gpu() -> bool:
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        print("  GPU skip: cupy not importable in this env")
        return True

    import numpy as np
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    _GPUKERNELS = (
        "correlation",
        "drawdown",
        "portfolio",
        "predictive",
        "quantile",
        "quantile_shape",
        "rank",
        "robustness",
        "stability",
        "temporal",
        "turnover",
    )
    for name in _GPUKERNELS:
        __import__(f"quant_evaluator.kernels.gpu.{name}")
        print(f"  import kernels.gpu.{name} ok")

    import quant_evaluator.kernels.gpu.correlation as corr
    import quant_evaluator.kernels.gpu.drawdown as dd

    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 8))
    sc = corr.batched_spearman_ic(a, rng.normal(size=(300, 8)))
    sh = dd.compute_sharpe_batch(rng.normal(size=(300, 8)), periods_per_year=252)
    print(f"  gpu kernel exec ok: spearman shape={getattr(sc, 'shape', None)} sharpe shape={getattr(sh, 'shape', None)}")

    fvals = rng.normal(size=(_T, _N, _F))
    labels = rng.normal(size=(_T, _N))
    fb = FactorBatch(
        factor_ids=("g0", "g1"),
        time_axis=AxisRef("t", "int", _T),
        asset_axis=AxisRef("a", "str", _N),
        values=np.ascontiguousarray(fvals),
    )
    lb = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
    )
    bundle = evaluate(fb, lb, metrics=("rank_ic",), backend="cuda")
    per_factor = bundle.scalar_metrics["rank_ic"]
    print(f"  GPU facade evaluate ok -> rank_ic per factor: {cp.asnumpy(per_factor).tolist()}")
    return True


def main() -> int:
    import quant_evaluator  # top-level clean import (core deps only)

    print("import quant_evaluator ok")
    ok_cpu = _run_cpu()
    ok_gpu = _run_gpu()
    print("SMOKE_PASS" if (ok_cpu and ok_gpu) else "SMOKE_FAIL")
    return 0 if (ok_cpu and ok_gpu) else 1


if __name__ == "__main__":
    sys.exit(main())
