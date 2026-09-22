"""R63 operator speed gate: no registered daily operator may be row-wise slow.

The R62 sweep timed every daily operator on a 750-row panel; the slow tail
(>=30ms) was entirely per-instrument / per-day Python loops and has been
vectorized in R63 (microstructure batch panel, MedRV prefix scan, quantile /
RQA sliding kernels).  This gate re-times the whole daily registry and fails
when any operator exceeds the budget, so a future "small fix" cannot
silently reintroduce a row-wise implementation.

Budget: 300ms on a 750-row single-column panel.
  - every vectorized operator in R63 sweep2 runs < 13ms;
  - the one documented intrinsic floor (ts_persistence_entropy_h1, 160ms of
    serial pivot-XOR over a 220-simplex triangulation) stays well under it;
  - the row-wise killer pattern measured ~385ms and is caught.
Override in CI via ``R63_SPEED_BUDGET_MS``.
"""
from __future__ import annotations

import inspect
import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators as _ce
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import operator_surface as _surface

_load_all = getattr(_ce, "load_all", None)
if callable(_load_all):
    _load_all()

warnings.filterwarnings("ignore")

_BUDGET_MS = float(os.environ.get("R63_SPEED_BUDGET_MS", "300"))

_IDX = pd.bdate_range("2020-01-01", periods=750)

# params that may be filled without operator-specific knowledge; anything
# outside this mapping and without a catalog default is skipped (PARAM_GAP) --
# the same policy the R62/R63 sweeps used.
_SAFE_WINDOW = {
    "window", "period", "lag", "length", "history", "count", "bins", "lookback",
    "span", "min_periods", "min_history", "min_breadth", "min_line", "min_nodes",
    "min_coverage_fraction", "min_transitions", "smooth", "decay_scale", "tau",
    "shift_sigma", "threshold", "r", "m", "max_agg", "embedding_dim", "dim",
    "order", "k", "theiler", "refractory", "q", "horizon", "d", "fast_window",
    "slow_window", "subsequence_length", "n_bins", "n_levels", "max_lag", "n_lags",
}


def _fill_params(canon: str) -> tuple[dict, list[str]]:
    cat = OperatorRegistry._catalog.get(canon, {})
    specs = cat.get("param_specs") or {}
    params: dict = {}
    missing: list[str] = []
    for pname, spec in specs.items():
        dflt = getattr(spec, "default", None)
        if dflt is not None and str(dflt) != "MISSING":
            params[pname] = dflt
            continue
        pl = pname.lower()
        if pl in _SAFE_WINDOW:
            if "min_periods" in pl or pl.startswith("min_"):
                params[pname] = 3
            elif pl in ("smooth", "decay_scale", "tau", "shift_sigma"):
                params[pname] = 2.0
            elif pl in ("threshold", "r", "m", "max_agg"):
                params[pname] = 0.5
            elif pl in ("embedding_dim", "dim", "order", "k", "bins", "theiler", "refractory"):
                params[pname] = 3
            else:
                params[pname] = 20
        elif pl == "session_tz":
            params[pname] = "Asia/Shanghai"
        elif pl in ("rank", "delay", "level", "topk", "n", "num", "count", "deg", "degree"):
            params[pname] = 1
        elif pl == "group":
            params[pname] = 0
        elif pl == "groups":
            params[pname] = 2
        elif getattr(spec, "choices", None):
            params[pname] = spec.choices[0]
        else:
            missing.append(pname)
    return params, missing


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.standard_normal(750)) + 100.0
    return pd.DataFrame({"s0": x}, index=_IDX)


def _daily_canonicals() -> set[str]:
    daily = set(getattr(_surface, "DAILY_CANONICALS", ()) or ())
    migrated = getattr(_surface, "DAILY_FACTOR_MIGRATED", None)
    if migrated is None:
        fn = getattr(_surface, "daily_factor_migrated", None)
        migrated = fn() if callable(fn) else ()
    return daily | set(migrated or ())


@pytest.mark.slow
def test_daily_operators_within_speed_budget():
    daily = _daily_canonicals()
    canon_list = sorted(c for c in OperatorRegistry._operators if c in daily)
    assert len(canon_list) > 1000, (
        f"daily registry looks wrong ({len(canon_list)}) -- gate is not real"
    )
    df = _panel()
    offenders: list[tuple[float, str]] = []
    timed = 0
    for canon in canon_list:
        pod = (OperatorRegistry._operators.get(canon) or {}).get("pandas_numpy")
        if pod is None or not hasattr(pod, "_calculate_series"):
            continue
        params, missing = _fill_params(canon)
        if missing:
            continue
        try:
            # warm call: JIT/compile noise must not pollute the measurement
            # (the sweep harness warmed once and took min-of-2 for the same
            # reason); the timed call below still catches row-wise regressions.
            pod._calculate_series(df.copy(), **params)
            t0 = time.perf_counter()
            pod._calculate_series(df.copy(), **params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            # data-shape gaps are the conformance suite's job, not this gate's
            continue
        timed += 1
        if elapsed_ms >= _BUDGET_MS:
            offenders.append((elapsed_ms, canon))
    # The R63 sweep timed 503/1429 (the rest hit harness call-convention
    # gaps: multi-input operators, intraday kernels needing session data).
    # 400 is a tripwire: a mass kernel/timing regression drops below it.
    assert timed >= 400, f"only {timed} operators timed -- gate is not real"
    offenders.sort(reverse=True)
    assert not offenders, (
        f"{len(offenders)} operators exceed the {_BUDGET_MS:.0f}ms speed budget "
        "(row-wise implementations regress the batch; vectorize them):\n  "
        + "\n  ".join(f"{ms:8.1f}ms  {canon}" for ms, canon in offenders[:40])
    )
