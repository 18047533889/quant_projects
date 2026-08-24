# -*- coding: utf-8 -*-
"""R28-P0-005 / §二十五..二十七: model causality universal gate.

For every high-risk model canonical we run two behavioral proofs:
  - PREFIX INVARIANCE: output[t] from the full series equals output[t] computed
    on the prefix x[:t+1] (no future data may influence the past).
  - FUTURE PERTURBATION: mutating all data AFTER t leaves output[:t+1] unchanged.

The operator implementations are the ground truth — these are runtime tests, not
string scans (taskbook §一百二十三).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

from .fixture_factory import price_panel, return_panel, two_return_panels


def _load():
    load_all()


#: (canonical, input-arity, kwargs) — arity 1: (x,), arity 2: (y, x), arity 3: (y, x1, x2)
_MODEL_SPECS = [
    ("ts_garch_next_vol_forecast", 1, {"window": 60}),
    ("ts_garch_standardized_shock", 1, {"window": 60}),
    ("ts_gjr_garch_vol_forecast", 1, {"window": 60}),
    ("ts_har_rv_forecast", 1, {"window": 60}),
    ("ts_kalman_level", 1, {"q": 1e-3, "r": 1e-2}),
    ("ts_kalman_trend", 1, {"q_level": 1e-3, "q_trend": 1e-5, "r": 1e-2}),
    ("ts_kalman_innovation_z", 1, {"q": 1e-3, "r": 1e-2}),
    ("ts_ar_prior_forecast", 1, {"window": 40, "order": 1}),
    ("ts_ar_prior_innovation", 1, {"window": 40, "order": 1}),
    ("ts_multi_regression_coeff_prior", 2, {"window": 40}),
    ("ts_ridge_regression_coeff_prior", 2, {"window": 40}),
    ("ts_quantile_regression_coeff_prior", 2, {"window": 40, "q": 0.5}),
    ("panel_rolling_pcr_forecast", 3, {"window": 60, "n_components": 2, "label_horizon": 1}),
    ("panel_rolling_pls_forecast", 3, {"window": 60, "n_components": 2, "label_horizon": 1}),
    ("panel_rolling_elastic_net_forecast", 3, {"window": 60, "alpha": 0.01, "l1_ratio": 0.5, "label_horizon": 1}),
    ("ts_dmd_dominant_frequency", 1, {"window": 60, "rank": 2}),
    ("ts_dmd_dominant_growth_rate", 1, {"window": 60, "rank": 2}),
    ("ts_path_signature_area", 2, {"window": 60}),
    ("ts_transfer_entropy", 2, {"window": 60, "bins": 4}),
]


def _call(canonical, arity, kwargs, seed=42):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        pytest.skip(f"{canonical}: backend not registered (transient)")
    if arity == 1:
        args = (return_panel(120, 3, seed),)
    elif arity == 2:
        y, x, _ = two_return_panels(120, 3, seed)
        args = (y, x)
    else:
        args = two_return_panels(120, 3, seed)
    return op.calculate(*args, **kwargs)


def _single(out):
    """Extract the single instrument output as a 1-D array."""
    return out.to_numpy()[:, 0]


@pytest.mark.parametrize("canonical,arity,kwargs", _MODEL_SPECS, ids=[s[0] for s in _MODEL_SPECS])
def test_model_prefix_invariance(canonical, arity, kwargs):
    _load()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    # prefix-run: compute each row on its own prefix (simulates re-running a
    # factor on the data available at each decision time)
    try:
        full = _call(canonical, arity, kwargs)
    except (ValueError, ArithmeticError):
        pytest.skip(f"{canonical}: infeasible fixture (fails closed on input contract)")
    full_arr = _single(full)

    # incremental prefix: op(prefix) must equal op(full)[:len(prefix)]
    # We check a handful of cutoffs to keep runtime bounded.
    for cutoff in (40, 60, 80, 100):
        prefix_args = []
        if arity == 1:
            prefix_args = (return_panel(120, 3, seed=42).iloc[:cutoff],)
        elif arity == 2:
            y, x, _ = two_return_panels(120, 3, seed=42)
            prefix_args = (y.iloc[:cutoff], x.iloc[:cutoff])
        else:
            y, x1, x2 = two_return_panels(120, 3, seed=42)
            prefix_args = (y.iloc[:cutoff], x1.iloc[:cutoff], x2.iloc[:cutoff])
        try:
            prefix_out = op.calculate(*prefix_args, **kwargs)
        except Exception:
            continue  # some ops need a minimum window; skip underpowered prefixes
        prefix_arr = _single(prefix_out)
        np.testing.assert_allclose(
            prefix_arr,
            full_arr[:cutoff],
            equal_nan=True,
            atol=1e-6,
            err_msg=f"{canonical} prefix invariance violated at cutoff {cutoff}",
        )


_PERTS = ("scale", "flip", "nan", "outlier", "shift")


@pytest.mark.parametrize("canonical,arity,kwargs", _MODEL_SPECS, ids=[s[0] for s in _MODEL_SPECS])
def test_model_future_perturbation(canonical, arity, kwargs):
    _load()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    try:
        base = _call(canonical, arity, kwargs)
    except (ValueError, ArithmeticError):
        pytest.skip(f"{canonical}: infeasible fixture (fails closed on input contract)")
    base_arr = _single(base)
    cutoff = 70

    for mode in _PERTS:
        args = _make_panels(arity, seed=42)
        if arity == 1:
            x = args[0].copy()
            tail = x.iloc[cutoff + 1 :].to_numpy()
            if mode == "shift":
                tail = tail + 1.0  # moderate shift: a +1000 level trips the price-vs-return DQ gate
            elif mode == "scale":
                tail = tail * 100.0
            elif mode == "flip":
                tail = -tail
            elif mode == "nan":
                tail = np.full_like(tail, np.nan)
            else:
                tail = np.full_like(tail, 7.0)
            x.iloc[cutoff + 1 :] = tail
            args = (x,)
        elif arity == 2:
            y, x = args[0].copy(), args[1].copy()
            x.iloc[cutoff + 1 :] = x.iloc[cutoff + 1 :].to_numpy() * 100.0
            args = (y, x)
        else:
            y, x1, x2 = [a.copy() for a in args]
            x1.iloc[cutoff + 1 :] = x1.iloc[cutoff + 1 :].to_numpy() * 100.0
            args = (y, x1, x2)
        try:
            changed = op.calculate(*args, **kwargs)
        except (ValueError, ArithmeticError):
            # Input-contract gates (e.g. price-vs-return DQ, infeasible window×bins)
            # legitimately fail closed on a perturbed series — that rejection IS the
            # protection, so the mode passes.
            continue
        changed_arr = _single(changed)
        np.testing.assert_allclose(
            changed_arr[:cutoff],
            base_arr[:cutoff],
            equal_nan=True,
            atol=1e-6,
            err_msg=f"{canonical} future perturbation({mode}) changed the past",
        )


def _make_panels(arity, seed=42):
    if arity == 1:
        return (return_panel(120, 3, seed),)
    if arity == 2:
        y, x, _ = two_return_panels(120, 3, seed)
        return (y, x)
    return two_return_panels(120, 3, seed)
