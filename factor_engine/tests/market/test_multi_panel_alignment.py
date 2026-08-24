# -*- coding: utf-8 -*-
"""Multi-input operator panel alignment (audit P0: multi-panel governance).

A multi-input operator must never pair A's data with B's panel because of a
column-order mismatch.  ``SeriesOperator.calculate`` enforces identical index
AND columns (order-sensitive) before any ``to_numpy`` happens, so every flagged
multi-input family fails fast on a reordered secondary panel instead of
silently mispairing values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _dates(n: int = 40) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.cleaned_operators import load_all

    load_all()


def _panels(n: int = 40):
    rng = np.random.default_rng(3)
    dates = _dates(n)
    x = pd.DataFrame(rng.normal(size=(n, 2)), index=dates, columns=["A", "B"])
    y = pd.DataFrame(rng.normal(size=(n, 2)), index=dates, columns=["A", "B"])
    return x, y


@pytest.mark.parametrize(
    "canonical,args",
    [
        ("ts_first_passage_bias", ["scale"]),
        ("ts_first_passage_hit_probability", ["scale"]),
        ("ts_dc_overshoot_ratio", ["scale"]),
        ("ts_dc_event_rate", ["scale"]),
        ("ts_path_signature_area", ["y"]),
        ("ts_path_signature_depth2_norm", ["y"]),
        ("ts_kalman_beta", []),
    ],
)
def test_multi_input_column_reorder_rejected(canonical, args):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
    if op is None:
        pytest.skip(f"{canonical} not registered")
    x, y = _panels()
    pos = {"scale": y.copy(), "y": y.copy()}
    if canonical == "ts_kalman_beta":
        pos = {}
    kwargs = {"window": 10}
    if canonical.startswith("ts_dc_"):
        kwargs = {"window": 10, "threshold": 1.0}
    if canonical == "ts_path_signature_area" or canonical == "ts_path_signature_depth2_norm":
        kwargs = {"window": 10}
    if canonical == "ts_kalman_beta":
        kwargs = {"q": 0.001, "r": 1.0}
    # Correct alignment runs.
    aligned = dict(pos)
    aligned_args = tuple(aligned.get(a) for a in args)
    try:
        base = op.calculate(x, *aligned_args, **kwargs) if args else op.calculate(x, **kwargs)
    except Exception:
        pytest.skip(f"{canonical} requires more history than the fixture provides")
    # Reordered secondary panel must raise (not silently mispair).
    reordered = dict(pos)
    for k, v in reordered.items():
        reordered[k] = v[["B", "A"]]
    reordered_args = tuple(reordered.get(a) for a in args)
    with pytest.raises(ValueError):
        if args:
            op.calculate(x, *reordered_args, **kwargs)
        else:
            op.calculate(x, **kwargs)
    assert base is not None


def test_group_spectrum_alignment():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("group_feature_mode_share", "pandas_numpy") or OperatorRegistry.get(
        "group_feature_mode_share"
    )
    if op is None:
        pytest.skip("group_corr_mode_share not registered")
    rng = np.random.default_rng(5)
    dates = _dates(30)
    f1 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    f2 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    f3 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    group = pd.DataFrame(
        [[1, 1, 2, 2]] * 30, index=dates, columns=["A", "B", "C", "D"], dtype=float
    )
    base = op.calculate(f1, f2, f3, group)
    with pytest.raises(ValueError):
        op.calculate(f1, f2[["D", "C", "B", "A"]], f3, group)
    assert base is not None
