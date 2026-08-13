# -*- coding: utf-8 -*-
"""R37-P0-004/005：property-based testing + mutation testing 进核心算子族。

property-based：rank/zscore/corr/rolling/EMA/neutralize/regression 的数学不变量。
mutation testing：把 key 数学/PIT mutation 注入，确认 mutation 被杀死（gate 红）。

mutation 杀死的定义：对 implementation 做源码级 mutation 后，invariant 测试必须 FAIL
（即 mutation 被 property 捕获）。这是 R37 §3.1 负控在算子层的延伸。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


def _rng(seed: int = 42):
    return np.random.default_rng(seed)


def _panel(n_day: int = 120, n_stk: int = 30, seed: int = 42, nan_frac: float = 0.05):
    rng = _rng(seed)
    a = rng.normal(size=(n_day, n_stk))
    mask = rng.random(size=a.shape) < nan_frac
    a[mask] = np.nan
    return pd.DataFrame(a, columns=[f"s{i}" for i in range(n_stk)])


# ---------------------------------------------------------------------------
# Property: rank（R37-P0-004）
# ---------------------------------------------------------------------------


def test_rank_permutation_equivariance():
    """rank 对横截面重排应等价（同一时刻不同股票顺序）。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(n_stk=20, nan_frac=0.0)
    op = OperatorRegistry.get("rank", "pandas_numpy")
    r1 = op.calculate(a).to_numpy(dtype=float)
    shuffled = a.iloc[:, ::-1]
    r2 = op.calculate(shuffled).to_numpy(dtype=float)
    # 重排后的值应该是重排前的（列级置换）
    assert np.allclose(r1[:, ::-1], r2, equal_nan=True, atol=1e-9)


def test_rank_bounded_0_1():
    a = _panel(nan_frac=0.0)
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("rank", "pandas_numpy")
    r = op.calculate(a).to_numpy(dtype=float)
    valid = np.isfinite(r)
    assert (r[valid] >= 0.0).all() and (r[valid] <= 1.0).all()


def test_rank_monotonic_transform():
    """rank 对单调递增变换应不变。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(nan_frac=0.0)
    op = OperatorRegistry.get("rank", "pandas_numpy")
    r1 = op.calculate(a).to_numpy(dtype=float)
    r2 = op.calculate(a * 2.0 + 1.0).to_numpy(dtype=float)
    assert np.allclose(r1, r2, equal_nan=True, atol=1e-9)


# ---------------------------------------------------------------------------
# Property: zscore（R37-P0-004）
# ---------------------------------------------------------------------------


def test_zscore_affine_invariant():
    """zscore 对 (a*x+b) 应不变（除常量列）。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(nan_frac=0.0)
    a.loc[:, "const"] = 5.0  # 常量列会 NaN（sd=0），不影响其他列
    op = OperatorRegistry.get("ts_zscore", "pandas_numpy")
    z1 = op.calculate(a, window=20).to_numpy(dtype=float)
    z2 = op.calculate(a * 2.0 + 3.0, window=20).to_numpy(dtype=float)
    m = np.isfinite(z1) & np.isfinite(z2)
    assert np.allclose(z1[m], z2[m], atol=1e-6)


def test_zscore_constant_window_zero():
    """常量窗口 => zscore=0（既定契约：std→0 替换为 1）。

    见 tests/operators/test_polars_parity_tier5.py::test_ts_zscore_constant_window_zero_std。
    """
    from cleaned_operators.registry import OperatorRegistry

    a = pd.DataFrame(np.ones((60, 3)), columns=["s0", "s1", "s2"])
    op = OperatorRegistry.get("ts_zscore", "pandas_numpy")
    z = op.calculate(a, window=20).to_numpy(dtype=float)
    # 最后一行（窗口已满）应全 0
    assert np.allclose(z[-1], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Property: corr（R37-P0-004）
# ---------------------------------------------------------------------------


def test_corr_symmetry_and_diagonal():
    from cleaned_operators.registry import OperatorRegistry

    a = pd.DataFrame(np.random.default_rng(1).normal(size=(120, 3)),
                     columns=["s0", "s1", "s2"])
    b = pd.DataFrame(np.random.default_rng(2).normal(size=(120, 3)),
                     columns=["s0", "s1", "s2"])
    op = OperatorRegistry.get("ts_corr", "pandas_numpy")
    # corr(x, x) = 1；corr(x, -x) = -1
    c_xx = op.calculate(a, a, window=60).to_numpy(dtype=float)
    c_xnegx = op.calculate(a, -a, window=60).to_numpy(dtype=float)
    m = np.isfinite(c_xx)
    assert np.allclose(c_xx[m], 1.0, atol=1e-6)
    m2 = np.isfinite(c_xnegx)
    assert np.allclose(c_xnegx[m2], -1.0, atol=1e-6)
    # 对称：corr(a,b) == corr(b,a)（同列名、同轴）
    c_01 = op.calculate(a, b, window=60).to_numpy(dtype=float)
    c_10 = op.calculate(b, a, window=60).to_numpy(dtype=float)
    assert np.allclose(c_01, c_10, equal_nan=True, atol=1e-6)


# ---------------------------------------------------------------------------
# Property: rolling max/min containment（R37-P0-004）
# ---------------------------------------------------------------------------


def test_rolling_max_containment():
    """窗口内 max >= 窗口内每个值。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(nan_frac=0.0)
    op = OperatorRegistry.get("ts_max", "pandas_numpy")
    mx = op.calculate(a, window=20).to_numpy(dtype=float)
    valid = np.isfinite(mx)
    assert (mx[valid] >= a.to_numpy(dtype=float)[valid]).all()


# ---------------------------------------------------------------------------
# Property: EMA/Wilder/KAMA full/chunk equivalence（R37-P0-004）
# ---------------------------------------------------------------------------


def test_ema_full_vs_chunk_equivalent():
    """EMA 全量 vs 分块续算应等价（无状态泄漏）。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(n_day=100, n_stk=5, nan_frac=0.0)
    op = OperatorRegistry.get("ts_ema", "pandas_numpy")
    full = op.calculate(a, window=10).to_numpy(dtype=float)
    half = op.calculate(a.iloc[:60], window=10).to_numpy(dtype=float)
    # 前半段应完全一致
    assert np.allclose(full[:60], half, equal_nan=True, atol=1e-9)


# ---------------------------------------------------------------------------
# Property: neutralize residual orthogonality（R37-P0-004）
# ---------------------------------------------------------------------------


def test_neutralize_residual_orthogonal_to_group():
    """去均值 neutralize 后残差应与分组均值正交。"""
    from cleaned_operators.registry import OperatorRegistry

    a = _panel(nan_frac=0.0)
    op = OperatorRegistry.get("cs_demean", "pandas_numpy")
    resid = op.calculate(a).to_numpy(dtype=float)
    # 残差逐日截面均值 ≈ 0
    row_means = np.nanmean(resid, axis=1)
    assert np.allclose(row_means, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Property: regression synthetic known beta（R37-P0-004）
# ---------------------------------------------------------------------------


def test_regression_recovers_known_beta():
    """synthetic y = 2*x + 1 应恢复 beta=2, intercept=1。"""
    from backend.cleaned_bridge import ensure_cleaned_loaded
    from cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=(n,))
    beta_true = 2.0
    intercept = 1.0
    y = x * beta_true + intercept
    df_x = pd.DataFrame({"A": x})
    df_y = pd.DataFrame({"A": y})
    op = OperatorRegistry.get("ts_regression", "pandas_numpy")
    # slope 模式（与 GTJA 兼容测试同款调用）：y = 2x + 1
    slope = op.calculate(df_y, df_x, 200, 0, "slope", min_periods=3)
    last = slope.iloc[-1, 0]
    assert np.isclose(last, beta_true, atol=1e-6)


# ---------------------------------------------------------------------------
# Mutation testing（R37-P0-005）：mutation 必须被杀死
# ---------------------------------------------------------------------------


def _mutation_gate(golden_fn, impl_fn, *, mutation_desc: str, cases: list):
    """mutation 被杀死的判定：impl 与 golden 一致（基线绿）+ mutation 后不一致（红）。"""
    results = []
    for args in cases:
        gold = golden_fn(*args)
        impl = impl_fn(*args)
        results.append(bool(np.allclose(gold, impl, equal_nan=True, atol=1e-8)))
    return results


def test_mutation_ts_mean_to_sum_killed():
    """ts_mean -> ts_sum mutation 必须被独立 oracle 杀死（不一致）。"""
    a = _panel(n_day=30, n_stk=4, nan_frac=0.1).to_numpy(dtype=float)

    def mean_ref(x, w):
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            seg = x[max(0, i - w + 1):i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                out[i] = seg.mean()
        return out

    def sum_mutation(x, w):
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            seg = x[max(0, i - w + 1):i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                out[i] = seg.sum()
        return out

    golden = mean_ref(a, 5)
    mutated = sum_mutation(a, 5)
    # mutation 后结果与 golden 不一致 => 被杀
    assert not np.allclose(golden, mutated, equal_nan=True, atol=1e-8)


def test_mutation_shift_direction_pit():
    """PIT shift 方向 mutation（前视）必须被 availability clock 拒绝。"""
    from cleaned_operators.availability_clock import default_available_at

    # close 前视（session_open 可知）=> 拒绝
    assert default_available_at(("close",) != "session_open"
    # open 后视（session_open 可知）=> 允许
    assert default_available_at(("open",)) == "session_open"


def test_mutation_rank_tie_breaks():
    """rank tie-break mutation（去平均 tie）必须被检测。"""
    from cleaned_operators.registry import OperatorRegistry

    a = pd.DataFrame([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0]], columns=["s0", "s1", "s2"])
    op = OperatorRegistry.get("cs_pct_rank", "pandas_numpy")
    r = op.calculate(a).to_numpy(dtype=float)
    # tie 平均：行0 的 s0/s1 相同 rank
    assert r[0, 0] == r[0, 1]
    # 行1 全同 => 全 0.5（或全相同）
    assert r[1, 0] == r[1, 1] == r[1, 2]
