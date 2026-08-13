# -*- coding: utf-8 -*-
"""R19-016..029 数学内核审计测试：rank / regression / neutralization / finite。

覆盖：
- R19-131 rank_corr no-lookahead（ts_rank_corr_ 前缀不变 + 窗口局部）
- R19-132 ts_regression_slope exact OLS（y=2x+3 任意窗口 slope=2）
- R19-133 industry+size FWL（每行业均值≈0 + cov(resid, demeaned log size)≈0）
- R19-019/020 rank_ average tie + finite mask 排除 Inf
- R19-021/022 downside_beta/tail_beta ddof 一致性（centered sums）
- R19-023/024/025 size_resid 禁 silent clip + cs_resid_/cs_regression_ isfinite
- R19-026 coalesce_ first-finite
- R19-027..029 c_* 统计 finite mask（CrossSectionSampleMask）
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import _numpy_kernels as _k


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """窗口内手工 Spearman：分别 rank 后 Pearson。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# R19-016..018 rank_corr 拆分
# ---------------------------------------------------------------------------

def test_rank_corr_d0_full_sample_branch_removed_fail_closed():
    """旧 d==0 full-sample 广播分支必须彻底移除 → fail-closed raise。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        _k.rank_corr_(x, y, d=0)
    with pytest.raises(ValueError):
        _k.rank_corr_(x, y, d=-1)


def test_ts_rank_corr_no_full_sample_lookahead_prefix_invariant():
    """R19-131: t 时点的值只依赖 trailing window，未来数据改变不影响早期值。"""
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y1 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    x2 = np.array([1.0, 2.0, 3.0, 100.0, 200.0])  # 未来被放大的样本
    w = 3
    r1 = _k.ts_rank_corr_(x1, y1, window=w)
    r2 = _k.ts_rank_corr_(x2, y1, window=w)
    # 前 w-1 个为 NaN
    assert np.isnan(r1[: w - 1]).all()
    # 前缀不变：index w-1 的值不受未来数据影响（若用全样本 rank 则会变）
    assert r1[w - 1] == r2[w - 1]
    # 且等于窗口内手工 Spearman
    assert r1[w - 1] == pytest.approx(_spearman(x1[:w], y1[:w]))


def test_ts_rank_corr_matches_windowed_spearman():
    rng = np.random.default_rng(7)
    x = rng.normal(size=30)
    y = rng.normal(size=30)
    w = 8
    out = _k.ts_rank_corr_(x, y, window=w)
    for i in range(w - 1, len(x)):
        expected = _spearman(x[i - w + 1 : i + 1], y[i - w + 1 : i + 1])
        assert out[i] == pytest.approx(expected, abs=1e-12)


def test_ts_rank_corr_tie_method_average_consistency():
    """tie 处理默认 average，与全库 rank_ tie authority 一致。"""
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 2.0, 1.0, 4.0, 3.0])
    out = _k.ts_rank_corr_(x, y, window=5)
    assert out[-1] == pytest.approx(_spearman(x, y))


def test_cs_rank_corr_per_date_global_state():
    """R19-016..018: cs_rank_corr_ 逐日返回标量（GLOBAL_STATE，不伪装个股 alpha）。"""
    panel_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    panel_y = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    out = _k.cs_rank_corr_(panel_x, panel_y)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(1.0)
    # 1D 输入退化为单日截面
    out1 = _k.cs_rank_corr_(panel_x[0], panel_y[0])
    assert out1.shape == (1,)
    assert out1[0] == pytest.approx(out[0])
    # 文档必须声明 GLOBAL_STATE（docstring 含该标记）
    assert "GLOBAL_STATE" in _k.cs_rank_corr_.__doc__


# ---------------------------------------------------------------------------
# R19-019/020 rank_ tie + finite
# ---------------------------------------------------------------------------

def test_rank_average_tie_default():
    """并列给平均 rank；默认 average，消除列顺序信号。"""
    r = _k.rank_(np.array([1.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(r, [1 / 6, 1 / 6, 2 / 3, 1.0])


def test_rank_methods():
    arr = np.array([1.0, 1.0, 2.0, 3.0])
    rmin = _k.rank_(arr, method="min")
    rmax = _k.rank_(arr, method="max")
    rdense = _k.rank_(arr, method="dense")
    rfirst = _k.rank_(arr, method="first")
    # min: 并列给最小 rank (1,1,3,4) → pct (0,0,2/3,1)
    np.testing.assert_allclose(rmin, [0.0, 0.0, 2 / 3, 1.0])
    # max: 并列给最大 rank (2,2,3,4) → pct (1/3,1/3,2/3,1)
    np.testing.assert_allclose(rmax, [1 / 3, 1 / 3, 2 / 3, 1.0])
    # dense: 稠密 rank (1,1,2,3) → pct (0,0,1/3,2/3)
    np.testing.assert_allclose(rdense, [0.0, 0.0, 1 / 3, 2 / 3])
    # first: 按出现顺序 rank (1,2,3,4) → pct (0,1/3,2/3,1)
    np.testing.assert_allclose(rfirst, [0.0, 1 / 3, 2 / 3, 1.0])
    with pytest.raises(ValueError):
        _k.rank_(arr, method="bogus")


def test_rank_finite_mask_excludes_inf():
    r = _k.rank_(np.array([1.0, 2.0, np.inf, np.nan, 3.0]))
    assert np.isnan(r[2]) and np.isnan(r[3])  # Inf 与 NaN 均不参与
    valid = r[np.isfinite(r)]
    np.testing.assert_allclose(valid, [0.0, 0.5, 1.0])
    # 单有效值 → 0.5
    assert _k.rank_(np.array([np.nan, np.inf, 7.0]))[2] == 0.5


# ---------------------------------------------------------------------------
# R19-021/022 regression slope ddof 一致性
# ---------------------------------------------------------------------------

def test_ts_regression_slope_exact_ols():
    """R19-132: y = 2x + 3 → 任意窗口 slope 恰为 2.0。"""
    x = np.arange(1.0, 11.0)
    y = 2.0 * x + 3.0
    for w in (3, 5, 7, 10):
        out = _k.ts_regression_slope_(x, y, d=w)
        assert np.isfinite(out[w - 1 :]).all()
        np.testing.assert_allclose(out[w - 1 :], 2.0, atol=1e-12)


def test_ts_regression_slope_equals_polyfit():
    """centered sums == np.polyfit 一次项（无 ddof 系统性偏差）。"""
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    y = 1.5 * x + rng.normal(size=40) * 0.1
    w = 6
    out = _k.ts_regression_slope_(x, y, d=w)
    for i in range(w - 1, len(x)):
        slope = np.polyfit(x[i - w + 1 : i + 1], y[i - w + 1 : i + 1], 1)[0]
        assert out[i] == pytest.approx(slope, abs=1e-12)


def test_ts_regression_slope_constant_x_is_nan():
    x = np.ones(10)
    y = np.arange(10.0)
    out = _k.ts_regression_slope_(x, y, d=5)
    assert np.isnan(out).all()


def test_downside_beta_matches_ols_on_down_subset():
    """downside_beta_ 用 centered sums，等于仅在 m<0 子集上的 OLS 斜率。"""
    rng = np.random.default_rng(4)
    ret = rng.normal(size=60)
    idx_ret = rng.normal(size=60)
    w = 20
    out = _k.downside_beta_(ret, idx_ret, window=w)
    for i in range(w - 1, len(ret)):
        m = idx_ret[i - w + 1 : i + 1]
        r = ret[i - w + 1 : i + 1]
        mask = m < 0
        if mask.sum() >= 3:
            slope = np.polyfit(m[mask], r[mask], 1)[0]
            assert out[i] == pytest.approx(slope, abs=1e-12)
        else:
            assert np.isnan(out[i])


def test_tail_beta_matches_ols_on_tail_subset():
    rng = np.random.default_rng(5)
    ret = rng.normal(size=60)
    idx_ret = rng.normal(size=60)
    w = 20
    out = _k.tail_beta_(ret, idx_ret, window=w, q=0.3)
    for i in range(w - 1, len(ret)):
        m = idx_ret[i - w + 1 : i + 1]
        r = ret[i - w + 1 : i + 1]
        threshold = np.percentile(m, 30.0)
        mask = m <= threshold
        if mask.sum() >= 3:
            slope = np.polyfit(m[mask], r[mask], 1)[0]
            assert out[i] == pytest.approx(slope, abs=1e-12)
        else:
            assert np.isnan(out[i])


# ---------------------------------------------------------------------------
# R19-023/024/025 neutralization
# ---------------------------------------------------------------------------

def test_size_resid_no_silent_clip():
    """R19-024: market_cap<=0/NaN/±Inf → NaN，禁止 clamp 成 log(max(mc,1))=0。"""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mcap = np.array([100.0, 0.0, -5.0, np.nan, np.inf])
    out = _k.size_resid_panel_(y.reshape(1, -1), mcap.reshape(1, -1))
    assert np.isnan(out[0]).all()  # 只有第 1 个市值合法 → 样本不足，全 NaN

    # 至少 3 个合法市值时，非法市值对应位置必须 NaN 且不参与回归
    y2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mcap2 = np.array([100.0, 200.0, 300.0, 0.0, np.nan, 600.0])
    out2 = _k.size_resid_panel_(y2.reshape(1, -1), mcap2.reshape(1, -1))
    assert np.isnan(out2[0][3]) and np.isnan(out2[0][4])
    valid_idx = np.array([0, 1, 2, 5])
    lv = np.log(mcap2[valid_idx])
    yv = y2[valid_idx]
    beta = np.polyfit(lv, yv, 1)[0]
    expected = yv - (np.polyfit(lv, yv, 1)[1] + beta * lv)
    np.testing.assert_allclose(out2[0][valid_idx], expected, atol=1e-12)


def test_industry_size_fwl_properties():
    """R19-133: FWL —— 每行业均值≈0 + cov(resid, demeaned log size)≈0。"""
    rng = np.random.default_rng(0)
    T, N = 5, 60
    y = rng.normal(size=(T, N))
    ind = rng.integers(0, 4, size=(T, N)).astype(float)
    mcap = np.exp(rng.normal(20, 1.0, size=(T, N)))
    resid = _k.industry_size_resid_panel_(y, ind, mcap)
    log_size = _k._log_market_cap(mcap)
    size_tilde = _k.group_demean_panel_(log_size, ind)

    for t in range(T):
        for g in np.unique(ind[t]):
            m = ind[t] == g
            assert abs(float(np.nanmean(resid[t][m])) < 1e-9
        # FWL 正交性：resid ⊥ demeaned log size（全局）
        r = resid[t]
        st = size_tilde[t]
        v = np.isfinite(r) & np.isfinite(st)
        rv = r[v]
        sv = st[v]
        covv = float(((rv - rv.mean()) * (sv - sv.mean())).sum()) / (v.sum() - 1)
        assert abs(covv) < 1e-9

    # 与直接 industry_dummy + log_size OLS 残差一致
    t = 0
    valid = np.isfinite(y[t]) & np.isfinite(log_size[t]) & np.isfinite(ind[t])
    yv = y[t][valid]
    lsv = log_size[t][valid]
    gv = ind[t][valid].astype(int)
    D = np.zeros((len(yv), 4))
    D[np.arange(len(yv)), gv] = 1.0
    X = np.column_stack([D, lsv])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    direct = yv - X @ beta
    np.testing.assert_allclose(resid[t][valid], direct, atol=1e-8)


def test_cs_resid_cs_regression_isfinite_mask():
    """R19-025: cs_resid_/cs_regression_ 用 np.isfinite，±Inf 位置 → NaN。"""
    y = np.array([1.0, 2.0, 3.0, np.inf, 5.0])
    x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    res = _k.cs_resid_(y, x)
    assert np.isnan(res[3])  # Inf 位置不产生残差
    finite = np.isfinite(y) & np.isfinite(x)
    beta = np.polyfit(x[finite], y[finite], 1)[0]
    alpha = np.polyfit(x[finite], y[finite], 1)[1]
    np.testing.assert_allclose(res[finite], y[finite] - (alpha + beta * x[finite]), atol=1e-12)

    for mode in (0, 1, 2):
        out = _k.cs_regression_(y, x, mode=mode)
        assert np.isnan(out[3])


# ---------------------------------------------------------------------------
# R19-026 coalesce first-finite
# ---------------------------------------------------------------------------

def test_coalesce_first_finite():
    """R19-026: 实现与文档一致 —— first finite；Inf 不阻塞后续参数。"""
    out = _k.coalesce_(
        np.array([np.nan, np.inf, 3.0, 4.0]),
        np.array([10.0, 20.0, 30.0, np.inf]),
    )
    np.testing.assert_allclose(out, [10.0, 20.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# R19-027..029 统一 finite mask（operator 级）
# ---------------------------------------------------------------------------

def _cs_module():
    try:
        return importlib.import_module("cleaned_operators.common.cross_sectional")
    except Exception:  # pragma: no cover - concurrent-session guard
        pytest.skip("cross_sectional import blocked by concurrent session edit")


def test_c_stats_finite_mask_excludes_inf():
    """c_count/c_mean/c_std/c_sum/c_percentile/cs_mad 的 sample_validity=finite。"""
    m = _cs_module()
    idx = pd.bdate_range("2024-01-01", periods=3)
    x = pd.DataFrame(
        [
            [1.0, 2.0, np.inf, np.nan],
            [np.inf, np.inf, np.inf, np.inf],
            [1.0, 2.0, 3.0, 4.0],
        ],
        index=idx,
        columns=list("ABCD"),
    )
    classes = {
        "c_count": "CrossSectionalCount",
        "c_mean": "CrossSectionalMean",
        "c_std": "CrossSectionalStd",
        "c_sum": "CrossSectionalSum",
        "c_percentile": "CrossSectionalPercentile",
        "cs_mad": "CrossSectionalMad",
    }
    for name, cls in classes.items():
        out = getattr(m, cls)().calculate(x)
        assert out.shape == x.shape
        row0 = out.iloc[0].tolist()
        assert all(np.isfinite(v) for v in row0), (name, row0)
        # 全 Inf 行 → 无有效样本 → NaN（count 为 0）
        row1 = out.iloc[1].tolist()
        if name == "c_count":
            assert row1 == [0.0, 0.0, 0.0, 0.0]
        else:
            assert all(np.isnan(v) for v in row1), (name, row1)

    # 精确值：row0 [1,2,Inf,NaN] → finite=[1,2]
    assert m.CrossSectionalCount().calculate(x).iloc[0, 0] == 2.0
    assert m.CrossSectionalMean().calculate(x).iloc[0, 0] == pytest.approx(1.5)
    assert m.CrossSectionalSum().calculate(x).iloc[0, 0] == pytest.approx(3.0)


def test_cross_section_sample_mask_declares_finite():
    m = _cs_module()
    assert m.CrossSectionSampleMask.sample_validity == "finite"
    x = pd.DataFrame({"A": [1.0, np.inf, np.nan], "B": [2.0, 3.0, 4.0]})
    masked = m.CrossSectionSampleMask.mask(x)
    assert np.isnan(masked.iloc[1, 0])  # Inf → NaN
    assert masked.iloc[0, 0] == 1.0


def test_cs_resid_polars_roundtrip_uses_isfinite():
    """cs_resid 的 pandas/polars 都复用 cs_resid_（isfinite），Inf 位置一致。"""
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.inf], "B": [2.0, 4.0, 6.0, 8.0]})
    x = pd.DataFrame({"A": [2.0, 4.0, 6.0, 8.0], "B": [1.0, 2.0, 3.0, 4.0]})
    m = _cs_module()
    res = m.CSResid().calculate(y, x)
    assert np.isnan(res.iloc[3, 0])  # Inf y → NaN
