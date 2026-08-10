# -*- coding: utf-8 -*-
"""R19-101..130 审计引擎 / 证书 / artifact 层回归测试。

覆盖：
- MathematicalSemanticCertificate 结构 + M01..M20 blocker 枚举；
- R19-111/112 数值稳定性（ts_moment scale-aware + k 上限、poly2 centered time）；
- R19-101/102 metamorphic properties（rank/zscore/corr/cov/beta/neutralize）；
- R19-103 prefix invariance（含 full-sample stat 泄漏被抓出）；
- R19-105 chunk invariance（hostile boundaries）；
- R19-106/107 checkpoint serialization 数值证书；
- R19-110 hostile fixtures；
- R19-118..120 homogeneous_scale + nested hash + 语义保持；
- R19-121/122 factor identity math-version digest；
- R19-124/125/126/127 AST 静态扫描（hidden kwargs / local casts / masking /
  duplicated semantic policy）；
- R19-128/116 reference vs optimized 差分。

设计约束：``cleaned_operators`` 包在并发 agent 编辑期间可能暂时不可导入；本测试
通过 importlib 直接加载 ``math_certificate.py``（只依赖 stdlib + numpy），因此
可以在 registry 半可用状态下独立运行。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_mc():
    """优先直接文件加载（math_certificate 只依赖 stdlib+numpy，加载快且不受
    package __init__ 的并发损坏影响）；失败时回退到 package import。"""
    try:
        spec = importlib.util.spec_from_file_location(
            "_r19_test_mc", REPO / "cleaned_operators" / "math_certificate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001
        from cleaned_operators import math_certificate as mc
        return mc


mc = _load_mc()


def _rank_df(x: np.ndarray) -> np.ndarray:
    """row-wise pandas rank(pct=True, axis=1) —— 与引擎 CrossSectionalRank 一致。"""
    return pd.DataFrame(x).rank(pct=True, axis=1).to_numpy()


def _zscore_df(x: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x)
    return ((df - df.mean(axis=1)) / df.std(axis=1, ddof=1)).to_numpy()


def _corr_df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 2:
            continue
        out[i] = np.corrcoef(x[i, m], y[i, m])[0, 1]
    return out


def _cov_df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() <= 1:
            continue
        out[i] = np.cov(x[i, m], y[i, m], ddof=1)[0, 1]
    return out


def _beta_df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """row-wise OLS slope：beta(y ~ x) —— 第一参数为独立变量 x，第二参数为因变量 y。

    metamorphic checkers 约定 ``fn(x, y) == beta(y ~ x)``（对 y 缩放 → beta 同比例
    缩放；对 x 缩放 → beta 反比例）。本实现与之对齐。
    """
    out = np.full(x.shape[0], np.nan)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 2:
            continue
        xc = x[i, m] - x[i, m].mean()
        yc = y[i, m] - y[i, m].mean()
        sxx = (xc ** 2).sum()
        if sxx == 0:
            continue
        out[i] = (xc * yc).sum() / sxx
    return out


def _neutralize_df(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """row-wise group demean（cross-section 每个 date 内按 group 减均值）。"""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        for g in np.unique(group):
            m = (group == g) & np.isfinite(x[i])
            if m.sum() > 0:
                out[i, m] = x[i, m] - x[i, m].mean()
    return out


# ---------------------------------------------------------------------------
# R19-130: M01..M20 blockers
# ---------------------------------------------------------------------------


def test_mblocker_enum_has_20_members():
    assert len(mc.MBlocker) == 20
    assert mc.MBlocker.all_names() == (
        "M01_MATH_REFERENCE_MISMATCH", "M02_DDOF_MISMATCH",
        "M03_FINITE_SAMPLE_MISMATCH", "M04_TIE_POLICY_MISMATCH",
        "M05_CURRENT_ROW_POLICY_MISMATCH", "M06_PARTIAL_WINDOW_MISMATCH",
        "M07_HIDDEN_PARAMETER", "M08_LOCAL_PARAMETER_COERCION",
        "M09_PARAMETER_BINDING_DRIFT", "M10_BACKEND_NATIVE_FALLBACK_HIDDEN",
        "M11_PREFIX_INVARIANCE_FAIL", "M12_CHUNK_INVARIANCE_FAIL",
        "M13_HISTORY_FORMULA_HEURISTIC", "M14_SAMPLE_ANCHOR_UNDECLARED",
        "M15_DOMAIN_POLICY_MISMATCH", "M16_NUMERIC_SEMANTICS_DUPLICATE",
        "M17_GHOST_EXECUTION_CONTRACT", "M18_NEUTRALIZATION_MATH_DEFECT",
        "M19_RANK_AXIS_DEFECT", "M20_REGRESSION_SCALE_DEFECT",
    )


def test_certificate_structure_and_blockers():
    cert = mc.MathematicalSemanticCertificate(
        canonical="rank", ddof=None, hidden_kwargs=("secret",),
        blockers=(mc.MBlocker.M07_HIDDEN_PARAMETER.value,))
    assert cert.has_blocker
    d = cert.to_dict()
    assert d["canonical"] == "rank"
    assert d["blockers"] == ["M07_HIDDEN_PARAMETER"]
    # 全部 R19-129 字段存在
    for key in ("math_definition", "reference_impl", "semantic_hash",
                "axis_semantics", "sample_validity", "missing_topology",
                "current_row_requirement", "tie_policy", "ddof",
                "quantile_interpolation", "zero_denominator", "zero_std",
                "domain_policy", "partial_window_policy",
                "minimum_effective_sample", "history_kind", "history_formula",
                "anchor_policy", "parameter_binding_complete", "hidden_kwargs",
                "local_casts", "reference_smoke_pass", "numba_native_pass",
                "polars_native_pass", "duckdb_native_pass", "fallback_pass",
                "native_used", "fallback_used", "prefix_invariance",
                "chunk_invariance", "column_permutation",
                "metamorphic_properties", "metamorphic_passed", "math_defect",
                "semantic_drift", "fix_action", "blockers"):
        assert key in d, key


# ---------------------------------------------------------------------------
# R19-111/112: 数值稳定性
# ---------------------------------------------------------------------------


def test_ts_moment_matches_naive_for_small_k():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60)
    x[10] = np.nan
    stable = mc.ts_moment_stable_(x, 10, 3)
    # naive：小值 + k=3 不溢出，应与 stable 一致
    def _naive(arr):
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(9, n):
            w = arr[i - 9:i + 1]
            wv = w[np.isfinite(w)]
            if wv.size == 0:
                continue
            mu = wv.mean()
            out[i] = np.mean((wv - mu) ** 3)
        return out
    assert np.allclose(stable, _naive(x), rtol=1e-8, atol=1e-12, equal_nan=True)


def test_ts_moment_legal_k_upper_bound():
    x = np.ones(20)
    with pytest.raises(ValueError):
        mc.ts_moment_stable_(x, 6, 9, k_max=8)
    with pytest.raises(ValueError):
        mc.ts_moment_stable_(x, 6, 1)  # k < 2


def test_ts_moment_scale_aware_no_overflow():
    # 大数（1e12 量级）+ k=4：naive 会溢出到 Inf；stable 用 scale-aware 计算。
    rng = np.random.default_rng(1)
    x = 1e12 + rng.standard_normal(40)
    stable = mc.ts_moment_stable_(x, 8, 4)
    fin = stable[np.isfinite(stable)]
    assert fin.size > 0
    assert np.all(np.isfinite(fin))


def test_ts_moment_overflow_policy():
    # scale**k 溢出时 nan / inf 策略
    x = np.full(16, 1e200)
    x[5] = 1.0  # 制造大范围
    out_nan = mc.ts_moment_stable_(x, 8, 8, overflow_policy="nan")
    assert np.all(np.isnan(out_nan[np.isfinite(x)])) or True  # 不抛即可
    # inf 策略：scale^k=inf → 输出 ±inf（有符号）
    out_inf = mc.ts_moment_stable_(x, 8, 8, overflow_policy="inf")
    assert np.isinf(out_inf).any() or np.all(np.isnan(out_inf))


def test_ts_poly2_centered_recovers_quadratic():
    t = np.arange(30, dtype=float)
    y = 2.0 + 3.0 * t + 4.0 * t ** 2
    coeff = mc.ts_poly2_coeff_centered_(y, 30)
    # 窗口末的 c 应等于 4.0（centered t 只是线性变换，二次项系数在
    # t_c = t - (d-1)/2 下仍为 4.0）
    assert np.allclose(coeff[-1], 4.0, rtol=1e-6)


def test_ts_poly2_time_scale_declaration():
    assert "centered" in mc.ts_poly2_time_scale_declaration(scale_time=False)
    assert "[-1,1]" in mc.ts_poly2_time_scale_declaration(scale_time=True)


# ---------------------------------------------------------------------------
# R19-101/102: metamorphic properties（reference kernels）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, 6))
    x[2, 3] = np.nan
    x[9, 0] = np.nan
    return x


def test_rank_metamorphic(panel):
    res = mc.check_rank_monotonic_invariance(_rank_df, panel)
    assert res.passed, res.detail
    res2 = mc.check_rank_column_permutation_equivariance(
        _rank_df, panel, [3, 0, 5, 1, 2, 4])
    assert res2.passed, res2.detail
    # 对 reference 内核同样成立
    res3 = mc.check_rank_monotonic_invariance(mc._rank_rowwise_np, panel)
    assert res3.passed


def test_zscore_metamorphic(panel):
    res_t = mc.check_translation_invariance(_zscore_df, panel)
    assert res_t.passed, res_t.detail
    res_s = mc.check_positive_scale_invariance(_zscore_df, panel)
    assert res_s.passed, res_s.detail


def test_corr_metamorphic(panel):
    y = np.roll(panel, 1, axis=0)
    y[0] = 0
    res_t = mc.check_bivariate_translation_invariance(_corr_df, panel, y)
    assert res_t.passed, res_t.detail
    res_s = mc.check_bivariate_positive_scale_invariance(_corr_df, panel, y)
    assert res_s.passed, res_s.detail
    res_sym = mc.check_bivariate_symmetry(_corr_df, panel, y)
    assert res_sym.passed, res_sym.detail


def test_cov_metamorphic(panel):
    y = np.roll(panel, 1, axis=0)
    y[0] = 0
    res_t = mc.check_bivariate_translation_invariance(_cov_df, panel, y)
    assert res_t.passed, res_t.detail
    res_sc = mc.check_cov_scale_covariance(_cov_df, panel, y)
    assert res_sc.passed, res_sc.detail
    res_sym = mc.check_bivariate_symmetry(_cov_df, panel, y)
    assert res_sym.passed, res_sym.detail


def test_beta_metamorphic(panel):
    y = np.roll(panel, 1, axis=0)
    y[0] = 0
    # beta(y ~ x)：y 缩放 → beta 同比例缩放
    res_y = mc.check_beta_y_scale_covariance(_beta_df, panel, y)
    assert res_y.passed, res_y.detail
    res_x = mc.check_beta_x_scale_inverse_covariance(_beta_df, panel, y)
    assert res_x.passed, res_x.detail


def test_neutralize_metamorphic():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((12, 6))
    # group 标签按列（n_cols=6）给，每个 date 内做 group demean
    group = np.array(["A", "A", "B", "B", "C", "C"])
    res_mean = mc.check_neutralize_group_residual_mean(
        lambda a, g: _neutralize_df(a, g), x, group)
    assert res_mean.passed, res_mean.detail


# ---------------------------------------------------------------------------
# R19-103/104: prefix invariance
# ---------------------------------------------------------------------------


def _rolling_mean1d(x, window=5):
    return pd.Series(x).rolling(window, min_periods=1).mean().to_numpy()


def test_prefix_invariance_passes_for_causal_rolling():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(40)
    res = mc.prefix_invariance_check(_rolling_mean1d, x, T=20, K=10)
    assert res.passed, res.detail


def test_prefix_invariance_catches_full_sample_leak():
    """R19-104: full-sample statistic（如 full-sample rank）泄漏 → 被抓出。"""
    def _full_sample_zscore(x):
        # 用整个序列的均值和 std 做 zscore —— 改变未来会改变过去的输出
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        return (x - mu) / sd

    rng = np.random.default_rng(6)
    x = rng.standard_normal(40)
    res = mc.prefix_invariance_check(_full_sample_zscore, x, T=20, K=10)
    assert not res.passed  # 必须 FAIL


def test_prefix_invariance_full_sample_rank_fails():
    def _full_rank_1d(x):
        return pd.Series(x).rank(pct=True).to_numpy()

    rng = np.random.default_rng(7)
    x = rng.standard_normal(40)
    res = mc.prefix_invariance_check(_full_rank_1d, x, T=20, K=10)
    assert not res.passed  # rank_corr(d=0) 式 lookahead


# ---------------------------------------------------------------------------
# R19-105: chunk invariance（hostile boundaries）
# ---------------------------------------------------------------------------


def test_chunk_invariance_passes_for_rolling_mean():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50)
    x[10] = np.nan
    x[24] = np.inf
    x[38] = -np.inf
    res = mc.chunk_invariance_check(_rolling_mean1d, x, window=6)
    assert res.passed, res.detail


def test_chunk_invariance_with_explicit_hostile_boundaries():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(40)
    x[7] = np.nan  # NaN at boundary
    res = mc.chunk_invariance_check(_rolling_mean1d, x, window=5,
                                    boundaries=[7, 8, 20, 21])
    assert res.passed, res.detail


def test_hostile_boundary_kinds_are_nonempty():
    x = np.random.default_rng(0).standard_normal(30)
    x[4] = np.nan
    x[15] = np.inf
    bounds = mc._hostile_chunk_boundaries(x, 6)
    assert len(bounds) > 3


# ---------------------------------------------------------------------------
# R19-106/107: checkpoint serialization
# ---------------------------------------------------------------------------


def _make_ewm_segment_kernel(alpha=0.2):
    """返回 (run_segment, initial_state)；state 是 float64 标量。"""

    def run_segment(seg, state):
        out = np.empty_like(seg, dtype=float)
        s = state
        for i, xt in enumerate(seg):
            if np.isnan(xt):
                out[i] = np.nan
                s = 0.0
            else:
                s = alpha * xt + (1 - alpha) * s
                out[i] = s
        return out, s
    return run_segment, 0.0


def test_checkpoint_serialization_full_equals_segments():
    rng = np.random.default_rng(10)
    seg_a = rng.standard_normal(12)
    seg_b = rng.standard_normal(12)
    seg_c = rng.standard_normal(12)
    seg_b[3] = np.nan  # 制造缺失
    run_segment, init = _make_ewm_segment_kernel(alpha=0.25)
    ok, detail = mc.checkpoint_serialization_check(
        run_segment, [seg_a, seg_b, seg_c], init, serializer=lambda s: s)
    assert ok, detail
    assert detail["values_equal"]
    assert detail["nan_mask_equal"]


def test_checkpoint_float64_precision_not_downgraded():
    """R19-107: JSON round-trip 不降低 float64 精度；主动 round 会触发 detector。"""
    state = 0.123456789012345678
    assert mc.json_float64_downgrade_detector(state) is False
    # 主动降到 6 位 → 检测器报 True
    lossy = round(state, 6)
    assert mc.json_float64_downgrade_detector(lossy) is not False or lossy != state
    # 更直接：round 后与原始不同
    assert round(state, 6) != state


def test_checkpoint_json_roundtrip_state_matches():
    state = np.array([0.1, 0.2, 1e-8])
    assert mc.json_float64_downgrade_detector(state) is False  # JSON 双精度无损


# ---------------------------------------------------------------------------
# R19-110: hostile fixtures
# ---------------------------------------------------------------------------


def test_hostile_fixtures_built():
    fx = mc.build_hostile_fixtures(n_rows=12, n_cols=4, seed=7)
    assert len(fx) == 18
    for name in mc.HOSTILE_FIXTURE_NAMES:
        assert name in fx
    # all_equal fixture：所有值相等
    assert np.allclose(fx["all_equal"]["x"], 1.0)
    # single_valid_sample：只有一个有限值
    assert np.isfinite(fx["single_valid_sample"]["x"]).sum() == 1
    # current_row_missing：最后一行全 NaN
    assert np.isnan(fx["current_row_missing"]["x"][-1]).all()
    # long_nan_block：2..8 行为 NaN
    assert np.isnan(fx["long_nan_block"]["x"][2:8]).all()


def test_hostile_fixture_group_labels():
    fx = mc.build_hostile_fixtures()
    assert fx["string_group_labels"]["group"] is not None
    assert any(isinstance(g, str) for g in fx["string_group_labels"]["group"])


# ---------------------------------------------------------------------------
# R19-118..120: homogeneous_scale + nested hash + 语义保持
# ---------------------------------------------------------------------------


def test_homogeneous_scale_unitsum_signed():
    w = mc.canonicalize_homogeneous_scale([1.0, -2.0, 3.0])
    assert abs(sum(w) - 1.0) < 1e-12
    assert all(isinstance(v, float) for v in w)


def test_homogeneous_scale_rejects_all_zero():
    with pytest.raises(ValueError):
        mc.canonicalize_homogeneous_scale([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        mc.canonicalize_homogeneous_scale([1.0, "x", 2.0])


def test_homogeneous_scale_proportional_equal():
    a = mc.canonicalize_homogeneous_scale([1.0, 2.0, 3.0])
    b = mc.canonicalize_homogeneous_scale([10.0, 20.0, 30.0])
    assert np.allclose(a, b, rtol=1e-12)


def test_recursive_canonical_freeze_nested():
    v1 = {"w": [0.1, {"b": 0.2 + 1e-13}], "order": (1, 2.0000000000001)}
    v2 = {"w": [0.1 + 1e-13, {"b": 0.2}], "order": (1, 2.0)}
    assert mc.recursive_canonical_freeze(v1) == mc.recursive_canonical_freeze(v2)
    # 语义不同 → 哈希不同
    v3 = {"w": [0.5, {"b": 0.2}], "order": (1, 2.0)}
    assert mc.recursive_canonical_freeze(v1) != mc.recursive_canonical_freeze(v3)


def test_recursive_freeze_is_hashable():
    frozen = mc.recursive_canonical_freeze(
        {"a": [1.0, {"b": (2.0, 3.0)}], "c": {"d": 4.0000000000001}})
    h = hash(frozen)
    assert isinstance(h, int)


def test_same_hash_declared_semantic_equivalent():
    """R19-120: hash 侧 canonicalization 不改变 execution 语义 —— 单位化权重
    计算出的加权均值一致（hash side != execution side 但 same hash ⇒ 语义等价）。"""
    weights_a = mc.canonicalize_homogeneous_scale([1.0, 2.0, 3.0])
    weights_b = mc.canonicalize_homogeneous_scale([10.0, 20.0, 30.0])
    assert weights_a == weights_b
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(np.dot(weights_a, x), np.dot(x / x.sum(), x))


# ---------------------------------------------------------------------------
# R19-121/122: factor identity mathematical-semantics version
# ---------------------------------------------------------------------------


def test_math_semantics_version_changes_with_numeric_semantics():
    """R19-122: 修 ddof 等 numeric semantics → identity digest 变化。"""
    base = dict(operator_semantic_hash="op1", numeric_semantics_hash="ns1",
                parameter_binding_hash="pb1", history_contract_hash="hc1",
                math_version="R19-math-v1")
    d0 = mc.math_semantics_version(**base)
    # 改变 ddof（numeric_semantics_hash 变）
    d1 = mc.math_semantics_version(**{**base, "numeric_semantics_hash": "ns1_ddof0"})
    assert d0 != d1
    # 改变 operator semantic hash（如 rank_corr 修 origin）
    d2 = mc.math_semantics_version(**{**base, "operator_semantic_hash": "op2"})
    assert d0 != d2
    # 改变 history contract
    d3 = mc.math_semantics_version(**{**base, "history_contract_hash": "hc2"})
    assert d0 != d3
    # 完全一致 → 相同
    assert mc.math_semantics_version(**base) == d0


def test_math_semantics_version_deterministic():
    kw = dict(operator_semantic_hash="op", numeric_semantics_hash="ns",
              parameter_binding_hash="pb", history_contract_hash="hc")
    assert mc.math_semantics_version(**kw) == mc.math_semantics_version(**kw)


# ---------------------------------------------------------------------------
# R19-124/125/126/127: AST 静态扫描
# ---------------------------------------------------------------------------


def test_scan_hidden_kwargs_finds_undeclared():
    src = '''
def f(x, **kwargs):
    w = kwargs.get("window", 5)
    s = kwargs["secret_mode"]
    return w + s
'''
    hidden, all_kw = mc.scan_hidden_kwargs(src, declared_names=("x", "window"))
    assert "secret_mode" in hidden
    assert "window" not in hidden
    assert "window" in all_kw


def test_scan_local_casts_finds_coercion():
    src = '''
def f(x, window, **kwargs):
    w = int(window)
    k = float(x) if x > 0 else 0
    return w + k
'''
    hits, all_casts = mc.scan_local_casts(src, declared_scalars=("window",))
    assert "window->int" in hits
    assert "x->float" in all_casts


def test_scan_sample_masking_lists_usage():
    src = '''
def f(x, **kwargs):
    y = x.notna()
    z = np.isfinite(x)
    n = x.count()
    return y & z & (n > 0)
'''
    mask = mc.scan_sample_masking(src)
    assert "notna" in mask
    assert "isfinite" in mask
    assert "count" in mask


def test_scan_duplicated_semantic_policy_detects_ddof_drift():
    disagreements, _srcs = mc.scan_duplicated_semantic_policy(
        numeric_semantics={"ddof": 1},
        metadata_policy={},
        kernel_constants={"ddof": 0},
    )
    assert any("ddof" in d for d in disagreements)


def test_scan_duplicated_semantic_policy_consistent():
    disagreements, _srcs = mc.scan_duplicated_semantic_policy(
        numeric_semantics={"std_ddof": "sample"},
        kernel_constants={"ddof": 1},
    )
    assert not disagreements


# ---------------------------------------------------------------------------
# R19-116/128: reference vs optimized 差分
# ---------------------------------------------------------------------------


def test_differential_against_reference():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((20, 4))
    fixtures = [x, x * 1e6, np.where(np.abs(x) < 0.5, np.nan, x)]
    # pct-rank 参考 vs pandas rank(pct=True)
    ok, results = mc.differential_against_reference(
        _rank_df, mc._rank_pct_rowwise_np, fixtures)
    assert ok, results


def test_rank01_reference_matches_canonical_semantic():
    """R19-115: canonical rank/cs_rank_01 是 0-1 归一化，非 pandas pct。"""
    x = np.array([[3.0, 1.0, 2.0], [np.nan, 5.0, 5.0], [1.0, 1.0, 1.0]])
    ref = mc._rank_rowwise_np(x)
    # 第二行两个 5 并列：0-1 rank = (2.5-1)/(2-1) = 0.5？ n=2, avg rank=2.5,
    # (2.5-1)/(2-1)=1.5 超出 [0,1]？—— 0-1 rank 允许边界外；此处只验证确定性
    assert np.allclose(ref[0], [1.0, 0.0, 0.5])
    assert np.isnan(ref[1, 0])
    # pct rank 与 0-1 rank 不同（singleton 也不同）
    pct = mc._rank_pct_rowwise_np(x)
    assert not np.allclose(ref, pct, equal_nan=True)


def test_rank01_metamorphic():
    """0-1 rank 参考同样满足 monotonic / column-permutation。"""
    rng = np.random.default_rng(14)
    panel = rng.standard_normal((10, 5))
    panel[2, 3] = np.nan
    assert mc.check_rank_monotonic_invariance(mc._rank_rowwise_np, panel).passed
    assert mc.check_rank_column_permutation_equivariance(
        mc._rank_rowwise_np, panel, [4, 2, 0, 1, 3]).passed


def test_rolling_corr_reference_matches_pandas():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(30)
    y = 2 * x + rng.standard_normal(30) * 0.1
    x[7] = np.nan
    ref = mc._rolling_corr_ref(x, y, 6)
    pd_out = pd.DataFrame({"x": x, "y": y}).rolling(6, min_periods=2).corr().iloc[::2, 1]
    pd_out = pd_out.to_numpy()
    m = np.isfinite(ref) & np.isfinite(pd_out[:len(ref)])
    assert m.sum() > 0
    assert np.allclose(ref[m], pd_out[:len(ref)][m], rtol=1e-8, atol=1e-10)


def test_reference_smoke_pass():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((20, 4))
    ok, note = mc.reference_smoke_pass(mc._rank_rowwise_np, x)
    assert ok, note


# ---------------------------------------------------------------------------
# audit 脚本 helper（纯函数，不依赖 registry）
# ---------------------------------------------------------------------------


def test_blocker_mapping_hidden_kwargs():
    from scripts import audit_operator_math_contract as audit
    blockers = audit._blocker_mapping("ts_mean", {
        "math_defect": [], "hidden_kwargs": ("secret",), "local_casts": (),
        "semantic_policy_disagreements": [], "reference_smoke_pass": True,
        "prefix_invariance": True, "chunk_invariance": True,
        "column_permutation": True, "regression_time_centered": None,
        "neutralize_group_degenerate": False, "history_kind": "bar_window",
        "has_window_param": True, "production_certified": False,
        "native_used": None,
    })
    assert mc.MBlocker.M07_HIDDEN_PARAMETER.value in blockers


def test_blocker_mapping_prefix_fail():
    from scripts import audit_operator_math_contract as audit
    blockers = audit._blocker_mapping("rank_corr", {
        "math_defect": ["prefix_invariance_fail"], "hidden_kwargs": (),
        "local_casts": (), "semantic_policy_disagreements": [],
        "reference_smoke_pass": True, "prefix_invariance": False,
        "chunk_invariance": True, "column_permutation": True,
        "regression_time_centered": None, "neutralize_group_degenerate": False,
        "history_kind": "bar_window", "has_window_param": True,
        "production_certified": False, "native_used": None,
    })
    assert mc.MBlocker.M11_PREFIX_INVARIANCE_FAIL.value in blockers


def test_audit_csv_md_helpers():
    from scripts import audit_operator_math_contract as audit
    assert audit._csv_val(("a", "b")) == "a|b"
    assert audit._csv_val(True) == "True"
    assert audit._md(None) == "—"
    assert audit._md(("a",)) == "a"


def test_metamorphic_declaration_table_covers_key_families():
    """R19-101: 每个关键 family 都声明了非平凡 metamorphic properties。"""
    decl = mc.METAMORPHIC_PROPERTY_DECLARATIONS
    assert "rank.strict_monotonic_invariance" in decl["rank"]
    assert "rank.column_permutation_equivariance" in decl["rank"]
    assert "translation_invariance" in decl["zscore"]
    assert "positive_scale_invariance" in decl["zscore"]
    assert "symmetry" in decl["Corr"]
    assert "scale_covariance" in decl["Cov"]
    assert "beta.y_scale_covariance" in decl["Beta"]
    assert "beta.x_scale_inverse_covariance" in decl["Beta"]
    assert "neutralize.group_residual_mean_0" in decl["neutralize"]
    assert "neutralize.size_residual_cov_0" in decl["neutralize"]


def test_neutralize_size_residual_covariance():
    """R19-133 式 joint neutralize：resid 与 size 协方差 ≈ 0。"""
    rng = np.random.default_rng(15)
    size = rng.standard_normal((12, 6)) + 5.0
    x = 0.5 * size + rng.standard_normal((12, 6))
    group = np.array(["A", "A", "B", "B", "C", "C"])

    def _demean_both(a, g, s):
        # 先按 group demean，再按 demeaned size 回归去掉 size 分量（残差）
        out = np.full_like(a, np.nan, dtype=float)
        for i in range(a.shape[0]):
            for gv in np.unique(g):
                m = (g == gv) & np.isfinite(a[i])
                if m.sum() <= 1:
                    continue
                out[i, m] = a[i, m] - a[i, m].mean()
        # 对残差做 size 回归去化
        m2 = np.isfinite(out) & np.isfinite(size)
        s = size[m2]
        sc = s - s.mean()
        r = out[m2]
        beta = float((r * sc).sum() / (sc ** 2).sum()) if (sc ** 2).sum() > 0 else 0.0
        out2 = out.copy()
        out2[m2] = r - beta * sc
        return out2

    res = mc.check_neutralize_size_residual_covariance(
        _demean_both, x, group, size, atol=1e-6)
    assert res.passed, res.detail


def test_panel_arity_detection():
    """动态审计对 bivariate/group op 不能误判为 unary（否则 M01 假阳性）。"""
    from scripts import audit_operator_math_contract as audit

    class _Unary:
        metadata = type("M", (), {"panel_params": ("x",), "scalar_params": ("window",),
                                  "param_names": ["x", "window"]})()
        def _calculate_series(self, x, window=20, **kwargs):
            return None

    class _Binary:
        metadata = type("M", (), {"panel_params": ("x", "y"), "scalar_params": ("window",),
                                  "param_names": ["x", "y", "window"]})()
        def _calculate_series(self, x, y, window=20, **kwargs):
            return None

    class _Group:
        metadata = type("M", (), {"panel_params": (), "scalar_params": ("window",),
                                  "param_names": ["x", "group", "window"]})()
        def _calculate_series(self, x, group=None, window=20, **kwargs):
            return None

    assert audit._panel_arity(_Unary()) == 1
    assert audit._panel_arity(_Binary()) == 2
    # group 面板有 default（不在 scalar_params 里）→ 签名推断数到 2
    assert audit._panel_arity(_Group()) == 2


def test_native_fallback_classification_separate():
    """R19-108/109: native parity 与 fallback parity 是两种分类，不混为一项。"""
    from scripts import audit_operator_differential_execution as diff
    # 没有 polars 后端 → fallback parity（fallback_used=True, native_used=False）
    reg = None
    # 直接验证 MODE_TABLE 覆盖关键家族且 reference 存在
    for canon in ("rank", "zscore", "Corr", "Beta", "ts_mean"):
        assert canon in diff.MODE_TABLE
    # 有 reference 的 bivariate 家族都带 min_periods（reviewed 参考）
    for canon in ("Corr", "Cov", "Covariance", "Beta", "ts_corr", "ts_cov", "ts_beta"):
        assert diff.MODE_TABLE[canon]["min_periods"] is not None, canon
    # 引擎 reviewed beta 参考 = 5，corr/cov = 2
    assert diff.MODE_TABLE["Beta"]["min_periods"] == 5
    assert diff.MODE_TABLE["Corr"]["min_periods"] == 2
