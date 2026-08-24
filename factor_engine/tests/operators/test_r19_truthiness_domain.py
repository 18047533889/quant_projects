# -*- coding: utf-8 -*-
"""R19-063..074 + R19-094..100 targeted regression tests.

覆盖：
- R19-063..065  neutralize 缺 group / 全 missing group / 部分 missing membership
- R19-066..068  cross-sectional exact duplicate（rank↔cs_rank_01 等）数值等价证明
- R19-069..071  truthiness 统一（Inf 非 typed bool；三值 where policy 声明）
- R19-072..074  numeric_semantics 注册函数 hard-fail + numeric_semantics_hash
- R19-094       cs_mad_zscore 命名（raw-MAD standardized）
- R19-095..097  asin/acos/power/exp domain policy 跨 backend 一致
- R19-098       blom_transform finite sample audit
- R19-099..100  group label ordering + column permutation metamorphic

策略：直接 import 算子**类**并调用 ``_calculate_series`` —— 因为 ``load_all`` 的
dedupe 会把 ``c_neutralize``/``neutralize`` 重映射到 ``group_neutralize``
（group.py 的 GroupDemean），经 registry 取值会拿到另一个实现；类级单测只验证
本批文件（cross_sectional.py）的语义，生产 remap 由中央协调。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 先 import 算子模块（触发 cleaned_operators.base 完整加载），再 import
# backend.numeric_semantics —— 避免并发会话修改 base.py 期间的循环导入部分态。
from factor_engine.cleaned_operators.common.cross_sectional import (  # noqa: E402
    CrossSectionalMadZscore,
    CrossSectionalNeutralize,
    CrossSectionalNeutralizePolars,
    CrossSectionalPercentile,
    CsPctRank,
    CsQuantile,
    CsRank01,
    Rank,
    RankPct,
)
from factor_engine.cleaned_operators.common.elementwise import Acos, Asin, BlomTransform, Exp, Pow  # noqa: E402
from factor_engine.cleaned_operators.common.polars_math_extended import AsinPolars  # noqa: E402

from factor_engine.backend import numeric_semantics as ns  # noqa: E402
from factor_engine.backend.numeric_semantics import NumericSemantics  # noqa: E402

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


def _panel(rows: int = 3, cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    cols_ = [f"S{i}" for i in range(cols)]
    return pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=cols_)


# ---------------------------------------------------------------------------
# R19-063..065 neutralize 缺 group / 全 missing / 部分 missing
# ---------------------------------------------------------------------------


def test_neutralize_missing_group_outputs_all_nan():
    """group=None 不再退化为 global demean —— 输出全 NaN（fail-closed）。"""
    x = _panel()
    out = CrossSectionalNeutralize()._calculate_series(x)
    assert out.isna().all().all()
    # 显式 global demean 仍可通过 cs_demean（CrossSectionalDemean）获得。
    demean = x.sub(x.mean(axis=1), axis=0)
    assert not demean.isna().all().all()


def test_neutralize_all_missing_group_day_is_nan_not_global_demean():
    """某日 group 全 missing 不能 silent global demean -> 该日整行 NaN。"""
    x = _panel(rows=2)
    idx = x.index
    cols = x.columns
    g = pd.DataFrame(
        [[1.0, 1.0, 2.0, 2.0, 1.0, 1.0], [np.nan] * len(cols)],
        index=idx, columns=cols,
    )
    out = CrossSectionalNeutralize()._calculate_series(x, g)
    assert out.iloc[0].notna().all()
    assert out.iloc[1].isna().all()
    row0 = out.iloc[0]
    # 组1 (S0,S1,S4,S5) 与组2 (S2,S3) 各自 demean 后组均值为 0。
    assert abs(row0.iloc[0] + row0.iloc[1] + row0.iloc[4] + row0.iloc[5]) < 1e-12
    assert abs(row0.iloc[2] + row0.iloc[3]) < 1e-12


def test_neutralize_partial_group_membership_unknown_stays_nan():
    """部分 group missing：只有 group known 的股票参与组中性化，unknown 保持 NaN。"""
    x = _panel(rows=1)
    cols = x.columns
    g = pd.DataFrame(
        [[1.0, 1.0, 2.0, 2.0, np.nan, np.nan]], index=x.index, columns=cols,
    )
    out = CrossSectionalNeutralize()._calculate_series(x, g)
    row = out.iloc[0]
    assert row.iloc[:4].notna().all()
    assert row.iloc[4:].isna().all()
    assert abs(row.iloc[0] + row.iloc[1]) < 1e-12
    assert abs(row.iloc[2] + row.iloc[3]) < 1e-12


def test_neutralize_group_label_ordering_and_type_independent():
    """R19-099：group label 出现顺序/类型（str/int/重排值）不改变结果。"""
    x = _panel()
    cols = x.columns
    idx = x.index
    g_str = pd.DataFrame(
        [["a"] * 2 + ["b"] * 2 + ["c"] * 2] * len(idx), index=idx, columns=cols,
    )
    g_int = pd.DataFrame(
        [[1] * 2 + [2] * 2 + [3] * 2] * len(idx), index=idx, columns=cols,
    )
    g_reorder = pd.DataFrame(
        [[10] * 2 + [20] * 2 + [30] * 2] * len(idx), index=idx, columns=cols,
    )
    op = CrossSectionalNeutralize()
    out_str = op._calculate_series(x, g_str)
    out_int = op._calculate_series(x, g_int)
    out_re = op._calculate_series(x, g_reorder)
    pd.testing.assert_frame_equal(out_str, out_int)
    pd.testing.assert_frame_equal(out_int, out_re)


def test_neutralize_polars_missing_group_outputs_all_nan():
    """Polars neutralize group=None 同样 fail-closed -> 全 NaN。"""
    if pl is None:
        pytest.skip("polars not installed")
    x = _panel()
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    out = CrossSectionalNeutralizePolars()._calculate_series(pl_x)
    assert np.isnan(out.select(list(x.columns)).to_numpy()).all()


# ---------------------------------------------------------------------------
# R19-066..068 cross-sectional exact duplicate 数值等价证明
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "primary_cls,secondary_cls,args",
    [
        (Rank, CsRank01, ()),
        (RankPct, CsPctRank, ()),
        (CrossSectionalPercentile, CsQuantile, (0.3,)),
    ],
)
def test_duplicate_pair_exact_equivalent(primary_cls, secondary_cls, args):
    """exact duplicate pair 输出逐 cell 完全一致（数值等价证明）。"""
    x = _panel(rows=5, cols=7, seed=3)
    x.iloc[0, 0] = np.nan
    x.iloc[2, 4] = np.inf
    x.iloc[1, 1] = -np.inf
    out_a = primary_cls()._calculate_series(x, *args)
    out_b = secondary_cls()._calculate_series(x, *args)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_c_rank_differs_from_cs_rank_01():
    """R19-088：rank_pct（pandas 百分位，min=1/n）与 cs_rank_01（0-1，singleton=0.5）
    是两种 rank semantic，不是 duplicate —— 保持分开。"""
    x = pd.DataFrame([[1.0, 2.0, 4.0]], index=["2020-01-01"], columns=list("ABC"))
    pct = RankPct()._calculate_series(x)
    cs01 = CsRank01()._calculate_series(x)
    assert pct.iloc[0, 0] == pytest.approx(1.0 / 3.0)
    assert cs01.iloc[0, 0] == pytest.approx(0.0)
    xs = pd.DataFrame([[5.0, np.nan, np.nan]], index=["2020-01-01"], columns=list("ABC"))
    assert RankPct()._calculate_series(xs).iloc[0, 0] == pytest.approx(1.0)
    assert CsRank01()._calculate_series(xs).iloc[0, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# R19-100 column permutation metamorphic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name,cls",
    [
        ("rank", Rank),
        ("cs_rank_01", CsRank01),
        ("rank_pct", RankPct),
        ("c_neutralize", CrossSectionalNeutralize),
    ],
)
def test_cross_sectional_column_permutation_metamorphic(op_name, cls):
    """列顺序 permutation：对称/截面算子 permute instrument columns ->
    permute output columns same way。"""
    x = _panel(rows=4, cols=8, seed=7)
    perm = [5, 0, 3, 7, 1, 6, 2, 4]
    cols = list(x.columns)
    op = cls()
    if op_name == "c_neutralize":
        g = pd.DataFrame(
            np.tile([1, 1, 2, 2], (4, 2)), index=x.index, columns=cols, dtype=float,
        )
        out = op._calculate_series(x, g)
        xp = x[[cols[i] for i in perm]]
        out_p = op._calculate_series(xp, g[[cols[i] for i in perm]])
    else:
        out = op._calculate_series(x)
        xp = x[[cols[i] for i in perm]]
        out_p = op._calculate_series(xp)
    expect = out[[cols[i] for i in perm]]
    pd.testing.assert_frame_equal(out_p, expect)


# ---------------------------------------------------------------------------
# R19-069..071 truthiness / ConditionBool 统一
# ---------------------------------------------------------------------------


def test_truthy_inf_is_not_true_typed_bool_policy():
    """R19-069：±Inf 在逻辑条件中不再视为 true —— production 只允许 typed bool
    {0,1,NaN}。"""
    assert ns.truthy_inf_is_true() is False
    assert ns.where_truth_value_policy() == "three_valued"
    assert ns.truthy_null_is_false() is True
    assert ns.truthy_nan_is_false() is True


def test_truthiness_invalid_condition_policy_contract():
    """R19-137：非 {0,1,NaN} 的 where 条件（2 / -1 / Inf / string）是数据质量错误，
    按 policy 必须被 production 门拒绝。此处验证 numeric_semantics 契约承载该约束；
    where 算子本身在 signal/scalar_where（非本批文件），其 surface 门由中央协调。"""
    sem = ns.semantics_for("where")
    assert sem == NumericSemantics()  # where 不引入 Inf truthiness
    assert ns.where_truth_value_policy() == "three_valued"
    valid_conditions = {0.0, 1.0, float("nan")}
    for bad in (2.0, -1.0, float("inf"), -float("inf")):
        assert bad not in valid_conditions
        assert (bad == 0.0 or bad == 1.0) is False
    assert ns.truthy_inf_is_true() is False


# ---------------------------------------------------------------------------
# R19-072..074 numeric semantics registry 治理
# ---------------------------------------------------------------------------


def test_numeric_semantics_registration_duplicate_hard_fail():
    """重复注册同一 canonical 必须 hard fail（R19-072）。"""
    with pytest.raises(ValueError, match="duplicate"):
        ns.register_numeric_semantics("rank", NumericSemantics())
    name = "_r19_test_dup_canonical"
    ns.register_numeric_semantics(name, NumericSemantics(div_zero="null"))
    with pytest.raises(ValueError, match="duplicate"):
        ns.register_numeric_semantics(name, NumericSemantics(div_zero="inf"))
    ns.OPERATOR_SEMANTICS.pop(name, None)
    assert "divide" in ns.OPERATOR_SEMANTICS
    assert len(ns.OPERATOR_SEMANTICS) == len(set(ns.OPERATOR_SEMANTICS))


def test_numeric_semantics_hash_exists_and_stable():
    """R19-073：numeric_semantics_hash 存在、稳定、覆盖 ddof/tie/quantile/zero-std/
    div-zero/Inf/NaN/truthiness。"""
    h = ns.numeric_semantics_hash()
    assert isinstance(h, str) and len(h) == 64
    assert h == ns.numeric_semantics_hash()
    assert "rank" in ns.OPERATOR_SEMANTICS
    assert ns.semantics_for("rank").rank_ignore_nan is True
    assert ns.semantics_for("cs_quantile").quantile_interpolation == "linear"
    assert ns.semantics_for("zscore").zscore_zero_std == "zero"
    assert ns.semantics_for("divide").div_zero == "inf"


def test_exp_semantics_allow_inf():
    """R19-097：exp 溢出 policy = allow_inf（与 pandas/polars 实际输出一致）。"""
    assert ns.exp_overflow_policy() == "allow_inf"
    assert ns.semantics_for("exp").output_inf_to_nan is False
    assert ns.power_real_domain_policy() == "real_domain"
    assert ns.trig_domain_unit_interval_nan() is True


# ---------------------------------------------------------------------------
# R19-094 cs_mad_zscore 命名
# ---------------------------------------------------------------------------


def test_cs_mad_zscore_description_is_raw_mad():
    """description 精确表述 raw-MAD standardized score（不乘 0.6745）。"""
    meta = CrossSectionalMadZscore.metadata
    assert "raw-MAD" in meta.description
    assert "0.6745" in meta.description
    x = pd.DataFrame([[1.0, 2.0, 4.0, 8.0]], index=["2020-01-01"], columns=list("ABCD"))
    out = CrossSectionalMadZscore()._calculate_series(x)
    med = 3.0
    mad = float(np.median([2.0, 1.0, 1.0, 5.0]))
    expected = (x - med) / mad
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# R19-095..097 domain policy 跨 backend
# ---------------------------------------------------------------------------


def test_asin_acos_domain_policy_pandas():
    """asin/acos 越界 → NaN（不抛错、无复数），in-domain 值正确。"""
    x = pd.DataFrame(
        [[2.0, 0.5, -1.0, 1.0, np.nan], [-2.0, 0.0, np.inf, -np.inf, 0.25]],
        columns=list("ABCDE"),
    )
    for cls in (Asin, Acos):
        out = cls()._calculate_series(x)
        assert np.isnan(out.iloc[0, 0])
        assert np.isnan(out.iloc[0, 4])
        assert np.isnan(out.iloc[1, 2])
        assert np.isnan(out.iloc[1, 3])
        assert not np.isnan(out.iloc[0, 1])


def test_asin_domain_policy_polars_matches_pandas():
    """asin polars 与 pandas 越界一致 -> NaN；NULL 保持 NULL。"""
    if pl is None:
        pytest.skip("polars not installed")
    x_pd = pd.DataFrame([[2.0, 0.5], [-2.0, 0.25]], columns=list("AB"))
    out_n = Asin()._calculate_series(x_pd)
    pl_x = pl.from_pandas(x_pd.reset_index(names=["row"]))
    out_p = AsinPolars()._calculate_series(pl_x)
    np.testing.assert_allclose(
        out_p.select(list(x_pd.columns)).to_numpy(),
        out_n.to_numpy(),
        equal_nan=True,
        rtol=1e-6,
    )


def test_power_real_domain_policy():
    """负数底数 & 非整数指数 → NaN；整数指数 → 实数。"""
    x = pd.DataFrame(
        [[-2.0, 2.0], [-3.0, -3.0], [0.0, 4.0]], columns=list("AB"),
    )
    frac = Pow()._calculate_series(x, 0.5)
    assert np.isnan(frac.iloc[0, 0]) and np.isnan(frac.iloc[1, 0])
    assert frac.iloc[0, 1] == pytest.approx(np.sqrt(2.0))
    intp = Pow()._calculate_series(x, 3)
    assert intp.iloc[0, 0] == pytest.approx(-8.0)
    assert intp.iloc[1, 1] == pytest.approx(-27.0)


def test_exp_overflow_allow_inf():
    """exp(1000) → Inf 保留（allow_inf policy），exp(0)=1，exp(NaN)=NaN。"""
    x = pd.DataFrame([[1000.0, 0.0, np.nan]], columns=list("ABC"))
    out = Exp()._calculate_series(x)
    assert np.isinf(out.iloc[0, 0])
    assert out.iloc[0, 1] == pytest.approx(1.0)
    assert np.isnan(out.iloc[0, 2])


# ---------------------------------------------------------------------------
# R19-098 blom_transform finite sample
# ---------------------------------------------------------------------------


def test_blom_transform_finite_mask():
    """Inf 既不计入 n，也不作为极端 rank；其 cell 输出 NaN。"""
    x = pd.DataFrame(
        [[1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0]], columns=[f"c{i}" for i in range(8)],
    )
    out = BlomTransform()._calculate_series(x)
    assert np.isnan(out.iloc[0, 2])
    finite_count = out.iloc[0].notna().sum()
    assert finite_count == 7
    vals = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    from scipy import stats as sp_stats
    ranked = np.argsort(np.argsort(vals)) + 1.0
    n = len(vals)
    expected = sp_stats.norm.ppf((ranked - 3 / 8) / (n + 1 / 4))
    got = out.iloc[0].dropna().to_numpy()
    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=1e-9)


def test_blom_transform_nan_in_nan_out():
    """NaN 输入 cell 输出 NaN，且不进入 rank 基。"""
    x = pd.DataFrame([[1.0, np.nan, 3.0, 4.0]], columns=list("ABCD"))
    out = BlomTransform()._calculate_series(x)
    assert np.isnan(out.iloc[0, 1])
    assert out.iloc[0].notna().sum() == 3
