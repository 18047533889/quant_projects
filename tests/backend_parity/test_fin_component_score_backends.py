# -*- coding: utf-8
"""fin_component_score 双后端（Pandas reference / Polars native）parity。

R25-184 轴契约：每个 component 是独立的 ``date x instrument`` 面板，score 逐
cell 跨 component slot 求和 —— 每个股票列拿到自己的真实 score（不再是"把
DataFrame 列当组件、把一行横截面 score 重复回所有列"）。SQL 长表下推模型里
每个输入绑定只有单一 ``_v`` 列，无法表达多面板组件 —— 因此该算子**有意不做
SQL pushdown**（不是缺失，而是语义不匹配）。本测试锁定 pandas 与 polars 两个
真实后端在相同数据上逐值一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.fundamental.component_score import FinComponentScore
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()


def _pandas_ref(*comp_panels: pd.DataFrame, **params) -> np.ndarray:
    out = FinComponentScore()._calculate_series(*comp_panels, **params)
    return out.to_numpy(dtype=float)


def _polars_ref(*comp_panels: pd.DataFrame, **params) -> np.ndarray:
    import polars as pl

    op = OperatorRegistry.get("fin_component_score", "polars")
    assert op is not None, "polars backend for fin_component_score not registered"
    pl_panels = []
    for pdf in comp_panels:
        pdf = pdf.copy()
        pdf.reset_index(drop=True, inplace=True)
        pdf["__row__"] = np.arange(len(pdf))
        plf = pl.from_pandas(pdf)
        pl_panels.append(plf.drop("__row__"))
    res = op._calculate_series(*pl_panels, **params)
    return np.asarray(res.to_numpy(), dtype=float)


def _assert_parity(*comp_panels: pd.DataFrame, **params):
    pd_out = _pandas_ref(*comp_panels, **params)
    pl_out = _polars_ref(*comp_panels, **params)
    assert pd_out.shape == pl_out.shape
    nan_mask = np.isnan(pd_out) | np.isnan(pl_out)
    np.testing.assert_array_equal(
        np.isnan(pd_out), np.isnan(pl_out), err_msg="NaN masks differ"
    )
    np.testing.assert_allclose(
        pd_out[~nan_mask], pl_out[~nan_mask], rtol=1e-9, atol=1e-12
    )


def _panel(values) -> pd.DataFrame:
    # Each component is its OWN date x instrument panel (one value per stock).
    return pd.DataFrame(
        values, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), dtype=float
    )


def test_fin_component_score_pandas_polars_parity_default_directions():
    # Three separate component panels (date x instrument), default "up" direction.
    _assert_parity(
        _panel({"A": [1.0, -2.0]}),
        _panel({"A": [3.0, 4.0]}),
        _panel({"A": [-1.0, 0.5]}),
    )


def test_fin_component_score_pandas_polars_parity_explicit_directions_and_weights():
    _assert_parity(
        _panel({"A": [1.0, -2.0], "B": [0.5, 1.0]}),
        _panel({"A": [3.0, 4.0], "B": [-1.0, 2.0]}),
        _panel({"A": [-1.0, 0.5], "B": [2.0, -3.0]}),
        component_directions=["up", "down", "down"],
        score_weights=[1.0, 2.0, 0.5],
    )


def test_fin_component_score_pandas_polars_parity_nan_rows():
    _assert_parity(
        _panel({"A": [1.0, np.nan], "B": [0.0, 1.0]}),
        _panel({"A": [-2.0, np.nan], "B": [1.0, -1.0]}),
        _panel({"A": [3.0, np.nan], "B": [2.0, 0.0]}),
    )


def test_fin_component_score_scores_per_instrument_not_cross_sectional():
    # R25-078/184: each instrument must get its OWN score — the operator must
    # NOT score across instruments and repeat one value into every column.
    comp = _panel({"A": [1.0, 2.0], "B": [-3.0, -4.0]})  # single component
    out = FinComponentScore()._calculate_series(comp)
    assert out["A"].iloc[0] == 1.0, "instrument A score must be 1 (its own value)"
    assert out["B"].iloc[0] == 0.0, "finite FALSE is a zero contribution, not missing"


def test_fin_component_score_polars_backend_registered():
    assert "polars" in OperatorRegistry.backends_for("fin_component_score")


def test_fin_component_score_sql_pushdown_is_deliberately_unsupported():
    """SQL 长表模型只有单一 _v 列，多面板组件语义不匹配。

    该断言防止未来有人误把 fin_component_score 加入 SQL_IMPLEMENTED ——
    除非 emitter 引入多面板 layer（大改动，需独立评审）。
    """
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    assert "fin_component_score" not in SQL_IMPLEMENTED_CANONICALS
