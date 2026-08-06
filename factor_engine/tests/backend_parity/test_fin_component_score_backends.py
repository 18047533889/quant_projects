# -*- coding: utf-8
"""fin_component_score 双后端（Pandas reference / Polars native）parity。

输入语义是「观测 × 组件」多列面板（一行一个观测，每列一个组件），而非单值
长表列。SQL 长表下推模型里每个输入绑定只有单一 ``_v`` 列，无法表达多列组件
面板 —— 因此该算子**有意不做 SQL pushdown**（不是缺失，而是语义不匹配）。
本测试锁定 pandas 与 polars 两个真实后端在相同数据上逐值一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.fundamental.component_score import FinComponentScore
from cleaned_operators.registry import OperatorRegistry

load_all()


def _pandas_ref(comp_panel: pd.DataFrame, **params) -> np.ndarray:
    out = FinComponentScore()._calculate_series(comp_panel, **params)
    return out.to_numpy(dtype=float)


def _polars_ref(comp_panel: pd.DataFrame, **params) -> np.ndarray:
    import polars as pl

    op = OperatorRegistry.get("fin_component_score", "polars")
    assert op is not None, "polars backend for fin_component_score not registered"
    pdf = comp_panel.copy()
    pdf.reset_index(drop=True, inplace=True)
    pdf["__row__"] = np.arange(len(pdf))
    plf = pl.from_pandas(pdf)
    res = op._calculate_series(plf.drop("__row__"), **params)
    return np.asarray(res.to_numpy(), dtype=float)


def _assert_parity(comp_panel: pd.DataFrame, **params):
    pd_out = _pandas_ref(comp_panel, **params)
    pl_out = _polars_ref(comp_panel, **params)
    assert pd_out.shape == pl_out.shape
    nan_mask = np.isnan(pd_out) | np.isnan(pl_out)
    np.testing.assert_array_equal(
        np.isnan(pd_out), np.isnan(pl_out), err_msg="NaN masks differ"
    )
    np.testing.assert_allclose(
        pd_out[~nan_mask], pl_out[~nan_mask], rtol=1e-9, atol=1e-12
    )


def _panel(values) -> pd.DataFrame:
    return pd.DataFrame(
        values, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), dtype=float
    )


def test_fin_component_score_pandas_polars_parity_default_directions():
    _assert_parity(_panel({"c1": [1.0, -2.0], "c2": [3.0, 4.0], "c3": [-1.0, 0.5]}))


def test_fin_component_score_pandas_polars_parity_explicit_directions_and_weights():
    _assert_parity(
        _panel({"c1": [1.0, -2.0], "c2": [3.0, 4.0], "c3": [-1.0, 0.5]}),
        component_directions=["up", "down", "down"],
        score_weights=[1.0, 2.0, 0.5],
    )


def test_fin_component_score_pandas_polars_parity_nan_rows():
    _assert_parity(
        _panel({"c1": [1.0, np.nan], "c2": [-2.0, np.nan], "c3": [3.0, np.nan]})
    )


def test_fin_component_score_polars_backend_registered():
    assert "polars" in OperatorRegistry.backends_for("fin_component_score")


def test_fin_component_score_sql_pushdown_is_deliberately_unsupported():
    """SQL 长表模型只有单一 _v 列，多列组件面板语义不匹配。

    该断言防止未来有人误把 fin_component_score 加入 SQL_IMPLEMENTED ——
    除非 emitter 引入多列 layer（大改动，需独立评审）。
    """
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    assert "fin_component_score" not in SQL_IMPLEMENTED_CANONICALS
