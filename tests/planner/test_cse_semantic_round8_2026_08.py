"""AI 审计 Round-8 P1 包：rolling CSE 语义键、执行作用域隔离、重复名 fail-fast。

覆盖审计项：
- #318/#320：``rolling_semantic_key`` 从 positional literal child input 解析
  window/lag（``ts_mean(close, 20)`` 的 20 是 ``op=="literal"`` child，不在 attrs）。
- #319：所有 active 参数（``ddof``/``min_periods``/``min_count``）进入语义键，
  不再被 ``_ROLLING_ATTR_IGNORE`` 忽略。
- #321：``FactorExecutionScope`` —— 不同 freq/universe 作用域 key 不同；同一
  作用域内相同 rolling 子树可被 ``apply_rolling_cse`` 共享。
- #322：``_dag_from_factors`` 对重复 ``factor.name`` fail-fast。
- run_mode 严格化：非法值（含 typo）抛 ``ValueError``。
"""

from __future__ import annotations

import pytest

from factor_engine.planner.dag import (
    DuplicateFactorNameError,
    FactorExecutionScope,
    assert_unique_factor_names,
)
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.rolling_cache import _window_from_literals
from factor_engine.planner.rolling_cse import apply_rolling_cse, rolling_semantic_key


# ---------------------------------------------------------------------------
# PlanNode 构造 helpers
# ---------------------------------------------------------------------------
def _col(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def _lit(value: float) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": value}, inputs=[])


def _ts_mean(x: PlanNode, window: int, **attrs: object) -> PlanNode:
    return PlanNode(op="ts_mean", inputs=[x, _lit(window)], attrs=attrs)


def _ts_std(x: PlanNode, window: int, **attrs: object) -> PlanNode:
    return PlanNode(op="ts_std", inputs=[x, _lit(window)], attrs=attrs)


def _ts_delay(x: PlanNode, lag: int, **attrs: object) -> PlanNode:
    return PlanNode(op="ts_delay", inputs=[x, _lit(lag)], attrs=attrs)


def _walk(root: PlanNode):
    for c in root.inputs:
        yield from _walk(c)
    yield root


def _has_plan_ref(root: PlanNode) -> bool:
    return any(n.op == "plan_ref" for n in _walk(root))


# ---------------------------------------------------------------------------
# #318/#320：literal child 窗口解析
# ---------------------------------------------------------------------------
def test_window_from_literals_parses_child_input():
    node = _ts_mean(_col(), 20)
    assert _window_from_literals(node) == 20


def test_window_from_literals_skips_non_literal_inputs():
    # ts_corr(x, y, 20)：第二路输入是 column，应被跳过，取 literal 20
    node = PlanNode(op="ts_corr", inputs=[_col(), _col("y"), _lit(20)], attrs={})
    assert _window_from_literals(node) == 20


def test_ts_mean_window_20_vs_60_key_differs():
    k20 = rolling_semantic_key(_ts_mean(_col(), 20))
    k60 = rolling_semantic_key(_ts_mean(_col(), 60))
    assert k20 is not None and k60 is not None
    assert k20 != k60
    assert '"window":20' in k20
    assert '"window":60' in k60


def test_ts_mean_window_20_vs_60_not_shared():
    r1 = _ts_mean(_col(), 20)
    r2 = _ts_mean(_col(), 60)
    new_roots, shared = apply_rolling_cse([r1, r2])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_ts_delay_lag_1_vs_5_key_differs():
    k1 = rolling_semantic_key(_ts_delay(_col(), 1))
    k5 = rolling_semantic_key(_ts_delay(_col(), 5))
    assert k1 is not None and k5 is not None
    assert k1 != k5


def test_ts_delta_literal_lag():
    node = PlanNode(op="ts_delta", inputs=[_col(), _lit(3)], attrs={})
    assert _window_from_literals(node) == 3


# ---------------------------------------------------------------------------
# #319：所有 active 参数进 key（ddof/min_periods/min_count 不再被忽略）
# ---------------------------------------------------------------------------
def test_ts_std_ddof_0_vs_1_key_differs():
    k0 = rolling_semantic_key(_ts_std(_col(), 20, ddof=0))
    k1 = rolling_semantic_key(_ts_std(_col(), 20, ddof=1))
    assert k0 is not None and k1 is not None
    assert k0 != k1
    assert '"ddof":0' in k0
    assert '"ddof":1' in k1


def test_ts_mean_min_periods_differs():
    k1 = rolling_semantic_key(_ts_mean(_col(), 20, min_periods=1))
    k20 = rolling_semantic_key(_ts_mean(_col(), 20, min_periods=20))
    assert k1 is not None and k20 is not None
    assert k1 != k20
    assert '"min_periods":1' in k1
    assert '"min_periods":20' in k20


def test_ts_mean_min_count_in_key():
    k = rolling_semantic_key(_ts_mean(_col(), 20, min_count=10))
    assert k is not None
    assert '"min_count":10' in k


# ---------------------------------------------------------------------------
# apply_rolling_cse 共享行为
# ---------------------------------------------------------------------------
def test_apply_rolling_cse_identical_nodes_register_shared():
    # 两个完全相同的 rolling 子树：结构键相同，apply_rolling_cse 登记为共享。
    a = _ts_mean(_col(), 20)
    b = _ts_mean(_col(), 20)
    _, shared = apply_rolling_cse([a, b])
    assert len(shared) >= 1


def test_apply_rolling_cse_unifies_semantically_equivalent_aliases():
    # ts_std 与 ts_std_dev canonical 到同一 op；结构不同但语义键相同 -> 统一为 plan_ref。
    a = PlanNode(op="ts_std", inputs=[_col(), _lit(20)], attrs={})
    b = PlanNode(op="ts_std_dev", inputs=[_col(), _lit(20)], attrs={})
    new_roots, shared = apply_rolling_cse([a, b])
    assert len(shared) >= 1
    assert new_roots[0].op == "ts_std"
    assert new_roots[1].op == "plan_ref"


# ---------------------------------------------------------------------------
# #321：FactorExecutionScope
# ---------------------------------------------------------------------------
def test_scope_key_differs_on_freq_and_universe():
    s1 = FactorExecutionScope(frequency="1d", universe_id="ALL")
    s2 = FactorExecutionScope(frequency="1h", universe_id="ALL")
    s3 = FactorExecutionScope(frequency="1d", universe_id="HS300")
    assert s1.scope_key() != s2.scope_key()
    assert s1.scope_key() != s3.scope_key()
    assert (
        s1.scope_key()
        == FactorExecutionScope(frequency="1d", universe_id="ALL").scope_key()
    )


def test_same_scope_identical_rolling_subtree_shared():
    # 同一作用域内，语义键相同的 rolling 子树被 apply_rolling_cse 统一为 plan_ref。
    a = PlanNode(op="ts_std", inputs=[_col(), _lit(20)], attrs={})
    b = PlanNode(op="ts_std_dev", inputs=[_col(), _lit(20)], attrs={})
    new_roots, shared = apply_rolling_cse([a, b])
    assert len(shared) >= 1
    assert new_roots[1].op == "plan_ref"


# ---------------------------------------------------------------------------
# #322：重复 factor name fail-fast
# ---------------------------------------------------------------------------
def test_assert_unique_factor_names_raises_on_duplicate():
    with pytest.raises(DuplicateFactorNameError):
        assert_unique_factor_names(["a", "b", "a"])
    assert_unique_factor_names(["a", "b", "c"])


def test_duplicate_factor_name_raises():
    import numpy as np
    import pandas as pd

    from factor_engine.api import ts_mean
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(
        data={"close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx)}
    )
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    f1 = Factor(name="dup", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="dup", expr=ts_mean(col("close"), 3))
    with pytest.raises(DuplicateFactorNameError):
        eng._dag_from_factors([f1, f2])


def test_unique_factor_names_ok():
    import numpy as np
    import pandas as pd

    from factor_engine.api import ts_mean
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(
        data={"close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx)}
    )
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=ts_mean(col("close"), 3))
    # 不会抛错；CSE 关闭以避开 backend 执行依赖。
    dag, _ = eng._dag_from_factors([f1, f2], enable_cse=False)
    assert [r.factor_name for r in dag.roots] == ["a", "b"]


# ---------------------------------------------------------------------------
# run_mode 严格化
# ---------------------------------------------------------------------------
def test_run_mode_strict_rejects_typo_at_engine_construction():
    from factor_engine.runtime.engine import FactorEngine, _validate_run_mode

    # 合法值
    _validate_run_mode("production")
    _validate_run_mode("research")
    _validate_run_mode("paper")
    # 非法值（含 typo）直接抛 ValueError，启动即失败，不静默回落。
    with pytest.raises(ValueError):
        _validate_run_mode("prod")
    with pytest.raises(ValueError):
        FactorEngine(backend=None, data_source=None, run_mode="prod")
