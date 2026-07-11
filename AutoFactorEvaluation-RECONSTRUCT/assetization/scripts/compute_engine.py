"""因子特征物化与核心计算引擎（spec 第 2 部分）。

包装 factor_engine，对外暴露 spec 约定的四个方法：
- parse_and_optimize   (2.2.1)
- execute_plan         (2.2.2)
- asof_join            (2.2.3)
- atomic_materialize   (2.2.4)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# factor_engine 内部使用 ``from api.xxx import ...`` 风格的相对导入，
# 因此需要同时把 monorepo 根和 factor_engine 本身加入 sys.path。
_MONOREPO_ROOT = Path(__file__).resolve().parents[1]  # AutoFactorEvaluation/
_FACTOR_ENGINE_ROOT = _MONOREPO_ROOT / "factor_engine"

for _p in (str(_MONOREPO_ROOT), str(_FACTOR_ENGINE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.dsl_parser import parse_expr                                                  # noqa: E402
from backend.context import ExecutionContext                                          # noqa: E402
from backend.factory import build_backend                                             # noqa: E402
from ir.analyzer import Analyzer                                                      # noqa: E402
from planner.logical_plan import PlanNode                                             # noqa: E402
from planner.lowerer import Lowerer                                                   # noqa: E402
from planner.optimizer import Optimizer                                               # noqa: E402
from storage.datasource import DataSource                                             # noqa: E402

from .models import PhysicalPlan                                   # noqa: E402


# ---------------------------------------------------------------------------
# In-memory data source —— 把 DataFrame 包装为 DataSource
# ---------------------------------------------------------------------------

class _InMemorySource(DataSource):
    """将 ``{col: Series}`` 字典包装为 DataSource，供 execute_plan 使用。"""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self._data = data

    def load_column(self, name: str) -> pd.Series:
        if name not in self._data:
            raise KeyError(
                f"Column {name!r} not found in data source. "
                f"Available: {list(self._data.keys())}"
            )
        return self._data[name]


# ---------------------------------------------------------------------------
# 2.2.1 DSL 解析与 DAG 优化
# ---------------------------------------------------------------------------

def parse_and_optimize(
    formula: str,
    *,
    backend: str = "pandas",
) -> PhysicalPlan:
    """解析 DSL 公式串，编译并优化为物理执行计划。

    spec 2.2.1 — ``parse_and_optimize``

    Args:
        formula: 因子表达式字符串，如 ``ts_mean(close, 5) / ts_mean(close, 10)``。
        backend: 目标执行后端，默认 ``pandas``。

    Returns:
        PhysicalPlan: 包含序列化 DAG、复用节点列表与后端标识。
    """
    # 1. Parse DSL → Expr tree
    expr = parse_expr(formula)

    # 2. Expr → IR（含依赖列分析）
    analyzer = Analyzer()
    analysis = analyzer.lower(expr)

    # 3. IR → LogicalPlan
    lowerer = Lowerer()
    logical_plan = lowerer.to_logical_plan(analysis.ir)

    # 4. Optimize（常量折叠等）
    opt = Optimizer()
    optimized = opt.optimize(logical_plan)

    # 5. 序列化 DAG 并封装
    dag = _plan_to_dict(optimized)
    return PhysicalPlan(
        dag=dag,
        reused_nodes=[],
        backend=backend,
        _plan_node=optimized,
        _ir_node=analysis.ir,
    )


# ---------------------------------------------------------------------------
# 2.2.2 向量化 / JIT 编译执行
# ---------------------------------------------------------------------------

def execute_plan(
    plan: PhysicalPlan,
    data: pd.DataFrame,
) -> np.ndarray:
    """在多后端上执行物理计划，返回张量结果。

    spec 2.2.2 — ``execute_plan``

    Args:
        plan: 由 ``parse_and_optimize`` 产出的物理计划。
        data: 输入 DataFrame，列名需覆盖计划中引用的所有列。

    Returns:
        np.ndarray: 因子值张量。
    """
    if plan._plan_node is None:
        raise ValueError(
            "PhysicalPlan has no attached plan node. "
            "Was it created by parse_and_optimize?"
        )

    backend = build_backend(plan.backend)

    # DataFrame 列 → DataSource
    data_dict: dict[str, pd.Series] = {}
    for col in data.columns:
        data_dict[col] = data[col]
    source = _InMemorySource(data_dict)

    ctx = ExecutionContext(data_source=source)
    result = backend.execute(plan._plan_node, ctx)

    # 统一返回 ndarray（spec 要求 np.ndarray 或 polars.Series）
    if hasattr(result, "to_numpy"):
        return result.to_numpy()
    return np.asarray(result)


# ---------------------------------------------------------------------------
# 2.2.3 双时态 ASOF JOIN
# ---------------------------------------------------------------------------

def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_time_col: str,
    right_time_col: str,
    *,
    by: str | None = None,
) -> pd.DataFrame:
    """双时态 ASOF JOIN：``WHERE right.time < left.time``，防止未来函数。

    spec 2.2.3 — ``asof_join``

    使用 ``pd.merge_asof(direction="backward")`` 取右表最后一条不晚于左表
    时间戳的记录，随后将恰好等价的匹配行掩码为 NaN，确保严格 ``<``。

    Args:
        left: 左表（交易流）。
        right: 右表（knowledge base）。
        left_time_col: 左表时间列名。
        right_time_col: 右表时间列名。
        by: 可选分组列（如 instrument），按组做 asof join。

    Returns:
        pd.DataFrame: JOIN 后的 DataFrame。
    """
    left = left.sort_values(left_time_col).reset_index(drop=True)
    right = right.sort_values(right_time_col).reset_index(drop=True)

    # 若左右时间列同名，右表时间列需临时重命名，以免被 merge_asof 吞掉
    right_time_join = right_time_col
    if left_time_col == right_time_col:
        right_time_join = "_right_time_tmp"
        right = right.rename(columns={right_time_col: right_time_join})

    merged = pd.merge_asof(
        left,
        right,
        left_on=left_time_col,
        right_on=right_time_join,
        by=by,
        direction="backward",
    )

    # 强制 strict <：恰好时间对齐的行，把右表值置 NaN
    right_cols = [
        c for c in right.columns
        if c != right_time_join and (by is None or c != by)
    ]
    mask = merged[right_time_join] >= merged[left_time_col]
    for c in right_cols:
        merged.loc[mask, c] = np.nan

    # 清理临时列
    if right_time_join != right_time_col:
        merged = merged.drop(columns=[right_time_join])

    return merged


# ---------------------------------------------------------------------------
# 2.2.4 特征张量写入 Staging 与原子落盘
# ---------------------------------------------------------------------------

def atomic_materialize(
    tensor: np.ndarray | pd.Series | pd.DataFrame,
    target_path: str | Path,
    schema: dict[str, str] | None = None,
    null_threshold: float = 0.05,
) -> tuple[str, bool]:
    """写入 staging，校验 schema 与空值率，通过后原子重命名到正式路径。

    spec 2.2.4 — ``atomic_materialize``

    流程：
    1. 写入 ``<target>.staging`` 临时文件
    2. 校验 schema（如提供）
    3. 计算空值率，超过 ``null_threshold`` 则清理 staging 并返回 False
    4. ``os.replace`` 原子覆盖到正式路径

    Args:
        tensor: 因子值张量。
        target_path: 最终落盘路径。
        schema: 可选的列名 → dtype 映射，用于校验。
        null_threshold: 最大允许空值率，默认 0.05。

    Returns:
        (final_path, success): 正式路径与是否成功。
    """
    target = Path(target_path)
    staging = target.with_name(f".{target.name}.staging")

    # 1. 写入 staging
    target.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        staging.unlink()

    _write_parquet(tensor, staging)

    try:
        written = pd.read_parquet(staging)

        # 2. Schema 校验
        if schema is not None:
            _validate_schema(written, schema)

        # 3. 空值率校验
        null_rate = _compute_null_rate(written)
        if null_rate > null_threshold:
            staging.unlink()
            return str(target), False

        # 4. 原子 rename
        if target.exists():
            target.unlink()
        os.replace(str(staging), str(target))
        return str(target), True

    except Exception:
        if staging.exists():
            staging.unlink()
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_to_dict(node: PlanNode) -> dict:
    """将 PlanNode 树序列化为 dict（供 PhysicalPlan.dag）。"""
    return {
        "op": node.op,
        "attrs": dict(node.attrs),
        "inputs": [_plan_to_dict(c) for c in node.inputs],
    }


def _write_parquet(
    tensor: np.ndarray | pd.Series | pd.DataFrame,
    path: Path,
) -> None:
    """统一写 parquet 入口。"""
    if isinstance(tensor, pd.DataFrame):
        tensor.to_parquet(path)
    elif isinstance(tensor, pd.Series):
        tensor.to_frame("value").to_parquet(path)
    else:
        pd.DataFrame({"value": np.asarray(tensor)}).to_parquet(path)


def _validate_schema(df: pd.DataFrame, schema: dict[str, str]) -> None:
    """校验 DataFrame 列名与 dtype 是否符合 schema。"""
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            raise ValueError(
                f"Schema validation failed: column {col!r} not found. "
                f"Existing columns: {list(df.columns)}"
            )
        actual = str(df[col].dtype)
        if actual != expected_dtype:
            raise ValueError(
                f"Schema validation failed for column {col!r}: "
                f"expected {expected_dtype}, got {actual}"
            )


def _compute_null_rate(df: pd.DataFrame) -> float:
    """计算 DataFrame 整体空值率。"""
    total = df.size
    if total == 0:
        return 0.0
    nulls = df.isna().sum().sum()
    return float(nulls / total)
