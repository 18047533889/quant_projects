"""Assetization 对当前仓库根 FactorEngine/DataAccess 的兼容门面。

旧版目录内复制的 ``factor_engine`` 已废弃。该模块保留原有四个公开函数，但所有
解析、计划、执行和正式因子写入均委托给 ``integrations.quant_platform``。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from integrations.quant_platform import (
    InMemoryFrameSource,
    bootstrap_quant_platform,
    materialize_factor_to_staging,
    validate_factor_formula,
)
from .models import PhysicalPlan

bootstrap_quant_platform()

from backend.context import ExecutionContext  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from planner.logical_plan import PlanNode  # noqa: E402


def parse_and_optimize(
    formula: str,
    *,
    backend: str = "pandas",
) -> PhysicalPlan:
    """用当前 FactorEngine 完整 parser/analyzer/lowerer/optimizer 编译 DSL。"""
    factor, plan, analysis = validate_factor_formula(
        formula,
        name="autofactor_assetization",
        backend=backend,
        run_mode="research",
    )
    return PhysicalPlan(
        dag=_plan_to_dict(plan),
        reused_nodes=[],
        backend=backend,
        _plan_node=plan,
        _ir_node=analysis.ir,
    )


def execute_plan(
    plan: PhysicalPlan,
    data: pd.DataFrame,
) -> np.ndarray:
    """在当前 FactorEngine 后端上执行已编译计划。"""
    if plan._plan_node is None:
        raise ValueError("PhysicalPlan 缺少可执行 PlanNode；请由 parse_and_optimize 创建")
    source = InMemoryFrameSource(data)
    context = ExecutionContext(data_source=source)
    result = build_backend(plan.backend).execute(plan._plan_node, context)
    if isinstance(result, pd.Series):
        return result.to_numpy(dtype=float, copy=False)
    if isinstance(result, pd.DataFrame):
        if result.shape[1] != 1:
            raise TypeError("执行结果 DataFrame 必须只有一列")
        return result.iloc[:, 0].to_numpy(dtype=float, copy=False)
    if hasattr(result, "to_numpy"):
        return np.asarray(result.to_numpy())
    return np.asarray(result)


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_time_col: str,
    right_time_col: str,
    *,
    by: str | None = None,
) -> pd.DataFrame:
    """严格 ``right.time < left.time`` 的 PIT-safe ASOF JOIN。"""
    left_sort = [by, left_time_col] if by else [left_time_col]
    right_sort = [by, right_time_col] if by else [right_time_col]
    ldf = left.sort_values(left_sort, kind="mergesort").copy()
    rdf = right.sort_values(right_sort, kind="mergesort").copy()
    return pd.merge_asof(
        ldf,
        rdf,
        left_on=left_time_col,
        right_on=right_time_col,
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )


def atomic_materialize(
    tensor: np.ndarray | pd.Series | pd.DataFrame,
    target_path: str | Path | None = None,
    schema: dict[str, str] | None = None,
    null_threshold: float = 0.05,
    *,
    factor_id: str | None = None,
    factor_version: str = "1",
    data_snapshot_id: str | None = None,
    publish: bool = False,
) -> tuple[str, bool]:
    """正式写入走 DataAccess staging；临时文件保留本地原子写兼容路径。"""
    if factor_id is not None:
        if not isinstance(tensor, pd.Series):
            raise TypeError("写 factor_lake_staging 时 tensor 必须是 MultiIndex Series")
        summary = materialize_factor_to_staging(
            tensor,
            factor_id=factor_id,
            factor_version=factor_version,
            snapshot_id=data_snapshot_id,
            publish=publish,
        )
        return f"data_access://factor_lake_staging/{factor_id}", bool(summary["rows"] >= 0)

    if target_path is None:
        raise ValueError("本地原子物化必须提供 target_path；正式因子写入请提供 factor_id")
    target = Path(target_path)
    staging = target.with_name(f".{target.name}.staging")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.unlink(missing_ok=True)
    _write_parquet(tensor, staging)
    try:
        written = pd.read_parquet(staging)
        if schema is not None:
            _validate_schema(written, schema)
        null_rate = _compute_null_rate(written)
        if null_rate > null_threshold:
            staging.unlink(missing_ok=True)
            return str(target), False
        os.replace(staging, target)
        return str(target), True
    except Exception:
        staging.unlink(missing_ok=True)
        raise


def _plan_to_dict(node: PlanNode) -> dict[str, Any]:
    return {
        "op": node.op,
        "attrs": dict(node.attrs),
        "inputs": [_plan_to_dict(child) for child in node.inputs],
    }


def _write_parquet(
    tensor: np.ndarray | pd.Series | pd.DataFrame,
    path: Path,
) -> None:
    if isinstance(tensor, pd.DataFrame):
        tensor.to_parquet(path, index=False)
    elif isinstance(tensor, pd.Series):
        tensor.to_frame("value").to_parquet(path)
    else:
        pd.DataFrame({"value": np.asarray(tensor)}).to_parquet(path, index=False)


def _validate_schema(df: pd.DataFrame, schema: dict[str, str]) -> None:
    for column, expected in schema.items():
        if column not in df.columns:
            raise ValueError(f"Schema 缺少列 {column!r}")
        actual = str(df[column].dtype)
        if actual != expected:
            raise ValueError(f"列 {column!r} dtype 期望 {expected}，实际 {actual}")


def _compute_null_rate(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return float(df.isna().to_numpy().sum() / df.size)
