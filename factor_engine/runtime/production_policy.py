"""Production 硬策略：research 宽松 / production 强制门禁。"""

from __future__ import annotations

import os
from typing import Any, Iterable

PRODUCTION_MODE = "production"


class ProductionPolicyViolation(ValueError):
    """production 模式下违反硬策略。"""


def resolve_run_mode(mode: str | None = None) -> str:
    if mode is not None and str(mode).strip():
        return str(mode).lower()
    return str(os.environ.get("FACTOR_ENGINE_RUN_MODE", "research")).lower()


def is_production_mode(mode: str | None = None) -> bool:
    return resolve_run_mode(mode) == PRODUCTION_MODE


def assert_columns_explicit(
    columns: Iterable[str] | None,
    *,
    mode: str | None = None,
    context: str = "read",
) -> None:
    """production 禁止 SELECT * / 空列列表。"""
    if not is_production_mode(mode):
        return
    if columns is None:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须显式指定 columns，禁止 SELECT *"
        )
    cols = list(columns)
    if not cols or "*" in cols:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须显式列名，禁止 SELECT * 或空 columns"
        )


def assert_production_run_flags(
    *,
    mode: str | None = None,
    input_dq_check: bool = False,
    auto_warmup: bool = False,
    pit_enforce: bool = False,
    context: str = "run",
) -> None:
    """production 强制 input_dq / auto_warmup / PIT。"""
    if not is_production_mode(mode):
        return
    missing: list[str] = []
    if not input_dq_check:
        missing.append("input_dq_check")
    if not auto_warmup:
        missing.append("auto_warmup")
    if not pit_enforce:
        missing.append("pit_enforce")
    if missing:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须开启: {', '.join(missing)}"
        )


def assert_no_stub_operators(plan: Any, *, mode: str | None = None) -> None:
    """production 禁止 ``*_stub`` 占位算子。"""
    if not is_production_mode(mode):
        return
    stub_ops: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op.endswith("_stub"):
            stub_ops.append(op)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    if stub_ops:
        unique = sorted(set(stub_ops))
        raise ProductionPolicyViolation(
            f"production 模式禁止 stub 算子: {', '.join(unique)}"
        )
