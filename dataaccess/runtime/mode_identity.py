"""R39 P0 #50/#51 —— RuntimeModeIdentity：request-scoped run-mode 单一权威。

FE 与 DA 历史上存在多套 run-mode truth：
    - ``query_budget._production_mode()`` 读 FACTOR_ENGINE_RUN_MODE / QUANT_PRODUCTION_MODE
    - ``query_budget._strict_read_mode()`` 读 DATA_ACCESS_STRICT_READ
    - ``security.run_mode.resolve_run_mode()`` 读 DATA_ACCESS_RUN_MODE
    - ``is_strict_semantics()`` 又把上面几套拼在一起
每层各自重读 env → 同一请求在不同层看到不同 mode，宽松/严格语义漂移。

本模块建立**一个** request-scoped 权威：``ContextVar`` 承载
``(RunMode, source)``。``current_runtime_mode()`` 只在**无 identity** 时才
回退环境变量（DATA_ACCESS_RUN_MODE + QUANT_PRODUCTION_MODE 兼容开关），
``is_strict_semantics()``（QueryBudget floor 判定）统一从它派生——一层写入，
所有层只读。
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from data_access.security.run_mode import RunMode, resolve_run_mode

#: request-scoped (RunMode, source) 身份。None = 未设置（回退 env）。
_mode_identity_var: ContextVar[tuple[RunMode, str] | None] = ContextVar(
    "data_access_runtime_mode_identity", default=None
)


def set_runtime_mode_identity(
    mode: Any,
    source: str = "explicit",
) -> Any:
    """绑定当前 request 的 run-mode 权威，返回 ContextVar token（退出时 reset）。

    ``mode`` 接受 RunMode 或字符串（production / automated_research /
    interactive_research）。嵌套调用：内层设置覆盖外层，退出恢复外层。
    """
    rm = mode if isinstance(mode, RunMode) else RunMode(str(mode).strip().lower())
    return _mode_identity_var.set((rm, source))


def reset_runtime_mode_identity(token: Any) -> None:
    """恢复上级 run-mode identity（或 None）。"""
    _mode_identity_var.reset(token)


def current_mode_identity() -> tuple[RunMode, str] | None:
    """当前 request 的 (RunMode, source)；None = 未设置（走 env 回退）。"""
    return _mode_identity_var.get()


def current_runtime_mode() -> RunMode:
    """权威 run mode：request identity 优先；无 identity 才回退 env。

    回退优先级（与 ``resolve_run_mode`` 一致）：DATA_ACCESS_RUN_MODE >
    QUANT_PRODUCTION_MODE=1 → production > interactive_research。
    """
    ident = _mode_identity_var.get()
    if ident is not None:
        return ident[0]
    return resolve_run_mode()


def current_mode_source() -> str | None:
    """当前 mode 来源（"explicit" / "contextvar" / None=env 回退）。"""
    ident = _mode_identity_var.get()
    return ident[1] if ident is not None else None


def is_strict_semantics_authority() -> bool:
    """R39 P0 #51：**唯一**严格语义判定（QueryBudget floor / fail-closed 门）。

    规则：
        - DATA_ACCESS_STRICT_READ=1 是独立全局 strict 开关（与 mode 无关）；
        - 否则 strict = 当前权威 mode 是 production / automated_research
          （``RunMode.strict_semantics``，INV-07：机器不看 warning）。
    所有 QueryBudget / PIT / calendar / required-filters 的 fail-closed 判定都
    从这一个函数派生，不再各有各的 env 拼接。
    """
    if _env_strict_read():
        return True
    return current_runtime_mode().strict_semantics


def is_production_authority() -> bool:
    """唯一 production 判定（host lease / publish / telemetry 用）。"""
    return current_runtime_mode().is_production


def _env_strict_read() -> bool:
    return os.environ.get("DATA_ACCESS_STRICT_READ", "").lower() in {
        "1",
        "true",
        "yes",
    }


__all__ = [
    "set_runtime_mode_identity",
    "reset_runtime_mode_identity",
    "current_mode_identity",
    "current_runtime_mode",
    "current_mode_source",
    "is_strict_semantics_authority",
    "is_production_authority",
]
