"""DSL 白名单构建：合法函数名 → ``CleanedCall`` 工厂。

``build_dsl_allowlist()`` 是投递校验、``parse_expr``、挖掘对接的 **唯一算子名来源**：
- ``col``：字段引用；
- 其余键来自 ``cleaned_operators`` 中 **已实现** 的 canonical 名与 ``_aliases.py`` 别名。

**动态白名单**：不再硬编码内建算子。白名单在每次调用时从 ``OperatorRegistry``
当前已注册的算子中动态生成 —— 无论内建算子还是通过 ``factor_engine_operators``
加载的外部算子，只要已注册到 ``OperatorRegistry`` 且拥有 runtime 实现，都会自动进入白名单。

服务启动时序（``FactorEngine.from_loaded_config``）:
  1. ``ensure_cleaned_loaded()`` — 加载内建算子
  2. ``maybe_load_operators_from_config()`` — 加载外部算子
  3. ``parse_expr()`` / ``parse_factor()`` — 基于当前 ``OperatorRegistry`` 构建白名单并解析
"""

from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

# 兼容旧代码中对 stub 集合的引用（Gateway 等）；cleaned 未实现的算子会在 runtime 报错。
STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist() -> dict[str, Callable[..., Any]]:
    """返回 ``{函数名: 工厂}``，供 ``parse_expr`` 与 ``validate_us_dsl`` 使用。

    每次调用都重新查询 ``OperatorRegistry`` 的当前状态，
    因此外部算子加载后自动生效，无需重启。
    """
    allow = {"col": col}
    allow.update(build_cleaned_dsl_allowlist(set()))
    return allow
