"""DSL 白名单构建：合法函数名 → ``CleanedCall`` 工厂。

``build_dsl_allowlist()`` 是投递校验、``parse_expr``、挖掘对接的 **唯一算子名来源**：
- ``col``：字段引用；
- 其余键来自 ``cleaned_operators`` 中 **已实现** 的 canonical 名与 ``_aliases.py`` 别名。

注意
----
- 白名单里有名但 ``OperatorRegistry.get`` 为 ``None`` 的条目不应出现（当前 loader 已过滤）。
- ``STUB_IR_OPS`` 保留为空集，仅兼容旧 Gateway 代码引用；无 stub runtime。
"""

from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

# 兼容旧代码中对 stub 集合的引用（Gateway 等）；cleaned 未实现的算子会在 runtime 报错。
STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist() -> dict[str, Callable[..., Any]]:
    """返回 ``{函数名: 工厂}``，供 ``parse_expr`` 与 ``validate_us_dsl`` 使用。

    合并 ``col`` 与 ``build_cleaned_dsl_allowlist`` 中已实现的 cleaned 算子；
    是投递校验、DSL 解析、挖掘对接的 **唯一算子名来源**。

    Returns
    -------
    dict[str, Callable[..., Any]]
        键为 DSL 可调用的函数名，值为对应的 ``CleanedCall`` 工厂。
    """
    allow = {"col": col}
    allow.update(build_cleaned_dsl_allowlist(set()))
    return allow
