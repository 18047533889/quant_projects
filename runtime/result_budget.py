# -*- coding: utf-8 -*-
"""Phase 5 R18：最终结果字节预算。

10 年分钟因子 × 5000 标的，即使中间计算很省，最终 pd.Series 本身也可能数 GB。
这里把「最终结果」也纳入资源预算：

- 显式配置了 ``result_budget_bytes``（``resources.results.max_in_memory_bytes`` /
  ``FACTOR_ENGINE_RESULT_BUDGET_BYTES``）时强制执行（production fail-closed，
  research 抛同错误但可经 ``FACTOR_ENGINE_RESULT_BUDGET_WARN_ONLY=1`` 降级为告警）。
- 未显式配置时不额外打扰（保持旧行为），避免破坏已有批跑。
"""
from __future__ import annotations

import logging
from typing import Any

from runtime.resource_errors import ResourceBudgetExceeded
from runtime.resource_governor import estimate_object_bytes

logger = logging.getLogger("runtime.result_budget")


def effective_result_budget_bytes(perf: Any) -> int | None:
    """显式结果字节预算（无显式配置返回 ``None`` = 不强制）。"""
    if perf is None:
        return None
    explicit = getattr(perf, "result_budget_bytes", None)
    if explicit is not None and explicit > 0:
        return int(explicit)
    return None


def enforce_result_budget(
    result: Any,
    perf: Any,
    *,
    factor_name: str = "",
    run_mode: str | None = None,
) -> bool:
    """检查最终结果是否超显式字节预算；返回是否放行。

    - 未显式配置 → 放行（返回 True）
    - 超限：
        - production：抛 ``ResourceBudgetExceeded``（要求改用 sink/stream/materialize）
        - research：默认抛（fail-closed）；``FACTOR_ENGINE_RESULT_BUDGET_WARN_ONLY=1``
          时仅告警并放行
    """
    budget = effective_result_budget_bytes(perf)
    if budget is None or result is None:
        return True
    size = estimate_object_bytes(result)
    if size <= budget:
        return True
    prod = str(run_mode or "").lower() == "production"
    warn_only = not prod and (
        __import__("os").environ.get("FACTOR_ENGINE_RESULT_BUDGET_WARN_ONLY", "").lower()
        in {"1", "true", "yes"}
    )
    if warn_only:
        logger.warning(
            "factor %s result %.1f MiB exceeds result_budget %.1f MiB (warn-only)",
            factor_name,
            size / 1024**2,
            budget / 1024**2,
        )
        return True
    raise ResourceBudgetExceeded(
        f"final result {size / 1024**2:.1f} MiB exceeds result_budget "
        f"{budget / 1024**2:.1f} MiB"
        + (f" (factor={factor_name})" if factor_name else "")
        + "; use result_policy=sink / run_many_iter / stream instead of "
        "returning the full series in memory"
    )
