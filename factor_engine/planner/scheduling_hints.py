"""基于 plan 代价摘要的执行调度建议。"""

from __future__ import annotations

from typing import Any


def derive_scheduling_hints(cost_summary: dict[str, Any] | None) -> dict[str, Any]:
    """从 ``summarize_plans`` 输出推导 run_many / 物化调度建议。

    参数：
        cost_summary: :func:`planner.cost_summary.summarize_plans` 的汇总字典；
            为 ``None`` 或空时返回空建议

    返回：
        含 ``hints``（建议动作列表）与 ``recommended``（推荐运行参数字典）的字典
    """
    if not cost_summary:
        return {"hints": [], "recommended": {}}

    hints: list[dict[str, str]] = []
    recommended: dict[str, Any] = {}

    high_memory = list(cost_summary.get("high_memory_ops") or [])
    tier_hist = cost_summary.get("tier_histogram") or {}
    tier3 = int(tier_hist.get("3", 0))
    factor_count = int(cost_summary.get("factor_count") or 0)

    if high_memory:
        hints.append(
            {
                "action": "materialize_sharded",
                "reason": f"高内存算子: {', '.join(high_memory[:5])}",
            }
        )
        recommended["shard_by"] = "asset_bucket"

    if tier3 > 0:
        hints.append(
            {
                "action": "prefer_polars_lazy",
                "reason": f"含 {tier3} 个 tier-3(pandas) 节点，建议 PolarsBackend(use_lazy=True)",
            }
        )
        recommended["backend"] = "polars_lazy"

    if factor_count > 4 and not high_memory:
        hints.append(
            {
                "action": "run_many_parallel",
                "reason": f"多因子 batch({factor_count}) 可并行根节点",
            }
        )
        recommended["parallel_roots"] = True

    if int(tier_hist.get("0", 0)) > 20:
        hints.append(
            {
                "action": "enable_cse",
                "reason": "大量 tier-0 滚动节点，CSE 收益高",
            }
        )
        recommended["enable_cse"] = True

    return {"hints": hints, "recommended": recommended}
