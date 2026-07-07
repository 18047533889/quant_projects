# -*- coding: utf-8 -*-
"""算子执行策略（Operator Policy）：企业级语义的标准化描述。

与 ``OperatorMetadata``（文档/catalog）互补：
- metadata：人类可读、LLM 提示词
- policy：机器可读、lookback 推断、PIT 审计、lineage hash
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Scope = Literal["ts", "cs", "elementwise", "aggregate", "hypothesis", "unknown"]
NanPolicy = Literal["propagate", "ignore", "zero", "ffill_only"]

# 显式声明的核心算子策略（其余由 infer_operator_policy 推断）
_EXPLICIT_POLICIES: dict[str, dict[str, Any]] = {
    "ts_delay": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_delta": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_std": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_rank": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_min": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_max": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quantile": {"scope": "cs", "pit_safe": True},
    "scale": {"scope": "cs", "pit_safe": True},
    "normalize": {"scope": "cs", "pit_safe": True},
    "standardize": {"scope": "cs", "pit_safe": True},
    "col": {"scope": "elementwise", "lookback_window": 0, "pit_safe": True},
    "Lead": {"scope": "ts", "lag": -1, "pit_safe": True},
    "next": {"scope": "ts", "lag": -1, "pit_safe": True},
    "bfill": {"scope": "elementwise", "pit_safe": True, "nan_policy": "ffill_only"},
    "rank": {"scope": "cs", "pit_safe": True},
    "zscore": {"scope": "cs", "pit_safe": True},
    "winsorize": {"scope": "cs", "pit_safe": True},
    "neutralize": {"scope": "cs", "pit_safe": True},
    "group_rank": {"scope": "cs", "pit_safe": True},
    "group_neutralize": {"scope": "cs", "pit_safe": True},
    "shuffle": {"scope": "unknown", "pit_safe": True},
    "avg": {"scope": "aggregate", "pit_safe": True},
    "corr_test": {"scope": "hypothesis", "pit_safe": True},
}


@dataclass
class OperatorPolicy:
    """算子执行语义（企业级 schema 子集）。"""

    scope: Scope = "unknown"
    lookback_window: int | None = None
    min_periods: int | None = None
    lag: int = 0
    pit_safe: bool = True
    nan_policy: NanPolicy = "propagate"
    includes_current_bar: bool = True
    calendar: str = "bar"  # 本引擎 bar = 每标的连续行，非自然日
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_operator_policy(op: Any, *, canonical: str | None = None) -> OperatorPolicy:
    """从算子实例 metadata/category/tags 推断默认 policy。"""
    canon = canonical or getattr(getattr(op, "metadata", None), "name", "") or ""
    if canon in _EXPLICIT_POLICIES:
        return OperatorPolicy(**{**{"scope": "unknown", "pit_safe": True}, **_EXPLICIT_POLICIES[canon]})

    meta = getattr(op, "metadata", None)
    category = (getattr(meta, "category", "") or "").lower()
    tags = [str(t).lower() for t in (getattr(meta, "tags", None) or [])]
    name = (getattr(meta, "name", "") or canon).lower()

    scope: Scope = "unknown"
    if category in ("cross_sectional",) or "cross_section" in tags:
        scope = "cs"
    elif category in ("group_neutralization",):
        scope = "cs"
    elif category in (
        "time_series",
        "shift_diff_cum",
        "technical_signal",
        "price_volume",
        "intraday_microstructure",
        "signal",
    ):
        scope = "ts"
    elif category in ("statistics",):
        if any(k in name for k in ("test", "corr_test", "granger", "ttest", "adf", "kpss")):
            scope = "hypothesis"
        else:
            scope = "aggregate"
    elif category in ("math", "elementwise_math", "data_handling"):
        scope = "elementwise"

    pit_safe = "future" not in tags
    if name in ("lead", "next", "shuffle"):
        pit_safe = True  # 已禁用或返回 NaN

    if scope == "ts" and any(k in name for k in ("mean", "std", "sum", "corr", "rank", "decay")):
        lookback: int | None = None  # 运行时由 window 参数决定
        min_periods = 1
    elif scope == "cs":
        lookback = 0
        min_periods = 1
    elif scope == "aggregate":
        lookback = None  # expanding
        min_periods = 1
    else:
        lookback = None
        min_periods = None

    return OperatorPolicy(
        scope=scope,
        lookback_window=lookback,
        min_periods=min_periods,
        pit_safe=pit_safe,
        tags=list(getattr(meta, "tags", None) or []),
    )


def compute_operator_catalog_hash(*, backend: str = "pandas_numpy") -> str:
    """对所有已实现算子的 canonical + policy 做确定性 hash，用于 lineage。"""
    from cleaned_operators.registry import OperatorRegistry

    entries: list[dict[str, Any]] = []
    for canon in sorted(OperatorRegistry._operators.keys()):
        op = OperatorRegistry.get(canon, backend=backend)
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canon)
        entries.append(
            {
                "canonical": canon,
                "category": getattr(op.metadata, "category", ""),
                "policy": policy.to_dict(),
            }
        )
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_lookback(analysis_lookback: int, *, lag_buffer: int = 1, extra: int = 5) -> int:
    """将 IR 分析 lookback 转为数据加载历史缓冲 bar 数。"""
    base = max(0, int(analysis_lookback))
    return base + max(0, int(lag_buffer)) + max(0, int(extra))
