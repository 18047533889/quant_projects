# -*- coding: utf-8 -*-
"""SLO 告警规则：基于 pipeline metrics 的阈值检查。"""

from __future__ import annotations

from typing import Any


DEFAULT_SLO_RULES: list[dict[str, Any]] = [
    {
        "name": "pipeline_success_rate",
        "metric": "pipeline_success_ratio",
        "op": ">=",
        "threshold": 0.99,
        "severity": "critical",
    },
    {
        "name": "dq_pass_rate",
        "metric": "dq_passed_ratio",
        "op": ">=",
        "threshold": 0.995,
        "severity": "warning",
    },
    {
        "name": "dual_write_open_failures",
        "metric": "dual_write_open_failures",
        "op": "==",
        "threshold": 0,
        "severity": "critical",
    },
    {
        "name": "materialize_p99_latency_sec",
        "metric": "materialize_p99_sec",
        "op": "<=",
        "threshold": 600,
        "severity": "warning",
    },
]


def evaluate_slo_rules(
    metrics: dict[str, float | int],
    *,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """评估 SLO 规则，返回 violations 列表。"""
    active = rules or DEFAULT_SLO_RULES
    violations: list[dict[str, Any]] = []
    for rule in active:
        key = str(rule["metric"])
        if key not in metrics:
            continue
        value = float(metrics[key])
        threshold = float(rule["threshold"])
        op = str(rule["op"])
        ok = (
            (op == ">=" and value >= threshold)
            or (op == "<=" and value <= threshold)
            or (op == "==" and value == threshold)
            or (op == ">" and value > threshold)
            or (op == "<" and value < threshold)
        )
        if not ok:
            violations.append(
                {
                    "name": rule["name"],
                    "metric": key,
                    "value": value,
                    "threshold": threshold,
                    "op": op,
                    "severity": rule.get("severity", "warning"),
                }
            )
    return {"ok": not violations, "violations": violations}
