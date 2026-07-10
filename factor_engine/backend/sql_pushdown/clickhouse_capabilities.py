# -*- coding: utf-8
"""ClickHouse 窗口函数能力探测（部署前降级 SQL tier）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.sql_pushdown.duckdb_capabilities import CapabilityLevel


@dataclass
class ClickhouseCapabilityReport:
    version: str = ""
    features: dict[str, CapabilityLevel] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def is_supported(self, feature: str) -> bool:
        return self.features.get(feature) == "supported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "features": dict(self.features),
            "errors": list(self.errors),
        }


_FEATURE_PROBES: tuple[tuple[str, str], ...] = (
    ("median_window", "SELECT median(x) OVER (ORDER BY x) FROM (SELECT arrayJoin([1,2,3]) AS x)"),
    (
        "quantile_cont_window",
        "SELECT quantile(0.5)(x) OVER () FROM (SELECT arrayJoin([1.0,2.0]) AS x)",
    ),
    (
        "corr_window",
        "SELECT corr(x, y) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
        "FROM (SELECT arrayJoin([1,2]) AS x, arrayJoin([2,3]) AS y)",
    ),
    (
        "stddev_samp_window",
        "SELECT stddevSamp(x) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
        "FROM (SELECT arrayJoin([1.0,2.0]) AS x)",
    ),
)


def probe_clickhouse_capabilities(client=None) -> ClickhouseCapabilityReport:
    """探测 ClickHouse；无 client 时标记为 unknown（不阻断静态 tier）。"""
    report = ClickhouseCapabilityReport()
    if client is None:
        for name, _ in _FEATURE_PROBES:
            report.features[name] = "unsupported"
        report.errors.append("clickhouse client not provided")
        return report

    try:
        row = client.query("SELECT version()").first_row
        report.version = str(row[0]) if row else ""
    except Exception as exc:
        report.errors.append(f"version probe failed: {exc}")

    for name, sql in _FEATURE_PROBES:
        try:
            client.query(sql)
            report.features[name] = "supported"
        except Exception as exc:
            report.features[name] = "unsupported"
            report.errors.append(f"{name}: {exc}")
    return report


def downgrade_clickhouse_canonicals(report: ClickhouseCapabilityReport) -> frozenset[str]:
    """根据 ClickHouse 探测结果降级 production SQL canonical。"""
    from backend.sql_tiers import CLICKHOUSE_SQL_PRODUCTION_SAFE

    downgrade: set[str] = set()
    if not report.is_supported("corr_window"):
        downgrade.update({"ts_corr", "ts_cov", "ts_beta", "rolling_beta"})
    if not report.is_supported("quantile_cont_window"):
        downgrade.update(
            {
                "cs_quantile",
                "c_percentile",
                "group_percentile",
                "winsorize",
                "group_winsorize",
            }
        )
    if not report.is_supported("median_window"):
        downgrade.update({"ts_median", "cs_mad", "cs_mad_zscore"})
    return frozenset(c for c in downgrade if c in CLICKHOUSE_SQL_PRODUCTION_SAFE)


_CACHED_REPORT: ClickhouseCapabilityReport | None = None


def get_clickhouse_capability_report(*, refresh: bool = False) -> ClickhouseCapabilityReport:
    global _CACHED_REPORT
    if _CACHED_REPORT is None or refresh:
        _CACHED_REPORT = probe_clickhouse_capabilities()
    return _CACHED_REPORT


def effective_clickhouse_production_safe(canon: str, *, refresh: bool = False) -> bool:
    from backend.sql_tiers import CLICKHOUSE_SQL_PRODUCTION_SAFE
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name not in CLICKHOUSE_SQL_PRODUCTION_SAFE:
        return False
    return name not in downgrade_clickhouse_canonicals(get_clickhouse_capability_report(refresh=refresh))
