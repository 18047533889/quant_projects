# -*- coding: utf-8
"""DuckDB 版本与窗口函数能力探测（部署前降级 SQL tier）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CapabilityLevel = str  # supported | unsupported | requires_version


@dataclass
class DuckdbCapabilityReport:
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
    ("median_window", "SELECT median(x) OVER (ORDER BY x) FROM (VALUES (1),(2),(3)) t(x)"),
    ("quantile_cont_window", "SELECT quantile_cont(x, 0.5) OVER () FROM (VALUES (1.0),(2.0)) t(x)"),
    ("corr_window", "SELECT corr(x, y) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM (VALUES (1,2),(2,3)) t(x,y)"),
    ("cov_window", "SELECT covar_samp(x, y) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM (VALUES (1,2),(2,3)) t(x,y)"),
    ("stddev_samp_window", "SELECT stddev_samp(x) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM (VALUES (1.0),(2.0)) t(x)"),
    ("last_value_ignore_nulls", "SELECT last_value(x IGNORE NULLS) OVER (ORDER BY x) FROM (VALUES (1),(NULL),(2)) t(x)"),
)


def probe_duckdb_capabilities(con=None) -> DuckdbCapabilityReport:
    """探测 DuckDB 连接；``con=None`` 时用内存连接。"""
    report = DuckdbCapabilityReport()
    owns = False
    try:
        import duckdb
    except ImportError:
        report.errors.append("duckdb not installed")
        for name, _ in _FEATURE_PROBES:
            report.features[name] = "unsupported"
        return report

    if con is None:
        con = duckdb.connect(":memory:")
        owns = True
    try:
        row = con.execute("SELECT version()").fetchone()
        report.version = str(row[0]) if row else ""
    except Exception as exc:
        report.errors.append(f"version probe failed: {exc}")

    for name, sql in _FEATURE_PROBES:
        try:
            con.execute(sql).fetchone()
            report.features[name] = "supported"
        except Exception as exc:
            report.features[name] = "unsupported"
            report.errors.append(f"{name}: {exc}")

    if owns:
        con.close()
    return report


def downgrade_sql_canonicals(report: DuckdbCapabilityReport) -> frozenset[str]:
    """根据探测结果返回应从 SQL production_safe 临时降级的 canonical。"""
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    downgrade: set[str] = set()
    if not report.is_supported("corr_window"):
        downgrade.update({"ts_corr", "ts_cov", "ts_beta", "rolling_beta", "ewm_corr", "ewm_cov"})
    if not report.is_supported("quantile_cont_window"):
        downgrade.update(
            {
                "cs_quantile",
                "c_percentile",
                "ts_quantile",
                "group_percentile",
                "winsorize",
                "group_winsorize",
            }
        )
    if not report.is_supported("median_window"):
        downgrade.update({"ts_median", "cs_mad", "cs_mad_zscore"})
    if not report.is_supported("last_value_ignore_nulls"):
        downgrade.add("ffill")
    return frozenset(c for c in downgrade if c in SQL_PRODUCTION_SAFE_CANONICALS)


_CACHED_REPORT: DuckdbCapabilityReport | None = None


def get_duckdb_capability_report(*, refresh: bool = False) -> DuckdbCapabilityReport:
    global _CACHED_REPORT
    if _CACHED_REPORT is None or refresh:
        _CACHED_REPORT = probe_duckdb_capabilities()
    return _CACHED_REPORT
