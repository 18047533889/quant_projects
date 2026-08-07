"""数据质量契约检查与 CLI（读侧走 DataAccess，不裸读 parquet）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import pyarrow as pa

from .contracts import QualityOptions, QualityReport, run_quality_checks


@dataclass(frozen=True)
class _LegacyReport:
    dataset: str
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    sampled: bool = False


def validate_table(
    dataset: str,
    table: pa.Table,
    *,
    required_columns: tuple[str, ...] = (),
    primary_key: tuple[str, ...] = (),
    quality_mode: str = "strict",
) -> QualityReport:
    """向后兼容入口：映射到 run_quality_checks。"""
    if quality_mode not in {"off", "warn", "strict"}:
        raise ValueError("quality_mode must be off, warn, or strict")
    report = run_quality_checks(
        dataset,
        table,
        options=QualityOptions(
            required_columns=tuple(required_columns),
            primary_key=tuple(primary_key),
        ),
    )
    if quality_mode == "off":
        return QualityReport(
            dataset=dataset, passed=True, checks=report.checks, failures=()
        )
    return report


def _parse_checks(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        for token in item.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a data_access dataset/URI quality contract (reads via DataAccess)"
    )
    parser.add_argument("path", help="dataset name or file URI (store.read_uri)")
    parser.add_argument("--dataset", default="uri", help="dataset label for report")
    parser.add_argument("--required-column", action="append", default=[])
    parser.add_argument("--primary-key", action="append", default=[])
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--instrument-column", default=None)
    parser.add_argument("--null-ratio-max", type=float, default=None)
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--declared-schema", default=None,
                        help='JSON {"col": "type"} schema alignment')
    parser.add_argument("--range", action="append", default=[],
                        help='col=min:max, e.g. "close=0.0:1000.0"')
    parser.add_argument("--finite", action="append", default=[],
                        help="columns that must be fully finite")
    parser.add_argument("--check", action="append", default=[],
                        help="extra checks: monotonic_time,duplicate_timestamp,future_timestamp,pit_leakage")
    parser.add_argument("--format", default="auto", help="file format override")
    parser.add_argument("--columns", default=None,
                        help="comma-separated columns to project")
    args = parser.parse_args(argv)

    # 读数据走 DataAccess（read_uri 或注册数据集）
    from data_access import get_store

    store = get_store()
    columns = None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    try:
        # 已注册数据集名优先
        if args.path in store.registry:
            handle = store.read(
                args.path, columns=columns, query_budget=__import__(
                    "data_access.read.query_budget", fromlist=["QueryBudget"]
                ).QueryBudget(max_rows=200_000_000)
            )
        else:
            handle = store.read_uri(
                args.path,
                columns=columns,
                format=args.format,
                time_column=args.time_column,
                instrument_column=args.instrument_column,
            )
    except Exception as exc:
        print(json.dumps({"dataset": args.path, "passed": False,
                          "checks": [], "failures": [f"read failed: {exc}"],
                          "details": {}}, ensure_ascii=False))
        return 1

    table = handle.to_arrow()

    declared_schema: dict[str, str] = {}
    if args.declared_schema:
        declared_schema = json.loads(args.declared_schema)

    ranges: dict[str, tuple[Any, Any]] = {}
    for spec in args.range:
        col, _, bounds = spec.partition("=")
        lo_s, _, hi_s = bounds.partition(":")
        ranges[col.strip()] = (
            float(lo_s) if lo_s.strip() else None,
            float(hi_s) if hi_s.strip() else None,
        )

    extra_checks = _parse_checks(args.check)
    options = QualityOptions(
        required_columns=tuple(args.required_column),
        primary_key=tuple(args.primary_key),
        declared_schema=declared_schema,
        null_ratio_max=args.null_ratio_max,
        range=ranges,
        finite_columns=tuple(args.finite),
        time_column=args.time_column,
        instrument_column=args.instrument_column,
        check_monotonic_time="monotonic_time" in extra_checks,
        check_duplicate_timestamp="duplicate_timestamp" in extra_checks,
        check_future_timestamp="future_timestamp" in extra_checks,
        min_rows=args.min_rows,
        check_pit_leakage="pit_leakage" in extra_checks,
    )
    report = run_quality_checks(args.dataset, table, options=options)
    print(json.dumps(report.to_dict(), ensure_ascii=False, default=str))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
