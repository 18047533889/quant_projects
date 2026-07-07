#!/usr/bin/env python3
"""校验 GTJA191 factor lake 落盘值：结构审计 + 重算对比。

用法:
  source ~/quant_projects/env.sh
  python3 scripts/verify_gtja191_factor_lake.py
  python3 scripts/verify_gtja191_factor_lake.py --sample-size 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

QUANT_ROOT = Path(__file__).resolve().parents[1]
FE_ROOT = QUANT_ROOT / "factor_engine"
GTJA_ROOT = QUANT_ROOT / "gtja191"
LAKE_ROOT = QUANT_ROOT / "data" / "factors" / "lake"

if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.dsl_parser import parse_factor  # noqa: E402
from scripts.materialize_gtja191_factors import (  # noqa: E402
    _FORMULA_OVERRIDES,
    build_engine,
    release_engine,
)
from storage.materializer import ParquetMaterializer  # noqa: E402
from workspace_paths import default_factor_lake_root  # noqa: E402


@dataclass
class AuditReport:
    structural_issues: list[str] = field(default_factory=list)
    recompute_mismatches: list[str] = field(default_factory=list)
    recompute_checked: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)


def load_catalog() -> dict[str, dict]:
    path = GTJA_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: v for k, v in payload.items()
        if not v.get("delivery_excluded")
    }


def factor_formula(fid: str, item: dict) -> str:
    return _FORMULA_OVERRIDES.get(fid, item["dsl_formula"].strip())


def load_lake_long(
    lake_root: Path,
    factor_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    factor_dir = lake_root / "factors" / factor_id
    if not factor_dir.is_dir():
        raise FileNotFoundError(factor_dir)
    frames: list[pd.DataFrame] = []
    for ydir in sorted(factor_dir.glob("year=*")):
        pq_path = ydir / "data.parquet"
        if not pq_path.is_file():
            continue
        df = pd.read_parquet(pq_path, columns=["datetime", "asset", "value"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["datetime", "asset", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    if start is not None:
        out = out[out["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["datetime"] <= pd.Timestamp(end)]
    return out


def audit_structure(catalog: dict[str, dict], lake_root: Path) -> tuple[list[str], dict]:
    issues: list[str] = []
    row_counts: list[int] = []
    year_counts: list[int] = []
    min_dates: list[pd.Timestamp] = []
    max_dates: list[pd.Timestamp] = []

    for fid in sorted(catalog):
        factor_dir = lake_root / "factors" / fid
        if not factor_dir.is_dir():
            issues.append(f"{fid}: 目录不存在")
            continue
        year_dirs = sorted(p for p in factor_dir.iterdir() if p.is_dir() and p.name.startswith("year="))
        if len(year_dirs) < 10:
            issues.append(f"{fid}: 年分区不足 ({len(year_dirs)})")
        total = 0
        f_min = None
        f_max = None
        for ydir in year_dirs:
            pq_path = ydir / "data.parquet"
            if not pq_path.is_file():
                issues.append(f"{fid}: 缺少 {ydir.name}/data.parquet")
                continue
            meta = pq.read_metadata(pq_path)
            total += meta.num_rows
            if meta.num_rows == 0:
                issues.append(f"{fid}: {ydir.name} 空分区")
                continue
            # 抽样读首尾时间 + 重复键检查
            table = pq.read_table(pq_path, columns=["datetime", "asset", "value"])
            df = table.to_pandas()
            df["datetime"] = pd.to_datetime(df["datetime"])
            dup = df.duplicated(subset=["datetime", "asset"]).sum()
            if dup:
                issues.append(f"{fid}: {ydir.name} 存在 {dup} 条重复 (datetime, asset)")
            if df["value"].dtype != np.float32:
                issues.append(f"{fid}: {ydir.name} value 类型非 float32 ({df['value'].dtype})")
            if not np.isfinite(df["value"].to_numpy()).all():
                issues.append(f"{fid}: {ydir.name} 含 inf（落盘应已清洗）")
            ts_min = df["datetime"].min()
            ts_max = df["datetime"].max()
            f_min = ts_min if f_min is None else min(f_min, ts_min)
            f_max = ts_max if f_max is None else max(f_max, ts_max)
        if total == 0:
            issues.append(f"{fid}: 总行数为 0")
        else:
            row_counts.append(total)
            year_counts.append(len(year_dirs))
            if f_min is not None:
                min_dates.append(f_min)
            if f_max is not None:
                max_dates.append(f_max)

    missing_in_lake = set(catalog) - {
        p.name for p in (lake_root / "factors").iterdir() if p.is_dir()
    }
    for fid in sorted(missing_in_lake):
        issues.append(f"{fid}: catalog 有但未落盘")

    stats = {
        "factor_count": len(catalog),
        "row_count_min": min(row_counts) if row_counts else 0,
        "row_count_max": max(row_counts) if row_counts else 0,
        "row_count_median": int(np.median(row_counts)) if row_counts else 0,
        "date_min": str(min(min_dates)) if min_dates else None,
        "date_max": str(max(max_dates)) if max_dates else None,
    }
    return issues, stats


def series_to_long(series: pd.Series) -> pd.DataFrame:
    df = series.reset_index()
    df.columns = ["datetime", "asset", "value"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["asset"] = df["asset"].astype("string")
    df["value"] = df["value"].astype("float32")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    return df


def compare_factor(
    fid: str,
    formula: str,
    *,
    lake_root: Path,
    start: str,
    end: str,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> tuple[bool, str, dict]:
    """重算必须用全量历史（与落盘一致），再截取对比窗口。"""
    engine = build_engine(max_files=None, start_date=None, end_date=None)
    try:
        factor = parse_factor(formula, name=fid)
        run_out = engine.run(factor)
        fresh = series_to_long(run_out["result"])
    finally:
        release_engine(engine)

    win_start = pd.Timestamp(start)
    win_end = pd.Timestamp(end)
    fresh = fresh[(fresh["datetime"] >= win_start) & (fresh["datetime"] <= win_end)]
    lake = load_lake_long(lake_root, fid, start=start, end=end)

    meta = {
        "fresh_rows": len(fresh),
        "lake_rows": len(lake),
        "inner_rows": 0,
        "lake_only_rows": 0,
        "fresh_only_rows": 0,
        "bad_rows": 0,
        "max_diff": 0.0,
    }
    if fresh.empty and lake.empty:
        return True, "both empty in window", meta
    if fresh.empty or lake.empty:
        meta["lake_only_rows"] = len(lake)
        meta["fresh_only_rows"] = len(fresh)
        return False, f"empty mismatch fresh={len(fresh)} lake={len(lake)}", meta

    merged = fresh.merge(
        lake,
        on=["datetime", "asset"],
        how="inner",
        suffixes=("_fresh", "_lake"),
    )
    meta["inner_rows"] = len(merged)
    meta["fresh_only_rows"] = len(fresh) - len(merged)
    meta["lake_only_rows"] = len(lake) - len(merged)

    if merged.empty:
        return False, "inner join 0 行（时间/标的键不对齐）", meta

    diff = (merged["value_fresh"].astype("float64") - merged["value_lake"].astype("float64")).abs()
    meta["max_diff"] = float(diff.max())
    tol = atol + rtol * merged["value_lake"].astype("float64").abs()
    bad = diff > tol
    meta["bad_rows"] = int(bad.sum())

    issues: list[str] = []
    if meta["fresh_only_rows"] > 0:
        issues.append(f"fresh_only={meta['fresh_only_rows']}")
    if meta["lake_only_rows"] > 0:
        issues.append(f"lake_only={meta['lake_only_rows']}（多为旧 smoke 残留，fresh 为 NaN 未覆盖）")
    if meta["bad_rows"] > 0:
        pct = 100.0 * meta["bad_rows"] / len(merged)
        issues.append(
            f"数值偏差 bad={meta['bad_rows']}/{len(merged)} ({pct:.4f}%) "
            f"max_diff={meta['max_diff']:.6g}"
        )

    if issues:
        return False, "; ".join(issues), meta
    extra = f", lake_only={meta['lake_only_rows']}" if meta["lake_only_rows"] else ""
    return True, f"match n={len(merged)} max_diff={meta['max_diff']:.6g}{extra}", meta


def pick_sample_ids(catalog: dict[str, dict], size: int, seed: int) -> list[str]:
    all_ids = sorted(catalog)
    must = [
        "gtja191_alpha_001",
        "gtja191_alpha_014",
        "gtja191_alpha_056",  # override 公式
        "gtja191_alpha_063",  # flex_max 修复
        "gtja191_alpha_070",  # amount
        "gtja191_alpha_110",  # flex_max
        "gtja191_alpha_190",  # max(..., 1e-8)
    ]
    must = [x for x in must if x in catalog]
    rng = random.Random(seed)
    rest = [x for x in all_ids if x not in must]
    extra = rng.sample(rest, k=min(size - len(must), len(rest)))
    return must + extra


def audit_catalog_vs_sqlite(catalog: dict[str, dict], lake_root: Path) -> list[str]:
    issues: list[str] = []
    mat = ParquetMaterializer(lake_root=lake_root)
    listed = {f["factor_id"]: f for f in mat.list_factors()}
    for fid in sorted(catalog):
        if fid not in listed:
            issues.append(f"{fid}: 未在 _catalog.sqlite 注册")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GTJA191 factor lake values")
    parser.add_argument("--lake-root", type=Path, default=default_factor_lake_root())
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window",
        default="2024-01-01:2024-03-31",
        help="重算对比时间窗 start:end",
    )
    args = parser.parse_args()
    start, end = args.window.split(":", 1)

    catalog = load_catalog()
    report = AuditReport()

    struct_issues, stats = audit_structure(catalog, args.lake_root)
    report.structural_issues.extend(struct_issues)
    report.stats.update(stats)

    catalog_issues = audit_catalog_vs_sqlite(catalog, args.lake_root)
    report.structural_issues.extend(catalog_issues)

    sample_ids = pick_sample_ids(catalog, args.sample_size, args.seed)
    lake_only_total = 0
    numeric_failures: list[str] = []
    for fid in sample_ids:
        formula = factor_formula(fid, catalog[fid])
        ok, msg, meta = compare_factor(
            fid,
            formula,
            lake_root=args.lake_root,
            start=start,
            end=end,
        )
        report.recompute_checked.append(fid)
        lake_only_total += int(meta.get("lake_only_rows", 0))
        if not ok:
            if meta.get("bad_rows", 0) > 0 or meta.get("fresh_only_rows", 0) > 0:
                numeric_failures.append(f"{fid}: {msg}")
            report.recompute_mismatches.append(f"{fid}: {msg}")
        else:
            print(f"[OK] {fid}: {msg}")

    report.stats["recompute_lake_only_rows_in_window"] = lake_only_total

    # 全量再算一遍：2024Q1 窗口内各因子行数应与 fresh 一致（抽 3 条快速扫）
    print("\n=== 结构审计 ===")
    print(json.dumps(report.stats, ensure_ascii=False, indent=2))
    if report.structural_issues:
        print(f"结构问题 {len(report.structural_issues)} 条:")
        for line in report.structural_issues[:30]:
            print(" ", line)
        if len(report.structural_issues) > 30:
            print(f"  ... 还有 {len(report.structural_issues) - 30} 条")
    else:
        print("结构审计: 通过")

    print(f"\n=== 重算对比 ({start} ~ {end}) ===")
    print(f"抽样 {len(report.recompute_checked)} 条")
    print(f"窗口内 lake-only 残留行合计: {lake_only_total}")
    if numeric_failures:
        print(f"数值/键 真正不一致: {len(numeric_failures)} 条")
        for line in numeric_failures:
            print(" [FAIL]", line)
    elif report.recompute_mismatches:
        print(f"仅 lake-only 残留: {len(report.recompute_mismatches)} 条（inner 数值已对齐）")
        for line in report.recompute_mismatches[:10]:
            print(" [WARN]", line)
    else:
        print("全部抽样因子 inner 数值对齐")

    out_path = args.lake_root / "_runs" / "gtja191_verify_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "structural_issues": report.structural_issues,
        "recompute_mismatches": report.recompute_mismatches,
        "numeric_failures": numeric_failures,
        "recompute_checked": report.recompute_checked,
        "stats": report.stats,
        "window": {"start": start, "end": end},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告 -> {out_path}")

    failed = bool(report.structural_issues or numeric_failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
