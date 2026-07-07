#!/usr/bin/env python3
"""将 gtja191 目录中的因子公式计算并落盘到 factor lake。

默认落盘 185 条可投递因子（跳过 6 条 benchmark/zero stub）。
不修改 gtja191/ 内任何文件。
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
FE_ROOT = QUANT_ROOT / "factor_engine"
GTJA_ROOT = QUANT_ROOT / "gtja191"

if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.dsl_parser import parse_factor  # noqa: E402
from api.factor import Factor  # noqa: E402
from api.mining_integration import default_ashare_pv_data_source_config  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from logging_utils import configure_logging, get_logger  # noqa: E402
from runtime.config import DataSourceConfig  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from runtime.perf_config import PerfConfig  # noqa: E402
from storage.cache import CacheManager  # noqa: E402
from storage.factory import build_data_source  # noqa: E402
from storage.materializer import ParquetMaterializer  # noqa: E402
from workspace_paths import default_factor_lake_root, quant_projects_root  # noqa: E402

logger = get_logger("scripts.materialize_gtja191_factors")

# 目录 DSL 转换错误时的落盘修正（不修改 gtja191/ 内文件）
_FORMULA_OVERRIDES: dict[str, str] = {
    # 原式把 power(13) 误嵌进 ts_corr 窗口位，应为 corr(...,13)^5
    "gtja191_alpha_056": (
        "(rank((open - ts_min(open, 12))) < rank((power(rank(ts_corr("
        "ts_sum(((high + low) / 2), 19), ts_sum(ts_mean(volume,40), 19), 13)), 5))))"
    ),
}

# 全 A 股 panel 落盘内存预算（30G 机器、无 swap 时偏保守）
_DEFAULT_RESERVE_GB = 6.0   # OS + IDE + 其他进程
_DEFAULT_BASE_GB = 3.0      # OHLCV 列缓存 + DuckDB 读缓冲
_DEFAULT_PER_FACTOR_GB = 5.0  # 单条复杂因子计算峰值


def _read_meminfo_kb(key: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def read_mem_available_gb() -> float:
    kb = _read_meminfo_kb("MemAvailable:")
    if kb is None:
        kb = _read_meminfo_kb("MemFree:")
    return (kb or 8 * 1024 * 1024) / (1024**2)


def read_mem_total_gb() -> float:
    kb = _read_meminfo_kb("MemTotal:")
    return (kb or 30 * 1024 * 1024) / (1024**2)


def recommend_batch_size(
    *,
    reserve_gb: float = _DEFAULT_RESERVE_GB,
    base_gb: float = _DEFAULT_BASE_GB,
    per_factor_gb: float = _DEFAULT_PER_FACTOR_GB,
) -> int:
    """按可用内存推荐 ``run_many`` 批大小（无 swap 时宁可小批、多批）。"""
    total_gb = read_mem_total_gb()
    avail_gb = read_mem_available_gb()
    budget = avail_gb - reserve_gb - base_gb
    if budget <= 0:
        return 1
    size = int(budget // per_factor_gb)
    if total_gb <= 32:
        cap = 2
    elif total_gb <= 48:
        cap = 3
    else:
        cap = 4
    chosen = max(1, min(size, cap))
    logger.info(
        "内存预算: total=%.1fG avail=%.1fG reserve=%.1fG base=%.1fG "
        "per_factor=%.1fG -> batch_size=%d (cap=%d)",
        total_gb,
        avail_gb,
        reserve_gb,
        base_gb,
        per_factor_gb,
        chosen,
        cap,
    )
    return chosen


def release_engine(engine: FactorEngine | None) -> None:
    if engine is None:
        return
    ds = getattr(engine, "data_source", None)
    if ds is not None:
        for attr in ("_column_cache", "_panel_cache"):
            cache = getattr(ds, attr, None)
            if isinstance(cache, dict):
                cache.clear()
    del engine
    gc.collect()
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def _data_snapshot_id() -> str | None:
    """落盘 lineage：绑定当前 ashare 数据源配置 hash。"""
    try:
        from runtime.lineage import hash_data_source_config

        cfg = default_ashare_pv_data_source_config()
        return hash_data_source_config(cfg)
    except Exception:
        return None


def _prepare_factor_dir(lake_root: Path, factor_id: str, *, force: bool) -> None:
    """算子/引擎变更后全量重算：先删旧分区，避免 upsert 残留 stale 行。"""
    if not force:
        return
    factor_dir = lake_root / "factors" / factor_id
    if factor_dir.is_dir():
        shutil.rmtree(factor_dir)
        logger.info("force: 已删除旧落盘 %s", factor_dir)


def load_catalog(*, include_excluded: bool) -> list[dict]:
    catalog_path = GTJA_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = list(payload.values())
    items.sort(key=lambda x: x["factor_name"])
    if not include_excluded:
        items = [x for x in items if not x.get("delivery_excluded")]
    return items


def build_engine(*, max_files: int | None, start_date: str | None, end_date: str | None) -> FactorEngine:
    ds_cfg = default_ashare_pv_data_source_config(
        max_files=max_files,
        start_date=start_date,
        end_date=end_date,
    )
    ds_type = str(ds_cfg.pop("type", "parquet"))
    data_source = build_data_source(DataSourceConfig(type=ds_type, options=ds_cfg))
    if hasattr(data_source, "load_columns"):
        data_source.load_columns(
            ["open", "high", "low", "close", "volume", "vwap", "amount", "ret", "preclose"]
        )
    backend = build_backend("pandas")
    return FactorEngine(backend=backend, data_source=data_source, cache=CacheManager())


def _chunked(items: list[dict], size: int) -> list[list[dict]]:
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def factor_partition_count(lake_root: Path, factor_id: str) -> int:
    factor_dir = lake_root / "factors" / factor_id
    if not factor_dir.is_dir():
        return 0
    return sum(1 for p in factor_dir.iterdir() if p.is_dir() and p.name.startswith("year="))


def is_factor_complete(lake_root: Path, factor_id: str, *, min_partitions: int = 10) -> bool:
    """全量 A 股日频约 11 个年分区；不足视为未完成或仅 smoke test。"""
    return factor_partition_count(lake_root, factor_id) >= min_partitions


def materialize_gtja191(
    *,
    lake_root: Path,
    author: str,
    max_files: int | None,
    start_date: str | None,
    end_date: str | None,
    factor_names: list[str] | None,
    include_excluded: bool,
    stop_on_error: bool,
    batch_size: int,
    parallel: bool,
    n_jobs: int | None,
    skip_existing: bool,
    min_partitions: int,
    force: bool,
) -> dict:
    catalog = load_catalog(include_excluded=include_excluded)
    if factor_names:
        wanted = set(factor_names)
        catalog = [item for item in catalog if item["factor_name"] in wanted]
        if not catalog:
            raise ValueError(f"未匹配到因子: {factor_names}")

    skipped: list[str] = []
    if skip_existing:
        pending: list[dict] = []
        for item in catalog:
            fid = item["factor_name"]
            if is_factor_complete(lake_root, fid, min_partitions=min_partitions):
                skipped.append(fid)
            else:
                pending.append(item)
        catalog = pending
        if skipped:
            logger.info("跳过已完整落盘 %d 条: %s ...", len(skipped), ", ".join(skipped[:5]))

    summary: dict = {
        "lake_root": str(lake_root),
        "factor_count": len(catalog),
        "skipped_existing": skipped,
        "succeeded": [],
        "failed": [],
        "skipped_excluded": 6 if not include_excluded else 0,
        "batch_size": batch_size,
        "parallel": parallel,
    }

    if not catalog:
        logger.info("无需落盘：全部因子已完整存在")
        summary["succeeded_count"] = 0
        summary["failed_count"] = 0
        summary["skipped_existing_count"] = len(skipped)
        return summary

    materializer = ParquetMaterializer(lake_root=lake_root)
    perf = PerfConfig.from_env()
    snapshot_id = _data_snapshot_id()

    logger.info(
        "开始落盘 %d 个 GTJA191 因子 -> %s (batch_size=%d, parallel=%s, force=%s, "
        "mem_total=%.1fG, mem_avail=%.1fG, snapshot=%s)",
        len(catalog),
        lake_root,
        batch_size,
        parallel,
        force,
        read_mem_total_gb(),
        read_mem_available_gb(),
        snapshot_id,
    )

    batches = _chunked(catalog, batch_size)
    done = 0
    for batch_idx, batch in enumerate(batches, start=1):
        logger.info(
            "批次 %d/%d: %d 条因子 (mem_avail=%.1fG)",
            batch_idx,
            len(batches),
            len(batch),
            read_mem_available_gb(),
        )
        engine = build_engine(max_files=max_files, start_date=start_date, end_date=end_date)
        factors: list[Factor] = []
        meta: dict[str, dict] = {}
        for item in batch:
            fid = item["factor_name"]
            formula = _FORMULA_OVERRIDES.get(fid, item["dsl_formula"].strip())
            desc = f"GTJA191 {fid} ({item.get('conversion', '')})"
            factors.append(
                parse_factor(
                    formula,
                    name=fid,
                    freq="1d",
                    universe="A_SHARE_ALL_A_EX_ST",
                    description=desc,
                )
            )
            meta[fid] = {"formula": formula, "description": desc}

        t_batch = time.time()
        try:
            if parallel and len(factors) > 1:
                out = engine.run_many_parallel(factors, perf=perf, n_jobs=n_jobs)
            else:
                # 统一走 run_many：启用 CSE + panel_native（PerfConfig）
                out = engine.run_many(factors, perf=perf)
        except Exception as exc:
            if stop_on_error:
                release_engine(engine)
                raise
            for item in batch:
                failure = {
                    "factor_id": item["factor_name"],
                    "error": str(exc),
                    "elapsed_sec": round(time.time() - t_batch, 2),
                }
                summary["failed"].append(failure)
                logger.error("  BATCH FAIL: %s", exc)
            release_engine(engine)
            continue

        analyses = out["analyses"]
        results = out["results"]
        for fid, result in results.items():
            done += 1
            info = meta[fid]
            t0 = time.time()
            try:
                _prepare_factor_dir(lake_root, fid, force=force)
                mat_summary = materializer.materialize(
                    factor_id=fid,
                    result=result,
                    ir_node=analyses[fid].ir,
                    author=author,
                    frequency="1d",
                    description=info["description"],
                    expression=info["formula"],
                    write_metadata=True,
                    data_snapshot_id=snapshot_id,
                )
                elapsed = time.time() - t0
                row = {
                    "factor_id": fid,
                    "rows_written": mat_summary["rows_written"],
                    "partitions": mat_summary["partitions"],
                    "elapsed_sec": round(elapsed, 2),
                }
                summary["succeeded"].append(row)
                logger.info(
                    "[%d/%d] OK %s: %d rows, %s partitions, write %.1fs",
                    done,
                    len(catalog),
                    fid,
                    mat_summary["rows_written"],
                    mat_summary["partitions"],
                    elapsed,
                )
            except Exception as exc:
                failure = {
                    "factor_id": fid,
                    "error": str(exc),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                summary["failed"].append(failure)
                logger.error("[%d/%d] FAIL %s: %s", done, len(catalog), fid, exc)
                logger.debug(traceback.format_exc())
                if stop_on_error:
                    release_engine(engine)
                    raise

        release_engine(engine)

    summary["succeeded_count"] = len(summary["succeeded"])
    summary["failed_count"] = len(summary["failed"])
    summary["skipped_existing_count"] = len(skipped)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize gtja191 factors to factor lake")
    parser.add_argument(
        "--lake-root",
        type=Path,
        default=default_factor_lake_root(),
        help="Factor lake root (default: ~/quant_projects/data/factors/lake)",
    )
    parser.add_argument("--author", default="shw", help="Author tag for catalog")
    parser.add_argument("--max-files", type=int, default=None, help="Limit parquet files (smoke test)")
    parser.add_argument("--start-date", default=None, help="Filter start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Filter end date YYYY-MM-DD")
    parser.add_argument(
        "--factor",
        action="append",
        dest="factor_names",
        help="Only run e.g. gtja191_alpha_014",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include 6 delivery-excluded factors (benchmark/zero stub)",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="每批因子数；0=按本机内存自动（30G 机器默认约 2）",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use run_many_parallel within each batch (needs joblib)",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Workers for --parallel")
    parser.add_argument(
        "--force",
        action="store_true",
        help="算子/引擎变更后全量重算：落盘前删除该因子旧目录（避免 upsert 残留）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip factors already materialized with >= min-partitions year dirs",
    )
    parser.add_argument(
        "--min-partitions",
        type=int,
        default=10,
        help="Min year= partitions to treat factor as complete (default 10)",
    )
    parser.add_argument(
        "--memory-reserve-gb",
        type=float,
        default=_DEFAULT_RESERVE_GB,
        help="为系统/IDE 预留内存 (GB)，自动 batch_size 时使用",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=quant_projects_root() / "logs" / "materialize_gtja191.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_file=args.log_file)

    data_root = quant_projects_root() / "data" / "a_share" / "lqtp_data" / "StockDailyBar"
    if not data_root.is_dir() or not any(data_root.glob("*.parquet")):
        logger.error("本地数据不存在: %s — 请先 sync COS StockDailyBar", data_root)
        return 1

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = recommend_batch_size(reserve_gb=args.memory_reserve_gb)

    summary = materialize_gtja191(
        lake_root=args.lake_root,
        author=args.author,
        max_files=args.max_files,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_names=args.factor_names,
        include_excluded=args.include_excluded,
        stop_on_error=args.stop_on_error,
        batch_size=batch_size,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        skip_existing=args.skip_existing and not args.force,
        min_partitions=args.min_partitions,
        force=args.force,
    )

    out_path = args.lake_root / "_runs" / f"gtja191_materialize_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nrun summary -> {out_path}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
