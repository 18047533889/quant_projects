#!/usr/bin/env python3
"""将 week2_pv_factors 目录中的因子公式计算并落盘到 factor lake。

不修改 week2_pv_factors/ 内任何文件；只读取 catalog 与公式定义。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
FE_ROOT = QUANT_ROOT / "factor_engine"
WEEK2_ROOT = QUANT_ROOT / "week2_pv_factors"

if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.dsl_parser import parse_factor  # noqa: E402
from api.factor import Factor  # noqa: E402
from api.mining_integration import default_ashare_pv_data_source_config  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from logging_utils import configure_logging, get_logger  # noqa: E402
from runtime.config import DataSourceConfig  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from storage.cache import CacheManager  # noqa: E402
from storage.factory import build_data_source  # noqa: E402
from storage.materializer import ParquetMaterializer  # noqa: E402
from workspace_paths import default_factor_lake_root, quant_projects_root  # noqa: E402

logger = get_logger("scripts.materialize_week2_factors")

CAMPAIGN_TAG = "ashare_week2_pv"


def load_catalog() -> list[dict]:
    catalog_path = WEEK2_ROOT / "source" / "week2_factors_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    return list(payload["factors"])


def build_engine(*, max_files: int | None, start_date: str | None, end_date: str | None) -> FactorEngine:
    ds_cfg = default_ashare_pv_data_source_config(
        max_files=max_files,
        start_date=start_date,
        end_date=end_date,
    )
    ds_type = str(ds_cfg.pop("type", "parquet"))
    data_source = build_data_source(DataSourceConfig(type=ds_type, options=ds_cfg))
    if hasattr(data_source, "load_columns"):
        data_source.load_columns(["open", "high", "low", "close", "volume", "vwap"])
    backend = build_backend("pandas")
    return FactorEngine(backend=backend, data_source=data_source, cache=CacheManager())


def factor_id_for(item: dict) -> str:
    return f"{CAMPAIGN_TAG}_{item['id']}"


def materialize_week2(
    *,
    lake_root: Path,
    author: str,
    max_files: int | None,
    start_date: str | None,
    end_date: str | None,
    factor_ids: list[str] | None,
    stop_on_error: bool,
) -> dict:
    catalog = load_catalog()
    if factor_ids:
        wanted = set(factor_ids)
        catalog = [item for item in catalog if item["id"] in wanted]
        if not catalog:
            raise ValueError(f"未匹配到因子: {factor_ids}")

    engine = build_engine(max_files=max_files, start_date=start_date, end_date=end_date)
    materializer = ParquetMaterializer(lake_root=lake_root)

    summary: dict = {
        "lake_root": str(lake_root),
        "factor_count": len(catalog),
        "succeeded": [],
        "failed": [],
    }

    logger.info("开始落盘 %d 个 Week2 因子 -> %s", len(catalog), lake_root)

    for idx, item in enumerate(catalog, start=1):
        fid = factor_id_for(item)
        formula = item["formula"].strip()
        desc = f"{item['id']} {item['name_zh']} ({item.get('cluster', '')})"
        logger.info("[%d/%d] %s", idx, len(catalog), fid)
        t0 = time.time()
        try:
            factor = parse_factor(
                formula,
                name=fid,
                freq="1d",
                universe="A_SHARE_ALL_A_EX_ST",
                description=desc,
            )
            output = engine.run(factor)
            mat_summary = materializer.materialize(
                factor_id=fid,
                result=output["result"],
                ir_node=output["analysis"].ir,
                author=author,
                frequency="1d",
                description=desc,
                expression=formula,
            )
            elapsed = time.time() - t0
            row = {
                "factor_id": fid,
                "catalog_id": item["id"],
                "rows_written": mat_summary["rows_written"],
                "partitions": mat_summary["partitions"],
                "elapsed_sec": round(elapsed, 2),
            }
            summary["succeeded"].append(row)
            logger.info(
                "  OK %s: %d rows, %s partitions, %.1fs",
                fid,
                mat_summary["rows_written"],
                mat_summary["partitions"],
                elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            failure = {
                "factor_id": fid,
                "catalog_id": item["id"],
                "error": str(exc),
                "elapsed_sec": round(elapsed, 2),
            }
            summary["failed"].append(failure)
            logger.error("  FAIL %s: %s (%.1fs)", fid, exc, elapsed)
            logger.debug(traceback.format_exc())
            if stop_on_error:
                raise

    summary["succeeded_count"] = len(summary["succeeded"])
    summary["failed_count"] = len(summary["failed"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize week2_pv_factors to factor lake")
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
    parser.add_argument("--factor-id", action="append", dest="factor_ids", help="Only run catalog id e.g. w2_001")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", type=Path, default=quant_projects_root() / "logs" / "materialize_week2.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_file=args.log_file)

    data_root = quant_projects_root() / "data" / "a_share" / "lqtp_data" / "StockDailyBar"
    if not data_root.is_dir() or not any(data_root.glob("*.parquet")):
        logger.error("本地数据不存在: %s — 请先 sync COS StockDailyBar", data_root)
        return 1

    summary = materialize_week2(
        lake_root=args.lake_root,
        author=args.author,
        max_files=args.max_files,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_ids=args.factor_ids,
        stop_on_error=args.stop_on_error,
    )

    out_path = args.lake_root / "_runs" / f"week2_materialize_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nrun summary -> {out_path}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
