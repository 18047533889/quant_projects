"""
data_access.cos_mirror —— COS 清洗数据本地镜像按需同步（A 股 / 美股）

读数据前由 store 调用 ``ensure_local_mirror_for_dataset``：
  - 本地缺文件时按 ``time_range`` 从 COS 增量拉取
  - ``DATA_ACCESS_SKIP_COS_MIRROR=1`` 可关闭
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .exceptions import ValidationError

if TYPE_CHECKING:
    from .registry import Dataset, StaticDataset

logger = logging.getLogger("data_access.cos_mirror")

COS_CLI = os.environ.get("DATA_ACCESS_COS_CLI", os.environ.get("ASHARE_COS_CLI", "clean-cos-ro"))

ASHARE_COS_PREFIX = os.environ.get(
    "ASHARE_LQTP_COS_PREFIX",
    "cos://qs-cold/clean_data/ashare/lqtp_data",
).rstrip("/")
ASHARE_LOCAL_ROOT = Path(
    os.environ.get(
        "ASHARE_PARQUET_ROOT",
        "/home/shw/quant_projects/data/a_share/lqtp_data",
    )
).expanduser()

US_MASSIVE_COS_PREFIX = os.environ.get(
    "US_MASSIVE_COS_PREFIX",
    "cos://qs-cold/clean_data/us_stock/massive_data",
).rstrip("/")
US_MASSIVE_LOCAL_ROOT = Path(
    os.environ.get(
        "US_MASSIVE_ROOT",
        "/home/shw/quant_projects/data/us_stock/massive_data",
    )
).expanduser()

US_CLEAN_COS_PREFIX = os.environ.get(
    "US_CLEAN_COS_PREFIX",
    "cos://qs-cold/clean_data",
).rstrip("/")
US_CLEAN_LOCAL_ROOT = Path(
    os.environ.get(
        "US_CLEAN_ROOT",
        "/home/shw/quant_projects/data/us_stock/clean_data",
    )
).expanduser()


@dataclass(frozen=True)
class MirrorSpec:
    """单个 datasets.yaml 条目对应的 COS 镜像规格。"""

    cos_prefix: str
    local_root: Path
    table: str | None = None
    # daily_parquet | single_full | hive_date | hive_year | root_file
    layout: str = "daily_parquet"
    file_name: str = "full.parquet"


def _spec(
    dataset: str,
    *,
    cos_prefix: str,
    local_root: Path,
    table: str | None = None,
    layout: str = "daily_parquet",
    file_name: str = "full.parquet",
) -> tuple[str, MirrorSpec]:
    return dataset, MirrorSpec(
        cos_prefix=cos_prefix.rstrip("/"),
        local_root=local_root.expanduser(),
        table=table,
        layout=layout,
        file_name=file_name,
    )


def _build_mirror_registry() -> dict[str, MirrorSpec]:
    ap, ar = ASHARE_COS_PREFIX, ASHARE_LOCAL_ROOT
    up, ur = US_MASSIVE_COS_PREFIX, US_MASSIVE_LOCAL_ROOT
    cp, cr = US_CLEAN_COS_PREFIX, US_CLEAN_LOCAL_ROOT

    entries: list[tuple[str, MirrorSpec]] = [
        # ---- A 股 lqtp ----
        _spec("ashare_calendar", cos_prefix=ap, local_root=ar, table="Calendar", layout="single_full"),
        _spec("ashare_etf_daily", cos_prefix=ap, local_root=ar, table="ETFDailyBar"),
        _spec("ashare_etf_list", cos_prefix=ap, local_root=ar, table="ETFList"),
        _spec("ashare_index_constituent", cos_prefix=ap, local_root=ar, table="IndexConstituent"),
        _spec("ashare_index_daily", cos_prefix=ap, local_root=ar, table="IndexDailyBar"),
        _spec("ashare_index_list", cos_prefix=ap, local_root=ar, table="IndexList"),
        _spec("ashare_stock_balance", cos_prefix=ap, local_root=ar, table="StockBalance"),
        _spec("ashare_stock_capital_daily", cos_prefix=ap, local_root=ar, table="StockCapitalDaily"),
        _spec("ashare_stock_cashflow", cos_prefix=ap, local_root=ar, table="StockCashFlow"),
        _spec("ashare_stock_daily", cos_prefix=ap, local_root=ar, table="StockDailyBar"),
        _spec("ashare_stock_dividend", cos_prefix=ap, local_root=ar, table="StockDividend"),
        _spec("ashare_stock_income", cos_prefix=ap, local_root=ar, table="StockIncome"),
        _spec("ashare_stock_indicator", cos_prefix=ap, local_root=ar, table="StockIndicator"),
        _spec("ashare_stock_industry", cos_prefix=ap, local_root=ar, table="StockIndustry"),
        _spec("ashare_stock_list", cos_prefix=ap, local_root=ar, table="StockList"),
        _spec("ashare_stock_minute", cos_prefix=ap, local_root=ar, table="StockMinuteBar"),
        _spec("ashare_stock_status", cos_prefix=ap, local_root=ar, table="StockStatus"),
        _spec("ashare_stock_topten_float_shareholder", cos_prefix=ap, local_root=ar, table="StockTopTenFloatShareholder"),
        _spec("ashare_stock_topten_shareholder", cos_prefix=ap, local_root=ar, table="StockTopTenShareholder"),
        _spec("ashare_stock_valuation_daily", cos_prefix=ap, local_root=ar, table="StockValuationDaily"),
        # ---- 美股 massive_data ----
        _spec("us_calendar", cos_prefix=up, local_root=ur, table="Calendar", layout="single_full"),
        _spec("us_etf_daily", cos_prefix=up, local_root=ur, table="ETFDailyBar"),
        _spec("us_etf_list", cos_prefix=up, local_root=ur, table="ETFList", layout="single_full"),
        _spec("us_security_master", cos_prefix=up, local_root=ur, table="SecurityMaster", layout="single_full"),
        _spec("us_stock_balance", cos_prefix=up, local_root=ur, table="StockBalance"),
        _spec("us_stock_capital_daily", cos_prefix=up, local_root=ur, table="StockCapitalDaily"),
        _spec("us_stock_cashflow", cos_prefix=up, local_root=ur, table="StockCashFlow"),
        _spec("us_stock_daily", cos_prefix=up, local_root=ur, table="StockDailyBar"),
        _spec("us_stock_dividend", cos_prefix=up, local_root=ur, table="StockDividend"),
        _spec("us_stock_income", cos_prefix=up, local_root=ur, table="StockIncome"),
        _spec("us_stock_indicator", cos_prefix=up, local_root=ur, table="StockIndicator"),
        _spec("us_stock_indices_components", cos_prefix=up, local_root=ur, table="StockIndicesComponents"),
        _spec("us_stock_industry", cos_prefix=up, local_root=ur, table="StockIndustry", layout="single_full"),
        _spec("us_stock_list", cos_prefix=up, local_root=ur, table="StockList"),
        _spec("us_stock_status", cos_prefix=up, local_root=ur, table="StockStatus", layout="single_full"),
        _spec("us_stock_valuation_daily", cos_prefix=up, local_root=ur, table="StockValuationDaily"),
        _spec("us_ticker_alias", cos_prefix=up, local_root=ur, table="TickerAlias", layout="single_full"),
        _spec("us_ticker_map", cos_prefix=up, local_root=ur, table="TickerMap", layout="single_full"),
        # ---- 美股 clean_data 衍生层（hive / 单文件）----
        _spec(
            "us_universe_daily",
            cos_prefix=f"{cp}/universe_daily",
            local_root=cr / "universe_daily",
            layout="hive_year",
            file_name="data.parquet",
        ),
        _spec(
            "us_adj_factor",
            cos_prefix=f"{cp}/adj_factor",
            local_root=cr / "adj_factor",
            layout="hive_date",
            file_name="data.parquet",
        ),
        _spec(
            "us_adj_factor_clamped",
            cos_prefix=f"{cp}/is_adj_factor_clamped",
            local_root=cr / "is_adj_factor_clamped",
            layout="hive_date",
            file_name="data.parquet",
        ),
        _spec(
            "us_is_early_close",
            cos_prefix=f"{cp}/is_early_close",
            local_root=cr / "is_early_close",
            layout="root_file",
            file_name="data.parquet",
        ),
        _spec(
            "us_is_ticker_halt_minute",
            cos_prefix=f"{cp}/is_ticker_halt/minute",
            local_root=cr / "is_ticker_halt" / "minute",
            layout="hive_date",
            file_name="data.parquet",
        ),
    ]
    return dict(entries)


DATASET_MIRROR_REGISTRY: dict[str, MirrorSpec] = _build_mirror_registry()

# 兼容旧导出
ASHARE_DATASET_TABLE_MAP: dict[str, str] = {
    name: spec.table  # type: ignore[misc]
    for name, spec in DATASET_MIRROR_REGISTRY.items()
    if name.startswith("ashare_") and spec.table
}

US_DATASET_TABLE_MAP: dict[str, str] = {
    name: spec.table  # type: ignore[misc]
    for name, spec in DATASET_MIRROR_REGISTRY.items()
    if name.startswith("us_") and spec.table
}


def skip_cos_mirror() -> bool:
    return os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR", "").lower() in (
        "1",
        "true",
        "yes",
    )


def mirror_spec_for_dataset(dataset_name: str) -> MirrorSpec | None:
    return DATASET_MIRROR_REGISTRY.get(dataset_name)


def table_for_dataset(dataset_name: str) -> str | None:
    spec = mirror_spec_for_dataset(dataset_name)
    return spec.table if spec else None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _iter_dates(start: date, end: date) -> list[date]:
    if start > end:
        return []
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _iter_years(start: date, end: date) -> list[int]:
    return list(range(start.year, end.year + 1))


def _run_cos_cli(args: Sequence[str]) -> None:
    cmd = [COS_CLI, *args]
    logger.info("cos_mirror: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _is_missing_object_error(exc: subprocess.CalledProcessError) -> bool:
    combined = f"{exc.stderr or ''}{exc.stdout or ''}".lower()
    return any(
        token in combined
        for token in ("nosuchkey", "not found", "404", "object not found")
    )


def _cos_table_uri(spec: MirrorSpec) -> str:
    if spec.table:
        return f"{spec.cos_prefix}/{spec.table}"
    return spec.cos_prefix


def _local_table_dir(spec: MirrorSpec) -> Path:
    if spec.table:
        return spec.local_root / spec.table
    return spec.local_root


def _sync_cos_file(cos_uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    try:
        _run_cos_cli(["cp", cos_uri, str(dest)])
    except subprocess.CalledProcessError as exc:
        if _is_missing_object_error(exc):
            logger.debug("cos_mirror: 跳过缺失文件 %s", cos_uri)
            return
        raise


def _sync_daily_file(spec: MirrorSpec, day: date) -> None:
    fname = f"{day.isoformat()}.parquet"
    dest = _local_table_dir(spec) / fname
    _sync_cos_file(f"{_cos_table_uri(spec)}/{fname}", dest)


def _sync_single_full(spec: MirrorSpec) -> None:
    dest = _local_table_dir(spec) / spec.file_name
    _sync_cos_file(f"{_cos_table_uri(spec)}/{spec.file_name}", dest)


def _sync_root_file(spec: MirrorSpec) -> None:
    dest = spec.local_root / spec.file_name
    _sync_cos_file(f"{spec.cos_prefix}/{spec.file_name}", dest)


def _sync_hive_date(spec: MirrorSpec, day: date) -> None:
    part = f"date={day.isoformat()}"
    dest = spec.local_root / part / spec.file_name
    _sync_cos_file(f"{spec.cos_prefix}/{part}/{spec.file_name}", dest)


def _sync_hive_year(spec: MirrorSpec, year: int) -> None:
    part = f"year={year}"
    dest = spec.local_root / part / spec.file_name
    _sync_cos_file(f"{spec.cos_prefix}/{part}/{spec.file_name}", dest)


def sync_dataset(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    full: bool = False,
) -> None:
    """同步指定 datasets.yaml 数据集到本地镜像。"""
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        raise ValueError(f"数据集 '{dataset_name}' 未配置 COS 镜像")

    if spec.layout == "single_full":
        _sync_single_full(spec)
        return
    if spec.layout == "root_file":
        _sync_root_file(spec)
        return

    if full or time_range is None:
        dest = _local_table_dir(spec) if spec.layout == "daily_parquet" else spec.local_root
        dest.mkdir(parents=True, exist_ok=True)
        _run_cos_cli(["sync", _cos_table_uri(spec) + "/", str(dest) + "/"])
        return

    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValueError(f"无法解析 time_range: {time_range}")

    if spec.layout == "daily_parquet":
        for day in _iter_dates(start, end):
            _sync_daily_file(spec, day)
    elif spec.layout == "hive_date":
        for day in _iter_dates(start, end):
            _sync_hive_date(spec, day)
    elif spec.layout == "hive_year":
        for year in _iter_years(start, end):
            _sync_hive_year(spec, year)
    else:
        raise ValueError(f"未知 layout: {spec.layout}")


def ensure_local_mirror(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
) -> None:
    """读前确保数据集在本地有足够 parquet 文件。"""
    if skip_cos_mirror():
        return

    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        return

    if spec.layout == "single_full":
        dest = _local_table_dir(spec) / spec.file_name
        if not dest.exists():
            logger.info("cos_mirror: 拉取单文件 %s", dataset_name)
            _sync_single_full(spec)
        return

    if spec.layout == "root_file":
        dest = spec.local_root / spec.file_name
        if not dest.exists():
            logger.info("cos_mirror: 拉取根文件 %s", dataset_name)
            _sync_root_file(spec)
        return

    local_dir = _local_table_dir(spec) if spec.layout == "daily_parquet" else spec.local_root
    existing = list(local_dir.rglob("*.parquet")) if local_dir.exists() else []

    if not existing:
        if time_range is not None:
            logger.info("cos_mirror: 本地无 %s，按 time_range 增量拉取", dataset_name)
            sync_dataset(dataset_name, time_range=time_range)
        else:
            raise ValidationError(
                f"cos_mirror: 本地无 {dataset_name} 且未指定 time_range，无法增量拉取。"
                f"请先执行 sync 脚本（如 scripts/sync_ashare_lqtp_cos.sh），"
                f"或在 data_source 配置 start_date/end_date。"
            )
        return

    if time_range is None:
        return

    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return

    if spec.layout == "daily_parquet":
        missing = [
            day
            for day in _iter_dates(start, end)
            if not (_local_table_dir(spec) / f"{day.isoformat()}.parquet").exists()
        ]
        for day in missing:
            _sync_daily_file(spec, day)
    elif spec.layout == "hive_date":
        missing = [
            day
            for day in _iter_dates(start, end)
            if not (spec.local_root / f"date={day.isoformat()}" / spec.file_name).exists()
        ]
        for day in missing:
            _sync_hive_date(spec, day)
    elif spec.layout == "hive_year":
        missing = [
            year
            for year in _iter_years(start, end)
            if not (spec.local_root / f"year={year}" / spec.file_name).exists()
        ]
        for year in missing:
            _sync_hive_year(spec, year)


def ensure_local_mirror_for_dataset(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None = None,
) -> None:
    from .registry import StaticDataset

    if not isinstance(ds, StaticDataset):
        return
    ensure_local_mirror(ds.name, time_range=time_range)
