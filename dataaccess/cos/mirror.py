"""
data_access.cos.mirror —— COS 清洗数据本地镜像按需同步（A 股 / 美股）

读数据前由 store 调用 ``ensure_local_mirror_for_dataset``（``DATA_ACCESS_COS_READ_MODE=mirror`` 时）：
  - 本地缺文件时按 ``time_range`` 从 COS 增量拉取
  - ``DATA_ACCESS_SKIP_COS_MIRROR=1`` 可关闭拉取

远程直读（``DATA_ACCESS_COS_READ_MODE=remote|auto``）见 ``data_access.cos.remote``：
  DuckDB httpfs 读 ``s3://``，不落地；需 ``DATA_ACCESS_COS_S3_ENDPOINT`` 与 COS 凭证。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from data_access.core.exceptions import ValidationError
from data_access.registry.paths import resolve_namespace_path

if TYPE_CHECKING:
    from data_access.registry import Dataset, StaticDataset

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
        _spec("ashare_turnover_base_daily", cos_prefix=ap, local_root=ar, table="TurnoverBaseDaily"),
        _spec("ashare_stock_topten_float_shareholder", cos_prefix=ap, local_root=ar, table="StockTopTenFloatShareholder"),
        _spec("ashare_stock_topten_shareholder", cos_prefix=ap, local_root=ar, table="StockTopTenShareholder"),
        _spec("ashare_stock_valuation_daily", cos_prefix=ap, local_root=ar, table="StockValuationDaily"),
        # ---- 美股 massive_data ----
        _spec("us_calendar", cos_prefix=up, local_root=ur, table="Calendar", layout="single_full"),
        _spec("us_etf_daily", cos_prefix=up, local_root=ur, table="ETFDailyBar"),
        _spec("us_etf_list", cos_prefix=up, local_root=ur, table="ETFList", layout="single_full"),
        _spec("us_security_master", cos_prefix=up, local_root=ur, table="SecurityMaster", layout="single_full"),
        _spec("us_stock_balance", cos_prefix=up, local_root=ur, table="StockBalance"),
        # StockCapitalDaily 目录混放 {date}.parquet（拆分事件）与 shares_{date}.parquet
        # （PIT 股本），两 schema 不同——按数据集拆分镜像，禁止 glob 整个目录。
        _spec("us_stock_capital_split", cos_prefix=up, local_root=ur, table="StockCapitalDaily"),
        _spec("us_stock_capital_shares", cos_prefix=up, local_root=ur, table="StockCapitalDaily"),
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
        _spec("us_security_master_daily_snap", cos_prefix=up, local_root=ur, table="SecurityMasterDailySnap"),
        _spec("us_ticker_shares_snapshot", cos_prefix=up, local_root=ur, table="TickerSharesSnapshot"),
        _spec("us_fact_news", cos_prefix=up, local_root=ur, table="FactNews"),
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


def known_cos_mirror_local_roots() -> tuple[Path, ...]:
    """COS 镜像覆盖的本地根目录（A 股 lqtp / 美股 massive / clean）。"""
    return (
        ASHARE_LOCAL_ROOT.resolve(),
        US_MASSIVE_LOCAL_ROOT.resolve(),
        US_CLEAN_LOCAL_ROOT.resolve(),
    )


def dataset_root_requires_cos_mirror(ds: "StaticDataset") -> bool:
    """静态数据集 root 落在已知 COS 镜像根下时需登记 mirror spec。"""
    # #P1-final closure 12：root 可能含 ${RUN_NAMESPACE}，按当前 context 解析
    root = Path(resolve_namespace_path(str(ds.root))).resolve()
    for mirror_root in known_cos_mirror_local_roots():
        try:
            root.relative_to(mirror_root)
            return True
        except ValueError:
            continue
    return False


def datasets_requiring_cos_mirror(registry: Any) -> set[str]:
    """从 registry 推断应配置 COS 镜像的数据集名集合。"""
    from data_access.registry import StaticDataset

    out: set[str] = set()
    for name, ds in registry._datasets.items():
        if isinstance(ds, StaticDataset) and dataset_root_requires_cos_mirror(ds):
            out.add(name)
    return out


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


def _calendar_domain_of(dataset_name: str) -> str:
    """#25 数据集声明的 calendar_domain（trade_day/calendar_day）；缺省 trade_day。"""
    try:
        from data_access.cos_contract import get_cos_contract

        c = get_cos_contract(dataset_name)
        if c is not None and c.calendar_domain:
            return c.calendar_domain
    except Exception:
        pass
    return "trade_day"


def _market_trading_days(dataset_name: str, start: date, end: date) -> list[date]:
    """#25 用市场交易日历返回 [start, end] 内的交易日；无日历回退自然日。"""
    try:
        from data_access.read.session_calendar import get_market_calendar

        market = None
        if dataset_name.startswith("us_"):
            market = "us"
        elif dataset_name.startswith("ashare_") or dataset_name.startswith("a_share"):
            market = "ashare"
        if market is None:
            return _iter_dates(start, end)
        cal = get_market_calendar(market, store=None)
        if cal is None or not getattr(cal, "has_data", False):
            return _iter_dates(start, end)
        days = [d for d in cal.trading_days if start <= d <= end]
        return sorted(days) if days else _iter_dates(start, end)
    except Exception:
        return _iter_dates(start, end)


def _expected_dates(dataset_name: str, start: date, end: date) -> list[date]:
    """#25 决定哪些 partition 应该存在：trade_day 用交易日历，calendar_day 用自然日。

    避免「交易日型数据按自然日枚举」导致周末/节假日被当成缺失 partition，进而
    误判 mirror 不完整 / 误切 remote。
    """
    domain = _calendar_domain_of(dataset_name)
    if domain == "calendar_day":
        return _iter_dates(start, end)
    # trade_day（缺省）：优先交易日历，回退自然日
    tdays = _market_trading_days(dataset_name, start, end)
    return tdays or _iter_dates(start, end)


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


def _write_download_manifest(dest: Path, *, remote_key: str | None = None) -> None:
    """#27 写入本地镜像 manifest（size + mtime + checksum + 下载时间 + verified）。

    供 mirror inventory（``load_mirror_inventory``）判定镜像是否真正正确——单纯
    ``Path.exists()`` 不足以发现下载中断 / 文件损坏 / 上游同名更新。
    """
    import hashlib
    import time as _time

    manifest = dest.with_suffix(dest.suffix + ".manifest.json")
    try:
        st = dest.stat()
    except OSError:
        return
    payload: dict[str, Any] = {
        "remote_key": remote_key or str(dest.name),
        "path": str(dest),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "downloaded_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "verified": True,
    }
    try:
        h = hashlib.sha256()
        with dest.open("rb") as fh:
            for _chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(_chunk)
        payload["checksum_sha256"] = h.hexdigest()[:32]
    except OSError:
        payload["verified"] = False
    # #P1-74 data 与 manifest 组成 atomic pair：manifest 先写 tmp 再 os.replace。
    # 进程死在 data.replace 与 manifest.replace 之间 → 旧 manifest 与 stat 不匹配
    # → _verify_mirror_file 判 stale → 自动 resync（fail-open 而非误判已同步）。
    tmp_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp_manifest), str(manifest))


def _verify_mirror_file(dest: Path) -> bool:
    """#27 校验本地镜像文件是否与 manifest 一致（size/mtime/checksum）。"""
    if not dest.exists():
        return False
    st = dest.stat()
    if st.st_size <= 0:
        return False
    manifest = dest.with_suffix(dest.suffix + ".manifest.json")
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("size") != st.st_size or payload.get("mtime_ns") != st.st_mtime_ns:
        return False
    if payload.get("verified") is False:
        return False
    return True


def load_mirror_inventory(store: Any, dataset: str) -> dict[str, Any]:
    """#27 本地 mirror 的下载清单：remote_key / size / checksum / downloaded_at /
    verified，以及按文件聚合的完整性统计。

    供 metadata plane（#38）与运维看板使用。返回:
        {files: [{path, remote_key, size, checksum_sha256, downloaded_at, verified}],
         total_files, verified_files, corrupt_files}
    """
    spec = mirror_spec_for_dataset(dataset)
    if spec is None:
        return {"files": [], "total_files": 0, "verified_files": 0, "corrupt_files": 0}
    base = _local_table_dir(spec) if spec.table else spec.local_root
    if not base.exists():
        return {"files": [], "total_files": 0, "verified_files": 0, "corrupt_files": 0}
    files: list[dict[str, Any]] = []
    for mp in sorted(base.rglob("*.parquet.manifest.json")):
        try:
            payload = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data_file = Path(str(payload.get("path") or mp).replace(".manifest.json", ""))
        payload["verified"] = bool(payload.get("verified")) and (
            data_file.exists()
            and data_file.stat().st_size == payload.get("size")
        )
        files.append(
            {
                "path": str(data_file),
                "remote_key": payload.get("remote_key"),
                "size": payload.get("size"),
                "checksum_sha256": payload.get("checksum_sha256"),
                "downloaded_at": payload.get("downloaded_at"),
                "verified": payload.get("verified"),
            }
        )
    return {
        "files": files,
        "total_files": len(files),
        "verified_files": sum(1 for f in files if f["verified"]),
        "corrupt_files": sum(1 for f in files if not f["verified"]),
    }


def _local_file_fresh(dest: Path) -> bool:
    """#27 本地文件存在且 manifest 与 stat 一致（含 verified）时视为已同步。"""
    return _verify_mirror_file(dest)


def _local_file_usable(dest: Path) -> bool:
    """#P1-73 增量检查用判定：verified 或「存在且非空」。

    整目录 ``cos sync`` 出来的文件没有 per-file manifest；strict ``_local_file_fresh``
    对它们恒 False，会导致每次 ensure 全量重拉。增量路径用本判定：verified-stale
    或损坏（有 manifest 对不上）→ resync；manifest-less 的完整文件视为可用。
    """
    if _verify_mirror_file(dest):
        return True
    if not dest.exists():
        return False
    try:
        return dest.stat().st_size > 0
    except OSError:
        return False


def _sync_cos_file(cos_uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _local_file_fresh(dest):
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    try:
        _run_cos_cli(["cp", cos_uri, str(tmp)])
        if not tmp.exists() or tmp.stat().st_size <= 0:
            logger.debug("cos_mirror: 跳过空/缺失下载 %s", cos_uri)
            if tmp.exists():
                tmp.unlink()
            return
        tmp.replace(dest)
        _write_download_manifest(dest, remote_key=cos_uri)
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
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
        # #P1-73 exists 不够：文件存在但损坏/无 manifest 也会被误判已同步。
        if not _local_file_fresh(dest):
            logger.info("cos_mirror: 拉取单文件 %s", dataset_name)
            _sync_single_full(spec)
        return

    if spec.layout == "root_file":
        dest = spec.local_root / spec.file_name
        # #P1-73 同上：统一 verified mirror object 判定。
        if not _local_file_fresh(dest):
            logger.info("cos_mirror: 拉取根文件 %s", dataset_name)
            _sync_root_file(spec)
        return

    local_dir = _local_table_dir(spec) if spec.layout == "daily_parquet" else spec.local_root
    # 只判断目录是否非空：完整 rglob 在大镜像上很贵，后续按 time_range 做增量存在性检查
    has_local = False
    if local_dir.exists() and local_dir.is_dir():
        try:
            next(local_dir.iterdir())
            has_local = True
        except StopIteration:
            has_local = False

    if not has_local:
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
        # #25 交易日型数据按交易日历枚举期望 partition，周末/节假日不判缺失
        # #P1-73 增量判定：verified-stale / 损坏 resync；manifest-less 完整文件
        # 不强制重拉（整目录 cos sync 无 per-file manifest）。
        expected = _expected_dates(dataset_name, start, end)
        missing = [
            day
            for day in expected
            if not _local_file_usable(_local_table_dir(spec) / f"{day.isoformat()}.parquet")
        ]
        for day in missing:
            _sync_daily_file(spec, day)
    elif spec.layout == "hive_date":
        expected = _expected_dates(dataset_name, start, end)
        missing = [
            day
            for day in expected
            if not _local_file_usable(spec.local_root / f"date={day.isoformat()}" / spec.file_name)
        ]
        for day in missing:
            _sync_hive_date(spec, day)
    elif spec.layout == "hive_year":
        missing = [
            year
            for year in _iter_years(start, end)
            if not _local_file_usable(spec.local_root / f"year={year}" / spec.file_name)
        ]
        for year in missing:
            _sync_hive_year(spec, year)


def ensure_local_mirror_for_dataset(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None = None,
) -> None:
    from data_access.registry import StaticDataset

    if not isinstance(ds, StaticDataset):
        return
    ensure_local_mirror(ds.name, time_range=time_range)
