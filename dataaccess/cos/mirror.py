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
    """单个 datasets.yaml 条目对应的 COS 镜像规格（R25 P0-001/002：只保留
    deployment/location；layout/filename template/partition clock 来自
    RuntimeDatasetContract.physical_partition，见 ``physical_partition_for``）。"""

    cos_prefix: str
    local_root: Path
    table: str | None = None
    # daily_parquet | period_files | prefixed_date_file | single_full | hive_date |
    # hive_year | root_file | event_files
    layout: str = "daily_parquet"
    file_name: str = "full.parquet"
    # R25 P0-002：同目录混放不同 schema 时用 file_selector 区分前缀
    # （StockCapital shares_2024-01-01.parquet vs 2024-01-01.parquet）。
    file_selector: str | None = None
    # R25 P0-002：显式文件名模板（如 "{date}.parquet" / "shares_{date}.parquet"）。
    filename_template: str | None = None


def _spec(
    dataset: str,
    *,
    cos_prefix: str,
    local_root: Path,
    table: str | None = None,
    layout: str = "daily_parquet",
    file_name: str = "full.parquet",
    file_selector: str | None = None,
    filename_template: str | None = None,
) -> tuple[str, MirrorSpec]:
    return dataset, MirrorSpec(
        cos_prefix=cos_prefix.rstrip("/"),
        local_root=local_root.expanduser(),
        table=table,
        layout=layout,
        file_name=file_name,
        file_selector=file_selector,
        filename_template=filename_template,
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
        _spec("us_stock_balance", cos_prefix=up, local_root=ur, table="StockBalance", layout="period_files", filename_template="{period_end}.parquet"),
        # StockCapitalDaily 目录混放 {date}.parquet（拆分事件）与 shares_{date}.parquet
        # （PIT 股本），两 schema 不同——按数据集拆分镜像，禁止 glob 整个目录。
        # R25 P0-002：用 file_selector 区分前缀，mirror/remote/auto 用同一个 locator。
        _spec("us_stock_capital_split", cos_prefix=up, local_root=ur, table="StockCapitalDaily", layout="prefixed_date_file", filename_template="{date}.parquet"),
        _spec("us_stock_capital_shares", cos_prefix=up, local_root=ur, table="StockCapitalDaily", layout="prefixed_date_file", file_selector="shares_", filename_template="shares_{date}.parquet"),
        _spec("us_stock_cashflow", cos_prefix=up, local_root=ur, table="StockCashFlow", layout="period_files", filename_template="{period_end}.parquet"),
        _spec("us_stock_daily", cos_prefix=up, local_root=ur, table="StockDailyBar"),
        _spec("us_stock_dividend", cos_prefix=up, local_root=ur, table="StockDividend"),
        _spec("us_stock_income", cos_prefix=up, local_root=ur, table="StockIncome", layout="period_files", filename_template="{period_end}.parquet"),
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


def _strict_mirror_mode() -> bool:
    """#P1-final closure 21：唯一严格模式判定（production OR strict_read）。"""
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        return True


# #P1-final closure 21：trade_day 期望日期在真实日历不可用时的回退标记。
# 自然日回退会高估期望 partition（把周末/节假日当缺失），strict 下应显式记录
# degraded，而不是静默当「完整期望」。


def _market_trading_days(
    dataset_name: str, start: date, end: date
) -> tuple[list[date], bool]:
    """#33 收官轮：返回 ``(交易日, used_calendar)``。

    ``used_calendar=True`` = 用了真实市场交易日历；``False`` = 日历不可用/市场未知/
    窗口内无交易日 → 回退自然日（调用方据此标记 degraded）。旧实现返回裸 list，
    日历不可用时自然日回退也是**非空列表**，调用方 ``if tdays`` 会误判成「用了
    日历」——degraded 检测从未真正生效。
    """
    try:
        from data_access.read.session_calendar import get_market_calendar

        market = None
        if dataset_name.startswith("us_"):
            market = "us"
        elif dataset_name.startswith("ashare_") or dataset_name.startswith("a_share"):
            market = "ashare"
        if market is None:
            return _iter_dates(start, end), False
        cal = get_market_calendar(market, store=None)
        if cal is None or not getattr(cal, "has_data", False):
            return _iter_dates(start, end), False
        days = sorted(d for d in cal.trading_days if start <= d <= end)
        if not days:
            # 日历可用但窗口内无交易日（长假/历史窗口）→ 自然日回退并标记 degraded
            return _iter_dates(start, end), False
        return days, True
    except Exception:
        return _iter_dates(start, end), False


def _expected_dates_with_degraded(
    dataset_name: str, start: date, end: date
) -> tuple[list[date], bool]:
    """#33 收官轮：期望 partition + degraded 标记一起返回（**线程安全**）。

    不再用 module-global ``_expected_dates_degraded``——两个线程同时查不同 dataset
    会互相覆盖 degraded 状态。degraded 随结果走，谁调用谁持有。

    ``degraded=True`` 表示 trade_day 真实交易日历不可用 → 自然日回退，期望
    partition 被高估（周末/节假日算缺失），不是权威交易日集合。
    """
    domain = _calendar_domain_of(dataset_name)
    if domain == "calendar_day":
        return _iter_dates(start, end), False
    # trade_day（缺省）：优先交易日历，回退自然日
    tdays, used_calendar = _market_trading_days(dataset_name, start, end)
    if used_calendar:
        return tdays, False
    logger.warning(
        "cos_mirror: %s 的 trade_day 期望日期回退为自然日（真实交易日历不可用）——"
        "期望 partition 会被高估（周末/节假日算缺失）。",
        dataset_name,
    )
    return tdays, True


def _expected_dates(dataset_name: str, start: date, end: date) -> list[date]:
    """#25 决定哪些 partition 应该存在：trade_day 用交易日历，calendar_day 用自然日。

    避免「交易日型数据按自然日枚举」导致周末/节假日被当成缺失 partition，进而
    误判 mirror 不完整 / 误切 remote。需要 degraded 标记时用
    ``_expected_dates_with_degraded``。**唯一 implementation**（R24 P1-1：删除
    历史重复定义）。
    """
    return _expected_dates_with_degraded(dataset_name, start, end)[0]


def expected_partitions(
    dataset_name: str, start: date, end: date, *, layout: str
) -> list[Any]:
    """#P0 收官（0.9.5）：**共享的期望 partition 编译器**。

    local/mirror/remote/hybrid 共用同一份「该数据集在 [start,end] 内应该存在哪些
    partition」：
        - ``daily_parquet`` / ``hive_date`` → 期望日期（trade_day 走交易日历，
          calendar_day 走自然日；日历不可用回退自然日并标记 degraded）；
        - ``hive_year`` → 年份（天然与自然日无关）；
        - ``period_files`` / ``prefixed_date_file`` / ``event_files`` → **不按
          request 时间轴枚举物理文件名**（R25 P0-001/016：partition_clock !=
          predicate_clock 时禁止瞎映射），返回空列表（调用方必须用完整目录
          sync / source index / manifest，宁可多扫不能漏）。
    remote 侧**不得**再实现一套「自然日逐日枚举」——交易日型数据跨周末会被误判
    缺失 → 误切 remote → 去请求不存在的周末对象。mirror 的
    ``_sync_daily_file`` / ``sync_dataset`` 与 remote 的路径构建全走这里，保证
    本地完整性判断与远程路径生成用同一份期望集合。
    """
    layout = str(layout or "").lower()
    if layout == "hive_year":
        return list(_iter_years(start, end))
    if layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/016：不能按 request time_range 展开 period/event 物理文件名
        # （StockIncome/2024-03-31.parquet 的 partition_clock=period_end，predicate
        # = filing_date）。返回空 → 调用方 fallback 完整目录 sync / source index。
        return []
    return _expected_dates(dataset_name, start, end)


def expected_partitions_with_degraded(
    dataset_name: str, start: date, end: date, *, layout: str
) -> tuple[list[Any], bool]:
    """共享期望 partition 编译器 + degraded 标记（线程安全，见 ``_expected_dates_with_degraded``）。

    ``hive_year`` 与自然日无关，永不 degraded；period/event/prefixed 布局按
    R25 P0-001/016 不枚举（空列表，非 degraded——调用方用完整 sync/source index）。
    """
    layout = str(layout or "").lower()
    if layout == "hive_year":
        return list(_iter_years(start, end)), False
    if layout in {"period_files", "prefixed_date_file", "event_files"}:
        return [], False
    return _expected_dates_with_degraded(dataset_name, start, end)


def physical_partition_for(dataset_name: str, registry: Any = None) -> Any:
    """R25 P0-001/002/016：**唯一物理分区 locator**（mirror/remote/auto 共用）。

    优先 RuntimeDatasetContract（契约声明，period_files / event_files 等）；无
    契约或布局未声明时从 MirrorSpec 编译。返回 ``data_access.contract.PhysicalPartitionSpec``。

    **禁止** mirror.py 自己拼 / remote.py 再拼一遍 / registry glob 又是一套——
    物理布局（layout/filename template/partition clock/缺失语义）只有这一个来源。

    R26-P1-030：消费调用方已绑定的 registry（不自行 ``load_registry()`` 再吞异常）——
    custom registry / test registry / namespace 不会漂移。未传 registry 时才尝试
    进程默认 registry。
    """
    from data_access.contract.physical_partition import (
        MissingPartitionSemantics,
        PhysicalLayout,
        PhysicalPartitionSpec,
        layout_from_mirror,
    )
    from data_access.contract.runtime_contract import compile_runtime_contract

    # 1) 契约优先：RuntimeDatasetContract 已把 storage_layout（period_files /
    #    event_files）编译成 PhysicalPartitionSpec。
    try:
        if registry is None:
            from data_access.registry import load_registry

            registry = load_registry()
        rc = compile_runtime_contract(dataset_name, registry)
        if rc is not None and rc.physical_partition is not None:
            return rc.physical_partition
    except Exception:
        pass

    # 2) 退路：MirrorSpec 编译（deployment 声明）。
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        return None
    # R26-P1-006：mirror 布局声明非法 → fail（不静默 fallback 到 DAILY_TRADE_DATE）。
    layout = layout_from_mirror(
        getattr(spec, "layout", None) or "daily_parquet"
    )
    filename_template = getattr(spec, "filename_template", None)
    file_selector = getattr(spec, "file_selector", None)
    # StockCapital 混放目录：file_selector 决定 filename template。
    if layout == PhysicalLayout.PREFIXED_DATE_FILE and not filename_template:
        sel = file_selector or ""
        filename_template = f"{sel}{{date}}.parquet"
    return PhysicalPartitionSpec(
        layout=layout,
        partition_clock="period_end"
        if layout == PhysicalLayout.PERIOD_END_FILE
        else "date",
        filename_template=filename_template or "{date}.parquet",
        file_selector=file_selector,
        completeness=MissingPartitionSemantics.ERROR,
    )


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


def _cache_scope_fields() -> dict[str, Any]:
    """R24 P0-S3 §5.5：cache manifest 的非 secret 权限身份（scope 绑定）。

    权限降低 / 策略变化后，旧的高权限 cache 不得自动复用（scope 不匹配 →
    ``_mirror_file_state`` 判 stale → 重新拉取）。只写非 secret 标识，绝不写
    credential 本身。
    """
    import hashlib as _hashlib

    fields: dict[str, Any] = {}
    try:
        from data_access.security.policy import get_authorizer
        from data_access.security.credentials import _global_credential_provider

        auth = get_authorizer()
        fields["access_policy_digest"] = getattr(
            getattr(auth, "policy", None), "digest", lambda: None
        )() or None
        principal = getattr(auth, "principal", None)
        fields["principal_scope_id"] = (
            getattr(principal, "principal_id", None) or None
        )
        provider = _global_credential_provider()
        if provider is not None:
            try:
                material = provider.resolve()
                fields["credential_scope_id"] = (
                    material.credential_scope_id or None
                )
            except Exception:
                fields["credential_scope_id"] = None
    except Exception:
        pass
    return fields


def _write_download_manifest(dest: Path, *, remote_key: str | None = None) -> None:
    """#27 写入本地镜像 manifest（size + mtime + checksum + 下载时间 + verified）。

    供 mirror inventory（``load_mirror_inventory``）判定镜像是否真正正确——单纯
    ``Path.exists()`` 不足以发现下载中断 / 文件损坏 / 上游同名更新。

    R24 P0-S3 §5.5：manifest 增加 ``principal_scope_id / access_policy_digest /
    credential_scope_id``——权限 scope 不匹配的旧缓存不得复用。
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
    payload.update(_cache_scope_fields())
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


def _mirror_file_state(dest: Path) -> str:
    """#P1-final closure 21：本地镜像文件完整性三态。

    - ``verified``       ：有 manifest，size/mtime/verified 一致（checksum 在
      deep 校验时才重算，见 ``_verify_mirror_file``）；
    - ``legacy_unverified``：manifest-less 但存在且非空（整目录 ``cos sync``
      产物）——**权威读取不能把「非空」当「完整」**；
    - ``corrupt``        ：有 manifest 但 size/mtime/verified 对不上，或文件为空；
    - ``missing``        ：不存在。
    """
    if not dest.exists():
        return "missing"
    try:
        st = dest.stat()
    except OSError:
        return "missing"
    if st.st_size <= 0:
        return "corrupt"
    manifest = dest.with_suffix(dest.suffix + ".manifest.json")
    if not manifest.exists():
        return "legacy_unverified"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "corrupt"
    if payload.get("size") != st.st_size or payload.get("mtime_ns") != st.st_mtime_ns:
        return "corrupt"
    if payload.get("verified") is False:
        return "corrupt"
    return "verified"


def _verify_mirror_file(dest: Path, *, deep: bool = False) -> bool:
    """#27 校验本地镜像文件是否与 manifest 一致（size/mtime；deep 再核 checksum）。

    ``deep=True`` 时若 manifest 记录 ``checksum_sha256``，重新读文件算 hash 对比——
    权威读取 / deep verify 用，杜绝「同 size/mtime 但内容被改」漏网。
    """
    import hashlib

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
    if deep:
        expected = payload.get("checksum_sha256")
        if expected:
            try:
                h = hashlib.sha256()
                with dest.open("rb") as fh:
                    for _chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(_chunk)
                if h.hexdigest()[:32] != str(expected):
                    return False
            except OSError:
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
        # R26-P0-008：verified 必须真实 bool——字符串 "false" 不得解析成 True。
        verified_raw = payload.get("verified")
        verified = (
            verified_raw
            if isinstance(verified_raw, bool)
            else str(verified_raw).lower() == "true"
        )
        payload["verified"] = verified and (
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

    #P1-final closure 21：**strict/production 下不把「非空」当「完整」**——
    ``legacy_unverified``（manifest-less）在严格模式必须 resync 成 verified 才
    可被权威读取使用；research/增量 sync 才允许宽松放行。
    """
    state = _mirror_file_state(dest)
    if state == "verified":
        return True
    if state == "corrupt":
        return False
    if state == "missing":
        return False
    # legacy_unverified：manifest-less 非空文件
    if _strict_mirror_mode():
        return False  # 权威读取：无法证明完整 ⇒ 必须 resync
    return True


def _missing_semantics_for_dataset(dataset_name: str) -> str:
    """R25 P0-015：缺失 partition 的契约语义（error / warn / empty_ok）。

    从 COS 契约 ``missing_partition_semantics`` 读取（缺省 error）；事件/稀疏表
    契约通常声明 empty_ok。Downloader **不自己决定**——按 contract 走。
    """
    try:
        from data_access.cos_contract import get_cos_contract

        c = get_cos_contract(dataset_name)
        if c is not None and getattr(c, "missing_partition_semantics", None):
            return str(c.missing_partition_semantics).lower()
    except Exception:
        pass
    return "error"


def _sync_cos_file(
    cos_uri: str,
    dest: Path,
    *,
    missing_semantics: str = "error",
    dataset_name: str | None = None,
) -> None:
    """从 COS 下载单个对象到本地（R24 P0-S3 §5.3 / T-S08 安全临时文件）。

    R25 P0-015：缺失对象按 dataset 契约语义处理，**不统一静默跳过**：
        - missing_semantics="error"  （dense 行情）→ 缺失抛 ``MissingRequiredPartition``；
        - missing_semantics="warn"   → 告警；
        - missing_semantics="empty_ok"（稀疏事件表）→ 静默跳过。

    - 目标文件是 symlink → 拒绝（symlink 替换攻击：恶意预建 symlink 把下载重定向
      到 root 之外）；
    - 用 ``tempfile.NamedTemporaryFile(dir=dest.parent, delete=False)``（O_EXCL
      原子创建随机名）下载，防止 tmp 抢占 / 并发 writer 覆盖 / symlink 替换；
    - 下载完成后 ``chmod 0600``（文件默认私有，绝不开 world-readable）；
    - ``os.replace`` 原子落盘 + 写 scope 绑定的 manifest。
    """
    import os as _os
    import tempfile as _tempfile

    from data_access.core.exceptions import MissingRequiredPartition

    dest.parent.mkdir(parents=True, exist_ok=True)
    if _local_file_fresh(dest):
        return
    if dest.is_symlink():
        raise ValidationError(
            f"本地镜像目标 {dest} 是 symlink，拒绝覆盖（T-S08 symlink 攻击 fail-closed）"
        )
    tmp = None
    try:
        # O_EXCL 原子创建随机名临时文件（防 tmp 抢占 / symlink 替换 / 并发覆盖）。
        tmp = _tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=".cosdl-", suffix=".tmp", delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        _run_cos_cli(["cp", cos_uri, str(tmp_path)])
        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            # 空下载：按语义处理（error 强制 / empty_ok 跳过）。
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if missing_semantics == "empty_ok":
                logger.debug("cos_mirror: 稀疏事件表空对象（empty_ok）跳过 %s", cos_uri)
                return
            if missing_semantics == "warn":
                logger.warning("cos_mirror: 对象下载为空（warn）%s", cos_uri)
                return
            raise MissingRequiredPartition(
                f"cos_mirror: 数据集 {dataset_name or ''} 必需 partition 下载为空："
                f"{cos_uri}（missing_semantics=error，production fail-closed）"
            )
        try:
            _os.chmod(tmp_path, 0o600)  # 文件默认 0600（私有）
        except OSError:
            pass
        if dest.exists() and dest.is_symlink():
            raise ValidationError(
                f"本地镜像目标 {dest} 是 symlink，拒绝覆盖（T-S08）"
            )
        tmp_path.replace(dest)
        _write_download_manifest(dest, remote_key=cos_uri)
    except subprocess.CalledProcessError as exc:
        if tmp is not None:
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass
        if _is_missing_object_error(exc):
            # R25 P0-015：404 / NoSuchKey 按契约语义处理，不统一 debug 跳过。
            if missing_semantics == "empty_ok":
                logger.debug("cos_mirror: 稀疏事件表缺失对象（empty_ok）跳过 %s", cos_uri)
                return
            if missing_semantics == "warn":
                logger.warning("cos_mirror: 对象缺失（warn）%s", cos_uri)
                return
            raise MissingRequiredPartition(
                f"cos_mirror: 数据集 {dataset_name or ''} 必需 partition 缺失："
                f"{cos_uri}（missing_semantics=error，production fail-closed；"
                "若确为稀疏事件表，请契约声明 missing_partition_semantics=empty_ok）"
            ) from exc
        raise
    except Exception:
        if tmp is not None:
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass
        raise


def _daily_filename(spec: MirrorSpec, day: date) -> str:
    """R25 P0-002：StockCapital split/shares 用同一个 locator 生成文件名。

    - split（file_selector=None）：``{date}.parquet``
    - shares（file_selector="shares_"）：``shares_{date}.parquet``
    mirror 与 remote **必须**走同一函数，禁止各自拼。
    """
    if spec.filename_template:
        if "{date}" in spec.filename_template:
            return spec.filename_template.replace("{date}", day.isoformat())
        # 无 {date} 占位符的模板：直接按 spec 拼接
        if spec.file_selector:
            return f"{spec.file_selector}{day.isoformat()}.parquet"
        return spec.filename_template
    if spec.file_selector:
        return f"{spec.file_selector}{day.isoformat()}.parquet"
    return f"{day.isoformat()}.parquet"


def _sync_daily_file(spec: MirrorSpec, day: date, dataset_name: str | None = None) -> None:
    fname = _daily_filename(spec, day)
    dest = _local_table_dir(spec) / fname
    _sync_cos_file(
        f"{_cos_table_uri(spec)}/{fname}",
        dest,
        missing_semantics=_missing_semantics_for_dataset(dataset_name or ""),
        dataset_name=dataset_name,
    )


def _sync_dir_full(spec: MirrorSpec) -> None:
    """R25 P0-001/016：period/event/prefixed 布局的完整目录同步。

    partition_clock != predicate_clock 时**不能**按 request time_range 枚举物理
    文件名（StockIncome/2024-03-31.parquet 的 partition 是 period_end，request
    knowledge 是 filing_date）。宁可多扫整个目录，不能漏读真实 filing。
    """
    dest = _local_table_dir(spec) if spec.table else spec.local_root
    dest.mkdir(parents=True, exist_ok=True)
    _run_cos_cli(["sync", _cos_table_uri(spec) + "/", str(dest) + "/"])


def _sync_single_full(spec: MirrorSpec) -> None:
    dest = _local_table_dir(spec) / spec.file_name
    _sync_cos_file(f"{_cos_table_uri(spec)}/{spec.file_name}", dest)


def publish_mirror_generation(
    spec: MirrorSpec,
    *,
    generation_id: str | None = None,
    source_generation: str | None = None,
    full: bool = False,
) -> str:
    """R25 P0-014 / §15：full mirror 的 **generation-based 原子发布**。

    不再 ``cos sync`` 直接写 live directory（读者可能看到 old/new mixture）。
    流程：
        1. 解析 generation G（缺省用 source_generation 或时间戳）；
        2. ``<mirror_root>/generation/G.tmp`` staging；
        3. sync 全部 + 验证（object inventory / checksum / schema 由调用方 verify）；
        4. fsync + ``G.tmp -> G`` rename；
        5. atomic manifest pointer -> G（``generation/current.json``）；
        6. 旧 generation 延迟 GC（reader 只 resolve pointer）。

    返回发布的 generation_id。读者（``resolve_current_mirror_generation``）只读
    pointer，**绝不**边 sync 边暴露。
    """
    import datetime as _dt

    if generation_id is None:
        stamp = source_generation or _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        generation_id = f"{stamp}-{_rand_suffix()}"
    mirror_root = spec.local_root
    staging = mirror_root / "generation" / f"{generation_id}.tmp"
    final = mirror_root / "generation" / generation_id
    staging.mkdir(parents=True, exist_ok=True)

    # sync 到 staging（不碰 live）。
    staging_spec = MirrorSpec(
        cos_prefix=spec.cos_prefix,
        local_root=staging,
        table=spec.table,
        layout=spec.layout,
        file_name=spec.file_name,
        file_selector=spec.file_selector,
        filename_template=spec.filename_template,
    )
    if spec.layout == "single_full":
        _sync_single_full(staging_spec)
    elif spec.layout == "root_file":
        _sync_root_file(staging_spec)
    elif spec.layout in {"period_files", "prefixed_date_file", "event_files", "daily_parquet"}:
        _sync_dir_full(staging_spec)
    elif spec.layout == "hive_date":
        for year in _iter_years(_dt.date(2000, 1, 1), _dt.date.today()):
            _sync_hive_year(
                MirrorSpec(
                    cos_prefix=spec.cos_prefix,
                    local_root=staging,
                    table=spec.table,
                    layout="hive_date",
                    file_name=spec.file_name,
                ),
                year,
            )
    else:
        _sync_dir_full(staging_spec)

    # 验证 + fsync + atomic rename + pointer。
    _verify_staged_generation(staging)
    os.replace(str(staging), str(final))
    _write_generation_pointer(mirror_root, generation_id, source_generation)
    logger.info(
        "cos_mirror: generation %s published (dataset=%s objects=%s)",
        generation_id,
        spec.table or spec.cos_prefix,
        _dir_object_count(final),
    )
    return generation_id


def _rand_suffix() -> str:
    import secrets

    return secrets.token_hex(3)


def _verify_staged_generation(staging: Path) -> None:
    """发布前验证 staging 非空且至少有一个 parquet（fail-closed，不发布空 generation）。"""
    if not staging.exists() or not staging.is_dir():
        raise ValidationError(f"mirror staging 不存在或非目录: {staging}")
    parquet_files = list(staging.rglob("*.parquet"))
    if not parquet_files:
        raise ValidationError(
            f"mirror staging 无任何 parquet 文件，拒绝发布空 generation: {staging}"
        )


def _write_generation_pointer(
    mirror_root: Path, generation_id: str, source_generation: str | None
) -> None:
    """原子写 current pointer（generation/current.json），reader 只 resolve 它。"""
    import time as _time

    gen_dir = mirror_root / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_generation": generation_id,
        "source_generation": source_generation,
        "published_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    tmp = gen_dir / "current.json.tmp"
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(gen_dir / "current.json"))


def resolve_current_mirror_generation(mirror_root: Path) -> str | None:
    """reader：读取当前 mirror generation pointer（R25 P0-014）。"""
    pointer = mirror_root / "generation" / "current.json"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        return payload.get("current_generation")
    except (OSError, json.JSONDecodeError):
        return None


def _dir_object_count(root: Path) -> int:
    try:
        return sum(1 for _ in root.rglob("*.parquet"))
    except OSError:
        return 0


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
    if spec.layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/002/016：period/event/prefixed 布局走完整目录 sync（不能用
        # request time_range 展开物理文件名）。StockCapital 混放目录会同时拉到
        # split 与 shares 两类文件——schema 不同，读取端用 file_selector 区分。
        _sync_dir_full(spec)
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
        for day in expected_partitions(dataset_name, start, end, layout=spec.layout):
            _sync_daily_file(spec, day, dataset_name=dataset_name)
    elif spec.layout == "hive_date":
        for day in expected_partitions(dataset_name, start, end, layout=spec.layout):
            _sync_hive_date(spec, day)
    elif spec.layout == "hive_year":
        for year in expected_partitions(dataset_name, start, end, layout=spec.layout):
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

    if spec.layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/002/016：period/event/prefixed 布局无法按 request time_range
        # 增量判断（partition_clock != predicate_clock）。目录为空/无 manifest 时
        # 直接完整目录 sync（宁可多扫不能漏）。存在 verified 文件即视为已同步。
        local_dir = _local_table_dir(spec) if spec.table else spec.local_root
        has_verified = False
        if local_dir.exists() and local_dir.is_dir():
            for p in local_dir.glob("*.parquet"):
                if _local_file_fresh(p):
                    has_verified = True
                    break
        if not has_verified:
            logger.info("cos_mirror: period/event 布局完整目录 sync %s", dataset_name)
            _sync_dir_full(spec)
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
            _sync_daily_file(spec, day, dataset_name=dataset_name)
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
    # #P0-15 门：数据集 root 必须落在已知 COS 镜像根下才触发 sync。
    # 旧实现按数据集**名字**查全局 mirror registry——测试/自选目录数据集同名但
    # root 在本地（非镜像）也会误触发 COS 拉取，还会因 dense 行情缺 partition
    # 误判 MissingRequiredPartition。
    if not dataset_root_requires_cos_mirror(ds):
        return
    ensure_local_mirror(ds.name, time_range=time_range)
