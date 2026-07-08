"""
data_access.store —— 对外唯一数据读写入口

对外 API（PR1 读 + PR2 写 + PR3 publish/upsert/sql）：
    read_arrow(dataset, columns=, time_range=, instrument_filter=, **params) -> pa.Table
    read_frame(dataset, ...) -> pd.DataFrame
    load_columns(dataset, keys, columns, time_range=, instrument_filter=, **params) -> dict[str, pd.Series]
    write_arrow(dataset, table, mode="overwrite"|"append", partition_by=None, **params) -> dict
    upsert(dataset, table, *, upsert_on=, partition_by=None, **params) -> dict          [PR3]
    publish_from_staging(staging, target, **params) -> dict                              [PR3]
    sql(query, *, read_datasets, read_params=None, params=None) -> pa.Table              [PR3, 有限]

仍未开放：
    流式 open_writer / 任意 DDL — PR4+

非职责：
    - 不做清洗（raw_data_layer 的事）
    - 不做因子计算（factor_engine 的事）
    - 不解析 YAML（registry.py 的事）

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa

if TYPE_CHECKING:
    import polars as pl

from . import audit
from .adapters import arrow_table_to_multiindex_columns
from .engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from .exceptions import DataError, ValidationError
from .namespace import is_namespace_explicit, resolve_namespace
from .paths import PathAuthorizer
from .predicate import Predicate, compile_predicate
from .schema_validation import (
    check_schema,
    enforce_schema_or_raise,
    mark_validated,
    reset_validated_cache,
)
from .registry import (
    Dataset,
    DatasetRegistry,
    ParametricDataset,
    StaticDataset,
    load_registry,
)
from .telemetry import record_polars_scan


logger = logging.getLogger("data_access.store")


_VALID_WRITE_MODES = {"overwrite", "append"}


def _assert_instrument_filter_supported(
    ds: Dataset,
    instrument_filter: Sequence[str] | None,
) -> None:
    """schema 已声明但缺少 instrument 列时，禁止 instrument_filter。"""
    if not instrument_filter:
        return
    schema = getattr(ds, "schema", None) or {}
    if schema and ds.instrument_column not in schema:
        raise ValidationError(
            f"数据集 '{ds.name}' 的 instrument_column='{ds.instrument_column}' "
            f"不在 schema 中，不支持 instrument_filter；请去掉 filter 或补 schema。"
        )


class DataAccessStore:
    """统一读取入口。进程内应只有一个实例（由 get_store() 管理）。"""

    def __init__(
        self,
        registry: DatasetRegistry,
        engine: DuckDBEngine,
    ) -> None:
        self._registry = registry
        self._engine = engine
        self._authorizer = PathAuthorizer(registry.allowed_roots())
        # PR8：首访 schema 自检的进程内缓存。同 dataset 只校验一次。
        # 说明：缓存 key 是 dataset 名而非 (name, params)——假设同名 dataset 的
        # 实际 schema 在不同参数下是一致的（factor_lake 不同 factor_id 列应相同）。
        # 如果未来出现 per-params schema 漂移的极端场景，再细化缓存 key。
        self._schema_checked: set[str] = set()

    # ---- 对外主 API ----

    def read_arrow(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        **params: Any,
    ) -> pa.Table:
        """读取已注册数据集，返回 Arrow Table（零拷贝，推荐路径）。

        参数：
            dataset: datasets.yaml 里注册的数据集名
            columns: 只读指定列（强烈推荐指定，避免宽表全扫）
            time_range: (start, end) 闭区间，按 registry 里的 time_column 过滤
            instrument_filter: 标的白名单
            **params: 参数化数据集的参数（如 factor_id=）

        返回：
            pa.Table；列顺序与 columns 一致；若 columns=None 则为 SELECT *。

        常见错误：
            ValidationError: 数据集未注册 / 参数缺失 / 路径越界
            DataError: 读出的结果为空（time_range 筛没了 → 通常是上游 bug）
            EngineError: DuckDB 内部错

        示例：
            >>> store.read_arrow(
            ...     "us_stocks_sip_day_aggs",
            ...     columns=["align_time", "ticker", "close"],
            ...     time_range=("2024-01-01", "2024-12-31"),
            ...     instrument_filter=["AAPL", "MSFT"],
            ... )
        """
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        paths = self._prepare_dataset_read(ds, time_range=time_range, params=params)
        self._ensure_schema(ds, paths)

        sql, sql_params = self._build_select_sql(
            ds=ds,
            paths=paths,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
        )

        start = time.perf_counter()
        table = self._engine.execute_arrow(sql, sql_params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "read_arrow dataset=%s rows=%d cols=%d elapsed_ms=%.1f",
            dataset, table.num_rows, table.num_columns, elapsed_ms,
        )
        return table

    def read_arrow_stream(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        batch_size: int = 100_000,
        **params: Any,
    ) -> Iterator[pa.RecordBatch]:
        """流式读取（PR7）：返回一个 Arrow RecordBatch 生成器。

        使用场景：
            - 单次查询结果大到不愿全部 materialize 到内存（几十 GB 回测输出）
            - ETL / 逐 batch 聚合后直接写下游（中间结果不留内存）
            - 处理慢但是连续可分：生成信号 → 发下游 → 丢 batch 继续读

        与 read_arrow 的取舍：
            read_arrow 一次全拿 = 零拷贝直接给 pandas/polars，延迟高但吞吐好；
            read_arrow_stream 按 batch = 延迟低、内存峰值平，但每 batch 有
            少量 overhead（cursor 状态 + batch 拼装）。百 MB 级别以内用 read_arrow，
            多 GB 扫描用 read_arrow_stream。

        参数：
            batch_size: 每个 RecordBatch 目标行数；默认 100k（与 DuckDB vector
                size 对齐）。太小会浪费向量化收益，太大内存波动大。

        注意：
            - 生成器用 for 循环消费；迭代完自动释放底层 cursor。
            - 中途 break 会让 Python GC 清理 reader，cursor 也会跟着释放。
            - 不要跨线程传 reader；DuckDB 的 record batch reader 线程不安全。

        示例：
            >>> total = 0
            >>> for batch in store.read_arrow_stream("factor_lake", factor_id="mom_3d"):
            ...     total += batch.num_rows
        """
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        paths = self._prepare_dataset_read(ds, time_range=time_range, params=params)
        self._ensure_schema(ds, paths)
        sql, sql_params = self._build_select_sql(
            ds=ds,
            paths=paths,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
        )
        reader = self._engine.execute_reader(sql, sql_params, batch_size=batch_size)
        logger.info(
            "read_arrow_stream dataset=%s batch_size=%d params=%s",
            dataset, batch_size, params or "{}",
        )
        # 包一层 iterator，便于调用方 `for batch in ...` 语义（reader 本身也可以迭代）
        for batch in reader:
            yield batch

    def scan_polars(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        **params: Any,
    ) -> "pl.LazyFrame":
        """返回 Polars LazyFrame，惰性扫描已注册数据集（PR7）。

        为什么提供：Polars 的 lazy engine 是 Rust 实现的 pushdown（列/谓词/
        partition 剪枝全套），对「多表 join + 窗口函数 + 大量复杂变换」远优于
        DuckDB 的 Arrow 结果 → pandas → polars 路径。尤其是跑复杂因子 pipeline
        时用 LazyFrame 能让 Polars 自己决定什么时候物化。

        我们做什么 / 不做什么：
            我们做：解析 registry 路径 + PathAuthorizer + 传 hive_partitioning
                    给 Polars；把 time_range / instrument_filter 转成 lazy
                    `.filter(...)` 附在 LazyFrame 上，供下游 pushdown 使用。
            我们不做：不替 Polars 做执行计划（LazyFrame 还没 .collect() 前不读数据）。
            遥测：记录 scan 建图 + schema 自检耗时（不含下游 .collect()）。

        参数：
            columns: 传给 pl.scan_parquet 的 n_rows / columns 优化；
                     None 表示不提前选列，完全交给 Polars optimizer
            time_range / instrument_filter: 转成 LazyFrame.filter 附加上

        示例：
            >>> lf = store.scan_polars("factor_lake", factor_id="mom_3d",
            ...                         time_range=("2024-01-01", None))
            >>> df = lf.filter(pl.col("asset").is_in(["AAPL"])).collect()

        依赖：
            需要 `polars` 已安装；没装会 raise ImportError 并提示 `pip install polars`。
        """
        try:
            import polars as pl_mod
        except ImportError as exc:
            raise ImportError(
                "scan_polars 需要 polars。pip install polars 后重试。"
            ) from exc

        scan_start = time.perf_counter()
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        paths = self._prepare_dataset_read(ds, time_range=time_range, params=params)
        # scan_polars 也走 schema 自检：发现声明漂移尽早报。
        # Polars 自己读 parquet 不经 DuckDB，但 DESCRIBE 用共享 engine 跑，一次性
        # 开销 < 50ms（只读 footer），比跑完一次 collect 才炸便宜得多。
        self._ensure_schema(ds, paths)

        # pl.scan_parquet 可以接 list[str]，也支持 glob。我们传 list 给它。
        # 跨年份列不一致时靠 union_by_name：
        #   - Polars 1.30+：参数是 missing_columns='insert'/'raise'，老的 allow_missing_columns 被弃用
        #   - Polars 1.28 ~ 1.29：参数名是 allow_missing_columns
        # 用 try/except 兜两边，避免硬编码版本号判断。
        scan_kwargs: dict[str, Any] = {"hive_partitioning": ds.hive_partitioning}
        try:
            lf = pl_mod.scan_parquet(
                paths,
                missing_columns="insert" if ds.union_by_name else "raise",
                **scan_kwargs,
            )
        except TypeError:
            lf = pl_mod.scan_parquet(
                paths,
                allow_missing_columns=ds.union_by_name,
                **scan_kwargs,
            )

        # 仅选列：给 Polars optimizer 做 column pushdown
        # time/instrument 列如果 columns 里没包含但有过滤条件，先留着让 filter
        # 用上，最后 Polars 自己会剪掉
        needed_cols: list[str] = []
        if columns:
            needed_cols.extend(columns)
            if time_range is not None and ds.time_column not in needed_cols:
                needed_cols.append(ds.time_column)
            if instrument_filter is not None and ds.instrument_column not in needed_cols:
                needed_cols.append(ds.instrument_column)
            lf = lf.select([pl_mod.col(c) for c in needed_cols])

        # Lazy filter：time_range + instrument_filter
        # 注意 Polars 不像 DuckDB 能把 '2024-01-01' 这种字符串自动 cast 成 datetime，
        # 对 datetime 列比较字符串会抛 InvalidOperationError。统一用 pandas.Timestamp
        # 把字符串 / datetime / date 归一化到 python datetime，再交给 pl.lit——
        # 这样无论下层是 Datetime 还是 Date 列，Polars 自己都能 coerce。
        if time_range is not None:
            start_val, end_val = time_range
            if start_val is not None:
                lf = lf.filter(
                    pl_mod.col(ds.time_column) >= pl_mod.lit(_to_pydatetime(start_val))
                )
            if end_val is not None:
                lf = lf.filter(
                    pl_mod.col(ds.time_column) <= pl_mod.lit(_to_pydatetime(end_val))
                )
        if instrument_filter is not None:
            lf = lf.filter(pl_mod.col(ds.instrument_column).is_in(list(instrument_filter)))

        # 如果 columns 指定了但我们追加过 time/instrument，在 filter 之后再把
        # 原始 columns 剪回来（filter 用过了就可以丢）
        if columns and needed_cols != list(columns):
            lf = lf.select([pl_mod.col(c) for c in columns])

        record_polars_scan(
            dataset=dataset,
            elapsed_ms=(time.perf_counter() - scan_start) * 1000,
            paths_count=len(paths),
        )
        return lf

    def read_frame(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        **params: Any,
    ):
        """同 read_arrow，但返回 pandas DataFrame。

        WHY 单独提供 read_frame：最末端要画图/落 CSV 的场景很多，每次都让
        调用方 .to_pandas() 也是重复劳动；但 pipeline 中段仍推荐 read_arrow。
        """
        table = self.read_arrow(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            **params,
        )
        return table.to_pandas(self_destruct=True, split_blocks=True)

    def load_columns(
        self,
        dataset: str,
        *,
        columns: Sequence[str],
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        output_names: dict[str, str] | None = None,
        normalize_timestamp: bool | None = None,
        timestamp_unit: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """批量读多列，每列转成 `(timestamp, instrument)` MultiIndex Series。

        这是 ParquetSource.load_column 的「批量版」，是 PR1 性能关键 API：
        老代码每列都重开 parquet、重读 ts/instrument，这里一次 SQL 拿齐，
        再在内存里按列拆分。

        参数：
            columns: 要读取的值列（不包括 time/instrument，它们 registry 里有）
            output_names: 可选列名映射 {源列: 目标 Series name}
            normalize_timestamp / timestamp_unit: 传给 adapter
            **params: 参数化数据集参数

        返回：
            {output_name: pd.Series}；Series 的 name 是 output_name 或源列名

        示例：
            >>> cols = store.load_columns(
            ...     "us_stocks_sip_day_aggs",
            ...     columns=["close", "volume", "open"],
            ...     time_range=("2024-01-01", "2024-12-31"),
            ... )
            >>> cols["close"].head()
        """
        ds = self._registry.get(dataset)
        output_names = output_names or {}
        adapter_opts = adapter_options_for_dataset(ds)
        if normalize_timestamp is None:
            normalize_timestamp = bool(adapter_opts.get("normalize_timestamp", False))
        if timestamp_unit is None:
            timestamp_unit = adapter_opts.get("timestamp_unit")

        # 一次 SQL 同时选所有需要的列 + 时间列 + 标的列
        all_cols = list(dict.fromkeys(
            [ds.time_column, ds.instrument_column, *columns]
        ))
        table = self.read_arrow(
            dataset,
            columns=all_cols,
            time_range=time_range,
            instrument_filter=instrument_filter,
            **params,
        )

        if table.num_rows == 0:
            raise DataError(
                f"数据集 '{dataset}' 在给定条件下读出 0 行；"
                f"检查 time_range / instrument_filter 或数据是否真的存在"
            )

        # 批量路径：一次 Arrow→pandas，多列拆分（避免每列重复转换）
        reverse_names = {src: tgt for src, tgt in output_names.items()}
        result = arrow_table_to_multiindex_columns(
            table,
            timestamp_column=ds.time_column,
            instrument_column=ds.instrument_column,
            value_columns=list(columns),
            output_names=reverse_names or None,
            normalize_timestamp=normalize_timestamp,
            timestamp_unit=timestamp_unit,
        )
        # output_names 映射的是 physical→logical，批量函数 key 用 target name
        if output_names:
            return {
                output_names.get(src, src): result[output_names.get(src, src)]
                for src in columns
            }
        return result

    # ---- 写入 API（PR2） ----

    def write_arrow(
        self,
        dataset: str,
        table: pa.Table,
        *,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """把 Arrow Table 写到已注册的数据集。只允许 namespaced/staging 两种 access_mode。

        参数：
            dataset: datasets.yaml 里登记的数据集名
            table: 要写入的 Arrow Table
            mode: "overwrite"（先清目标目录再写）| "append"（加新文件不动旧的）
            partition_by: 分区列，如 ["year"]。传了之后用 hive 布局落盘
            **params: 参数化数据集的参数（如 factor_id=）

        返回：
            {"rows": int, "path": str, "mode": str}，同时写一行审计日志

        拒绝规则：
            - published 数据集：直写永远 raise ValidationError，必须走 publish 流程（PR3）
            - namespace 兜底：写 namespaced/staging 时若未显式 export QUANT_RUN_NAMESPACE，
              只 warning 不拦（避免阻塞脚本调试），但日志里标记 namespace_explicit=false

        示例：
            >>> import pyarrow as pa
            >>> tbl = pa.table({"ts": [...], "symbol": [...], "pnl": [...]})
            >>> store.write_arrow(
            ...     "single_asset_backtest_runs",
            ...     tbl,
            ...     strategy_id="mom_3d",
            ...     version="v1",
            ...     mode="overwrite",
            ... )
        """
        if mode not in _VALID_WRITE_MODES:
            raise ValidationError(
                f"write_arrow mode 必须是 {_VALID_WRITE_MODES}，收到 {mode!r}"
            )
        if not isinstance(table, pa.Table):
            raise ValidationError(
                f"write_arrow 只接受 pyarrow.Table，收到 {type(table).__name__}；"
                f"如有 DataFrame 请先 pyarrow.Table.from_pandas(df)"
            )

        ds = self._registry.get(dataset)

        if ds.access_mode == "published":
            raise ValidationError(
                f"数据集 '{dataset}' 是 published，不允许直写。"
                f"请写对应的 staging 数据集，再用 publish_from_staging 晋升（PR3）"
            )
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' 的 access_mode={ds.access_mode!r} 不支持写入"
            )

        if not is_namespace_explicit():
            logger.warning(
                "写入 namespaced/staging 数据集 '%s' 但 QUANT_RUN_NAMESPACE 未显式设置，"
                "用的是兜底 namespace=%s。生产脚本请 export。",
                dataset, resolve_namespace(),
            )

        target_dir = self._resolve_write_dir(ds, params)
        self._authorizer.resolve_and_authorize(str(target_dir))

        with audit.AuditTimer() as timer:
            try:
                if mode == "overwrite":
                    self._clear_dir(target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                files_written = self._write_table_to_dir(
                    table, target_dir, partition_by=partition_by,
                )
                ok = True
                err_msg = None
            except Exception as exc:
                ok = False
                err_msg = f"{type(exc).__name__}: {exc}"
                files_written = []
                # 出错仍然先记审计再抛，方便事后追查
                raise
            finally:
                audit.record(
                    op="write",
                    dataset=dataset,
                    ok=ok if 'ok' in locals() else False,
                    mode=mode,
                    rows=table.num_rows,
                    paths=[str(p) for p in files_written] if files_written else [str(target_dir)],
                    params=params or None,
                    elapsed_ms=timer.elapsed_ms,
                    error=err_msg if not ok else None,
                    extra={"partition_by": list(partition_by)} if partition_by else None,
                )

        logger.info(
            "write_arrow dataset=%s rows=%d mode=%s files=%d elapsed_ms=%.1f",
            dataset, table.num_rows, mode, len(files_written), timer.elapsed_ms,
        )
        return {
            "rows": table.num_rows,
            "path": str(target_dir),
            "files": [str(p) for p in files_written],
            "mode": mode,
        }

    def upsert(
        self,
        dataset: str,
        table: pa.Table,
        *,
        upsert_on: Sequence[str],
        partition_by: Sequence[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """读-合-写的幂等合并写入。只允许 namespaced/staging。

        参数：
            dataset: datasets.yaml 里登记的 namespaced/staging 数据集
            table: 新增/更新的数据；列里必须包含 upsert_on + partition_by
            upsert_on: 合并键（必填）。同 upsert_on 组合已存在 → 新值覆盖旧值
            partition_by: 分区列。传了之后按 hive 布局逐分区合并（每分区一个
                data.parquet，合并时做 read-merge-tmp-rename 原子写）
            **params: 参数化数据集的参数（如 factor_id=）

        返回：
            {"rows": int, "path": str, "partitions": list[str], "elapsed_ms": float}

        与 write_arrow 的区别：
            - write_arrow(overwrite) 把目标目录清空再整写；upsert 保留旧数据，
              按 upsert_on 合并
            - write_arrow(append) 单纯加文件，不处理重复；upsert 会 dedup
            - upsert 按分区做读-合-写，比 overwrite 贵，但避免了全量重算

        示例：
            >>> store.upsert(
            ...     "factor_lake_staging", tbl,
            ...     factor_id="mom_3d",
            ...     upsert_on=["datetime", "asset"],
            ...     partition_by=["year"],
            ... )
        """
        if not isinstance(table, pa.Table):
            raise ValidationError(
                f"upsert 只接受 pyarrow.Table，收到 {type(table).__name__}；"
                f"DataFrame 请先 pyarrow.Table.from_pandas(df)"
            )

        ds = self._registry.get(dataset)

        if ds.access_mode == "published":
            raise ValidationError(
                f"数据集 '{dataset}' 是 published，不允许 upsert。"
                f"请 upsert 对应的 staging 数据集，再 publish_from_staging 晋升"
            )
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' 的 access_mode={ds.access_mode!r} 不支持 upsert"
            )

        if not is_namespace_explicit():
            logger.warning(
                "upsert 到 namespaced/staging 数据集 '%s' 但 QUANT_RUN_NAMESPACE 未显式设置，"
                "用的是兜底 namespace=%s。生产脚本请 export。",
                dataset, resolve_namespace(),
            )

        target_dir = self._resolve_write_dir(ds, params)

        from . import upsert as upsert_mod

        return upsert_mod.upsert_table(
            ds=ds,
            authorizer=self._authorizer,
            target_dir=target_dir,
            new_table=table,
            upsert_on=upsert_on,
            partition_by=partition_by,
            params=params,
        )

    def delete_rows(
        self,
        dataset: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
        after: Any | None = None,
        time_column: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """从 namespaced/staging 数据集删除时间范围内的行（行级补偿删除）。"""
        ds = self._registry.get(dataset)
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' access_mode={ds.access_mode!r} 不支持 delete_rows"
            )
        tc = time_column or ds.time_column
        target_dir = self._resolve_write_dir(ds, params)
        from . import upsert as upsert_mod

        return upsert_mod.delete_rows_from_dataset(
            ds=ds,
            authorizer=self._authorizer,
            target_dir=target_dir,
            time_column=tc,
            start=start,
            end=end,
            after=after,
            params=params,
        )

    def resolve_dataset_path(self, dataset: str, **params: Any) -> Path:
        """解析已登记数据集在当前 params 下的物理目录。"""
        from .publish import _resolve_dataset_dir

        ds = self._registry.get(dataset)
        return _resolve_dataset_dir(ds, params)

    def dataset_axis_columns(self, dataset: str) -> tuple[str, str]:
        """返回数据集的时间列与标的列名（供 factor_engine SQL 下推使用）。"""
        ds = self._registry.get(dataset)
        return ds.time_column, ds.instrument_column

    def publish_from_staging(
        self,
        staging_dataset: str,
        target_dataset: str,
        **params: Any,
    ) -> dict[str, Any]:
        """把 staging 数据集的内容晋升为 published 版本（PR3 实装）。

        参数：
            staging_dataset: 源 staging 数据集（access_mode=staging）
            target_dataset:  目标 published 数据集（access_mode=published）
            **params:        参数化数据集的参数（两边必须接受同一组）

        返回：
            {"source": {...}, "target_path": str, "archive_path": str | None,
             "rows": int, "elapsed_ms": float}

        行为：
            1. 校验两数据集的 access_mode / schema / params_schema 匹配
            2. 把 staging 的内容 copytree 到 published 父目录下的候选目录
            3. 加并发锁；把旧 published（如存在）rename 到 _archive/...
            4. rename 候选 → published，做发布后读取验证
            5. 任何阶段失败都尽力回滚到一致状态；全程写审计日志

        示例：
            >>> store.write_arrow("factor_lake_staging", tbl, factor_id="mom_3d")
            >>> store.publish_from_staging(
            ...     "factor_lake_staging", "factor_lake",
            ...     factor_id="mom_3d",
            ... )
        """
        from . import publish

        return publish.publish_from_staging(
            registry=self._registry,
            authorizer=self._authorizer,
            staging_name=staging_dataset,
            target_name=target_dataset,
            **params,
        )

    def sql(
        self,
        query: str,
        *,
        read_datasets: Sequence[str],
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        params: Sequence[Any] | None = None,
    ) -> pa.Table:
        """有限 SQL 逃生口：只允许 SELECT，FROM 的表必须是 read_datasets 里预声明的数据集。

        参数：
            query: 用户提供的 SELECT（可含 WITH / JOIN / GROUP BY 等）。
                禁用 INSERT/UPDATE/DELETE/COPY/ATTACH/PRAGMA/read_parquet 等关键字。
            read_datasets: 本次查询要访问的数据集名列表（必填，非空）。
                每个数据集会以其注册名作为 TEMP VIEW 暴露给 query 的 FROM。
            read_params: 参数化数据集的参数 {dataset_name: {param: value}}，
                例如 {"factor_lake": {"factor_id": "mom_3d"}}
            params: query 自身的 ? 绑定参数（用户层面的查询参数，不是路径）

        返回：
            pa.Table —— 用户 SQL 的结果

        示例：
            >>> tbl = store.sql(
            ...     "SELECT asset, AVG(value) FROM factor_lake "
            ...     "WHERE datetime >= ? GROUP BY asset",
            ...     read_datasets=["factor_lake"],
            ...     read_params={"factor_lake": {"factor_id": "mom_3d"}},
            ...     params=["2024-01-01"],
            ... )

        为什么要这一层：
            大部分临时分析需求（GROUP BY / window / 多表 JOIN）用标准 read_arrow
            表达不了，但我们又不想开放裸 read_parquet —— 路径白名单/审计/未来的
            配额限流就没着落了。这里通过 TEMP VIEW 收敛：你能读的表仅限 registry
            注册的，路径照旧走 PathAuthorizer，每次执行都记审计。
        """
        from . import sql_escape

        return sql_escape.run_sql(
            registry=self._registry,
            authorizer=self._authorizer,
            engine=self._engine,
            query=query,
            read_datasets=read_datasets,
            read_params=read_params,
            params=params,
            build_select_sql=self._build_select_sql,
            resolve_paths=self._resolve_paths,
        )

    # ---- 内部 helpers ----

    def _prepare_dataset_read(
        self,
        ds: Dataset,
        *,
        time_range: tuple[Any, Any] | None,
        params: dict[str, Any],
    ) -> list[str]:
        """COS 镜像 + 路径解析（read_arrow / stream / scan_polars 共用）。"""
        from .cos_mirror import ensure_local_mirror_for_dataset

        ensure_local_mirror_for_dataset(ds, time_range=time_range)
        return self._resolve_paths(ds, params)

    def _ensure_schema(self, ds: Dataset, paths: list[str]) -> None:
        """首次访问 dataset 时做一次 schema 对齐校验（PR8）。

        缓存策略：同 dataset 只校验一次，无论成功失败。失败时：
          - strict 模式：抛 ValidationError，不入缓存（下次还会再校验）
          - warn 模式：打日志入缓存（只叫一次，不刷屏）
          - off 模式：跳过

        为什么失败 strict 下不入缓存：让调用方自己决定修不修，修了再试一次就对了；
        不缓存失败态避免「本地修了 yaml，进程没重启 ⇒ 还是 raise」。
        """
        if not getattr(ds, "schema", None):
            return
        if ds.name in self._schema_checked:
            return

        result = check_schema(self._engine, ds, paths)
        if result.ok:
            self._schema_checked.add(ds.name)
            mark_validated(ds.name)
            return

        # 失败：按模式处理。strict 会 raise；warn 只记一次。
        from .schema_validation import _resolve_mode

        mode = _resolve_mode()
        try:
            enforce_schema_or_raise(result, mode=mode)
        except Exception:
            # strict: 不缓存，让调用方修好 yaml/数据后重试
            raise
        # warn / off：把 dataset 标为已检查，避免每次读都刷日志
        self._schema_checked.add(ds.name)
        if mode == "warn":
            mark_validated(ds.name)

    def _resolve_paths(self, ds: Dataset, params: dict[str, Any]) -> list[str]:
        """把 dataset + params 解析成传给 DuckDB 的 path glob 列表，并做白名单校验。"""
        glob_paths = ds.resolve_paths(**params)
        # 对每个 glob 的静态前缀部分做白名单校验；DuckDB 展开 glob 时
        # 不会跑出这个前缀，所以等价于对展开后的所有文件校验。
        for g in glob_paths:
            static_part = g.split("*", 1)[0].rstrip("/")
            if static_part:
                self._authorizer.resolve_and_authorize(static_part)
        return glob_paths

    def _resolve_write_dir(self, ds: Dataset, params: dict[str, Any]) -> Path:
        """write 的目标目录 = 把 glob 按 '/' 分段，取第一个含 '*' 的段之前的所有段。

        例子：
            '/xx/runs/mom_3d/**/*.parquet'      → '/xx/runs/mom_3d'
            '/xx/factors/mom_3d/year=*/*.parquet' → '/xx/factors/mom_3d'
                （hive 分区 year= 由 pyarrow.dataset 写入时生成，不属于写入根）
            '/xx/factors/mom_3d/*.parquet'       → '/xx/factors/mom_3d'
        """
        glob_paths = ds.resolve_paths(**params)
        if len(glob_paths) != 1:
            raise ValidationError(
                f"数据集 '{ds.name}' 解析出 {len(glob_paths)} 个 glob，"
                f"write_arrow 只支持唯一目标目录的数据集"
            )
        glob = glob_paths[0]
        segments = glob.split("/")
        # 找第一个含通配符的段
        clean_segments: list[str] = []
        for seg in segments:
            if "*" in seg or "?" in seg:
                break
            clean_segments.append(seg)
        if not clean_segments:
            raise ValidationError(f"数据集 '{ds.name}' 的 glob 无静态前缀: {glob}")
        return Path("/".join(clean_segments))

    @staticmethod
    def _clear_dir(path: Path) -> None:
        """overwrite 模式下清空目标目录。不存在就跳过，不递归到白名单外。"""
        if not path.exists():
            return
        if not path.is_dir():
            raise ValidationError(f"目标路径不是目录，不能 overwrite: {path}")
        # 逐项删（比 rmtree 稍慢但更安全，避免 symlink 逃逸）
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    @staticmethod
    def _write_table_to_dir(
        table: pa.Table,
        target_dir: Path,
        *,
        partition_by: Sequence[str] | None,
    ) -> list[Path]:
        """把 Arrow Table 写成 parquet；带分区用 pyarrow.dataset。

        不带 partition_by：写成 `{dir}/part-{uuid}.parquet` 单文件。
            append 时 uuid 保证不撞，overwrite 时目录已清空。
        带 partition_by：用 pyarrow.dataset.write_dataset 生成 hive 分区。
        """
        import pyarrow.parquet as pq

        target_dir.mkdir(parents=True, exist_ok=True)

        if partition_by:
            # 写分区：用 pyarrow.dataset 的 hive partitioning
            import pyarrow.dataset as pads

            # basename_template 必须含 "{i}"；用 uuid 前缀避免 append 时同分区下撞文件
            prefix = uuid.uuid4().hex[:8]
            pads.write_dataset(
                table,
                base_dir=str(target_dir),
                format="parquet",
                partitioning=list(partition_by),
                partitioning_flavor="hive",
                existing_data_behavior="overwrite_or_ignore",
                basename_template=f"part-{prefix}-{{i}}.parquet",
            )
            # 返回新写入的文件列表
            return sorted(target_dir.rglob("*.parquet"))

        # 单文件写：先写 .tmp 再 rename，保证原子
        out_name = f"part-{uuid.uuid4().hex[:8]}.parquet"
        out_path = target_dir / out_name
        tmp_path = target_dir / f".{out_name}.tmp"
        pq.write_table(table, tmp_path)
        os.replace(str(tmp_path), str(out_path))
        return [out_path]

    def _build_select_sql(
        self,
        *,
        ds: Dataset,
        paths: list[str],
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
    ) -> tuple[str, list[Any]]:
        """组装 SELECT 语句。路径用 ? 参数绑定，列名/谓词用 registry 控制。"""
        if columns:
            col_clause = ", ".join(_quote_ident(c) for c in columns)
        else:
            col_clause = "*"

        # read_parquet 的 path 参数：单路径直接 ?，多路径用 list
        # DuckDB 支持 read_parquet([...], hive_partitioning=..., union_by_name=...)
        # 注意 hive_partitioning / union_by_name 是 read_parquet 的命名参数，
        # 不能用 ? 占位符传（DuckDB SQL 语法限制），必须直接拼到 SQL 里
        # —— 这些值来自 registry，可信。
        path_param = paths if len(paths) > 1 else paths[0]

        read_parquet_opts = []
        if ds.hive_partitioning:
            read_parquet_opts.append("hive_partitioning=true")
        if ds.union_by_name:
            read_parquet_opts.append("union_by_name=true")
        opts_suffix = (", " + ", ".join(read_parquet_opts)) if read_parquet_opts else ""

        from_clause = f"read_parquet(?{opts_suffix})"

        predicate = Predicate(
            time_range=time_range,
            instrument_filter=instrument_filter,
        )
        compiled = compile_predicate(
            predicate,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
        )

        sql = f"SELECT {col_clause} FROM {from_clause} {compiled.where_sql}".strip()
        params: list[Any] = [path_param, *compiled.params]
        return sql, params


# ---- 进程单例管理 ----

_store: DataAccessStore | None = None
_store_lock = threading.Lock()


def get_store() -> DataAccessStore:
    """获取进程级 DataAccessStore 实例。第一次调用会初始化 registry + engine。

    线程安全：双重检查 + Lock。
    """
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        registry = load_registry()
        # 共享 DuckDBEngine：ParquetSource 也用同一个，buffer pool 复用
        engine = get_shared_engine()
        _store = DataAccessStore(registry=registry, engine=engine)
        logger.info(
            "DataAccessStore 初始化：%d 个数据集注册",
            len(registry.names()),
        )
        return _store


def adapter_options_for_dataset(ds: Dataset) -> dict[str, Any]:
    """按 datasets.yaml schema 推断 Arrow→MultiIndex 适配参数（A 股 date / 美股 timestamp）。"""
    schema = getattr(ds, "schema", None) or {}
    if not schema:
        return {}
    time_dtype = str(schema.get(ds.time_column, "")).lower()
    opts: dict[str, Any] = {}
    if time_dtype in {"date", "timestamp"}:
        opts["normalize_timestamp"] = True
    return opts


def reset_store() -> None:
    """主要给测试用。重置进程单例（会丢弃 DuckDB 连接里的缓存）。"""
    global _store
    with _store_lock:
        _store = None
    reset_shared_engine()
    reset_validated_cache()


def _quote_ident(name: str) -> str:
    """本地复制一份列名引用，避免 store 依赖 predicate 的私有函数。"""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _to_pydatetime(value: Any):
    """把 time_range 的端点统一成 python `datetime.datetime`，方便 Polars coerce。

    为什么：Polars 对 Datetime/Date 列不做 str->timestamp 自动 cast，
    直接 `pl.lit("2024-01-01")` 会报 InvalidOperationError。DuckDB 则会自动 cast，
    所以 read_arrow 这条路走字符串没问题，只有 scan_polars 需要转。

    支持的输入：
        - 字符串（走 pd.Timestamp 解析）
        - `datetime.datetime` / `pandas.Timestamp`：原样返回 .to_pydatetime()
        - `datetime.date`：直接返回（polars Date 列可以直接比）
        - int / float：视为 unix 秒或毫秒时戳是 DuckDB 专属语义，
          scan_polars 里不做猜测，原样丢回让 Polars 自己报错
    """
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        import pandas as _pd  # 本地 import：pd 只在这个分支用，避免拉顶层依赖

        return _pd.Timestamp(value).to_pydatetime()
    # 其它类型交给 polars，它会给一个更有信息量的错误
    return value
