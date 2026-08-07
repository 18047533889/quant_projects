"""
data_access.engine —— DuckDB 连接与 PRAGMA 管理

职责：
    1. 进程内单例 DuckDB 实例（in-memory）
    2. 启动时设置性能相关 PRAGMA（threads / memory_limit / object_cache）
    3. 暴露线程安全的查询接口 execute_arrow / execute_df / explain

设计要点（重要，PR 里还会反复提）：
    1. 一个进程一个 duckdb.connect(":memory:")，避免多实例各自 footer 缓存
    2. 多线程执行查询时用 conn.cursor() 分支出独立执行上下文，
       buffer pool / catalog / PRAGMA 共享
    3. PRAGMA enable_object_cache=true —— parquet footer 跨查询复用，
       factor_engine 重复读同一批文件时收益显著
    4. threads 默认取 min(可用 CPU 数, DUCKDB_MAX_THREADS)；memory_limit 默认按主机可用内存自动计算，
       可用 DUCKDB_MEMORY_LIMIT 显式覆盖；无法探测时保持 DuckDB 默认行为

非职责：
    不负责 SQL 组装（store.py）、不负责 predicate 编译（predicate.py）、
    不负责读 YAML（registry.py）。

维护人：quant 基础平台组    最后更新：2026-07-09
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator, Sequence

import duckdb
import pyarrow as pa

from .duckdb_config import DuckDBConfig, apply_pragmas, resolve_duckdb_config
from .exceptions import EngineError
from .retry import retry_io
from data_access.read.telemetry import maybe_log_slow_query_plan, record_query


logger = logging.getLogger("data_access.engine")


class _DeadlineConnectionPool:
    """隔离连接池（#33）：deadline 查询复用连接，避免每查询 ``connect(":memory:")``。

    每次 acquire 的查询都可被 ``interrupt()`` 单查询取消；连接归还池子后复用
    （保住 httpfs 配置 / metadata cache / 启动成本）。池满时溢出连接临时借用、
    归还即关。线程安全。
    """

    def __init__(self, size: int = 4) -> None:
        self._max = max(1, size)
        self._lock = threading.Lock()
        self._idle: list[duckdb.DuckDBPyConnection] = []
        self._created = 0
        self._overflow_ids: dict[int, bool] = {}

    def acquire(self, config: Any) -> "duckdb.DuckDBPyConnection":
        with self._lock:
            if self._idle:
                return self._idle.pop()
            if self._created < self._max:
                conn = duckdb.connect(":memory:")
                self._created += 1
                overflow = False
            else:
                conn = duckdb.connect(":memory:")
                overflow = True
        apply_pragmas(conn, config)
        # DuckDB 连接是 C 扩展对象，不支持属性赋值 → 用内部 dict 按 id 跟踪
        self._overflow_ids[id(conn)] = overflow
        return conn

    def release(self, conn: "duckdb.DuckDBPyConnection") -> None:
        overflow = bool(self._overflow_ids.pop(id(conn), False))
        with self._lock:
            if overflow or len(self._idle) >= self._max:
                try:
                    conn.close()
                except Exception:
                    logger.debug("deadline pool: conn.close 失败", exc_info=True)
                if overflow and self._created > self._max:
                    self._created -= 1
                return
            self._idle.append(conn)

    def close(self) -> None:
        with self._lock:
            for conn in self._idle:
                try:
                    conn.close()
                except Exception:
                    pass
            self._idle.clear()
            self._overflow_ids.clear()
            self._created = 0


class DuckDBEngine:
    """进程级 DuckDB 封装。线程安全读（通过 cursor），写操作上锁串行。

    用法：
        engine = DuckDBEngine()   # 进程一次就够
        tbl = engine.execute_arrow("SELECT * FROM read_parquet(?)", ["path"])

    WHY 不每线程一个 connect(":memory:")：
        每个 :memory: 连接是独立数据库，footer cache 不共享；我们的负载是
        「全局少量注册数据集 + 很多线程并发读相同 parquet」，跨线程共享
        buffer pool 的收益 >> cursor 带来的额外开销。

    WHY 保留 `_write_lock`：
        DuckDB 某些操作（CREATE VIEW、PRAGMA）要避免并发写 catalog；
        sql_escape 只在 register/drop TEMP VIEW 时持锁，查询本身用 cursor 并行。
    """

    def __init__(
        self,
        *,
        threads: int | None = None,
        memory_limit: str | None = None,
        enable_object_cache: bool = True,
        config: DuckDBConfig | None = None,
    ) -> None:
        self._config = config or resolve_duckdb_config(
            threads=threads,
            memory_limit=memory_limit,
            enable_object_cache=enable_object_cache,
        )
        self._conn = duckdb.connect(":memory:")
        self._write_lock = threading.Lock()
        self._deadline_pool = _DeadlineConnectionPool(size=4)

        with self._write_lock:
            apply_pragmas(self._conn, self._config)

        logger.info(
            "DuckDB 初始化完成: threads=%d memory_limit=%s object_cache=%s "
            "temp_directory=%s version=%s",
            self._config.threads,
            self._config.memory_limit or "(default)",
            self._config.enable_object_cache,
            self._config.temp_directory or "(default)",
            duckdb.__version__,
        )

    @property
    def config(self) -> DuckDBConfig:
        return self._config

    @property
    def is_closed(self) -> bool:
        """连接是否已被关闭（reset_shared_engine 后缓存的 store 需重建引擎）。"""
        try:
            self._conn.execute("SELECT 1")
            return False
        except duckdb.Error:
            return True

    def ensure_s3_configured(self) -> None:
        """COS 远程直读前配置 DuckDB httpfs（进程内一次）。"""
        from data_access.cos.s3_duckdb import ensure_duckdb_s3

        with self._write_lock:
            ensure_duckdb_s3(self._conn)

    # ---- catalog 写（短锁） ----

    def register_temp_views(self, views: Sequence[tuple[str, str]]) -> None:
        """注册 TEMP VIEW；views = [(view_name, inlined_sql), ...]。"""
        with self._write_lock:
            for view_name, inlined_sql in views:
                self._conn.execute(f"DROP VIEW IF EXISTS {view_name}")
                self._conn.execute(f"CREATE TEMP VIEW {view_name} AS {inlined_sql}")

    def drop_temp_views(self, view_names: Sequence[str]) -> None:
        """删除 TEMP VIEW；失败只记日志。"""
        with self._write_lock:
            for view_name in view_names:
                try:
                    self._conn.execute(f"DROP VIEW IF EXISTS {view_name}")
                except duckdb.Error as exc:
                    logger.warning("清理 TEMP VIEW %s 失败：%s", view_name, exc)

    # ---- 查询接口 ----

    @retry_io()
    def _execute_arrow_core(self, sql: str, params: Sequence[Any] | None) -> pa.Table:
        cursor = self._conn.cursor()
        try:
            if params is not None:
                result = cursor.execute(sql, params)
            else:
                result = cursor.execute(sql)
            return (
                result.to_arrow_table()
                if hasattr(result, "to_arrow_table")
                else result.fetch_arrow_table()
            )
        finally:
            cursor.close()

    @retry_io()
    def _execute_reader_core(
        self,
        sql: str,
        params: Sequence[Any] | None,
        *,
        batch_size: int,
    ) -> "ManagedBatchReader":
        from data_access.read.managed_reader import ManagedBatchReader

        cursor = self._conn.cursor()
        try:
            if params is not None:
                result = cursor.execute(sql, params)
            else:
                result = cursor.execute(sql)
            if hasattr(result, "to_arrow_reader"):
                reader = result.to_arrow_reader(batch_size)
            else:
                reader = result.fetch_record_batch(batch_size)
            return ManagedBatchReader(reader, cursor=cursor)
        except Exception:
            cursor.close()
            raise

    def explain(self, sql: str, params: Sequence[Any] | None = None) -> str:
        """返回 EXPLAIN 计划文本（诊断用，默认不在热路径调用）。"""
        cursor = self._conn.cursor()
        try:
            if params is not None:
                rows = cursor.execute(f"EXPLAIN {sql}", list(params)).fetchall()
            else:
                rows = cursor.execute(f"EXPLAIN {sql}").fetchall()
        except duckdb.Error as exc:
            raise EngineError(f"DuckDB EXPLAIN 失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            cursor.close()
        if not rows:
            return ""
        if len(rows) == 1 and len(rows[0]) == 1:
            return str(rows[0][0])
        return "\n".join(str(row[0]) if len(row) == 1 else str(row) for row in rows)

    def profile_analyze(self, sql: str, params: Sequence[Any] | None = None) -> str:
        """返回 EXPLAIN ANALYZE 文本（慢查询诊断，成本高于 EXPLAIN）。"""
        cursor = self._conn.cursor()
        try:
            if params is not None:
                rows = cursor.execute(f"EXPLAIN ANALYZE {sql}", list(params)).fetchall()
            else:
                rows = cursor.execute(f"EXPLAIN ANALYZE {sql}").fetchall()
        except duckdb.Error as exc:
            raise EngineError(
                f"DuckDB EXPLAIN ANALYZE 失败: {exc}\nSQL: {sql[:500]}"
            ) from exc
        finally:
            cursor.close()
        if not rows:
            return ""
        if len(rows) == 1 and len(rows[0]) == 1:
            return str(rows[0][0])
        return "\n".join(str(row[0]) if len(row) == 1 else str(row) for row in rows)

    def execute_arrow(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        deadline_ms: float | None = None,
    ) -> pa.Table:
        """执行 SQL，返回 Arrow Table（零拷贝路径，性能最优）。

        ``deadline_ms``：查询超时主动取消。使用独立连接 + watchdog 线程在
        截止后调用 ``interrupt()``（不打扰共享连接的并发查询）。超时抛
        ``DeadlineExceeded``。
        """
        start = time.perf_counter()
        elapsed_ms = 0.0
        try:
            if deadline_ms is not None:
                table = self._execute_isolated_with_deadline(
                    sql, params, deadline_ms=deadline_ms
                )
            else:
                table = self._execute_arrow_core(sql, params)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return table
        except duckdb.Error as exc:
            raise EngineError(f"DuckDB 查询失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            if not elapsed_ms:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            record_query(elapsed_ms=elapsed_ms, sql=sql, op="arrow")
            maybe_log_slow_query_plan(
                self,
                elapsed_ms=elapsed_ms,
                sql=sql,
                params=params,
                op="arrow",
            )

    def _execute_isolated_with_deadline(
        self,
        sql: str,
        params: Sequence[Any] | None,
        *,
        deadline_ms: float,
    ) -> pa.Table:
        """独立连接池 + watchdog：超时 interrupt，避免拖死共享连接。

        #33 连接来自 ``_deadline_pool``（复用 warm 连接），查询结束归还。
        """
        from data_access.core.exceptions import DeadlineExceeded

        deadline_sec = max(0.001, deadline_ms / 1000.0)
        conn = self._deadline_pool.acquire(self._config)
        timed_out = threading.Event()

        def _watchdog() -> None:
            if not timed_out.wait(deadline_sec):
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass

        thread = threading.Thread(target=_watchdog, daemon=True)
        thread.start()
        try:
            if "s3://" in sql:
                from data_access.cos.s3_duckdb import configure_fresh_duckdb_s3

                configure_fresh_duckdb_s3(conn)
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            return (
                result.to_arrow_table()
                if hasattr(result, "to_arrow_table")
                else result.fetch_arrow_table()
            )
        except duckdb.Error as exc:
            if timed_out.is_set():
                raise DeadlineExceeded(
                    f"查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                    "请缩小 time_range / instrument_filter / 指定 columns，"
                    "或提高 query_budget.max_elapsed_ms。"
                ) from exc
            raise
        finally:
            timed_out.set()
            self._deadline_pool.release(conn)

    def _execute_isolated_reader_with_deadline(
        self,
        sql: str,
        params: Sequence[Any] | None,
        *,
        batch_size: int,
        deadline_ms: float,
    ) -> "ManagedBatchReader":
        """流式读 + deadline（#32）：独立池连接 + watchdog，ManagedBatchReader
        close 时取消 watchdog 并归还连接。"""
        from data_access.core.exceptions import DeadlineExceeded
        from data_access.read.managed_reader import ManagedBatchReader

        deadline_sec = max(0.001, deadline_ms / 1000.0)
        conn = self._deadline_pool.acquire(self._config)
        timed_out = threading.Event()
        finished = threading.Event()

        def _watchdog() -> None:
            if not finished.wait(deadline_sec):
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass

        thread = threading.Thread(target=_watchdog, daemon=True)
        thread.start()

        def _on_close() -> None:
            finished.set()
            self._deadline_pool.release(conn)

        try:
            if "s3://" in sql:
                from data_access.cos.s3_duckdb import configure_fresh_duckdb_s3

                configure_fresh_duckdb_s3(conn)
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            if hasattr(result, "to_arrow_reader"):
                reader = result.to_arrow_reader(batch_size)
            else:
                reader = result.fetch_record_batch(batch_size)
            return ManagedBatchReader(reader, cursor=conn, on_close=_on_close)
        except duckdb.Error as exc:
            finished.set()
            self._deadline_pool.release(conn)
            if timed_out.is_set():
                raise DeadlineExceeded(
                    f"流式查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                ) from exc
            raise
        except Exception:
            finished.set()
            self._deadline_pool.release(conn)
            raise

    def execute_reader(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        batch_size: int = 100_000,
        deadline_ms: float | None = None,
    ) -> "ManagedBatchReader":
        """执行 SQL，返回 ManagedBatchReader（显式托管 reader+cursor 生命周期）。

        ``deadline_ms``（#32）：流式查询也主动超时取消——独立池连接 + watchdog
        interrupt；ManagedBatchReader 关闭时归还连接。
        """
        start = time.perf_counter()
        try:
            if deadline_ms is not None:
                return self._execute_isolated_reader_with_deadline(
                    sql, params, batch_size=batch_size, deadline_ms=deadline_ms
                )
            return self._execute_reader_core(sql, params, batch_size=batch_size)
        except duckdb.Error as exc:
            raise EngineError(f"DuckDB 流式查询失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            record_query(
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                sql=sql,
                op="reader",
            )

    def relation(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """返回一个 DuckDB Relation 对象（``RelationHandle`` 的底层扫描）。

        仅供 schema/explain 等只读检查使用；大数据 fetch 请走受控路径
        （RelationHandle.collect / execute_arrow）。relation 对象本身惰性，
        不会立即执行查询。
        """
        try:
            if params:
                return self._conn.sql(sql, params=list(params))
            return self._conn.sql(sql)
        except duckdb.Error as exc:
            raise EngineError(f"DuckDB relation 构建失败: {exc}\nSQL: {sql[:500]}") from exc

    @retry_io()
    def execute_isolated_arrow(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> pa.Table:
        """独立 :memory: 连接执行（ad-hoc SQL，不污染共享 catalog）。"""
        conn = duckdb.connect(":memory:")
        try:
            apply_pragmas(conn, self._config)
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            return (
                result.to_arrow_table()
                if hasattr(result, "to_arrow_table")
                else result.fetch_arrow_table()
            )
        except duckdb.Error as exc:
            raise EngineError(
                f"DuckDB isolated 查询失败: {exc}\nSQL: {sql[:500]}"
            ) from exc
        finally:
            conn.close()

    @staticmethod
    def _configure_scoped_s3_if_needed(
        conn,
        register_specs: Sequence[tuple[str, str]],
    ) -> None:
        """scoped 连接是全新 :memory:，共享 engine 上的 S3 配置不会继承。"""
        if not any("s3://" in inlined for _, inlined in register_specs):
            return
        from data_access.cos.s3_duckdb import configure_fresh_duckdb_s3

        configure_fresh_duckdb_s3(conn)

    def execute_scoped_sql_arrow(
        self,
        register_specs: Sequence[tuple[str, str]],
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        deadline_ms: float | None = None,
    ) -> pa.Table:
        """独立连接注册 TEMP VIEW 后执行 SQL（sql_escape 专用，支持并发 sql）。

        ``deadline_ms``：watchdog 超时 interrupt 本连接（scoped 连接是独立的，
        不影响共享连接上的并发查询）。
        """
        from data_access.core.exceptions import DeadlineExceeded

        start = time.perf_counter()
        conn = duckdb.connect(":memory:")
        timed_out = threading.Event()

        def _watchdog() -> None:
            if deadline_ms is not None and not timed_out.wait(deadline_ms / 1000.0):
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass

        thread: threading.Thread | None = None
        if deadline_ms is not None:
            thread = threading.Thread(target=_watchdog, daemon=True)
            thread.start()
        try:
            apply_pragmas(conn, self._config)
            self._configure_scoped_s3_if_needed(conn, register_specs)
            for view_name, inlined_sql in register_specs:
                conn.execute(f"CREATE TEMP VIEW {view_name} AS {inlined_sql}")
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            table = (
                result.to_arrow_table()
                if hasattr(result, "to_arrow_table")
                else result.fetch_arrow_table()
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            record_query(elapsed_ms=elapsed_ms, sql=sql, op="scoped_sql")
            maybe_log_slow_query_plan(
                self,
                elapsed_ms=elapsed_ms,
                sql=sql,
                params=params,
                op="scoped_sql",
            )
            return table
        except duckdb.Error as exc:
            if deadline_ms is not None and timed_out.is_set():
                raise DeadlineExceeded(
                    f"sql() 查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                    "请缩小 time_range / 指定 view_columns，或提高 max_elapsed_ms。"
                ) from exc
            raise EngineError(f"DuckDB scoped sql 失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            timed_out.set()
            conn.close()

    def execute_scoped_sql_stream(
        self,
        register_specs: Sequence[tuple[str, str]],
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        batch_size: int = 100_000,
    ) -> Iterator[pa.RecordBatch]:
        """独立连接流式 scoped sql；返回 batch iterator，迭代完自动关闭连接。"""
        from data_access.read.managed_reader import ManagedBatchReader

        def _iter() -> Iterator[pa.RecordBatch]:
            conn = duckdb.connect(":memory:")
            try:
                apply_pragmas(conn, self._config)
                self._configure_scoped_s3_if_needed(conn, register_specs)
                for view_name, inlined_sql in register_specs:
                    conn.execute(f"CREATE TEMP VIEW {view_name} AS {inlined_sql}")
                if params is not None:
                    rel = conn.execute(sql, list(params))
                else:
                    rel = conn.execute(sql)
                if hasattr(rel, "to_arrow_reader"):
                    reader = rel.to_arrow_reader(batch_size)
                else:
                    reader = rel.fetch_record_batch(batch_size)
                # ManagedBatchReader 显式托管 reader + conn 生命周期
                mbr = ManagedBatchReader(reader, on_close=conn.close)
                try:
                    for batch in mbr:
                        yield batch
                finally:
                    mbr.close()
            except duckdb.Error as exc:
                raise EngineError(
                    f"DuckDB scoped sql stream 失败: {exc}\nSQL: {sql[:500]}"
                ) from exc
            finally:
                conn.close()

        return _iter()

    def execute_df(self, sql: str, params: Sequence[Any] | None = None):
        """执行 SQL 并返回 pandas DataFrame。"""
        table = self.execute_arrow(sql, params)
        return table.to_pandas(self_destruct=True, split_blocks=True)

    def register_view(self, view_name: str, sql: str) -> None:
        """注册一个 VIEW。用在静态数据集启动时登记。"""
        with self._write_lock:
            self._conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")

    def close(self) -> None:
        """显式关闭。一般不用调，进程退出时 Python 会自动回收。"""
        try:
            self._conn.close()
        except duckdb.Error:
            pass


# ---- 进程共享单例 -----------------------------------------------------------

_shared_engine: DuckDBEngine | None = None
_shared_lock = threading.Lock()


def get_shared_engine() -> DuckDBEngine:
    """获取进程共享的 DuckDBEngine。第一次调用时初始化。"""
    global _shared_engine
    if _shared_engine is not None:
        return _shared_engine
    with _shared_lock:
        if _shared_engine is not None:
            return _shared_engine
        _shared_engine = DuckDBEngine()
        return _shared_engine


def reset_shared_engine() -> None:
    """主要给测试用，重置共享 engine。"""
    global _shared_engine
    from data_access.cos.s3_duckdb import reset_duckdb_s3_state

    with _shared_lock:
        if _shared_engine is not None:
            _shared_engine.close()
        _shared_engine = None
        reset_duckdb_s3_state()
