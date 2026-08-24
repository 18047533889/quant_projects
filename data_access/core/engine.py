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
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence

import duckdb
import pyarrow as pa

from .duckdb_config import DuckDBConfig, apply_pragmas, resolve_duckdb_config
from .exceptions import AccessDeniedError, EngineError, ValidationError
from .retry import retry_io
from data_access.read.telemetry import maybe_log_slow_query_plan, record_query


logger = logging.getLogger("data_access.engine")


class StorageKind(str, Enum):
    """远程存储需求种类（R46：从 SQL 字符串 sniff 升级为 typed 需求）。

    ``StorageRequirement`` 显式声明一次查询需要何种远程存储，而不是靠扫
    SQL/params 里的 ``s3://`` 字符串推断（#P0-11 的短期方案的完整修复方向）。
    """

    LOCAL = "local"
    REMOTE_S3 = "s3"
    REMOTE_GCS = "gcs"
    REMOTE_AZURE = "azure"


# 已实现真实远程配置的存储种类；其余 fail-closed（不静默当 local 读）。
_REMOTE_CONFIGURABLE = frozenset({StorageKind.REMOTE_S3})


@dataclass(frozen=True)
class StorageRequirement:
    """一次查询的 typed 远程存储需求（R46）。

    - ``kind``：存储种类，构造时 fail-closed——未知值直接抛 ``ValidationError``，
      绝不静默 fallback 成 local（与 ``StorageSpec.__post_init__`` 同策略）。
    - ``credential_scope``：凭证权限范围标识，绑定 cache scope / credential
      chain 用（可空，缺省取当前 principal scope）。
    """

    kind: StorageKind
    credential_scope: str | None = None

    def __post_init__(self) -> None:
        raw = self.kind
        if isinstance(raw, StorageKind):
            normalized = raw
        else:
            try:
                normalized = StorageKind(str(raw).strip().lower())
            except ValueError:
                raise ValidationError(
                    f"未知 storage kind={raw!r}。支持: "
                    f"{[k.value for k in StorageKind]}"
                ) from None
        if normalized is not self.kind:
            object.__setattr__(self, "kind", normalized)

    @property
    def is_remote(self) -> bool:
        return self.kind is not StorageKind.LOCAL


class _DeadlineConnectionPool:
    """隔离连接池（#33 + #P0-6..8）：deadline 查询复用连接，避免每查询 ``connect``。

    #P0-8 **真正限并发**：``_active + len(_idle)`` 上限为 ``_max``，达上限后
    ``acquire`` 在 admission deadline 内等待（``Condition``），超时抛
    ``ResourceBudgetExceeded``——不再无限溢出创建连接（每个连接都占
    threads/memory_limit，overflow 会放大并发内存峰值）。

    #P0-7 **interrupt/timeout 连接必须 discard**：``release(healthy=False)``
    直接关闭连接，不回到 idle 队列——被 interrupt 的 DuckDB 连接没有证明回到
    干净可复用状态前不能重新进池。

    #P0-10 关闭后 acquire → ``EngineClosedError``。
    """

    def __init__(
        self,
        size: int = 4,
        *,
        db_path: str | None = None,
        threads_per_conn: int | None = None,
    ) -> None:
        self._max = max(1, size)
        # R28-13：pool 连接不再各自 ``duckdb.connect(':memory:')``（独立数据库实例
        # → 各自的 footer/object cache）。改连**同一共享临时文件数据库**——DuckDB
        # DatabaseManager 对同一 path 复用同一个 DatabaseInstance，object cache /
        # buffer pool / catalog 全局共享，FactorEngine 重复读同一批 parquet 时从
        # 「4 个独立 cache」变成「1 个热 cache」。
        self._db_path = db_path
        # R28-14：每连接 threads 上限（总 worker ≈ max_concurrency × threads_per_conn）。
        self._threads_per_conn = threads_per_conn
        self._cond = threading.Condition()
        self._idle: list[duckdb.DuckDBPyConnection] = []
        self._active = 0
        self._closed = False

    def _new_connection(self) -> "duckdb.DuckDBPyConnection":
        if self._db_path:
            return duckdb.connect(self._db_path, read_only=False)
        return duckdb.connect(":memory:")

    def _conn_config(self, config: Any) -> Any:
        """R28-14：单连接线程上限（约束 concurrent×per_query ≈ physical cores）。

        ``threads`` 是每个 DatabaseInstance 全局的 worker 数。若 4 个并发连接各自
        ``apply_pragmas(config)`` 都用主机 CPU 数，16 核机器会调度 64 个 worker。
        这里把每个连接缩到 ``max(1, total_threads // max_concurrency)``。
        """
        if self._threads_per_conn is None:
            return config
        try:
            from dataclasses import replace

            return replace(config, threads=self._threads_per_conn)
        except Exception:
            return config

    def acquire(self, config: Any, *, wait_seconds: float = 5.0) -> "duckdb.DuckDBPyConnection":
        from data_access.core.exceptions import EngineClosedError, ResourceBudgetExceeded

        with self._cond:
            if self._closed:
                raise EngineClosedError("deadline connection pool 已关闭")
            deadline = time.monotonic() + max(0.0, wait_seconds)
            while True:
                if self._idle:
                    conn = self._idle.pop()
                    self._active += 1
                    break
                if self._active + len(self._idle) < self._max:
                    conn = self._new_connection()
                    self._active += 1
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResourceBudgetExceeded(
                        f"Deadline connection pool 已达并发上限 {self._max}，"
                        f"等待 {wait_seconds:.1f}s 后仍无空闲连接。"
                        "请降低并发或提高查询 deadline。"
                    )
                self._cond.wait(min(remaining, 0.1))
        try:
            apply_pragmas(conn, self._conn_config(config))
        except Exception:
            # #P0-38 pragma 配置失败：conn 已从池里取出（_active += 1），必须
            # 丢回/销毁并递减，否则反复失败会把池容量永久占满。
            with self._cond:
                try:
                    conn.close()
                except Exception:
                    pass
                self._active -= 1
                self._cond.notify()
            raise
        return conn

    def release(self, conn: "duckdb.DuckDBPyConnection", *, healthy: bool = True) -> None:
        """归还连接。``healthy=False``（interrupt/timeout/错误）→ 直接丢弃不重用。"""
        with self._cond:
            if self._closed:
                try:
                    conn.close()
                except Exception:
                    pass
                return
            if not healthy:
                try:
                    conn.close()
                except Exception:
                    logger.debug("deadline pool: 丢弃 unhealthy 连接 close 失败", exc_info=True)
                self._active -= 1
                self._cond.notify()
                return
            if len(self._idle) >= self._max:
                try:
                    conn.close()
                except Exception:
                    logger.debug("deadline pool: conn.close 失败", exc_info=True)
                self._active -= 1
                self._cond.notify()
                return
            self._idle.append(conn)
            self._active -= 1
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            for conn in self._idle:
                try:
                    conn.close()
                except Exception:
                    pass
            self._idle.clear()
            self._active = 0
            self._cond.notify_all()


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
        max_concurrency: int | None = None,
        governor: Any = None,
    ) -> None:
        self._config = config or resolve_duckdb_config(
            threads=threads,
            memory_limit=memory_limit,
            enable_object_cache=enable_object_cache,
        )
        # R28-10/14：DuckDB 并发 **单一 source of truth**——pool 容量与
        # ``_exec_sem`` 都取同一个 ``max_concurrency``（缺省读进程级 governor 的
        # ``max_duckdb_concurrency``）。旧实现 governor 默认 8、deadline pool 却
        # 固定 4：配置 8 实际 4 个在跑 + 4 个等 pool。现在统一成一个数。
        if max_concurrency is None:
            try:
                from data_access.runtime.resource_governor import get_global_governor

                max_concurrency = get_global_governor().max_duckdb_concurrency
            except Exception:
                max_concurrency = 8
        self._max_concurrency = max(1, int(max_concurrency))
        # R29-P0 #201：DuckDB 并发**单一信号量**。
        #   - 显式注入 governor → engine ``_exec_sem`` 直接就是 governor 的
        #     ``_duckdb_sem``（同一个对象，无双闸）；
        #   - 否则 standalone engine 用自带闸（``max_concurrency`` 语义保持：
        #     显式 5 → 闸 5、pool 5）；Store 建 pipeline 后经 ``set_governor``
        #     重链到 pipeline governor → 单一 admission point。
        from data_access.runtime.resource_governor import get_global_governor

        if governor is not None:
            self._governor = governor
            self._exec_sem = governor.duckdb_semaphore
        else:
            self._governor = get_global_governor()
            self._exec_sem = threading.Semaphore(self._max_concurrency)

        self._conn = duckdb.connect(":memory:")
        self._write_lock = threading.Lock()
        # R28-13：pool 连接共享同一临时文件数据库（object/footer cache 跨连接共享）。
        self._pool_db_path: str | None = None
        try:
            import tempfile

            _fd, _path = tempfile.mkstemp(
                prefix="da_duckdb_pool_", suffix=".db"
            )
            os.close(_fd)
            # mkstemp 产生的是 0 字节空文件，DuckDB 要求「不存在」或「合法 DB 文件」。
            # 删掉后由首个 pool 连接创建；同 path 复用同一个 DatabaseInstance
            # （object cache / buffer pool 跨连接共享，R28-13）。
            os.remove(_path)
            self._pool_db_path = _path
        except OSError:
            self._pool_db_path = None
        # R28-14：每连接 threads 上限 ≈ total_threads // max_concurrency。
        _tpc = None
        if self._config.threads:
            _tpc = max(1, int(self._config.threads) // self._max_concurrency)
        self._deadline_pool = _DeadlineConnectionPool(
            size=self._max_concurrency,
            db_path=self._pool_db_path,
            threads_per_conn=_tpc,
        )
        # R26-P1-018：fork/PID 防护——gunicorn --preload / multiprocessing fork /
        # factor mining process pool 会继承 fork 前创建的 DuckDB native connection /
        # lock / pool。入口检查 owner_pid，非 owner 进程禁止复用。
        self.owner_pid = os.getpid()

        with self._write_lock:
            apply_pragmas(self._conn, self._config)

        # R28-15：初始化完成日志移回 ``__init__``——旧实现在 ``_check_pid()`` 里，
        # 而 ``execute_arrow`` 每次先 ``_check_pid()``，普通 query 会不断刷
        # 「DuckDB 初始化完成」（日志噪音 + 无谓开销）。
        logger.info(
            "DuckDB 初始化完成: threads=%d memory_limit=%s object_cache=%s "
            "temp_directory=%s version=%s max_concurrency=%d",
            self._config.threads,
            self._config.memory_limit or "(default)",
            self._config.enable_object_cache,
            self._config.temp_directory or "(default)",
            duckdb.__version__,
            self._max_concurrency,
        )

    def set_governor(self, governor: Any) -> None:
        """R29-P0 #201：把 DuckDB 并发闸重链到指定 governor 的**同一信号量**。

        Store 建 pipeline 后调用，保证自定义（测试/多租户）governor 也被 engine
        尊重——engine 的 ``_exec_sem`` 就是 governor 的 ``_duckdb_sem``，单一闸。
        """
        from data_access.runtime.resource_governor import get_global_governor

        self._governor = governor if governor is not None else get_global_governor()
        self._exec_sem = self._governor.duckdb_semaphore

    def _check_pid(self) -> None:
        """R26-P1-018：fork 后子进程继续用父进程 native connection 会损坏状态。

        R28-15：这里只做 fork 防护（不含初始化日志——已移回 ``__init__``）。
        """
        import os as _os

        if _os.getpid() != self.owner_pid:
            raise RuntimeError(
                f"DuckDBEngine 在 fork 后被子进程（pid={_os.getpid()}，owner="
                f"{self.owner_pid}）复用——native connection/lock/pool 不可继承。"
                "请在 fork 后 reset 并重建 engine/store/pool（R26-P1-018）。"
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

    # ---- R29-P0 #204：组合读锚点跨连接注册（去临时 parquet round-trip）----

    def register_anchor_relation(self, view_name: str, table: Any) -> None:
        """把 Arrow 表注册成**共享 pool 数据库里的持久表**（跨连接可见）。

        pool 连接（deadline/isolated）与这里开的短连接连同一数据库文件
        （``_pool_db_path``）——同 path → 同 DatabaseInstance → 同一 catalog，
        持久表对全部 pool 连接可见。组合读（聚合锚点 + join）因此免写临时
        parquet、免压缩/解压 round-trip。

        用 ``CREATE TABLE ... AS SELECT``（**物化数据**，不是 VIEW）：VIEW 会
        把 ``__da_anchor_src`` 的注册引用存进定义，短连接关闭后跨连接查询变悬空
        引用；TABLE 真正复制数据进共享库，无悬空。用后必须
        ``drop_anchor_relation`` 清理。失败（pool 文件不可用）→ 抛异常，调用方
        回退临时 parquet。
        """
        if not self._pool_db_path:
            # R39 P0 #43：能力**明确缺失** → typed CapabilityUnavailableError（调用方
            # 可以精确回退临时 parquet）；数据库损坏/schema 错误/pool 状态错误仍走
            # 底层异常原样传播，禁止 ``except Exception`` 一把抓当能力缺失回退。
            from data_access.core.exceptions import CapabilityUnavailableError

            raise CapabilityUnavailableError(
                "DuckDB 共享 pool 数据库不可用，无法注册组合锚点关系"
                "（调用方应回退临时 parquet）"
            )
        import duckdb as _ddb

        con = _ddb.connect(self._pool_db_path)
        try:
            con.register("__da_anchor_src", table)
            con.execute(
                f'CREATE OR REPLACE TABLE "{view_name}" AS '
                "SELECT * FROM __da_anchor_src"
            )
        finally:
            try:
                con.unregister("__da_anchor_src")
            except Exception:
                pass
            con.close()

    def drop_anchor_relation(self, view_name: str) -> None:
        """删除组合锚点持久表（finally 清理）；失败只记日志。"""
        if not self._pool_db_path:
            return
        import duckdb as _ddb

        con = _ddb.connect(self._pool_db_path)
        try:
            con.execute(f'DROP TABLE IF EXISTS "{view_name}"')
        except Exception as exc:
            logger.warning("清理组合锚点表 %s 失败：%s", view_name, exc)
        finally:
            con.close()

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

        setup_start = time.perf_counter()
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
            setup_ms = (time.perf_counter() - setup_start) * 1000.0
            # #P1-7 setup_ms 只记 reader 构建耗时；真实 stream lifetime 由 reader
            # close 时上报（含 first_batch / stream_duration）。
            return ManagedBatchReader(
                reader,
                cursor=cursor,
                telemetry_fn=self._record_reader_stream,
                setup_ms=setup_ms,
            )
        except Exception:
            cursor.close()
            raise

    @staticmethod
    def _wrap_query_error(sql: str, exc: Exception) -> Exception:
        """把 DuckDB 查询错误包装成 DataAccess 异常类型（R24 P1-S9）。

        403 / Forbidden / PermissionDenied / AccessDenied / unauthorized 属于
        授权层错误，**不是 backend 错误**——必须原样传播为 ``AccessDeniedError``，
        禁止任何调用方 ``except Exception: return None`` 或降级成空 universe，
        更禁止换更高权限 credential 重试（T-S02）。
        其余 duckdb 错误保持 ``EngineError``（外部信息脱敏：不打印完整 SQL）。
        """
        from data_access.core.retry import ErrorClass, classify_exception

        if classify_exception(exc) == ErrorClass.AUTH:
            return AccessDeniedError(f"resource is not authorized (cloud denied)")
        # 外部错误不泄露 SQL / 完整 local/COS path（P1-S6 §8）。
        return EngineError(f"DuckDB 查询失败: {exc}")

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
        storage_requirement: StorageRequirement | None = None,
    ) -> pa.Table:
        """执行 SQL，返回 Arrow Table（零拷贝路径，性能最优）。

        ``deadline_ms``：查询超时自动取消。使用独立连接 + watchdog 线程在
        截止后调用 ``interrupt()``（不打扰共享连接的并发查询）。超时抛
        ``DeadlineExceeded``。

        ``storage_requirement``（R46）：typed 远程存储需求。显式声明优于
        字符串 sniff（#P0-11 短期方案的完整修复方向）。未配置的远程种类
        fail-closed（抛 ``ValidationError``），绝不静默当 local 读。

        R28-11：engine 内部统一 ``_exec_sem`` 并发门（不再要求调用方记得
        ``with duckdb_slot()``）。
        """
        self._check_pid()
        with self._exec_sem:
            start = time.perf_counter()
            elapsed_ms = 0.0
            try:
                if deadline_ms is not None:
                    table = self._execute_isolated_with_deadline(
                        sql, params, deadline_ms=deadline_ms
                    )
                else:
                    table = self._execute_arrow_core(sql, params)
                if storage_requirement is not None:
                    self._configure_storage(self._conn, storage_requirement)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return table
            except duckdb.Error as exc:
                raise self._wrap_query_error(sql, exc) from exc
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

    def _requires_remote_storage(self, sql: str, params: Sequence[Any] | None) -> bool:
        """#P0-11 判断本次执行是否需要 S3 配置。

        **不再只 sniff SQL 字符串**：远程 URI 常常在绑定 params 里（``?`` 传
        ``read_parquet(?)`` 的路径）。这里扫 sql + bound URI params。完整修复是
        PhysicalPlan 显式携带 ``requires_remote_storage``（本轮短期方案）。
        """
        if "s3://" in sql:
            return True
        if params:
            for p in params:
                if isinstance(p, str) and "s3://" in p:
                    return True
                if isinstance(p, (list, tuple)) and any(
                    isinstance(x, str) and "s3://" in x for x in p
                ):
                    return True
        return False

    def _configure_storage(
        self, conn, requirement: StorageRequirement
    ) -> None:
        """R46：按 typed ``StorageRequirement`` 为连接配置远程存储。

        fail-closed：未知/未实现 remote kind 抛 ``ValidationError``，绝不静默
        当 local 读。LOCAL 无操作。
        """
        if requirement.kind is StorageKind.LOCAL:
            return
        if requirement.kind in _REMOTE_CONFIGURABLE:
            from data_access.cos.s3_duckdb import ensure_duckdb_s3

            with self._write_lock:
                ensure_duckdb_s3(conn)
            return
        raise ValidationError(
            f"storage kind={requirement.kind.value!r} 的远程配置尚未实现"
            f"（已支持: {sorted(k.value for k in _REMOTE_CONFIGURABLE)}）。"
            "请不要对未支持后端声明远程存储需求。"
        )

    def _configure_isolated_s3_if_needed(self, conn, sql: str, params: Sequence[Any] | None) -> None:
        from data_access.cos.s3_duckdb import configure_fresh_duckdb_s3

        if self._requires_remote_storage(sql, params):
            configure_fresh_duckdb_s3(conn)

    def _record_reader_stream(self, metrics: dict[str, Any]) -> None:
        """#P1-7 流式 reader 真实生命周期上报（close 时调用）。"""
        try:
            record_query(
                elapsed_ms=float(metrics.get("total_duration_ms", 0.0) or 0.0),
                sql=f"reader_stream rows={metrics.get('rows')} "
                    f"ttf={metrics.get('time_to_first_batch_ms')}",
                op="reader_stream",
            )
        except Exception:  # noqa: BLE001 — telemetry 失败不能影响主路径
            pass

    def _execute_isolated_with_deadline(
        self,
        sql: str,
        params: Sequence[Any] | None,
        *,
        deadline_ms: float,
    ) -> pa.Table:
        """独立连接池 + watchdog：超时 interrupt，避免拖死共享连接。

        #33 连接来自 ``_deadline_pool``（复用 warm 连接），查询结束归还。

        #P0-9 deadline 用 **absolute monotonic**，查询正常返回后**再检查**一次
        timed_out / absolute deadline——watchdog 已触发但查询恰好正常结束时，丢弃
        结果抛 DeadlineExceeded（否则会返回「超时却成功」的错结果）。

        #P0-7 超时 / interrupt / 任何异常路径连接标记 unhealthy → 直接丢弃，
        不重新进池。

        # R28-12：**request absolute deadline 贯穿整条链**（governor slot →
        # connection pool → S3 credential → execute）。旧实现：pool acquire 默认
        # 最多等 5s，deadline 只有 1s 时 pool 等了 3s，watchdog 算出的 wait < 0，
        # ``if wait > 0`` 为 False 反而不 interrupt——查询继续真正跑完，最后才
        # 抛 DeadlineExceeded。现在：
        #   1. deadline 已过 → 直接 fail-fast（根本不执行）；
        #   2. pool acquire 只允许消费剩余时间（wait_seconds=剩余）；
        #   3. acquire 后 deadline 已过 → 归还连接并 fail（不执行）；
        #   4. watchdog 在 wait <= 0 时立即 interrupt（不再静默放行）。
        """
        from data_access.core.exceptions import DeadlineExceeded, ResourceBudgetExceeded

        deadline_sec = max(0.001, deadline_ms / 1000.0)
        absolute_deadline = time.monotonic() + deadline_sec
        if time.monotonic() >= absolute_deadline:
            raise DeadlineExceeded(
                f"查询超过 deadline={deadline_ms:.0f}ms（请求 deadline 已耗尽，"
                "未开始执行）。请降低并发或提高 query_budget.max_elapsed_ms。"
            )
        remaining = max(0.0, absolute_deadline - time.monotonic())
        try:
            conn = self._deadline_pool.acquire(
                self._config, wait_seconds=min(5.0, remaining)
            )
        except ResourceBudgetExceeded:
            # R28-12：pool 等待耗尽**请求 deadline** → 语义上是查询超时（不是并发
            # 误报）。旧实现这里漏转换，查询会继续跑完才报超时。
            raise DeadlineExceeded(
                f"查询超过 deadline={deadline_ms:.0f}ms（连接池等待耗尽请求 "
                "deadline）。请降低并发或提高 query_budget.max_elapsed_ms。"
            ) from None
        if time.monotonic() >= absolute_deadline:
            self._deadline_pool.release(conn, healthy=True)
            raise DeadlineExceeded(
                f"查询超过 deadline={deadline_ms:.0f}ms（连接池等待耗尽请求 "
                "deadline）。请降低并发或提高 query_budget.max_elapsed_ms。"
            )
        timed_out = threading.Event()
        healthy = True

        def _watchdog() -> None:
            wait = absolute_deadline - time.monotonic()
            if wait <= 0:
                # R28-12：deadline 已过（pool 等待吃掉了预算）→ 立即 interrupt，
                # 不再静默放行让查询跑完。
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass
                return
            if not timed_out.wait(wait):
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass

        thread = threading.Thread(target=_watchdog, daemon=True)
        thread.start()
        try:
            self._configure_isolated_s3_if_needed(conn, sql, params)
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            table = (
                result.to_arrow_table()
                if hasattr(result, "to_arrow_table")
                else result.fetch_arrow_table()
            )
            # #P0-9 deadline race：超时标记已触发但查询正常返回 → 丢弃结果 fail。
            if timed_out.is_set() or time.monotonic() >= absolute_deadline:
                healthy = False
                raise DeadlineExceeded(
                    f"查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                    "请缩小 time_range / instrument_filter / 指定 columns，"
                    "或提高 query_budget.max_elapsed_ms。"
                )
            return table
        except duckdb.Error as exc:
            if timed_out.is_set():
                healthy = False
                raise DeadlineExceeded(
                    f"查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                    "请缩小 time_range / instrument_filter / 指定 columns，"
                    "或提高 query_budget.max_elapsed_ms。"
                ) from exc
            healthy = False
            raise
        finally:
            timed_out.set()
            self._deadline_pool.release(conn, healthy=healthy)

    def _execute_isolated_reader_with_deadline(
        self,
        sql: str,
        params: Sequence[Any] | None,
        *,
        batch_size: int,
        deadline_ms: float,
    ) -> "ManagedBatchReader":
        """流式读 + deadline（#32）：独立池连接 + watchdog，ManagedBatchReader
        close 时取消 watchdog 并归还连接。

        #P0-6 pooled 连接**不**作为 ``cursor`` 交给 reader（reader.close() 会关
        cursor）——传 ``cursor=None``，归还完全由 ``on_close`` 负责（reader 关闭
        本身不会碰连接）。health 由 reader 的 ``had_error`` + timed_out 判定：
        interrupt / 异常 → unhealthy → 连接丢弃不重用（#P0-7）。
        """
        from data_access.core.exceptions import DeadlineExceeded
        from data_access.read.managed_reader import ManagedBatchReader

        deadline_sec = max(0.001, deadline_ms / 1000.0)
        absolute_deadline = time.monotonic() + deadline_sec
        conn = self._deadline_pool.acquire(self._config)
        timed_out = threading.Event()
        finished = threading.Event()

        def _watchdog() -> None:
            wait = absolute_deadline - time.monotonic()
            if wait > 0 and not finished.wait(wait):
                timed_out.set()
                try:
                    conn.interrupt()
                except Exception:
                    pass

        thread = threading.Thread(target=_watchdog, daemon=True)
        thread.start()

        def _on_close(reader: ManagedBatchReader) -> None:
            finished.set()
            unhealthy = timed_out.is_set() or getattr(reader, "had_error", False)
            self._deadline_pool.release(conn, healthy=not unhealthy)

        def _on_before_read() -> None:
            # deadline 触发后下一次 read 立即失败，不返回「超时却成功」的剩余 batch。
            if timed_out.is_set():
                raise DeadlineExceeded(
                    f"流式查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                )

        setup_start = time.perf_counter()
        try:
            self._configure_isolated_s3_if_needed(conn, sql, params)
            if params is not None:
                result = conn.execute(sql, list(params))
            else:
                result = conn.execute(sql)
            if hasattr(result, "to_arrow_reader"):
                reader = result.to_arrow_reader(batch_size)
            else:
                reader = result.fetch_record_batch(batch_size)
            setup_ms = (time.perf_counter() - setup_start) * 1000.0
            return ManagedBatchReader(
                reader,
                cursor=None,  # #P0-6 pooled 连接绝不交给 reader.close
                on_close=_on_close,
                on_before_read=_on_before_read,
                telemetry_fn=self._record_reader_stream,
                setup_ms=setup_ms,
            )
        except duckdb.Error as exc:
            finished.set()
            self._deadline_pool.release(conn, healthy=False)
            if timed_out.is_set():
                raise DeadlineExceeded(
                    f"流式查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                ) from exc
            raise
        except Exception:
            finished.set()
            self._deadline_pool.release(conn, healthy=False)
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

        R28-11/15：``execute_reader`` 同样过 ``_check_pid()``（fork 防护覆盖所有
        public execution entry），并持有 ``_exec_sem``——信号量随 reader **stream
        生命周期**持有，reader 关闭（含 GC / break / 异常）才释放。
        """
        self._check_pid()
        self._exec_sem.acquire()
        released = threading.Event()

        def _release_sem(_reader: Any = None) -> None:
            if not released.is_set():
                released.set()
                self._exec_sem.release()

        start = time.perf_counter()
        try:
            if deadline_ms is not None:
                reader = self._execute_isolated_reader_with_deadline(
                    sql, params, batch_size=batch_size, deadline_ms=deadline_ms
                )
            else:
                reader = self._execute_reader_core(sql, params, batch_size=batch_size)
        except duckdb.Error as exc:
            _release_sem()
            raise self._wrap_query_error(sql, exc) from exc
        except Exception:
            _release_sem()
            raise
        finally:
            record_query(
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                sql=sql,
                op="reader",
            )
        # 组合 on_close：先执行 reader 原有回调（归还连接），再释放信号量。
        original_on_close = getattr(reader, "_on_close", None)

        def _composed_on_close(rd: Any) -> None:
            try:
                if original_on_close is not None:
                    original_on_close(rd)
            finally:
                _release_sem(rd)

        reader._on_close = _composed_on_close
        return reader

    def relation(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """返回一个 DuckDB Relation 对象（``RelationHandle`` 的底层扫描）。

        仅供 schema/explain 等只读检查使用；大数据 fetch 请走受控路径
        （RelationHandle.collect / execute_arrow）。relation 对象本身惰性，
        不会立即执行查询。

        R28-15：relation 也是 public execution entry——同样过 fork 防护。
        """
        self._check_pid()
        try:
            if params:
                return self._conn.sql(sql, params=list(params))
            return self._conn.sql(sql)
        except duckdb.Error as exc:
            raise self._wrap_query_error(sql, exc) from exc

    @retry_io()
    def execute_isolated_arrow(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> pa.Table:
        """独立 :memory: 连接执行（ad-hoc SQL，不污染共享 catalog）。

        R28-15：同样过 fork 防护（public execution entry 统一覆盖）。
        """
        self._check_pid()
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

    def _configure_scoped_s3_if_needed(
        self,
        conn,
        register_specs: Sequence[tuple[str, str]],
        params: Sequence[Any] | None = None,
    ) -> None:
        """scoped 连接是全新 :memory:，共享 engine 上的 S3 配置不会继承。

        #P0-11 不再只 sniff 视图内联 SQL——用户绑定 params（``?`` 路径）含
        ``s3://`` 同样触发 S3 配置。
        """
        if any("s3://" in inlined for _, inlined in register_specs):
            from data_access.cos.s3_duckdb import configure_fresh_duckdb_s3

            configure_fresh_duckdb_s3(conn)
            return
        if params and self._requires_remote_storage("", params):
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

        R28-11/15：scoped sql 同样持有 ``_exec_sem`` + ``_check_pid``（所有
        public execution entry 统一并发/fork 防护）。
        """
        from data_access.core.exceptions import DeadlineExceeded

        self._check_pid()
        with self._exec_sem:
            return self._execute_scoped_sql_arrow_locked(
                register_specs, sql, params, deadline_ms=deadline_ms
            )

    def _execute_scoped_sql_arrow_locked(
        self,
        register_specs: Sequence[tuple[str, str]],
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        deadline_ms: float | None = None,
    ) -> pa.Table:
        """``_exec_sem`` 已持有时的 scoped sql 执行体（R28-11 拆分）。"""
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
            self._configure_scoped_s3_if_needed(conn, register_specs, params)
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
            raise self._wrap_query_error(sql, exc) from exc
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
        deadline_ms: float | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """独立连接流式 scoped sql；返回 batch iterator，迭代完自动关闭连接。

        #P0-41 ``deadline_ms`` 下推：watchdog 在绝对 deadline 触发 ``interrupt()``，
        ManagedBatchReader 的 on_before_read 在下次 read 时抛 ``DeadlineExceeded``
        ——流式查询第一批数据不返回时外层 budget 也能强制取消（不再只能等
        第一批出来才查 elapsed）。

        R28-15：public execution entry 同样过 fork 防护。
        """
        from data_access.core.exceptions import DeadlineExceeded
        from data_access.read.managed_reader import ManagedBatchReader

        self._check_pid()

        def _iter() -> Iterator[pa.RecordBatch]:
            conn = duckdb.connect(":memory:")
            timed_out = threading.Event()
            finished = threading.Event()
            watchdog: threading.Thread | None = None

            def _watch() -> None:
                if deadline_ms is not None:
                    wait = max(0.001, deadline_ms / 1000.0)
                    if not finished.wait(wait):
                        timed_out.set()
                        try:
                            conn.interrupt()
                        except Exception:
                            pass

            if deadline_ms is not None:
                watchdog = threading.Thread(target=_watch, daemon=True)
                watchdog.start()
            try:
                apply_pragmas(conn, self._config)
                self._configure_scoped_s3_if_needed(conn, register_specs, params)
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

                def _on_before_read() -> None:
                    if timed_out.is_set():
                        raise DeadlineExceeded(
                            f"sql_stream 查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                            "请缩小 time_range / 指定 view_columns，或提高 max_elapsed_ms。"
                        )

                # ManagedBatchReader 显式托管 reader + conn 生命周期（非 pooled，
                # on_close=conn.close 直接关）
                mbr = ManagedBatchReader(
                    reader,
                    on_close=lambda _r: conn.close(),
                    on_before_read=_on_before_read if deadline_ms is not None else None,
                )
                try:
                    for batch in mbr:
                        yield batch
                finally:
                    mbr.close()
            except duckdb.Error as exc:
                if deadline_ms is not None and timed_out.is_set():
                    raise DeadlineExceeded(
                        f"sql_stream 查询超过 deadline={deadline_ms:.0f}ms 被取消。"
                    ) from exc
                raise EngineError(
                    f"DuckDB scoped sql stream 失败: {exc}\nSQL: {sql[:500]}"
                ) from exc
            finally:
                finished.set()
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

    def register_arrow_table(self, table_name: str, table: Any) -> None:
        """把已物化 Arrow Table 注册为 DuckDB 虚拟表（PhysicalPlanExecutor 用）。

        供节点式计划把「已聚合的锚点」作为中间表参与后续 join。用 ``_tmp`` 前缀
        避免与数据表冲突；连接关闭自动失效。
        """
        with self._write_lock:
            self._conn.register(table_name, table)

    def unregister_table(self, table_name: str) -> None:
        """注销 DuckDB 虚拟表（PhysicalPlanExecutor 组合执行后清理）。"""
        try:
            with self._write_lock:
                self._conn.unregister(table_name)
        except (duckdb.Error, Exception):
            pass

    def close(self) -> None:
        """#P0-10 显式关闭：先关 deadline 连接池（idle 连接一起 shutdown），
        再关共享连接。关闭后 acquire/execute → EngineClosedError fail-fast。

        R28-13：关闭后清理共享 pool 临时数据库文件。
        """
        self._deadline_pool.close()
        try:
            self._conn.close()
        except duckdb.Error:
            pass
        if self._pool_db_path:
            try:
                if os.path.exists(self._pool_db_path):
                    os.remove(self._pool_db_path)
            except OSError:
                pass
            self._pool_db_path = None


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
