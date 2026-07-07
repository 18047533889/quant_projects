"""
data_access.engine —— DuckDB 连接与 PRAGMA 管理

职责：
    1. 进程内单例 DuckDB 实例（in-memory）
    2. 启动时设置性能相关 PRAGMA（threads / memory_limit / object_cache）
    3. 暴露线程安全的查询接口 execute_arrow / execute_df

设计要点（重要，PR 里还会反复提）：
    1. 一个进程一个 duckdb.connect(":memory:")，避免多实例各自 footer 缓存
    2. 多线程执行查询时用 conn.cursor() 分支出独立执行上下文，
       buffer pool / catalog / PRAGMA 共享
    3. PRAGMA enable_object_cache=true —— parquet footer 跨查询复用，
       factor_engine 重复读同一批文件时收益显著
    4. threads 默认用 os.cpu_count()；memory_limit 默认从 env 读，没设就不限
       （DuckDB 默认是物理内存 80%）

非职责：
    不负责 SQL 组装（store.py）、不负责 predicate 编译（predicate.py）、
    不负责读 YAML（registry.py）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Sequence

import duckdb
import pyarrow as pa

from .exceptions import EngineError
from .retry import retry_io
from .telemetry import record_query


logger = logging.getLogger("data_access.engine")


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
        虽然 PR1 还没写路径，但 DuckDB 某些操作（CREATE VIEW、PRAGMA）
        要避免并发写 catalog；提前把锁备好，上 PR2 的 publish 不用大改。
    """

    def __init__(
        self,
        *,
        threads: int | None = None,
        memory_limit: str | None = None,
        enable_object_cache: bool = True,
    ) -> None:
        self._conn = duckdb.connect(":memory:")
        self._write_lock = threading.Lock()

        # 默认用所有可用 CPU；用户想限可以 export DUCKDB_THREADS
        effective_threads = (
            threads
            if threads is not None
            else int(os.environ.get("DUCKDB_THREADS", os.cpu_count() or 4))
        )
        effective_mem = memory_limit or os.environ.get("DUCKDB_MEMORY_LIMIT")

        with self._write_lock:
            self._conn.execute(f"PRAGMA threads={effective_threads}")
            if effective_mem:
                # DuckDB 的 memory_limit 接受 '8GB' '2GB' 这种单位
                self._conn.execute(f"PRAGMA memory_limit='{effective_mem}'")
            if enable_object_cache:
                self._conn.execute("PRAGMA enable_object_cache=true")
            # 聚合查询默认不保留插入顺序，小幅提速；显式 ORDER BY 的不受影响
            self._conn.execute("PRAGMA preserve_insertion_order=false")

        logger.info(
            "DuckDB 初始化完成: threads=%d memory_limit=%s object_cache=%s version=%s",
            effective_threads, effective_mem or "(default)", enable_object_cache,
            duckdb.__version__,
        )

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
    ) -> pa.RecordBatchReader:
        cursor = self._conn.cursor()
        if params is not None:
            result = cursor.execute(sql, params)
        else:
            result = cursor.execute(sql)
        if hasattr(result, "to_arrow_reader"):
            return result.to_arrow_reader(batch_size)
        return result.fetch_record_batch(batch_size)

    def execute_arrow(self, sql: str, params: Sequence[Any] | None = None) -> pa.Table:
        """执行 SQL，返回 Arrow Table（零拷贝路径，性能最优）。

        每次调用开一个 cursor，线程间不共享执行状态；查询本身并行由 DuckDB
        的 PRAGMA threads 控制。

        PR6 起：执行完（含失败）把耗时送 `telemetry.record_query`，
        便于慢查询告警和 per-operator 配额观测。telemetry 内部吞异常，
        不会影响主路径。
        """
        start = time.perf_counter()
        try:
            return self._execute_arrow_core(sql, params)
        except duckdb.Error as exc:
            # 把 DuckDB 内部错包成我们的错误类型，上游好 except
            raise EngineError(f"DuckDB 查询失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            record_query(
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                sql=sql,
                op="arrow",
            )

    def execute_reader(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        batch_size: int = 100_000,
    ) -> pa.RecordBatchReader:
        """执行 SQL，返回 Arrow RecordBatchReader（真正流式，按 batch 拉）。

        PR7 起引入：大表扫描 / 离线 ETL 用这个代替 execute_arrow，峰值内存
        只占一个 batch；配合 for-loop 处理完一个 batch 就可以丢。

        参数：
            batch_size: 每个 RecordBatch 的目标行数；默认 100k（经验上对
                DuckDB 的 morsel 大小友好，小 batch 会损失向量化收益）。

        注意：返回的 reader 底层持有 cursor 句柄；迭代完（或 reader.close()）
        才会释放。不要拿 reader 存起来长期 hold——其他线程的 cursor 不受影响，
        但进程内 open cursor 太多会让 DuckDB 花更多时间维护状态。

        遥测仍然记录，但 elapsed_ms 只包含「拿到 reader 的时间」——batch
        的实际消费时间由调用方控制，record_query 没法精确感知。
        这是个 trade-off：下推到 reader 的 __iter__ 里会把正常 API 性能
        拖慢几个百分点，收益和复杂度不匹配。
        """
        start = time.perf_counter()
        try:
            return self._execute_reader_core(sql, params, batch_size=batch_size)
        except duckdb.Error as exc:
            raise EngineError(f"DuckDB 流式查询失败: {exc}\nSQL: {sql[:500]}") from exc
        finally:
            record_query(
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                sql=sql,
                op="reader",
            )

    def execute_df(self, sql: str, params: Sequence[Any] | None = None):
        """执行 SQL 并返回 pandas DataFrame。

        内部走 Arrow → to_pandas 路径，而不是 DuckDB 的 .df()，因为 Arrow 这条
        path 在大表上快且省内存。
        """
        table = self.execute_arrow(sql, params)
        # self_destruct=True：转完之后立刻释放 Arrow 底层 buffer，减一半内存峰值
        return table.to_pandas(self_destruct=True, split_blocks=True)

    def register_view(self, view_name: str, sql: str) -> None:
        """注册一个 VIEW。用在静态数据集启动时登记。

        WHY：注册成 VIEW 后业务代码 SELECT 可直接 FROM <view_name>，
             SQL 更干净，DuckDB 也能把元数据扫描摊销到一次。
        """
        # 列名/视图名已被 registry 清洗过，这里不重复清洗
        with self._write_lock:
            self._conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")

    def close(self) -> None:
        """显式关闭。一般不用调，进程退出时 Python 会自动回收。"""
        try:
            self._conn.close()
        except duckdb.Error:
            pass


# ---- 进程共享单例 -----------------------------------------------------------
# WHY：store 和 factor_engine 的 ParquetSource 都需要跑 DuckDB 查询；
#      用同一个 DuckDBEngine 实例可以共享 buffer pool / object_cache，
#      显著提升连续读同一批 parquet 的速度。
# 不和 store._store 合并到一个单例：engine 本身不依赖 registry，可以更早初始化。
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
    with _shared_lock:
        if _shared_engine is not None:
            _shared_engine.close()
        _shared_engine = None
