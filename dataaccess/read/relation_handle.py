"""
data_access.read.relation_handle —— 受控 SQL Relation 句柄

用途（P1）：让 FactorEngine 的 DuckDB SQL backend 能在 DataAccess 的 scan 之上
直接追加表达式（scan + factor expression 融合），不经过 Arrow→Pandas；但真正
collect 时仍强制 QueryBudget（deadline + 行数/字节）、audit、snapshot/lineage。

设计要点
    1. ``RelationHandle`` 持有 SQL + 绑定参数 + store 引用。``.sql()`` 在现有
       SELECT 之上再包一层（``SELECT <expr> FROM (<prev>) AS _sub``），表达式
       原地追加，数据仍在 DuckDB 内。
    2. 治理点集中在 ``.arrow() / .collect() / .pandas()``：内部走
       ``store._engine.execute_arrow(sql, params, deadline_ms=...)`` +
       ``enforce_arrow_budget`` + ``audit.record``。跨查询取消由 deadline 看门狗
       在独立连接上完成，不污染共享连接。
    3. ``.relation`` 是逃生口（返回活 DuckDB Relation 供 schema/explain 检查）；
       不要在它上面直接 fetch 大数据——取数据请走受控 ``.collect()``。
    4. 不暴露任何 ``_lf`` / ``_df`` 私有字段；本句柄没有私有扫描对象。

非职责
    不做谓词编译（用 store.read(..., filters=) 或在此之上写 SQL）；不做格式转换
    的治理（转 polars 后由调用方负责）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

from typing import Any, Sequence

from data_access.read.query_budget import QueryBudget, enforce_arrow_budget, resolve_query_budget


class RelationHandle:
    """一次受控 SQL 读的句柄。可嵌套 ``.sql()`` 追加表达式，collect 时受治理。"""

    def __init__(
        self,
        store: Any,
        sql: str,
        *,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
        snapshot: Any = None,
        lineage: Any = None,
    ) -> None:
        self._store = store
        self._sql = sql
        self._params: list[Any] = list(params or [])
        self._budget = resolve_query_budget(query_budget)
        self._snapshot = snapshot
        self._lineage = lineage

    # ---- 追加表达式 ----

    def sql(self, select_sql: str, *, params: Sequence[Any] | None = None) -> "RelationHandle":
        """在当前 SELECT 之上追加一层表达式，返回新句柄（原句柄不变）。

        ``select_sql`` 可以是完整 SELECT（引用 ``_sub`` 作为 FROM 源），或仅表达式列表
        （自动包成 ``SELECT <expr> FROM (<current>) AS _sub``）。
        """
        text = select_sql.strip()
        if text.lower().startswith("select"):
            # 用户写完整 SELECT：把第一个 "FROM _sub" 替换为当前句柄的子查询
            import re

            new_sql = re.sub(
                r"(?i)\bFROM\s+_sub\b",
                f"FROM ({self._sql}) AS _sub",
                text,
                count=1,
            )
            if "_sub" not in new_sql and "_SUB" not in new_sql.upper():
                raise ValueError("追加的 SELECT 必须引用 _sub 作为 FROM 源")
        else:
            new_sql = f"SELECT {text} FROM ({self._sql}) AS _sub"
        combined = [*self._params, *(list(params or []))]
        return RelationHandle(
            self._store,
            new_sql,
            params=combined,
            query_budget=self._budget,
            snapshot=self._snapshot,
            lineage=self._lineage,
        )

    # ---- 逃生口（仅检查用） ----

    @property
    def relation(self) -> Any:
        """返回底层 DuckDB Relation（用于 schema/explain 检查，不直接 fetch 大数据）。"""
        return self._store._engine.relation(self._sql, self._params)

    # ---- 受控 collect ----

    def arrow(self) -> Any:
        """执行 SQL，返回 Arrow Table（强制 budget + audit）。"""
        import time

        from data_access.core import audit

        start = time.perf_counter()
        table = self._store._engine.execute_arrow(
            self._sql, self._params, deadline_ms=self._budget.max_elapsed_ms
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        enforce_arrow_budget(self._budget, table, elapsed_ms=elapsed_ms)
        audit.record(
            op="relation_collect",
            dataset=getattr(self._snapshot, "dataset", None) or "<sql>",
            ok=True,
            rows=table.num_rows,
            elapsed_ms=elapsed_ms,
            extra={"sql": self._sql[:300], "params_count": len(self._params)},
        )
        return table

    def collect(self) -> Any:
        """``arrow()`` 的别名。"""
        return self.arrow()

    def pandas(self):
        import pandas as pd  # noqa: F401

        return self.arrow().to_pandas(split_blocks=True)

    def polars(self):
        import polars as pl

        return pl.from_arrow(self.arrow())

    def to_polars_lazy(self):
        import polars as pl

        return pl.from_arrow(self.arrow()).lazy()

    # ---- 检查 ----

    def explain(self) -> str:
        """返回 DuckDB 的物理计划文本（只读检查，不 collect 数据）。"""
        try:
            return str(self.relation.explain())
        except Exception as exc:
            return f"<explain failed: {exc}>"

    @property
    def sql_text(self) -> str:
        return self._sql

    def __repr__(self) -> str:
        return f"RelationHandle(sql={self._sql[:80]!r}, params={len(self._params)})"
