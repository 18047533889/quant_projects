"""轻量级 SQLite 元数据目录：因子注册表 + 数据水位线。

设计目标
--------
- **因子户口本** (``factor_registry``)：记录因子 ID、作者、频率、AST Hash
  — Hash 防呆：公式变了但 ID 没变 → ``FactorHashMismatchError``。
- **水位线** (``factor_watermark``)：追踪每个因子已落盘的时间区间与行数。
- **并发安全**：开启 WAL 模式 + 30 s 锁等待超时，适配社团多人同时跑批。

Notes
-----
连接在整个 ``FactorCatalog`` 生命周期内保持长连接，调用 ``close()`` 或
上下文管理器退出时释放。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .exceptions import FactorHashMismatchError, FactorNotFoundError


def _parse_json_field(raw: Any) -> dict[str, Any]:
    """将 JSON 字段解析为字典（容错空值）。
    
    参数:
        raw: 见函数签名
    
    返回:
        dict[str, Any]
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# IR Hash 工具
# ---------------------------------------------------------------------------

def compute_ir_hash(ir_node, *, structural_only: bool = False) -> str:
    """将 IR 树序列化后计算 SHA-256 摘要。

    参数:
        ir_node: 因子 IR 树根节点
        structural_only: 若为 True，忽略 FieldRef 特有的 catalog 元数据，
                        只比较算子拓扑结构（用于 source_expr 一致性检查）

    返回:
        str
    """
    _CATALOG_ATTRS = frozenset({
        "field_id", "field_registry_hash", "source_table", "source_field",
        "domain", "frequency", "cardinality", "temporal_model", "pit_safe"
    })

    def _serialize(node) -> dict:
        """递归序列化 IR 节点为字典。

        参数:
            node: 见函数签名

        返回:
            dict
        """
        attrs = node.attrs
        if structural_only:
            attrs = {k: v for k, v in attrs.items() if k not in _CATALOG_ATTRS}
        return {
            "op": node.op,
            "attrs": {k: _normalize(v) for k, v in sorted(attrs.items())},
            "inputs": [_serialize(inp) for inp in node.inputs],
        }

    def _normalize(v: Any) -> Any:
        """规范化不可 JSON 序列化的属性值。

        参数:
            v: 见函数签名

        返回:
            Any
        """
        if isinstance(v, float):
            # 处理 NaN / Inf 的 JSON 兼容
            if v != v:  # NaN
                return "__NaN__"
            if v == float("inf"):
                return "__Inf__"
            if v == float("-inf"):
                return "__-Inf__"
        return v

    payload = json.dumps(_serialize(ir_node), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS factor_registry (
    factor_id   TEXT PRIMARY KEY,
    author      TEXT NOT NULL,
    frequency   TEXT NOT NULL,
    description TEXT,
    ast_hash    TEXT NOT NULL,
    expression  TEXT,
    data_source_json TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_watermark (
    factor_id    TEXT PRIMARY KEY,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    row_count    INTEGER,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id)
);

CREATE TABLE IF NOT EXISTS factor_run (
    run_id                  TEXT PRIMARY KEY,
    factor_id               TEXT NOT NULL,
    factor_name             TEXT,
    ast_hash                TEXT NOT NULL,
    operator_catalog_hash   TEXT,
    field_catalog_hash      TEXT,
    expression              TEXT,
    lookback                INTEGER,
    referenced_columns_json TEXT,
    dq_passed               INTEGER,
    row_count               INTEGER,
    non_null_count          INTEGER,
    created_at              TEXT NOT NULL,
    extra_json              TEXT,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id)
);

CREATE TABLE IF NOT EXISTS factor_materialize_checkpoint (
    factor_id       TEXT NOT NULL,
    partition_year  INTEGER NOT NULL,
    run_id          TEXT NOT NULL,
    status          TEXT NOT NULL,
    error_message   TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (factor_id, partition_year)
);

CREATE TABLE IF NOT EXISTS factor_dependency (
    factor_id               TEXT PRIMARY KEY,
    referenced_columns_json TEXT NOT NULL,
    lookback                INTEGER NOT NULL DEFAULT 0,
    frequency               TEXT,
    source_dataset          TEXT,
    updated_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_column_dep (
    column_name TEXT NOT NULL,
    factor_id   TEXT NOT NULL,
    PRIMARY KEY (column_name, factor_id),
    FOREIGN KEY (factor_id) REFERENCES factor_dependency(factor_id)
);
"""


class FactorCatalog:
    """SQLite 因子元数据目录（注册、水位线、血缘）。
    
    参数:
        db_path: SQLite 数据库路径
    """

    def __init__(self, db_path: str | Path) -> None:
        """初始化实例。
        
        参数:
            db_path: SQLite 数据库路径
        
        返回:
            无
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        # WAL 模式：大幅提升多进程并发读写能力
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """向后兼容：为旧 catalog 补列。
        
        参数:
            无
        
        返回:
            无
        """
        cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(factor_registry)")
        }
        if "data_source_json" not in cols:
            self._conn.execute(
                "ALTER TABLE factor_registry ADD COLUMN data_source_json TEXT"
            )
        run_cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(factor_run)")
        }
        if "field_catalog_hash" not in run_cols:
            self._conn.execute(
                "ALTER TABLE factor_run ADD COLUMN field_catalog_hash TEXT"
            )
        ck_cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(factor_materialize_checkpoint)"
            )
        }
        if "partition_key" not in ck_cols:
            self._conn.execute(
                "ALTER TABLE factor_materialize_checkpoint ADD COLUMN partition_key TEXT"
            )
            self._conn.execute(
                "UPDATE factor_materialize_checkpoint "
                "SET partition_key = 'year=' || CAST(partition_year AS TEXT) "
                "WHERE partition_key IS NULL"
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_factor_mat_ck_partition_key "
                "ON factor_materialize_checkpoint(factor_id, partition_key)"
            )

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self):
        """__enter__。
        
        参数:
            无
        
        返回:
            无
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__。
        
        参数:
            exc_type: 见函数签名
            exc_val: 见函数签名
            exc_tb: 见函数签名
        
        返回:
            无
        """
        self.close()
        return False

    def close(self) -> None:
        """显式关闭数据库连接。
        
        参数:
            无
        
        返回:
            无
        """
        self._conn.close()

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(
        self,
        factor_id: str,
        author: str,
        frequency: str,
        ast_hash: str,
        *,
        description: str | None = None,
        expression: str | None = None,
        data_source_config: dict | None = None,
    ) -> None:
        """注册因子。如 factor_id 已存在且 Hash 一致则静默跳过；不一致则报错。
        
        参数:
            factor_id: 因子唯一标识
            author: 见函数签名
            frequency: 因子频率
            ast_hash: 因子 AST 哈希
            description: 见函数签名（可选）
            expression: 见函数签名（可选）
            data_source_config: 见函数签名（可选）
        
        返回:
            无
        
        
        Raises
                ------
                FactorHashMismatchError
                    ``factor_id`` 已注册但 ``ast_hash`` 与既存记录不同。
        """
        existing = self.get_factor_info(factor_id)
        ds_json = (
            json.dumps(data_source_config, sort_keys=True, default=str, ensure_ascii=False)
            if data_source_config
            else None
        )
        if existing is not None:
            if existing["ast_hash"] != ast_hash:
                raise FactorHashMismatchError(
                    f"因子 '{factor_id}' 已注册（Hash={existing['ast_hash'][:12]}…），"
                    f"但当前公式 Hash 为 {ast_hash[:12]}…。"
                    f"请升级版本号（如改为 '{factor_id}_v2'）后重新落盘。"
                )
            if ds_json is not None:
                self._conn.execute(
                    "UPDATE factor_registry SET data_source_json = ? WHERE factor_id = ?",
                    (ds_json, factor_id),
                )
                self._conn.commit()
            return  # Hash 一致 → 幂等，不做任何变更

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_registry "
            "(factor_id, author, frequency, description, ast_hash, expression, "
            "data_source_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (factor_id, author, frequency, description, ast_hash, expression, ds_json, now),
        )
        self._conn.commit()

    def verify_hash(self, factor_id: str, ast_hash: str) -> bool:
        """校验因子 Hash 是否与注册一致。未注册返回 True（尚无冲突）。
        
        参数:
            factor_id: 因子唯一标识
            ast_hash: 因子 AST 哈希
        
        返回:
            bool
        """
        existing = self.get_factor_info(factor_id)
        if existing is None:
            return True
        return existing["ast_hash"] == ast_hash

    # ------------------------------------------------------------------
    # 水位线
    # ------------------------------------------------------------------

    def get_watermark(self, factor_id: str) -> dict | None:
        """返回水位线字典 ``{factor_id, start_date, end_date, last_updated, row_count}``。
        
        参数:
            factor_id: 因子唯一标识
        
        返回:
            dict | None
        
        
        未找到返回 ``None``。
        """
        row = self._conn.execute(
            "SELECT * FROM factor_watermark WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_watermark(
        self,
        factor_id: str,
        start_date: str,
        end_date: str,
        *,
        row_count: int | None = None,
    ) -> None:
        """插入或更新水位线（UPSERT）。
        
        参数:
            factor_id: 因子唯一标识
            start_date: 起始日期
            end_date: 结束日期
            row_count: 见函数签名（可选）
        
        返回:
            无
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_watermark (factor_id, start_date, end_date, last_updated, row_count) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id) DO UPDATE SET "
            "start_date=excluded.start_date, end_date=excluded.end_date, "
            "last_updated=excluded.last_updated, row_count=excluded.row_count",
            (factor_id, start_date, end_date, now, row_count),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_factor_info(self, factor_id: str) -> dict | None:
        """返回因子注册信息字典，未找到返回 ``None``。
        
        参数:
            factor_id: 因子唯一标识
        
        返回:
            dict | None
        """
        row = self._conn.execute(
            "SELECT * FROM factor_registry WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_factors(self) -> list[dict]:
        """返回所有已注册因子的列表。
        
        参数:
            无
        
        返回:
            list[dict]
        """
        rows = self._conn.execute(
            "SELECT r.*, w.start_date, w.end_date, w.last_updated, w.row_count "
            "FROM factor_registry r "
            "LEFT JOIN factor_watermark w ON r.factor_id = w.factor_id "
            "ORDER BY r.factor_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_factor(self, factor_id: str) -> None:
        """从 Catalog 中删除因子注册信息和水位线（不删除物理文件）。
        
        参数:
            factor_id: 因子唯一标识
        
        返回:
            无
        """
        self._conn.execute(
            "DELETE FROM factor_run WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_column_dep WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_dependency WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_watermark WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_registry WHERE factor_id = ?", (factor_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 运行血缘
    # ------------------------------------------------------------------

    def record_run(self, lineage: dict) -> None:
        """记录一次因子运行血缘（append-only）。
        
        参数:
            lineage: 运行血缘记录字典
        
        返回:
            无
        """
        now = lineage.get("created_at") or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_run "
            "(run_id, factor_id, factor_name, ast_hash, operator_catalog_hash, field_catalog_hash, expression, "
            "lookback, referenced_columns_json, dq_passed, row_count, non_null_count, "
            "created_at, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lineage["run_id"],
                lineage["factor_id"],
                lineage.get("factor_name"),
                lineage["ast_hash"],
                lineage.get("operator_catalog_hash"),
                lineage.get("field_catalog_hash"),
                lineage.get("expression"),
                lineage.get("lookback"),
                json.dumps(lineage.get("referenced_columns", []), ensure_ascii=False),
                int(lineage["dq_passed"]) if lineage.get("dq_passed") is not None else None,
                lineage.get("row_count"),
                lineage.get("non_null_count"),
                now,
                json.dumps(lineage.get("extra", {}), ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    def list_runs(self, factor_id: str, *, limit: int = 20) -> list[dict]:
        """list_runs。
        
        参数:
            factor_id: 因子唯一标识
            limit: 返回条数上限（可选）
        
        返回:
            list[dict]
        """
        rows = self._conn.execute(
            "SELECT * FROM factor_run WHERE factor_id = ? ORDER BY created_at DESC LIMIT ?",
            (factor_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_dual_write_failures(
        self,
        *,
        factor_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """列出 extra_json 中 ``dual_write_failed=true`` 的物化 run。
        
        参数:
            factor_id: 因子唯一标识（可选）
            limit: 返回条数上限（可选）
        
        返回:
            list[dict]
        """
        if factor_id:
            rows = self.list_runs(factor_id, limit=max(limit * 10, 100))
        else:
            rows = self.list_recent_runs(limit=max(limit * 10, 200))
        failures: list[dict] = []
        for row in rows:
            extra = _parse_json_field(row.get("extra_json"))
            if not extra.get("dual_write_failed"):
                continue
            failures.append({**row, "extra": extra})
            if len(failures) >= limit:
                break
        return failures

    def list_recent_runs(self, *, limit: int = 200) -> list[dict]:
        """list_recent_runs。
        
        参数:
            limit: 返回条数上限（可选）
        
        返回:
            list[dict]
        """
        rows = self._conn.execute(
            "SELECT * FROM factor_run ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 分区落盘 checkpoint（断点续跑）
    # ------------------------------------------------------------------

    def record_partition_checkpoint(
        self,
        *,
        factor_id: str,
        partition_year: int,
        run_id: str,
        status: str,
        error_message: str | None = None,
        partition_key: str | None = None,
    ) -> None:
        """record_partition_checkpoint。
        
        参数:
            factor_id: 因子唯一标识（可选）
            partition_year: 分区年份（可选）
            run_id: 运行 ID（可选）
            status: checkpoint 状态（可选）
            error_message: 见函数签名（可选）
            partition_key: 分区键字符串（可选）
        
        返回:
            无
        """
        now = datetime.now(timezone.utc).isoformat()
        pkey = partition_key or f"year={int(partition_year)}"
        self._conn.execute(
            "INSERT INTO factor_materialize_checkpoint "
            "(factor_id, partition_year, partition_key, run_id, status, error_message, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id, partition_key) DO UPDATE SET "
            "partition_year=excluded.partition_year, run_id=excluded.run_id, "
            "status=excluded.status, error_message=excluded.error_message, "
            "updated_at=excluded.updated_at",
            (factor_id, int(partition_year), pkey, run_id, status, error_message, now),
        )
        self._conn.commit()

    def get_partition_checkpoint(
        self,
        factor_id: str,
        partition_year: int,
    ) -> dict | None:
        """get_partition_checkpoint。
        
        参数:
            factor_id: 因子唯一标识
            partition_year: 分区年份
        
        返回:
            dict | None
        """
        row = self._conn.execute(
            "SELECT * FROM factor_materialize_checkpoint "
            "WHERE factor_id = ? AND partition_year = ?",
            (factor_id, int(partition_year)),
        ).fetchone()
        return dict(row) if row else None

    def get_partition_checkpoint_by_key(
        self,
        factor_id: str,
        partition_key: str,
    ) -> dict | None:
        """get_partition_checkpoint_by_key。
        
        参数:
            factor_id: 因子唯一标识
            partition_key: 分区键字符串
        
        返回:
            dict | None
        """
        row = self._conn.execute(
            "SELECT * FROM factor_materialize_checkpoint "
            "WHERE factor_id = ? AND partition_key = ?",
            (factor_id, partition_key),
        ).fetchone()
        return dict(row) if row else None

    def list_partition_checkpoints(
        self,
        factor_id: str,
        *,
        status: str | None = None,
    ) -> list[dict]:
        """list_partition_checkpoints。
        
        参数:
            factor_id: 因子唯一标识
            status: checkpoint 状态（可选）
        
        返回:
            list[dict]
        """
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM factor_materialize_checkpoint "
                "WHERE factor_id = ? ORDER BY partition_year",
                (factor_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM factor_materialize_checkpoint "
                "WHERE factor_id = ? AND status = ? ORDER BY partition_year",
                (factor_id, status),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_partition_checkpoints(self, factor_id: str) -> None:
        """clear_partition_checkpoints。
        
        参数:
            factor_id: 因子唯一标识
        
        返回:
            无
        """
        self._conn.execute(
            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?",
            (factor_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 因子依赖 catalog（增量 by data event）
    # ------------------------------------------------------------------

    def record_factor_dependency(
        self,
        factor_id: str,
        *,
        referenced_columns: Iterable[str],
        lookback: int = 0,
        frequency: str | None = None,
        source_dataset: str | None = None,
    ) -> None:
        """物化/run 后登记因子对数据列的依赖。
        
        参数:
            factor_id: 因子唯一标识
            referenced_columns: 依赖的数据列名集合（可选）
            lookback: 回看 bar 数（可选）
            frequency: 因子频率（可选）
            source_dataset: 来源数据集名称（可选）
        
        返回:
            无
        """
        cols = sorted(set(str(c) for c in referenced_columns if c))
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_dependency "
            "(factor_id, referenced_columns_json, lookback, frequency, source_dataset, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id) DO UPDATE SET "
            "referenced_columns_json=excluded.referenced_columns_json, "
            "lookback=excluded.lookback, frequency=excluded.frequency, "
            "source_dataset=excluded.source_dataset, updated_at=excluded.updated_at",
            (
                factor_id,
                json.dumps(cols, ensure_ascii=False),
                int(lookback),
                frequency,
                source_dataset,
                now,
            ),
        )
        self._conn.execute(
            "DELETE FROM factor_column_dep WHERE factor_id = ?",
            (factor_id,),
        )
        for col in cols:
            self._conn.execute(
                "INSERT OR IGNORE INTO factor_column_dep (column_name, factor_id) VALUES (?, ?)",
                (col, factor_id),
            )
        self._conn.commit()

    def get_factor_dependency(self, factor_id: str) -> dict | None:
        """get_factor_dependency。
        
        参数:
            factor_id: 因子唯一标识
        
        返回:
            dict | None
        """
        row = self._conn.execute(
            "SELECT * FROM factor_dependency WHERE factor_id = ?",
            (factor_id,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["referenced_columns"] = json.loads(out.pop("referenced_columns_json", "[]"))
        return out

    def list_factors_for_column(self, column_name: str) -> list[dict]:
        """list_factors_for_column。
        
        参数:
            column_name: 数据列名
        
        返回:
            list[dict]
        """
        rows = self._conn.execute(
            "SELECT d.* FROM factor_dependency d "
            "JOIN factor_column_dep c ON d.factor_id = c.factor_id "
            "WHERE c.column_name = ? ORDER BY d.factor_id",
            (str(column_name),),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            item["referenced_columns"] = json.loads(
                item.pop("referenced_columns_json", "[]")
            )
            out.append(item)
        return out

    def list_factors_for_dataset(self, dataset: str) -> list[dict]:
        """list_factors_for_dataset。
        
        参数:
            dataset: data_access 数据集名称
        
        返回:
            list[dict]
        """
        rows = self._conn.execute(
            "SELECT * FROM factor_dependency WHERE source_dataset = ? ORDER BY factor_id",
            (str(dataset),),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            item["referenced_columns"] = json.loads(
                item.pop("referenced_columns_json", "[]")
            )
            out.append(item)
        return out

    def list_dependency_columns(self) -> list[str]:
        """list_dependency_columns。
        
        参数:
            无
        
        返回:
            list[str]
        """
        rows = self._conn.execute(
            "SELECT DISTINCT column_name FROM factor_column_dep ORDER BY column_name"
        ).fetchall()
        return [str(r[0]) for r in rows]
