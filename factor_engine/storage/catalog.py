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
from typing import Any

from .exceptions import FactorHashMismatchError, FactorNotFoundError


def _parse_json_field(raw: Any) -> dict[str, Any]:
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

def compute_ir_hash(ir_node) -> str:
    """将 ``IRNode`` 树递归序列化为确定性 JSON，取 SHA-256 摘要。

    序列化规则
    ----------
    - ``sort_keys=True`` 保证字典顺序确定性。
    - 仅包含 ``op``、``attrs``、``inputs``（递归），排除内存地址等运行时信息。
    """

    def _serialize(node) -> dict:
        return {
            "op": node.op,
            "attrs": {k: _normalize(v) for k, v in sorted(node.attrs.items())},
            "inputs": [_serialize(inp) for inp in node.inputs],
        }

    def _normalize(v: Any) -> Any:
        """将不可 JSON 序列化的值规范化。"""
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
"""


class FactorCatalog:
    """SQLite 元数据目录，管理因子注册与水位线。

    Parameters
    ----------
    db_path : str | Path
        SQLite 文件路径（如 ``lake_root/_catalog.sqlite``）。
        若文件不存在，首次连接时自动创建并建表。
    """

    def __init__(self, db_path: str | Path) -> None:
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
        """向后兼容：为旧 catalog 补列。"""
        cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(factor_registry)")
        }
        if "data_source_json" not in cols:
            self._conn.execute(
                "ALTER TABLE factor_registry ADD COLUMN data_source_json TEXT"
            )

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """显式关闭数据库连接。"""
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
        """校验因子 Hash 是否与注册一致。未注册返回 True（尚无冲突）。"""
        existing = self.get_factor_info(factor_id)
        if existing is None:
            return True
        return existing["ast_hash"] == ast_hash

    # ------------------------------------------------------------------
    # 水位线
    # ------------------------------------------------------------------

    def get_watermark(self, factor_id: str) -> dict | None:
        """返回水位线字典 ``{factor_id, start_date, end_date, last_updated, row_count}``。

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
        """插入或更新水位线（UPSERT）。"""
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
        """返回因子注册信息字典，未找到返回 ``None``。"""
        row = self._conn.execute(
            "SELECT * FROM factor_registry WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_factors(self) -> list[dict]:
        """返回所有已注册因子的列表。"""
        rows = self._conn.execute(
            "SELECT r.*, w.start_date, w.end_date, w.last_updated, w.row_count "
            "FROM factor_registry r "
            "LEFT JOIN factor_watermark w ON r.factor_id = w.factor_id "
            "ORDER BY r.factor_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_factor(self, factor_id: str) -> None:
        """从 Catalog 中删除因子注册信息和水位线（不删除物理文件）。"""
        self._conn.execute(
            "DELETE FROM factor_run WHERE factor_id = ?", (factor_id,)
        )
        self._conn.execute(
            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?", (factor_id,)
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
        """记录一次因子运行血缘（append-only）。"""
        now = lineage.get("created_at") or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_run "
            "(run_id, factor_id, factor_name, ast_hash, operator_catalog_hash, expression, "
            "lookback, referenced_columns_json, dq_passed, row_count, non_null_count, "
            "created_at, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lineage["run_id"],
                lineage["factor_id"],
                lineage.get("factor_name"),
                lineage["ast_hash"],
                lineage.get("operator_catalog_hash"),
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
        """列出 extra_json 中 ``dual_write_failed=true`` 的物化 run。"""
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
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO factor_materialize_checkpoint "
            "(factor_id, partition_year, run_id, status, error_message, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id, partition_year) DO UPDATE SET "
            "run_id=excluded.run_id, status=excluded.status, "
            "error_message=excluded.error_message, updated_at=excluded.updated_at",
            (factor_id, int(partition_year), run_id, status, error_message, now),
        )
        self._conn.commit()

    def get_partition_checkpoint(
        self,
        factor_id: str,
        partition_year: int,
    ) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM factor_materialize_checkpoint "
            "WHERE factor_id = ? AND partition_year = ?",
            (factor_id, int(partition_year)),
        ).fetchone()
        return dict(row) if row else None

    def list_partition_checkpoints(
        self,
        factor_id: str,
        *,
        status: str | None = None,
    ) -> list[dict]:
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
        self._conn.execute(
            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?",
            (factor_id,),
        )
        self._conn.commit()
