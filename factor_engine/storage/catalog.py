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

import contextlib
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, TypeVar

from .exceptions import (
    CatalogCorruptionError,
    CatalogMigrationError,
    CatalogSerializationError,
    FactorHashMismatchError,
    FactorNotFoundError,
    FactorRetiredError,
    FactorSemanticIdentityMismatchError,
    ProductionModeResolutionError,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Typed JSON catalog fields (NEW-P0-44 / NEW-P0-45 / NEW-P1-74)
#
# production 下 catalog 的全量定义 / data-source config 必须走 typed JSON
# schema：未知对象（callable / 自定义类 / enum-like 配置）直接抛
# ``CatalogSerializationError``，绝不 ``default=str`` 字符串化（字符串化后无法
# 还原重建）。损坏的 JSON 抛 ``CatalogCorruptionError``，绝不伪装成 ``{}``。
# ---------------------------------------------------------------------------


def _resolve_strict(strict: bool | None) -> bool:
    """解析 strict 解码开关：显式值优先，缺省从运行模式推断。

    R20-220..225: 运行模式解析**意外异常**必须 fail closed（抛错），绝不静默
    返回 False——否则 production 下 catalog 解码会退化成 permissive decode，
    把「catalog 损坏/类型非法」伪装成「config 缺失」，导致 corruption 被吞掉。
    """
    if strict is not None:
        return bool(strict)
    from runtime.production_policy import is_production_mode

    return is_production_mode()


def _strict_default(obj: Any) -> Any:
    """``json.dumps`` 的 default 钩子：production 下未知类型直接报错。

    绝不返回 ``str(obj)``——callable / 自定义类 / enum-like 配置字符串化后无法
    从字符串重建，等于把「不可序列化」伪装成「字符串 config」。
    """
    raise CatalogSerializationError(
        f"catalog typed JSON 序列化遇到不支持的类型 {type(obj).__name__}: {obj!r}。"
        f"production 全量定义必须使用 typed JSON schema，禁止 default=str 字符串化"
        f"（字符串化后的对象无法重建）。"
    )


def catalog_strict_dumps(
    obj: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> str:
    """严格 JSON 序列化：未知类型 → ``CatalogSerializationError``。

    参数:
        obj: 待序列化对象
        sort_keys: 是否按键排序（可选）
        ensure_ascii: 是否转义非 ASCII（可选）

    返回:
        str
    """
    try:
        return json.dumps(
            obj, sort_keys=sort_keys, ensure_ascii=ensure_ascii, default=_strict_default
        )
    except CatalogSerializationError:
        raise
    except TypeError as exc:  # pragma: no cover - json.dumps 兜底
        raise CatalogSerializationError(
            f"catalog typed JSON 序列化失败: {exc}"
        ) from exc


def _canonical_checksum(value: Any) -> str:
    """对 typed JSON 字段的值做 canonical SHA-256 校验和（NEW-P1-74）。"""
    raw = catalog_strict_dumps(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CatalogJsonField(Generic[T]):
    """Typed JSON catalog 字段（NEW-P1-74）。

    信封格式::

        {"schema_version": N, "checksum": sha256(value), "value": <typed payload>}

    - ``dumps()`` 使用严格编码器——未知类型抛 ``CatalogSerializationError``；
    - ``loads(raw, strict=...)`` 严格解码：损坏 JSON 抛 ``CatalogCorruptionError``，
      legacy 裸 JSON（无信封）经 ``migrate`` 迁移，checksum 不匹配抛
      ``CatalogCorruptionError``；
    - production 缺省 forbidden permissive decode（``_resolve_strict`` 自动按
      运行模式推断）。

    参数:
        value: 字段值
        schema_version: schema 版本（可选）
    """

    value: T
    schema_version: int = 1

    _ENVELOPE_KEYS = frozenset({"schema_version", "checksum", "value"})

    def dumps(self, *, checksum: bool = True) -> str:
        """严格序列化（含可选 checksum 信封）。"""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "value": self.value,
        }
        if checksum:
            payload["checksum"] = _canonical_checksum(self.value)
        return catalog_strict_dumps(payload)

    @classmethod
    def loads(
        cls,
        raw: Any,
        *,
        schema_version: int = 1,
        strict: bool | None = None,
        migrate: Any = None,
    ) -> "CatalogJsonField[T]":
        """严格解码；损坏/checksum 不匹配在 strict 下抛 ``CatalogCorruptionError``。"""
        parsed = _parse_json_field(raw, strict=strict)
        if not parsed:
            return cls(None, schema_version=schema_version)  # type: ignore[arg-type]
        if not set(parsed).issuperset({"value"}):
            # legacy 裸 JSON payload（无信封）：整体迁移到当前 schema。
            value: Any = parsed
            if migrate is not None:
                value = migrate(value)
            return cls(value, schema_version=schema_version)
        sv = int(parsed.get("schema_version", 1))
        value = parsed.get("value")
        if migrate is not None:
            while sv < schema_version:
                value = migrate(value)
                sv += 1
        checksum = parsed.get("checksum")
        if checksum:
            actual = _canonical_checksum(value)
            if actual != str(checksum):
                raise CatalogCorruptionError(
                    f"catalog typed JSON 字段 checksum 不匹配（数据损坏或篡改）: "
                    f"expected={checksum}, actual={actual}"
                )
        return cls(value, schema_version=sv)


def _parse_json_field(
    raw: Any,
    *,
    strict: bool | None = None,
) -> dict[str, Any]:
    """将 JSON 字段解析为字典。

    参数:
        raw: 见函数签名
        strict: 显式 strict 标志；缺省从运行模式推断（可选）。

    返回:
        dict[str, Any]

    NEW-P0-45：损坏 JSON 在 strict（production）下抛 ``CatalogCorruptionError``，
    绝不返回 ``{}`` 把「catalog 数据损坏」伪装成「config 缺失」。
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    strict_effective = _resolve_strict(strict)
    if not isinstance(raw, (str, bytes)):
        if strict_effective:
            raise CatalogCorruptionError(
                f"catalog JSON 字段类型非法 {type(raw).__name__}（期望 str/dict）"
            )
        return {}
    try:
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        parsed = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if strict_effective:
            raise CatalogCorruptionError(
                f"catalog JSON 字段损坏（{exc}）: {str(raw)[:80]!r}"
            ) from exc
        return {}
    if isinstance(parsed, dict):
        return parsed
    if strict_effective:
        raise CatalogCorruptionError(
            f"catalog JSON 字段类型非法（期望 dict，实际 {type(parsed).__name__}）"
        )
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
        "domain", "frequency", "cardinality", "temporal_model", "pit_safe",
        # Field-schema metadata attached by the catalog-bound DSL parser path
        # (``api.dsl_parser`` -> ``field()`` -> FieldRef): a bare ``col()``
        # ColumnRef carries only ``name``, so structural source_expr parity must
        # ignore these to compare operator topology alone.
        "field", "dtype", "unit",
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
    factor_version TEXT,
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

-- Review-8 #438: the primary key is (factor_id, partition_key), NOT
-- (factor_id, partition_year).  ``checkpoint_year`` collapses day partitions
-- (20260801 and 20260802 both encode to 202608), so a legacy PK on
-- partition_year collides as soon as a day-partitioned factor records a second
-- day.  partition_year/month/day are retrieval columns only.
CREATE TABLE IF NOT EXISTS factor_materialize_checkpoint (
    factor_id       TEXT NOT NULL,
    partition_year  INTEGER NOT NULL,
    run_id          TEXT NOT NULL,
    status          TEXT NOT NULL,
    error_message   TEXT,
    updated_at      TEXT NOT NULL,
    partition_key   TEXT NOT NULL,
    PRIMARY KEY (factor_id, partition_key)
);

-- R32-P0-012: schema-versioned migration authority。每个迁移版本一行，
-- checksum 绑定迁移报告；catalog 打开时从当前版本顺序迁移到最新。
CREATE TABLE IF NOT EXISTS catalog_schema_version (
    version     INTEGER NOT NULL PRIMARY KEY,
    migrated_at TEXT NOT NULL,
    checksum    TEXT NOT NULL
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

-- NEW-P0-56/57: 已删除（退役）因子的墓碑。delete_factor 只删 catalog 元数据、
-- 不删物理分区文件，因此复用同一 factor_id 会混合旧世代分区。退役墓碑记录
-- 删除时刻 + 全量语义 digest，register 时必须显式 rebuild=True 才能清墓碑复用。
CREATE TABLE IF NOT EXISTS factor_retired (
    factor_id       TEXT PRIMARY KEY,
    retired_at      TEXT NOT NULL,
    factor_version  TEXT,
    ast_hash        TEXT,
    retired_generation TEXT
);
"""


def _version_prefix(version: Any) -> str:
    """取 factor_version 的 16 位显示前缀（兼容 legacy 16 位与 full SHA-256）。"""
    return str(version or "")[:16]


class _ThreadSafeConnection:
    """R20-224: 单个 SQLite 连接 + RLock，供多线程 worker 共享。

    ``check_same_thread=False`` 允许跨线程使用同一连接，但每次调用（execute /
    executemany / executescript / commit / close）都用 RLock 串行化——SQLite 单写
    锁语义下，线程池并行 worker 不会出现 ``SQLite objects created in a thread can
    only be used in that same thread`` 的竞态。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(*args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(*args, **kwargs)

    def executescript(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executescript(*args, **kwargs)

    def commit(self) -> None:
        with self._lock:
            return self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            return self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            return self._conn.close()

    @contextlib.contextmanager
    def transaction(self) -> "Iterator[None]":
        """R32-P0-011: 锁住的事务 —— BEGIN/COMMIT 全程持同一 RLock。

        ``execute()`` 与 ``commit()`` 各自独立锁：线程 A execute 后释放锁，
        线程 B 可在 A 的 commit 前插入 statement，被 A 的 commit 一起提交。
        本 context manager 把 ``BEGIN IMMEDIATE`` → 全部写操作 → ``COMMIT``
        （或失败 ``ROLLBACK``）包在**同一个 RLock** 临界区内，杜绝 interleave。
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise
            else:
                self._conn.execute("COMMIT")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


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
        raw_conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, check_same_thread=False
        )
        raw_conn.row_factory = sqlite3.Row
        # R20-224: 统一走线程安全连接代理（RLock 串行化）。
        self._conn = _ThreadSafeConnection(raw_conn)
        # WAL 模式：大幅提升多进程并发读写能力
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=30000;")
        # R32-P0-010: SQLite foreign keys 默认关闭 —— Schema 虽写了 FK，若没有
        # ``PRAGMA foreign_keys=ON`` 约束实际不执行。启动必须开启并验证 == 1。
        self._conn.execute("PRAGMA foreign_keys=ON;")
        fk_enabled = self._conn.execute("PRAGMA foreign_keys;").fetchone()
        if int(fk_enabled[0]) != 1:
            raise CatalogMigrationError(
                f"SQLite foreign_keys could not be enabled for catalog {self._db_path}"
            )
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate_schema()
        # 清掉 _migrate_schema 遗留的隐式事务（DML 触发），否则 BEGIN IMMEDIATE
        # 报 "cannot start a transaction within a transaction"。
        self._conn.commit()
        # R32-P0-012: schema-versioned migration（独占事务 + checksum + backup）。
        self._run_versioned_migration()
        self._conn.commit()

    def _exec_commit(self, sql: str, params: tuple = ()) -> None:
        """Execute a write statement with bounded SQLITE_BUSY retry.

        Multiple materializer processes contend on the single SQLite write
        lock; even with ``busy_timeout`` a writer can surface
        ``database is locked`` under burst contention.  Bounded sleep-retry
        (``#443`` multiprocess race) makes the write path resilient without
        masking real errors.
        """
        import time

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        raise last_error  # type: ignore[misc]

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
        # R11 #6: factor_version（语义身份 digest 前缀）用于检测「公式没变但执行
        # 语义变了」——production 下禁止沿用同一 factor_id 静默覆盖。
        if "factor_version" not in cols:
            self._conn.execute(
                "ALTER TABLE factor_registry ADD COLUMN factor_version TEXT"
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
        self._migrate_checkpoint_primary_key()

    def _migrate_checkpoint_primary_key(self) -> None:
        """Review-8 #438: rebuild ``factor_materialize_checkpoint`` so the
        primary key is ``(factor_id, partition_key)`` instead of the legacy
        ``(factor_id, partition_year)``.

        ``checkpoint_year`` encodes day partitions to ``year*100+month``
        (20260801 and 20260802 both -> 202608), so the legacy PK collides as
        soon as a day-partitioned factor records a second day.  SQLite cannot
        ``ALTER TABLE`` a primary key, so the table is rebuilt in place; any
        rows that would violate the new key are ignored (they are unreachable
        duplicates of the same logical partition).
        """
        pk = self._conn.execute(
            "SELECT group_concat(name) FROM pragma_table_info('factor_materialize_checkpoint') "
            "WHERE pk > 0"
        ).fetchone()
        if pk is not None and pk[0] == "factor_id,partition_year":
            self._conn.execute("DROP INDEX IF EXISTS idx_factor_mat_ck_partition_key")
            self._conn.execute(
                "CREATE TABLE factor_materialize_checkpoint__new ("
                " factor_id       TEXT NOT NULL,"
                " partition_year  INTEGER NOT NULL,"
                " run_id          TEXT NOT NULL,"
                " status          TEXT NOT NULL,"
                " error_message   TEXT,"
                " updated_at      TEXT NOT NULL,"
                " partition_key   TEXT NOT NULL,"
                " PRIMARY KEY (factor_id, partition_key)"
                ")"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO factor_materialize_checkpoint__new "
                "(factor_id, partition_year, run_id, status, error_message, updated_at, partition_key) "
                "SELECT factor_id, partition_year, run_id, status, error_message, updated_at, "
                "       COALESCE(partition_key, 'year=' || CAST(partition_year AS TEXT)) "
                "FROM factor_materialize_checkpoint"
            )
            self._conn.execute("DROP TABLE factor_materialize_checkpoint")
            self._conn.execute(
                "ALTER TABLE factor_materialize_checkpoint__new "
                "RENAME TO factor_materialize_checkpoint"
            )

    # ------------------------------------------------------------------
    # R32-P0-012/013: schema-versioned migration
    # ------------------------------------------------------------------

    #: R32 catalog schema 版本。每个破坏性迁移递增一次；``catalog_schema_version``
    #: 表记录已应用版本 + checksum。
    CATALOG_SCHEMA_VERSION = 2

    def _run_versioned_migration(self) -> None:
        """从 ``catalog_schema_version`` 记录的当前版本顺序迁移到最新。

        - 独占事务（BEGIN IMMEDIATE）内执行迁移，避免两个进程并发迁移 race；
        - destructive 迁移前 backup catalog 文件；
        - 每个版本记录 checksum + migrated_at。
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM catalog_schema_version"
        ).fetchone()
        current = int(row[0]) if row[0] is not None else 0
        target = self.CATALOG_SCHEMA_VERSION
        if current >= target:
            return
        # destructive 迁移前 backup（灾难恢复：迁移失败可从 backup 回滚）。
        try:
            backup_path = self._db_path.with_name(
                f"{self._db_path.name}.backup-v{target}"
            )
            import shutil

            shutil.copy2(str(self._db_path), str(backup_path))
        except OSError:  # pragma: no cover - backup 失败仍继续（迁移本身仍安全）
            backup_path = None
        for version in range(current + 1, target + 1):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if version == 2:
                    self._migrate_checkpoint_partition_key_not_null()
                migrated_at = datetime.now(timezone.utc).isoformat()
                checksum = hashlib.sha256(
                    json.dumps(
                        {"version": version, "migrated_at": migrated_at},
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:16]
                self._conn.execute(
                    "INSERT OR IGNORE INTO catalog_schema_version "
                    "(version, migrated_at, checksum) VALUES (?, ?, ?)",
                    (version, migrated_at, checksum),
                )
                self._conn.execute("COMMIT")
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise CatalogMigrationError(
                    f"catalog schema migration v{current}->v{target} failed at "
                    f"v{version}; catalog rolled back to pre-migration state. "
                    f"Backup (if taken) at {backup_path}"
                )

    def _migrate_checkpoint_partition_key_not_null(self) -> None:
        """R32-P0-013: partition_key 作为逻辑主键一部分必须 NOT NULL。

        legacy 表定义 ``partition_key TEXT``（可空）；已存在 NULL partition_key
        的行迁移为 ``'year=' || partition_year``。重建（SQLite 无法 ALTER PK）。
        """
        rows = self._conn.execute(
            "PRAGMA table_info(factor_materialize_checkpoint)"
        ).fetchall()
        notnull = {r[1]: r[3] for r in rows}
        if notnull.get("partition_key"):
            return  # 已是 NOT NULL
        self._conn.execute("DROP INDEX IF EXISTS idx_factor_mat_ck_partition_key")
        self._conn.execute(
            "CREATE TABLE factor_materialize_checkpoint__nn ("
            " factor_id       TEXT NOT NULL,"
            " partition_year  INTEGER NOT NULL,"
            " run_id          TEXT NOT NULL,"
            " status          TEXT NOT NULL,"
            " error_message   TEXT,"
            " updated_at      TEXT NOT NULL,"
            " partition_key   TEXT NOT NULL,"
            " PRIMARY KEY (factor_id, partition_key)"
            ")"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO factor_materialize_checkpoint__nn "
            "(factor_id, partition_year, run_id, status, error_message, updated_at, partition_key) "
            "SELECT factor_id, partition_year, run_id, status, error_message, updated_at, "
            "       COALESCE(partition_key, 'year=' || CAST(partition_year AS TEXT)) "
            "FROM factor_materialize_checkpoint"
        )
        self._conn.execute("DROP TABLE factor_materialize_checkpoint")
        self._conn.execute(
            "ALTER TABLE factor_materialize_checkpoint__nn "
            "RENAME TO factor_materialize_checkpoint"
        )

    def catalog_integrity_check(self) -> dict[str, Any]:
        """R32 §5: ``PRAGMA quick_check`` + ``PRAGMA foreign_key_check``。

        返回 ``{"quick_check": "ok"|..., "foreign_key_check": [...], "version": N,
        "foreign_keys_enabled": bool}``。
        """
        quick = self._conn.execute("PRAGMA quick_check;").fetchone()
        fk_violations = [
            dict(r)
            for r in self._conn.execute("PRAGMA foreign_key_check;").fetchall()
        ]
        version_row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM catalog_schema_version"
        ).fetchone()
        fk_enabled = self._conn.execute("PRAGMA foreign_keys;").fetchone()
        return {
            "quick_check": str(quick[0]) if quick else "error",
            "foreign_key_check": fk_violations,
            "schema_version": int(version_row[0]) if version_row and version_row[0] else 0,
            "foreign_keys_enabled": int(fk_enabled[0]) == 1,
        }

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
        semantic_identity_digest: str | None = None,
        production: bool | None = None,
        rebuild: bool = False,
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
            semantic_identity_digest: ``FactorSemanticIdentity.identity_digest()``
                （R11 #6，可选；缺省用 ``ast_hash``）。
            production: 显式 production 标志；缺省从运行模式推断（可选）。
            rebuild: 显式全量重建声明（NEW-P0-56/57）。factor 曾被
                ``delete_factor`` 退役（墓碑存在）时，复用同一 factor_id 必须
                传 ``rebuild=True`` 清除墓碑——否则旧世代物理分区会与新世代数据
                混在同一个目录（抛 ``FactorRetiredError``）。

        返回:
            无


        Raises
                ------
                FactorHashMismatchError
                    ``factor_id`` 已注册但 ``ast_hash`` 与既存记录不同。
                FactorSemanticIdentityMismatchError
                    production 下 ``factor_id`` 已注册且 ``factor_version``
                    （语义身份 digest）与当前不一致——AST 没变但执行语义
                    变了（data_source/universe/market/pit/backend/dialect…），
                    必须新版本或显式全量重建。
                FactorRetiredError
                    ``factor_id`` 已被删除/退役且未传 ``rebuild=True``——复用会
                    混合旧世代物理分区。
                CatalogSerializationError
                    production 下 ``data_source_config`` 含不可 JSON 序列化对象
                    （NEW-P0-44，typed JSON schema，禁止 default=str 字符串化）。
        """
        effective_production = production
        if effective_production is None:
            try:
                from runtime.production_policy import is_production_mode

                effective_production = is_production_mode()
            except Exception as exc:  # R32-P0-014
                # Production authority 解析失败不能 fail-open 到 research —— 那会
                # 绕过 identity / typed-JSON / precision 等全部生产硬门。
                raise ProductionModeResolutionError(
                    f"production authority resolution failed for catalog register "
                    f"of '{factor_id}': {exc!r}"
                ) from exc

        # NEW-P0-56/57: deleted-factor 复用守卫。物理分区文件仍在磁盘上，直接
        # 复用 factor_id 会混合旧世代（2020-2025）与新世代（2026）数据。
        retired = self._get_retired(factor_id)
        if retired is not None:
            if not rebuild:
                raise FactorRetiredError(
                    f"因子 '{factor_id}' 已被删除/退役（retired_at="
                    f"{retired.get('retired_at')}, factor_version="
                    f"{retired.get('factor_version')}）。物理分区文件仍在磁盘上，"
                    f"复用同一 factor_id 会混合旧世代数据。请使用新 factor_id，"
                    f"或显式 rebuild=True 强制全量重建。"
                )
            self._exec_commit(
                "DELETE FROM factor_retired WHERE factor_id = ?", (factor_id,)
            )

        # NEW-P0-44: production 下 data_source_config 走 typed JSON schema——
        # 未知对象直接抛 CatalogSerializationError，绝不 default=str 字符串化。
        ds_json = None
        if data_source_config:
            if effective_production:
                ds_json = catalog_strict_dumps(data_source_config)
            else:
                ds_json = json.dumps(
                    data_source_config, sort_keys=True, default=str, ensure_ascii=False
                )
        # NEW-P0-57: 权威 catalog 存**全量** SHA-256 digest；16 位前缀只用于
        # Parquet 行 / ClickHouse / matrix manifest 等显示侧。
        new_version = str((semantic_identity_digest or ast_hash))
        # Atomic register (Review-8 #443): ``INSERT OR IGNORE`` makes the
        # existence check + insert one statement, so concurrent materializers
        # writing the same factor_id cannot both pass a read-then-insert check
        # and crash on ``UNIQUE constraint failed``.  The hash-conflict verdict
        # is taken against the row that actually won the insert.
        now = datetime.now(timezone.utc).isoformat()
        self._exec_commit(
            "INSERT OR IGNORE INTO factor_registry "
            "(factor_id, author, frequency, description, ast_hash, expression, "
            "data_source_json, factor_version, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                factor_id, author, frequency, description, ast_hash, expression,
                ds_json, new_version, now,
            ),
        )

        existing = self.get_factor_info(factor_id)
        if existing is None:  # pragma: no cover - insert above must have created it
            raise RuntimeError(f"factor '{factor_id}' register did not materialize a row")
        if existing["ast_hash"] != ast_hash:
            raise FactorHashMismatchError(
                f"因子 '{factor_id}' 已注册（Hash={existing['ast_hash'][:12]}…），"
                f"但当前公式 Hash 为 {ast_hash[:12]}…。"
                f"请升级版本号（如改为 '{factor_id}_v2'）后重新落盘。"
            )
        existing_version = existing.get("factor_version")
        # 语义身份版本门：production 下公式没变但执行语义变了 → 拒绝沿用同 ID。
        # 用 16 位前缀比较，兼容 legacy 16 位记录与新的 full SHA-256。
        if existing_version and _version_prefix(existing_version) != _version_prefix(new_version):
            if effective_production:
                raise FactorSemanticIdentityMismatchError(
                    f"因子 '{factor_id}' 已注册（factor_version={existing_version}），"
                    f"但当前语义身份版本为 {new_version}。公式 AST 未变但执行语义"
                    f"（data_source/universe/market/pit/backend/dialect/算子或字段"
                    f"catalog）已变化：production 禁止沿用同一 factor_id 静默覆盖，"
                    f"请升级版本号或显式全量重建。"
                )
        if ds_json is not None and existing.get("data_source_json") != ds_json:
            self._exec_commit(
                "UPDATE factor_registry SET data_source_json = ?, factor_version = ? "
                "WHERE factor_id = ?",
                (ds_json, new_version, factor_id),
            )
        elif existing_version != new_version:
            # NEW-P0-57: 顺带把 legacy 16 位前缀升级为 full SHA-256。
            self._exec_commit(
                "UPDATE factor_registry SET factor_version = ? WHERE factor_id = ?",
                (new_version, factor_id),
            )

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
        self._exec_commit(
            "INSERT INTO factor_watermark (factor_id, start_date, end_date, last_updated, row_count) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id) DO UPDATE SET "
            "start_date=excluded.start_date, end_date=excluded.end_date, "
            "last_updated=excluded.last_updated, row_count=excluded.row_count",
            (factor_id, start_date, end_date, now, row_count),
        )

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

    def delete_factor(self, factor_id: str, *, retire: bool = True) -> None:
        """从 Catalog 中删除因子注册信息和水位线（不删除物理文件）。

        NEW-P0-56/57：删除前写入 ``factor_retired`` 墓碑（退役时刻 + 全量语义
        digest）。物理分区文件仍在磁盘上，复用同一 factor_id 会混合旧世代数据；
        ``register(..., rebuild=True)`` 才能清除墓碑显式全量重建。

        参数:
            factor_id: 因子唯一标识
            retire: 是否写入退役墓碑（可选，缺省 True）

        返回:
            无
        """
        info = self.get_factor_info(factor_id)
        # R32-P0-011: 多语句删除必须整体持同一 RLock 事务（BEGIN IMMEDIATE →
        # 全部 DELETE → COMMIT），线程 B 不能在 A 的 delete 语句之间插入 statement
        # 被 A 的 commit 一起提交。
        with self._conn.transaction():
            if retire and info is not None:
                self._conn.execute(
                    "INSERT INTO factor_retired "
                    "(factor_id, retired_at, factor_version, ast_hash) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(factor_id) DO UPDATE SET "
                    "retired_at=excluded.retired_at, "
                    "factor_version=excluded.factor_version, "
                    "ast_hash=excluded.ast_hash",
                    (
                        factor_id,
                        datetime.now(timezone.utc).isoformat(),
                        info.get("factor_version"),
                        info.get("ast_hash"),
                    ),
                )
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
            # R11 #8: 新增的依赖 edge / full definition 表也要同步删除，否则 factor_id
            # 删除再复用时会残留旧 edge / full-spec（DependencyCatalog 查询按 factor_id
            # 命中，导致「新因子」被旧依赖误触发重算）。
            for table in ("factor_dependency_edge", "factor_full_definition"):
                try:
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE factor_id = ?", (factor_id,)
                    )
                except sqlite3.OperationalError:
                    # 表可能尚未创建（首次运行从未触发过 _ensure_tables）。
                    pass
            self._conn.execute(
                "DELETE FROM factor_registry WHERE factor_id = ?", (factor_id,)
            )

    def _get_retired(self, factor_id: str) -> dict | None:
        """返回 ``factor_retired`` 墓碑记录（None 表示未退役）。"""
        row = self._conn.execute(
            "SELECT * FROM factor_retired WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        return dict(row) if row else None

    def is_factor_retired(self, factor_id: str) -> bool:
        """该 factor_id 是否已被 ``delete_factor`` 退役（NEW-P0-56/57）。

        退役墓碑意味着物理分区文件仍在磁盘上；复用该 factor_id 会混合旧世代
        数据，必须显式 ``register(..., rebuild=True)`` 或换新 factor_id。
        """
        return self._get_retired(factor_id) is not None

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
        self._exec_commit(
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
        self._exec_commit(
            "INSERT INTO factor_materialize_checkpoint "
            "(factor_id, partition_year, partition_key, run_id, status, error_message, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id, partition_key) DO UPDATE SET "
            "partition_year=excluded.partition_year, run_id=excluded.run_id, "
            "status=excluded.status, error_message=excluded.error_message, "
            "updated_at=excluded.updated_at",
            (factor_id, int(partition_year), pkey, run_id, status, error_message, now),
        )

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
        self._exec_commit(
            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?",
            (factor_id,),
        )

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
        # R32-P0-011: upsert + 全量替换 column_dep 必须整体持同一 RLock 事务。
        with self._conn.transaction():
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
