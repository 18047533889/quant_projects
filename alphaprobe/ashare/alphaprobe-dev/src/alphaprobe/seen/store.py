"""Seen Index SQLite 存储（Phase B 有状态持久化包）。

独立 db 文件（默认 ``artifacts/global_seen_index.sqlite3``，env
``ALPHAPROBE_SEEN_DB`` 可覆盖），与 GlobalMemoryStore 的 §39 全表分离，
全部表带 ``seen_`` 前缀。WAL + ``threading.RLock`` 线程安全，
所有语句走索引。

版本化 namespace（§45）：``canonical_ast_hash`` / ``signal_equivalence_id``
的 UNIQUE 以 ``identity_version`` 为前缀 —— 不同 identity_version 的因子
不互判 duplicate，互不覆盖。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS seen_factors (
    factor_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT NOT NULL UNIQUE,
    canonical_ast_hash TEXT NOT NULL,
    signal_equivalence_id TEXT NOT NULL,
    parameter_family_id TEXT,
    orientation INTEGER NOT NULL DEFAULT 1,
    orientations_seen TEXT NOT NULL DEFAULT '1',
    canonical_dsl TEXT,
    identity_version TEXT NOT NULL DEFAULT '1',
    operator_semantics_version TEXT,
    first_seen_at REAL,
    last_seen_at REAL,
    times_seen INTEGER NOT NULL DEFAULT 1,
    source_system TEXT,
    run_id TEXT,
    status TEXT NOT NULL DEFAULT 'NEW',
    UNIQUE (identity_version, canonical_ast_hash),
    UNIQUE (identity_version, signal_equivalence_id)
);
CREATE INDEX IF NOT EXISTS idx_seen_factors_signal
    ON seen_factors(identity_version, signal_equivalence_id);
CREATE INDEX IF NOT EXISTS idx_seen_factors_family
    ON seen_factors(identity_version, parameter_family_id);
CREATE INDEX IF NOT EXISTS idx_seen_factors_status
    ON seen_factors(status);

CREATE TABLE IF NOT EXISTS seen_factor_alias (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_pk INTEGER NOT NULL,
    alias_dsl_hash TEXT NOT NULL,
    alias_dsl TEXT,
    source_system TEXT,
    source_external_id TEXT,
    first_seen_at REAL,
    times_seen INTEGER DEFAULT 1,
    UNIQUE (alias_dsl_hash, factor_pk)
);
CREATE INDEX IF NOT EXISTS idx_seen_alias_hash
    ON seen_factor_alias(alias_dsl_hash);

CREATE TABLE IF NOT EXISTS seen_fingerprints (
    fingerprint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_pk INTEGER NOT NULL,
    fp BLOB NOT NULL,
    fingerprint_version TEXT NOT NULL,
    sample_date_set_id TEXT,
    band_index INTEGER NOT NULL,
    band_value TEXT NOT NULL,
    created_at REAL,
    UNIQUE (factor_pk, fingerprint_version)
);
CREATE INDEX IF NOT EXISTS idx_seen_fp_band
    ON seen_fingerprints(fingerprint_version, band_index, band_value);

CREATE TABLE IF NOT EXISTS seen_subtree_stats (
    subtree_hash TEXT PRIMARY KEY,
    canonical_subtree TEXT,
    appearance_count INTEGER NOT NULL DEFAULT 0,
    unique_factor_count INTEGER NOT NULL DEFAULT 0,
    successful_factor_count INTEGER NOT NULL DEFAULT 0,
    elite_factor_count INTEGER NOT NULL DEFAULT 0,
    success_denominator INTEGER NOT NULL DEFAULT 0,
    last_seen_at REAL
);

CREATE TABLE IF NOT EXISTS seen_subtree_members (
    subtree_hash TEXT NOT NULL,
    factor_pk INTEGER NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    elite INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (subtree_hash, factor_pk)
);
CREATE INDEX IF NOT EXISTS idx_seen_subtree_member
    ON seen_subtree_members(subtree_hash);

CREATE TABLE IF NOT EXISTS seen_families (
    family_id TEXT PRIMARY KEY,
    family_template TEXT,
    tested_parameter_ranges TEXT,
    member_count INTEGER NOT NULL DEFAULT 0,
    best_factor_id TEXT,
    best_search_fitness REAL,
    historical_success_rate REAL,
    saturation REAL NOT NULL DEFAULT 0,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS seen_family_members (
    family_id TEXT NOT NULL,
    factor_pk INTEGER NOT NULL,
    factor_id TEXT NOT NULL,
    parameter_fingerprint TEXT,
    search_fitness REAL,
    outcome TEXT,
    created_at REAL,
    PRIMARY KEY (family_id, factor_pk)
);
CREATE INDEX IF NOT EXISTS idx_seen_fm_family
    ON seen_family_members(family_id);

CREATE TABLE IF NOT EXISTS seen_actions (
    action_key TEXT PRIMARY KEY,
    parent_signal_id TEXT,
    action_type TEXT,
    payload_json TEXT,
    grammar_version TEXT,
    first_seen_at REAL,
    times_seen INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS seen_parse_failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula TEXT,
    reason TEXT,
    created_at REAL
);
"""


class SeenStore:
    """SQLite-backed seen index store。线程安全，版本化 schema。

    只提供底层 SQL 原语 + 事务上下文；语义编排在
    ``dedup_service.DedupService`` / ``reservation`` 等上层模块。
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get(
                "ALPHAPROBE_SEEN_DB", "artifacts/global_seen_index.sqlite3"
            )
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._in_transaction = False
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.isolation_level = None  # 显式事务管理
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_meta(key, value, updated_at) VALUES (?,?,?)",
            ("schema_version", str(SCHEMA_VERSION), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    def __enter__(self) -> "SeenStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 事务 -------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """BEGIN IMMEDIATE ... COMMIT/ROLLBACK（SQLite 串行写天然原子）。

        嵌套规则：方法内 auto-commit 仅在非嵌套时执行（``_in_transaction``
        标志）。事务内调用 bump_times_seen / incr_meta 等不会提前 COMMIT。
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            try:
                yield
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass  # 事务已被内部异常吞掉
                raise
            else:
                try:
                    self._conn.execute("COMMIT")
                except sqlite3.OperationalError:
                    pass  # 内部方法已在末尾提交（理论上不会发生）
            finally:
                self._in_transaction = False

    # -- 通用查询 ---------------------------------------------------------

    def q(self, sql: str, params: tuple = ()) -> list[tuple]:
        """锁内 SELECT 全行（小结果集）。"""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def q1(self, sql: str, params: tuple = ()) -> tuple | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def ex(self, sql: str, params: tuple = ()) -> int:
        """锁内写（调用方负责在事务内或 autocommit 语义下使用）。"""
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._commit()
            return cur.rowcount

    def _commit(self) -> None:
        """提交：嵌套事务内不提交（外层 transaction() 统一 COMMIT）。"""
        if not self._in_transaction:
            self._conn.commit()

    # -- seen_meta key-value ------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.q1("SELECT value FROM seen_meta WHERE key=?", (key,))
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO seen_meta(key, value, updated_at) VALUES (?,?,?)",
                (key, value, time.time()),
            )
            self._commit()

    def incr_meta(self, key: str, by: int = 1) -> int:
        """自增 seen_meta 数值键（不存在则置 by）。返回新值。

        嵌套事务内不提交（外层 transaction() 统一 COMMIT）。
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO seen_meta(key, value, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET"
                " value=CAST(seen_meta.value AS INTEGER)+?, updated_at=?",
                (key, str(by), time.time(), by, time.time()),
            )
            self._commit()
            row = self._conn.execute(
                "SELECT value FROM seen_meta WHERE key=?", (key,)
            ).fetchone()
            return int(row[0])

    def get_meta_json(self, key: str, default: Any = None) -> Any:
        raw = self.get_meta(key)
        return json.loads(raw) if raw is not None else default

    def set_meta_json(self, key: str, value: Any) -> None:
        self.set_meta(key, json.dumps(value, sort_keys=True))

    # -- seen_factors 原语 ---------------------------------------------------

    def insert_factor(self, **kw: Any) -> int:
        """插入因子行，返回 factor_pk。冲突时抛 sqlite3.IntegrityError。

        嵌套事务内不提交（外层 transaction() 统一 COMMIT）。
        """
        cols = [
            "factor_id", "canonical_ast_hash", "signal_equivalence_id",
            "parameter_family_id", "orientation", "orientations_seen",
            "canonical_dsl", "identity_version", "operator_semantics_version",
            "first_seen_at", "last_seen_at", "times_seen", "source_system",
            "run_id", "status",
        ]
        vals = {k: kw.get(k) for k in cols}
        now = time.time()
        vals["first_seen_at"] = now
        vals["last_seen_at"] = now
        vals["times_seen"] = 1
        vals["status"] = kw.get("status", "NEW")
        vals["orientation"] = int(kw.get("orientation", 1) or 1)
        vals["orientations_seen"] = kw.get("orientations_seen", str(vals["orientation"]))
        placeholders = ", ".join("?" for _ in cols)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO seen_factors({', '.join(cols)}) VALUES ({placeholders})",
                [vals[c] for c in cols],
            )
            self._commit()
            return int(cur.lastrowid)

    def bump_times_seen(self, factor_pk: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE seen_factors SET times_seen=times_seen+1, last_seen_at=? "
                "WHERE factor_pk=?",
                (time.time(), factor_pk),
            )
            self._commit()

    def update_status(self, factor_pk: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE seen_factors SET status=? WHERE factor_pk=?", (status, factor_pk)
            )
            self._commit()

    def update_orientations(self, factor_pk: int, orientation: int) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT orientations_seen FROM seen_factors WHERE factor_pk=?",
                (factor_pk,),
            ).fetchone()
            seen = set((row[0] or "").split(",")) if row else set()
            seen.add(str(int(orientation)))
            self._conn.execute(
                "UPDATE seen_factors SET orientations_seen=? WHERE factor_pk=?",
                (",".join(sorted(seen, key=int)), factor_pk),
            )
            self._commit()

    def get_factor_by_pk(self, factor_pk: int) -> dict[str, Any] | None:
        row = self.q1("SELECT * FROM seen_factors WHERE factor_pk=?", (factor_pk,))
        return self._row_to_dict("seen_factors", row)

    def get_factor_by_id(self, factor_id: str) -> dict[str, Any] | None:
        row = self.q1("SELECT * FROM seen_factors WHERE factor_id=?", (factor_id,))
        return self._row_to_dict("seen_factors", row)

    def find_factor_by_hash(
        self, identity_version: str, canonical_ast_hash: str
    ) -> dict[str, Any] | None:
        row = self.q1(
            "SELECT * FROM seen_factors WHERE identity_version=? AND canonical_ast_hash=?",
            (identity_version, canonical_ast_hash),
        )
        return self._row_to_dict("seen_factors", row)

    def find_factor_by_signal(
        self, identity_version: str, signal_equivalence_id: str
    ) -> dict[str, Any] | None:
        row = self.q1(
            "SELECT * FROM seen_factors WHERE identity_version=? AND signal_equivalence_id=?",
            (identity_version, signal_equivalence_id),
        )
        return self._row_to_dict("seen_factors", row)

    def find_factor_by_hash_any_version(self, canonical_ast_hash: str) -> dict[str, Any] | None:
        row = self.q1(
            "SELECT * FROM seen_factors WHERE canonical_ast_hash=? "
            "ORDER BY identity_version LIMIT 1",
            (canonical_ast_hash,),
        )
        return self._row_to_dict("seen_factors", row)

    def find_family_members(
        self, identity_version: str, parameter_family_id: str
    ) -> list[dict[str, Any]]:
        rows = self.q(
            "SELECT * FROM seen_factors WHERE identity_version=? AND parameter_family_id=?",
            (identity_version, parameter_family_id),
        )
        return [self._row_to_dict("seen_factors", r) for r in rows]

    def count_factors(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_factors")
        return int(row[0]) if row else 0

    def factor_pk_by_id(self, factor_id: str) -> int | None:
        row = self.q1("SELECT factor_pk FROM seen_factors WHERE factor_id=?", (factor_id,))
        return int(row[0]) if row else None

    # -- seen_factor_alias ---------------------------------------------------

    def insert_alias(
        self, factor_pk: int, alias_dsl_hash: str, alias_dsl: str | None,
        source_system: str | None, source_external_id: str | None,
    ) -> bool:
        """INSERT OR IGNORE；返回 True 表示新增 alias 行。"""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO seen_factor_alias"
                "(factor_pk, alias_dsl_hash, alias_dsl, source_system,"
                " source_external_id, first_seen_at, times_seen) VALUES (?,?,?,?,?,?,1)",
                (factor_pk, alias_dsl_hash, alias_dsl, source_system, source_external_id, time.time()),
            )
            if cur.rowcount == 0:
                self._conn.execute(
                    "UPDATE seen_factor_alias SET times_seen=times_seen+1"
                    " WHERE alias_dsl_hash=? AND factor_pk=?",
                    (alias_dsl_hash, factor_pk),
                )
            self._commit()
            return cur.rowcount > 0

    def count_aliases_for(self, factor_pk: int) -> int:
        row = self.q1(
            "SELECT COUNT(*) FROM seen_factor_alias WHERE factor_pk=?", (factor_pk,)
        )
        return int(row[0]) if row else 0

    def count_aliases(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_factor_alias")
        return int(row[0]) if row else 0

    # -- seen_fingerprints ----------------------------------------------------

    def upsert_fingerprint(
        self, factor_pk: int, fp: bytes, fingerprint_version: str,
        sample_date_set_id: str | None, bands: list[int],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM seen_fingerprints WHERE factor_pk=? AND fingerprint_version=?",
                (factor_pk, fingerprint_version),
            )
            for bi, bv in enumerate(bands):
                self._conn.execute(
                    "INSERT OR REPLACE INTO seen_fingerprints"
                    "(factor_pk, fp, fingerprint_version, sample_date_set_id,"
                    " band_index, band_value, created_at) VALUES (?,?,?,?,?,?,?)",
                    (factor_pk, fp, fingerprint_version, sample_date_set_id, bi, str(bv), time.time()),
                )
            self._commit()

    def fingerprint_versions(self) -> dict[str, dict[str, Any]]:
        rows = self.q(
            "SELECT fingerprint_version, sample_date_set_id, COUNT(DISTINCT factor_pk)"
            " FROM seen_fingerprints GROUP BY fingerprint_version, sample_date_set_id"
        )
        out: dict[str, dict[str, Any]] = {}
        for ver, sds, cnt in rows:
            key = ver
            prev = out.get(key, {})
            prev["count"] = prev.get("count", 0) + int(cnt)
            prev["sample_date_set_id"] = sds
            out[key] = prev
        return out

    # -- seen_subtree -----------------------------------------------------------

    def upsert_subtree_stat(
        self, subtree_hash: str, canonical_subtree: str | None, factor_pk: int,
        success: bool, elite: bool, now: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_subtree_stats"
                "(subtree_hash, canonical_subtree, appearance_count,"
                " unique_factor_count, successful_factor_count, elite_factor_count,"
                " success_denominator, last_seen_at) VALUES (?,?,0,0,0,0,0,?)",
                (subtree_hash, canonical_subtree, now),
            )
            self._conn.execute(
                "UPDATE seen_subtree_stats SET appearance_count=appearance_count+1,"
                " last_seen_at=? WHERE subtree_hash=?",
                (now, subtree_hash),
            )
            # 先查 member 行是否存在，再决定是否算新 unique factor / 新 success
            member_row = self._conn.execute(
                "SELECT success, elite FROM seen_subtree_members"
                " WHERE subtree_hash=? AND factor_pk=?",
                (subtree_hash, factor_pk),
            ).fetchone()
            is_new_member = member_row is None
            was_success = bool(member_row[0]) if member_row else False
            was_elite = bool(member_row[1]) if member_row else False
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_subtree_members"
                "(subtree_hash, factor_pk, success, elite) VALUES (?,?,?,?)",
                (subtree_hash, factor_pk, int(success), int(elite)),
            )
            if is_new_member:
                self._conn.execute(
                    "UPDATE seen_subtree_stats SET unique_factor_count=unique_factor_count+1"
                    " WHERE subtree_hash=?",
                    (subtree_hash,),
                )
            # 只有首次成功/精英才计数（同一因子重复记录不重复计数）
            if success and not was_success:
                self._conn.execute(
                    "UPDATE seen_subtree_stats SET successful_factor_count="
                    "successful_factor_count+1 WHERE subtree_hash=?",
                    (subtree_hash,),
                )
            if success:
                self._conn.execute(
                    "UPDATE seen_subtree_stats SET success_denominator=success_denominator+1"
                    " WHERE subtree_hash=?",
                    (subtree_hash,),
                )
            if elite and not was_elite:
                self._conn.execute(
                    "UPDATE seen_subtree_stats SET elite_factor_count=elite_factor_count+1"
                    " WHERE subtree_hash=?",
                    (subtree_hash,),
                )
            self._commit()

    def get_subtree_stat(self, subtree_hash: str) -> dict[str, Any] | None:
        row = self.q1("SELECT * FROM seen_subtree_stats WHERE subtree_hash=?", (subtree_hash,))
        return self._row_to_dict("seen_subtree_stats", row)

    def count_subtrees(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_subtree_stats")
        return int(row[0]) if row else 0

    # -- seen_families ---------------------------------------------------------

    def upsert_family(
        self, family_id: str, family_template: str | None, factor_pk: int,
        factor_id: str, parameter_fingerprint: str | None = None,
        search_fitness: float | None = None, outcome: str | None = None,
        tested_parameter_ranges: str | None = None, commit: bool = True,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_families"
                "(family_id, family_template, tested_parameter_ranges, member_count,"
                " best_search_fitness, updated_at) VALUES (?,?,?,0,NULL,?)",
                (family_id, family_template, tested_parameter_ranges, time.time()),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_family_members"
                "(family_id, factor_pk, factor_id, parameter_fingerprint,"
                " search_fitness, outcome, created_at) VALUES (?,?,?,?,?,?,?)",
                (family_id, factor_pk, factor_id, parameter_fingerprint,
                 search_fitness, outcome, time.time()),
            )
            self._conn.execute(
                "UPDATE seen_families SET member_count="
                "(SELECT COUNT(*) FROM seen_family_members WHERE family_id=?),"
                " updated_at=? WHERE family_id=?",
                (family_id, time.time(), family_id),
            )
            if search_fitness is not None:
                self._conn.execute(
                    "UPDATE seen_families SET best_search_fitness=?, best_factor_id=?"
                    " WHERE family_id=? AND"
                    " (best_search_fitness IS NULL OR ? > best_search_fitness)",
                    (search_fitness, factor_id, family_id, search_fitness),
                )
            # 成功率 = 有 fitness 的成员里 outcome 为成功（success/ok/evaluated）的比例
            self._conn.execute(
                "UPDATE seen_families SET historical_success_rate="
                "(SELECT CASE WHEN COUNT(*) = 0 THEN NULL ELSE"
                " SUM(CASE WHEN outcome IN ('success','ok','evaluated') THEN 1 ELSE 0 END)"
                " * 1.0 / COUNT(*) END FROM seen_family_members WHERE family_id=?)"
                " WHERE family_id=?",
                (family_id, family_id),
            )
            if commit:
                self._commit()

    def get_family(self, family_id: str) -> dict[str, Any] | None:
        row = self.q1("SELECT * FROM seen_families WHERE family_id=?", (family_id,))
        return self._row_to_dict("seen_families", row)

    def count_families(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_families")
        return int(row[0]) if row else 0

    # -- seen_actions ----------------------------------------------------------

    def try_record_action(
        self, action_key: str, parent_signal_id: str | None, action_type: str | None,
        payload_json: str, grammar_version: str | None,
    ) -> bool:
        """返回 True 表示首次见到（first_time）。"""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO seen_actions(action_key, parent_signal_id,"
                " action_type, payload_json, grammar_version, first_seen_at, times_seen)"
                " VALUES (?,?,?,?,?,?,1)",
                (action_key, parent_signal_id, action_type, payload_json,
                 grammar_version, time.time()),
            )
            if cur.rowcount == 0:
                self._conn.execute(
                    "UPDATE seen_actions SET times_seen=times_seen+1 WHERE action_key=?",
                    (action_key,),
                )
            self._commit()
            return cur.rowcount > 0

    def count_actions(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_actions")
        return int(row[0]) if row else 0

    # -- parse failures ----------------------------------------------------------

    def record_parse_failure(self, formula: str | None, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO seen_parse_failures(formula, reason, created_at) VALUES (?,?,?)",
                (formula, reason, time.time()),
            )
            self._commit()

    def count_parse_failures(self) -> int:
        row = self.q1("SELECT COUNT(*) FROM seen_parse_failures")
        return int(row[0]) if row else 0

    # -- helpers ------------------------------------------------------------------

    def _row_to_dict(self, table: str, row: tuple | None) -> dict[str, Any] | None:
        if row is None:
            return None
        with self._lock:
            cols = [
                d[0] for d in self._conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            ]
        return dict(zip(cols, row))

    def raw_signal_count(self) -> int:
        """去重后的 signal 数（= factor 行数，因 signal 每 version UNIQUE）。"""
        return self.count_factors()
