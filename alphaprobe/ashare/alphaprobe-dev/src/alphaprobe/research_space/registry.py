"""SchemaRegistry —— schema 级持久注册表（plan Task 12 / Part G #29）。

设计
----
- **独立 SQLite**（默认 ``:memory:``；文件路径经 ``ALPHAPROBE_SCHEMA_DB``），
  不往 GlobalMemoryStore / ledger 加表。建表风格参考 memory/__init__.py
  与 ledger.py（sqlite3 + threading.Lock + executescript + 幂等 upsert）。
- schema 统计**独立于公式层**：registry 只按 schema_id 聚合实现层的
  reward / success / elite / failure 计数；公式层的参数/身份不在本表。
- failed implementation **不自动标记 schema failed**：记录 failure_reason 到
  implementation 行、schema 计数 +1 n_failed_impl；schema 的 ``is_failed``
  仅当显式 ``mark_schema_failed`` 或全部实现均失败才为 True。
- schema reward 聚合**带置信收缩**（Part G #29：无/极少实现 → 收缩向中性，
  绝不当最好）。与 fitness/confidence 同款公式（本地实现，避免 import 细节）。
- 版本化：``schema_versions`` 表维护 schema_id 每个语义版本的统计行；
  ``current_stats`` 冗余最新行便于读取；bump 语义版本后旧统计保留不混。

确定性 / schema_id
------------------
schema_id 由 SchemaPlan 九维语义派生（contracts.SchemaPlan.derive_schema_id）；
同名/描述/owner 变化不改变 id → 同一 schema 不同公式实现共享 schema_id。
registry.plan_for(schema_id) 幂等 upsert（首次登记 canonical tags）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from alphaprobe.research_space.contracts import (
    DEFAULT_SHRINKAGE_K,
    SCHEMA_ID_VERSION,
    ImplementationOutcome,
    ImplementationRecord,
    SchemaIdentity,
    SchemaPlan,
    SchemaStats,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALPHAPROBE_SCHEMA_DB",
    "SchemaRegistry",
    "default_schema_db_path",
    "reliability_of",
    "shrink_reward",
]

#: 环境变量默认 DB 路径（与 memory / ledger 同风格；缺省 :memory: 由构造器处理）
ALPHAPROBE_SCHEMA_DB = "ALPHAPROBE_SCHEMA_DB"


def default_schema_db_path() -> str:
    """返回文件 DB 缺省路径（``data/alphaprobe/schema_registry.sqlite3``）。"""
    return str(Path.home() / "quant_projects" / "data" / "alphaprobe" / "schema_registry.sqlite3")


def reliability_of(n_eff: int | float | None, *, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """reliability = sqrt(n_eff/(n_eff+k))；n_eff<=0 → 0（收缩向中性）。"""
    n = float(n_eff) if n_eff is not None else 0.0
    n = max(0.0, n)
    if not math.isfinite(n):
        n = 0.0
    k = max(float(k), 1e-9)
    return math.sqrt(n / (n + k))


def shrink_reward(
    raw_mean: float,
    *,
    n_eff: int | float | None,
    k: float = DEFAULT_SHRINKAGE_K,
) -> float:
    """U_conf = 0.5 + reliability*(U-0.5)；n_eff=0 → 0.5（中性，绝不当最好）。"""
    u = max(0.0, min(1.0, float(raw_mean)))
    r = reliability_of(n_eff, k=k)
    return round(0.5 + r * (u - 0.5), 6)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schemas (
    schema_id TEXT PRIMARY KEY,
    semantic_version TEXT NOT NULL,
    canonical_tags TEXT NOT NULL,        -- JSON dict[str,str]
    name TEXT DEFAULT '',
    description TEXT DEFAULT '',
    owner TEXT DEFAULT '',
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS implementations (
    implementation_id TEXT PRIMARY KEY,
    schema_id TEXT NOT NULL,
    formula TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT DEFAULT '',
    created_at REAL,
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_impl_schema ON implementations(schema_id);

CREATE TABLE IF NOT EXISTS impl_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    implementation_id TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    reward REAL NOT NULL,
    reward_conf REAL NOT NULL,
    fitness REAL,
    rank_ic REAL,
    long_short_quality REAL,
    stability REAL,
    elite INTEGER DEFAULT 0,
    n_eval INTEGER DEFAULT 1,
    payload TEXT DEFAULT '{}',
    recorded_at REAL
);
CREATE INDEX IF NOT EXISTS idx_result_impl ON impl_results(implementation_id);

-- schema 级统计（版本化：每个 schema_id 每个语义版本一行）
CREATE TABLE IF NOT EXISTS schema_stats (
    schema_id TEXT NOT NULL,
    version TEXT NOT NULL,
    n_impl INTEGER DEFAULT 0,
    n_eval INTEGER DEFAULT 0,
    n_success INTEGER DEFAULT 0,
    n_elite INTEGER DEFAULT 0,
    n_failed_impl INTEGER DEFAULT 0,
    reward_sum REAL DEFAULT 0,
    reward_conf_sum REAL DEFAULT 0,
    is_failed INTEGER DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (schema_id, version)
);

CREATE TABLE IF NOT EXISTS schema_flags (
    schema_id TEXT NOT NULL,
    flag TEXT NOT NULL,
    value TEXT NOT NULL,
    recorded_at REAL,
    PRIMARY KEY (schema_id, flag)
);

CREATE INDEX IF NOT EXISTS idx_stats_schema ON schema_stats(schema_id);
"""

#: 需要 n_eff 才计入的计数（实现数；schema 无实现不更新均值）
_N_REWARD = "reward"


@dataclass
class SchemaRegistry:
    """schema 级持久注册表（内存 + SQLite 可持久化）。

    ``db_path=None`` → 内存 SQLite（测试默认）；传入文件路径则持久化
    （生产接 ALPHAPROBE_SCHEMA_DB 环境变量）。
    """

    db_path: str | None = None
    _conn: Any | None = None
    _lock: Any | None = None
    #: 收缩强度（Part J3：中性默认，不宣称最优）
    shrinkage_k: float = DEFAULT_SHRINKAGE_K

    def __post_init__(self) -> None:
        if self._conn is None:
            path = self.db_path
            if path is None:
                path = os.environ.get(ALPHAPROBE_SCHEMA_DB, ":memory:")
            self.db_path = str(path)
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        if self._lock is None:
            self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # schema 登记 / 查询
    # ------------------------------------------------------------------

    def register_plan(
        self,
        plan: SchemaPlan,
        *,
        vocabulary: Any | None = None,
    ) -> SchemaIdentity:
        """登记一个 SchemaPlan（校验语义 → 派生 schema_id → 幂等 upsert）。

        Returns
        -------
        SchemaIdentity
            稳定身份（schema_id + semantic_version + canonical_tags）。
        """
        from alphaprobe.research_space.schema import validate_schema_plan

        validate_schema_plan(plan, vocabulary)
        identity = SchemaIdentity(
            schema_id=plan.derive_schema_id(),
            semantic_version=SCHEMA_ID_VERSION,
            canonical_tags=dict(plan.canonical_tags()),
        )
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO schemas(schema_id, semantic_version, canonical_tags,"
                " name, description, owner, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    identity.schema_id,
                    identity.semantic_version,
                    json.dumps(identity.canonical_tags, sort_keys=True),
                    plan.name,
                    plan.description,
                    plan.owner,
                    now,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE schemas SET updated_at=? WHERE schema_id=?",
                (now, identity.schema_id),
            )
            self._conn.commit()
        return identity

    def identity_of(self, plan: SchemaPlan) -> SchemaIdentity:
        """只派生身份（不落库）——测试/检索用。"""
        return SchemaIdentity(
            schema_id=plan.derive_schema_id(),
            semantic_version=SCHEMA_ID_VERSION,
            canonical_tags=dict(plan.canonical_tags()),
        )

    def plan_for(self, schema_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT canonical_tags, name, description, owner FROM schemas WHERE schema_id=?",
                (schema_id,),
            ).fetchone()
        if row is None:
            return None
        tags_json, name, desc, owner = row
        return {
            "schema_id": schema_id,
            "canonical_tags": json.loads(tags_json or "{}"),
            "name": name or "",
            "description": desc or "",
            "owner": owner or "",
        }

    def schema_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT schema_id FROM schemas ORDER BY schema_id").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # implementation 层
    # ------------------------------------------------------------------

    def register_implementation(
        self,
        *,
        implementation_id: str,
        plan: SchemaPlan | None = None,
        schema_id: str | None = None,
        formula: str = "",
        vocabulary: Any | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> SchemaIdentity:
        """登记一个实现（必须先有 schema）。返回 schema 身份。

        failed 实现在评估后由 ``record_implementation_outcome`` 更新；此处
        status 默认 ``recorded``（plan：失败实现不自动标记 schema failed）。
        """
        if plan is not None:
            identity = self.register_plan(plan, vocabulary=vocabulary)
            sid = identity.schema_id
        elif schema_id:
            sid = str(schema_id)
            existing = self.plan_for(sid)
            if existing is None:
                # 允许只给 schema_id 登记（调用方已自行保证 schema 存在）
                raise ValueError(f"schema_id {sid!r} 未登记（先 register_plan）")
        else:
            raise ValueError("register_implementation 需要 plan 或 schema_id")
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO implementations(implementation_id, schema_id, formula,"
                " status, failure_reason, created_at, meta) VALUES (?,?,?,?,?,?,?)",
                (
                    str(implementation_id),
                    sid,
                    str(formula or ""),
                    "recorded",
                    "",
                    now,
                    json.dumps(dict(meta or {})) if meta else "{}",
                ),
            )
            self._conn.commit()
        return SchemaIdentity(
            schema_id=sid,
            semantic_version=SCHEMA_ID_VERSION,
            canonical_tags=dict((self.plan_for(sid) or {}).get("canonical_tags") or {}),
        )

    def implementations_of(self, schema_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT implementation_id, formula, status, failure_reason, created_at, meta"
                " FROM implementations WHERE schema_id=? ORDER BY created_at",
                (schema_id,),
            ).fetchall()
        out = []
        for impl_id, formula, status, reason, created, meta in rows:
            out.append(
                {
                    "implementation_id": impl_id,
                    "schema_id": schema_id,
                    "formula": formula,
                    "status": status,
                    "failure_reason": reason,
                    "created_at": created,
                    "meta": json.loads(meta or "{}"),
                }
            )
        return out

    # ------------------------------------------------------------------
    # outcome 聚合（schema 层统计独立于公式层）
    # ------------------------------------------------------------------

    def record_implementation_outcome(
        self,
        *,
        implementation_id: str,
        outcome: ImplementationOutcome,
        formula: str = "",
        schema_id: str | None = None,
        vocabulary: Any | None = None,
    ) -> SchemaStats:
        """记录一个实现结果并聚合 schema 级统计（带置信收缩）。

        幂等语义：同一 implementation 重复 record → 同一结果行（DB 主键
        按 implementation_id + recorded_at 轮询——见下：以 implementation_id
        最新行 upsert，重复调用只更新不新增）。

        Returns
        -------
        SchemaStats
            更新后的 schema 聚合统计。
        """
        if schema_id is None:
            sid = self._schema_id_for_implementation(implementation_id)
            if sid is None:
                raise ValueError(f"implementation {implementation_id!r} 未登记")
        else:
            sid = str(schema_id)
        n_eval = max(1, int(outcome.n_eval or 1))
        raw = max(0.0, min(1.0, float(outcome.reward)))
        # 单条 reward 先按自身 n_eval 收缩，再进 schema 聚合（两次收缩语义：
        # 实现层缺样本也向中性；schema 层按 n_impl 再收缩 → 版本化置信计数）。
        conf = shrink_reward(raw, n_eff=n_eval, k=self.shrinkage_k)
        now = time.time()
        with self._lock:
            # upsert 最新一条 result（同一 implementation 只留一条最新）
            prev = self._conn.execute(
                "SELECT result_id FROM impl_results WHERE implementation_id=?",
                (implementation_id,),
            ).fetchone()
            if prev is not None:
                self._conn.execute(
                    "UPDATE impl_results SET reward=?, reward_conf=?, fitness=?, rank_ic=?,"
                    " long_short_quality=?, stability=?, elite=?, n_eval=?, payload=?,"
                    " recorded_at=? WHERE result_id=?",
                    (
                        raw,
                        conf,
                        _opt(outcome.fitness),
                        _opt(outcome.rank_ic),
                        _opt(outcome.long_short_quality),
                        _opt(outcome.stability),
                        int(bool(outcome.elite)),
                        n_eval,
                        json.dumps(dict(outcome.payload or {})),
                        now,
                        prev[0],
                    ),
                )
            else:
                self._conn.execute(
                    "INSERT INTO impl_results(implementation_id, schema_id, reward, reward_conf,"
                    " fitness, rank_ic, long_short_quality, stability, elite, n_eval, payload,"
                    " recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        implementation_id,
                        sid,
                        raw,
                        conf,
                        _opt(outcome.fitness),
                        _opt(outcome.rank_ic),
                        _opt(outcome.long_short_quality),
                        _opt(outcome.stability),
                        int(bool(outcome.elite)),
                        n_eval,
                        json.dumps(dict(outcome.payload or {})),
                        now,
                    ),
                )
            # status 更新
            status = "failed" if outcome.failed else ("promoted" if outcome.elite else "evaluated")
            self._conn.execute(
                "UPDATE implementations SET status=? WHERE implementation_id=?",
                (status, implementation_id),
            )
            self._conn.commit()
        return self._recompute_stats(sid)

    def _schema_id_for_implementation(self, implementation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT schema_id FROM implementations WHERE implementation_id=?",
                (implementation_id,),
            ).fetchone()
        return row[0] if row else None

    def _recompute_stats(self, schema_id: str, version: str = SCHEMA_ID_VERSION) -> SchemaStats:
        """全量重算一个 schema 的统计（简单正确优先；实现数 < 1k 足够快）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT reward, reward_conf, elite, n_eval FROM impl_results WHERE schema_id=?",
                (schema_id,),
            ).fetchall()
            impl_rows = self._conn.execute(
                "SELECT status, failure_reason FROM implementations WHERE schema_id=?",
                (schema_id,),
            ).fetchall()
        n_impl = len(impl_rows)
        n_eval = int(sum(max(1, int(r[3] or 1)) for r in rows))
        n_elite = int(sum(1 for r in rows if r[2]))
        n_failed = sum(1 for i in impl_rows if i[0] == "failed" or i[1])
        n_success = max(0, n_impl - n_failed)
        if rows:
            reward_sum = float(sum(r[0] for r in rows))
            conf_sum = float(sum(r[1] for r in rows))
            # schema 层聚合：实现层 raw reward 均值 按「有效实现数 n_impl」
            # 一次置信收缩（Part G #29：实现少 → 向中性 0.5；实现多 →
            # 收敛向 raw）。raw 已含单条 n_eval 收缩的行，这里不二次收缩
            # n_eval 维度（防止 n_eval 小导致永远贴近 0.5 —— 那会让实现
            # 多也没法收敛，违背「大样本强置信」#20）。
            raw_mean = reward_sum / len(rows)
            agg = shrink_reward(raw_mean, n_eff=n_impl, k=self.shrinkage_k)
        else:
            reward_sum = 0.0
            conf_sum = 0.0
            agg = 0.5
        # is_failed：显式 flag 或 全部实现均 failed（非空实现集）
        with self._lock:
            flag = self._conn.execute(
                "SELECT value FROM schema_flags WHERE schema_id=? AND flag='failed'",
                (schema_id,),
            ).fetchone()
        is_failed = bool(flag and flag[0] == "1") or (n_impl > 0 and n_failed == n_impl and n_failed > 0)
        stats = SchemaStats(
            schema_id=schema_id,
            version=version,
            n_impl=n_impl,
            n_eval=n_eval,
            n_success=n_success,
            n_elite=n_elite,
            n_failed_impl=n_failed,
            reward_sum=round(reward_sum, 6),
            reward_conf_sum=round(conf_sum, 6),
            aggregate_reward=agg,
            is_failed=is_failed,
            updated_at=time.time(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_stats(schema_id, version, n_impl, n_eval,"
                " n_success, n_elite, n_failed_impl, reward_sum, reward_conf_sum, is_failed,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    schema_id,
                    version,
                    stats.n_impl,
                    stats.n_eval,
                    stats.n_success,
                    stats.n_elite,
                    stats.n_failed_impl,
                    stats.reward_sum,
                    stats.reward_conf_sum,
                    int(stats.is_failed),
                    stats.updated_at,
                ),
            )
            self._conn.commit()
        return stats

    def stats_for(self, schema_id: str) -> SchemaStats:
        """读取 schema 最新统计（无实现 → 中性缺省，不落库）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT version, n_impl, n_eval, n_success, n_elite, n_failed_impl,"
                " reward_sum, reward_conf_sum, is_failed, updated_at"
                " FROM schema_stats WHERE schema_id=? ORDER BY version DESC LIMIT 1",
                (schema_id,),
            ).fetchone()
        if row is None:
            return SchemaStats(schema_id=schema_id)
        return SchemaStats(
            schema_id=schema_id,
            version=row[0],
            n_impl=row[1],
            n_eval=row[2],
            n_success=row[3],
            n_elite=row[4],
            n_failed_impl=row[5],
            reward_sum=row[6],
            reward_conf_sum=row[7],
            aggregate_reward=shrink_reward(
                (row[6] / row[1]) if row[1] else 0.5,
                n_eff=row[1],
                k=self.shrinkage_k,
            ),
            is_failed=bool(row[8]),
            updated_at=row[9],
        )

    # ------------------------------------------------------------------
    # failed 语义
    # ------------------------------------------------------------------

    def mark_implementation_failed(
        self,
        *,
        implementation_id: str,
        failure_reason: str,
    ) -> SchemaStats:
        """把单个实现标 failed（**不**自动把 schema 标 failed；计数 +1）。"""
        sid = self._schema_id_for_implementation(implementation_id)
        if sid is None:
            raise ValueError(f"implementation {implementation_id!r} 未登记")
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE implementations SET status='failed', failure_reason=? WHERE implementation_id=?",
                (str(failure_reason), implementation_id),
            )
            self._conn.commit()
        return self._recompute_stats(sid)

    def mark_schema_failed(self, schema_id: str, *, reason: str = "") -> SchemaStats:
        """显式把 schema 标记 failed（调用方明确判定整个 schema 死亡时才用）。"""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_flags(schema_id, flag, value, recorded_at)"
                " VALUES (?, 'failed', '1', ?)",
                (schema_id, now),
            )
            self._conn.commit()
        return self._recompute_stats(schema_id)

    # ------------------------------------------------------------------
    # saturation / elite rate（供 Retriever 消费）
    # ------------------------------------------------------------------

    def saturation(self, schema_id: str) -> float | None:
        """schema 饱和指标（实现数口径）：n_impl/(n_impl+k)。缺实现 → None。"""
        st = self.stats_for(schema_id)
        if st.n_impl <= 0:
            return None
        return st.n_impl / (st.n_impl + self.shrinkage_k)

    def summary(self) -> list[dict[str, Any]]:
        """全部 schema 的统计摘要（供 Retriever / 报告）。"""
        out: list[dict[str, Any]] = []
        for sid in self.schema_ids():
            st = self.stats_for(sid)
            d = st.as_dict()
            d["schema_id"] = sid
            out.append(d)
        return out


def _opt(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
