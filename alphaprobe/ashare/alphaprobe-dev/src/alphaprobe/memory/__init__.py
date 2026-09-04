"""Global Factor Intelligence Memory（任务书 §33-§43 / §76）。

机器记忆（完整 formula/AST/metrics/lineage/failure/survival）与 LLM Working
Memory（MemoryPacket 2k-4k token）分离。存储：SQLite/DuckDB 事务 + Parquet 大表。
冷启动 seed（用户尚未放入文件）→ 兼容空目录：ingest 0 条不报错。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _fitness_of(schema_json: Any) -> float | None:
    """从 factor_nodes.schema_json 提取 fitness（single_ic / 归一化值）。

    模块级函数：GlobalMemoryStore（build_memory_packet）与
    MemoryRetriever（_fitness_of 静态方法）共用同一实现。
    """
    if not schema_json:
        return None
    try:
        if isinstance(schema_json, str):
            schema_json = json.loads(schema_json)
    except Exception:
        return None
    if not isinstance(schema_json, dict):
        return None
    for key in ("fitness", "single_ic", "search_fitness", "norm_ic"):
        v = schema_json.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_nodes (
    factor_id TEXT PRIMARY KEY,
    canonical_formula TEXT NOT NULL,
    canonical_ast_hash TEXT NOT NULL,
    signal_equivalence_id TEXT NOT NULL UNIQUE,
    parameter_family_id TEXT,
    orientation INTEGER NOT NULL DEFAULT 1,
    source_system TEXT NOT NULL DEFAULT 'alphaprobe',
    source_external_id TEXT,
    source_snapshot TEXT,
    source_type TEXT,
    field_set TEXT,
    operator_set TEXT,
    schema_json TEXT,
    mechanism TEXT,
    horizon TEXT,
    complexity INTEGER,
    lookback INTEGER,
    first_seen_at REAL,
    last_seen_at REAL,
    times_seen INTEGER DEFAULT 1,
    is_seed INTEGER DEFAULT 0,
    lineage_root INTEGER DEFAULT 0,
    exportable INTEGER DEFAULT 0,
    is_exported INTEGER DEFAULT 0,
    status TEXT DEFAULT 'NEW',
    rediscovery_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_nodes_signal ON factor_nodes(signal_equivalence_id);
CREATE INDEX IF NOT EXISTS idx_nodes_family ON factor_nodes(parameter_family_id);

CREATE TABLE IF NOT EXISTS factor_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    source_snapshot TEXT,
    external_factor_id TEXT NOT NULL,
    UNIQUE(source_system, source_snapshot, external_factor_id)
);

CREATE TABLE IF NOT EXISTS action_edges (
    action_id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    parent_factor_ids TEXT NOT NULL,   -- JSON list，支持多父
    child_factor_id TEXT NOT NULL,
    round_id TEXT,
    campaign_id TEXT,
    generation INTEGER,
    prompt_version TEXT,
    llm_model TEXT,
    cost REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_child ON action_edges(child_factor_id前后);
"""

# 修正：上面占位符在 f-string 外，直接用合法 SQL 重写索引
SCHEMA = SCHEMA.replace(
    "CREATE INDEX IF NOT EXISTS idx_edges_child ON action_edges(child_factor_id前后);",
    "CREATE INDEX IF NOT EXISTS idx_edges_child ON action_edges(child_factor_id);",
)

SCHEMA += """
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    action_id TEXT,
    candidate_factor_id TEXT,
    outcome TEXT,
    rejection_reason TEXT,
    delta_search_fitness REAL,
    novelty_gain REAL,
    pool_utility_gain REAL,
    eval_cost REAL DEFAULT 0,
    llm_cost REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT NOT NULL,
    ts REAL,
    action TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_factor ON retrieval_events(factor_id);

CREATE TABLE IF NOT EXISTS evaluations (
    factor_id TEXT,
    segment TEXT,
    fidelity TEXT,
    metric_bundle TEXT,
    artifact_refs TEXT,
    evaluator_version TEXT,
    data_snapshot_id TEXT,
    universe_snapshot_id TEXT,
    label_spec_hash TEXT,
    created_at REAL,
    PRIMARY KEY (factor_id, segment, fidelity)
);

CREATE TABLE IF NOT EXISTS exploration_state (
    factor_id TEXT,
    action_family TEXT,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    elites INTEGER DEFAULT 0,
    mean_delta_fitness REAL DEFAULT 0,
    mean_novelty_gain REAL DEFAULT 0,
    saturation REAL DEFAULT 0,
    uncertainty REAL DEFAULT 1,
    version_key TEXT DEFAULT '',
    PRIMARY KEY (factor_id, action_family)
);

CREATE TABLE IF NOT EXISTS failure_memory (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT,
    action_family TEXT,
    reason TEXT NOT NULL,
    structural INTEGER DEFAULT 0,
    operator_version_key TEXT,
    created_at REAL,
    expires_at REAL
);

CREATE TABLE IF NOT EXISTS survival_profiles (
    factor_id TEXT PRIMARY KEY,
    status TEXT,
    payload TEXT,
    confidence REAL,
    support_periods INTEGER,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS direction_clusters (
    cluster_id TEXT PRIMARY KEY,
    version INTEGER DEFAULT 1,
    member_count INTEGER DEFAULT 0,
    alias_name TEXT,
    saturation REAL DEFAULT 0,
    survival_rate REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS factor_direction_membership (
    factor_id TEXT,
    cluster_id TEXT,
    confidence REAL,
    PRIMARY KEY (factor_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS market_regime_events (
    event_id TEXT PRIMARY KEY,
    start_date TEXT,
    end_date TEXT,
    description TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS search_run_events (
    event_id TEXT PRIMARY KEY,
    round_id TEXT,
    description TEXT,
    payload TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_search_run_events_round ON search_run_events(round_id);

CREATE TABLE IF NOT EXISTS source_ingestions (
    source_key TEXT PRIMARY KEY,
    source_system TEXT,
    snapshot TEXT,
    ingested_at REAL,
    count INTEGER
);

CREATE TABLE IF NOT EXISTS export_registry (
    factor_id TEXT PRIMARY KEY,
    campaign_id TEXT,
    exported_at REAL,
    manifest_path TEXT
);

CREATE TABLE IF NOT EXISTS system_versions (
    key TEXT PRIMARY KEY,
    value TEXT,
    recorded_at REAL
);
"""


@dataclass
class MemoryPacket:
    """§41：Retriever 只给当前必要信息，约 2k-4k token。"""

    parent: dict[str, Any]
    structural_neighbors: list[dict[str, Any]] = field(default_factory=list)
    numerical_neighbors: list[dict[str, Any]] = field(default_factory=list)
    successful_offspring: list[dict[str, Any]] = field(default_factory=list)
    representative_failures: list[dict[str, Any]] = field(default_factory=list)
    ancestry_summary: str = ""
    unexplored_actions: list[str] = field(default_factory=list)
    saturated_actions: list[str] = field(default_factory=list)
    rare_directions: list[dict[str, Any]] = field(default_factory=list)
    survival_exemplars: list[dict[str, Any]] = field(default_factory=list)
    allowed_fields: list[str] = field(default_factory=list)
    allowed_operators: list[str] = field(default_factory=list)
    cluster_context: dict[str, Any] | None = None

    def to_prompt_text(self) -> str:
        """定长摘要（不塞完整 lineage，§32）。"""
        parts = [
            f"[Parent] {self.parent.get('formula','?')} "
            f"(fitness={self.parent.get('fitness','?')})",
        ]
        if self.structural_neighbors:
            parts.append(
                "[Structural neighbors] "
                + "; ".join(n.get("formula", "?") for n in self.structural_neighbors[:5])
            )
        if self.numerical_neighbors:
            parts.append(
                "[Numerical neighbors] "
                + "; ".join(
                    f"{n.get('formula','?')}(fitness={n.get('fitness','?')})"
                    for n in self.numerical_neighbors[:5]
                )
            )
        if self.successful_offspring:
            parts.append(
                "[Successful offspring] "
                + "; ".join(o.get("formula", "?") for o in self.successful_offspring[:3])
            )
        if self.representative_failures:
            parts.append(
                "[Failures to avoid] "
                + "; ".join(
                    f"{f.get('reason','?')}:{f.get('formula','?')}"
                    for f in self.representative_failures[:3]
                )
            )
        if self.saturated_actions:
            parts.append("[Saturated actions] " + ",".join(self.saturated_actions[:8]))
        if self.unexplored_actions:
            parts.append("[Unexplored actions] " + ",".join(self.unexplored_actions[:8]))
        if self.survival_exemplars:
            parts.append(
                "[2026 survival exemplars] "
                + "; ".join(
                    f"{e.get('formula','?')}(survival_rate={e.get('survival_rate','?')})"
                    for e in self.survival_exemplars[:3]
                )
            )
        if self.cluster_context:
            parts.append(f"[Cluster context] {self.cluster_context}")
        if self.allowed_fields:
            parts.append("[Allowed fields] " + ",".join(self.allowed_fields[:12]))
        if self.allowed_operators:
            parts.append("[Allowed operators] " + ",".join(self.allowed_operators[:12]))
        return "\n".join(parts)


class GlobalMemoryStore:
    """SQLite-backed Global Memory（§38：前期不上 Neo4j）。线程安全。

    LLM 不直接读库（§40）：Retriever → MemoryPacket。
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get(
                "ALPHAPROBE_MEMORY_DB",
                str(Path.home() / "quant_projects/data/alphaprobe/global_memory.sqlite3"),
            )
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- §34 Factor Registry -------------------------------------------------

    def upsert_factor_node(
        self,
        *,
        factor_id: str,
        canonical_formula: str,
        canonical_ast_hash: str,
        signal_equivalence_id: str,
        parameter_family_id: str | None = None,
        orientation: int = 1,
        source_system: str = "alphaprobe",
        source_external_id: str | None = None,
        source_snapshot: str | None = None,
        source_type: str | None = None,
        field_set: list[str] | None = None,
        operator_set: list[str] | None = None,
        schema_json: dict | None = None,
        mechanism: str | None = None,
        horizon: str | None = None,
        complexity: int | None = None,
        lookback: int | None = None,
        is_seed: bool = False,
        lineage_root: bool = False,
        exportable: bool = False,
    ) -> tuple[bool, str | None]:
        """插入或 rediscovery 合并。返回 (created, existing_factor_id)。"""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT factor_id FROM factor_nodes WHERE signal_equivalence_id=?",
                (signal_equivalence_id,),
            ).fetchone()
            if row is not None:
                existing = row[0]
                self._conn.execute(
                    "UPDATE factor_nodes SET times_seen=times_seen+1, last_seen_at=?, "
                    "rediscovery_count=rediscovery_count+1 WHERE factor_id=?",
                    (now, existing),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO factor_aliases"
                    "(factor_id, source_system, source_snapshot, external_factor_id) "
                    "VALUES (?,?,?,?)",
                    (existing, source_system, source_snapshot, source_external_id or factor_id),
                )
                self._conn.commit()
                return False, existing
            self._conn.execute(
                "INSERT INTO factor_nodes(factor_id, canonical_formula, canonical_ast_hash,"
                " signal_equivalence_id, parameter_family_id, orientation, source_system,"
                " source_external_id, source_snapshot, source_type, field_set, operator_set,"
                " schema_json, mechanism, horizon, complexity, lookback, first_seen_at,"
                " last_seen_at, times_seen, is_seed, lineage_root, exportable, status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    factor_id, canonical_formula, canonical_ast_hash, signal_equivalence_id,
                    parameter_family_id, orientation, source_system, source_external_id,
                    source_snapshot, source_type,
                    json.dumps(field_set or []), json.dumps(operator_set or []),
                    json.dumps(schema_json or {}), mechanism, horizon, complexity, lookback,
                    now, now, 1, int(is_seed), int(lineage_root), int(exportable), "NEW",
                ),
            )
            self._conn.commit()
            return True, None

    def get_node(self, factor_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM factor_nodes WHERE factor_id=?", (factor_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM factor_nodes LIMIT 0").description]
        d = dict(zip(cols, row))
        for k in ("field_set", "operator_set", "schema_json"):
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        return d

    # -- §37 Persistent Multi-Parent Lineage ---------------------------------

    def insert_action_edge(
        self,
        *,
        action_id: str,
        action_type: str,
        parent_factor_ids: list[str],
        child_factor_id: str,
        round_id: str,
        campaign_id: str,
        generation: int,
        prompt_version: str = "",
        llm_model: str | None = None,
        cost: float = 0.0,
        latency_ms: int = 0,
        outcome: str = "ok",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO action_edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    action_id, action_type, json.dumps(parent_factor_ids), child_factor_id,
                    round_id, campaign_id, generation, prompt_version, llm_model,
                    cost, latency_ms, outcome,
                ),
            )
            self._conn.commit()

    def lineage_parents(self, child_factor_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT action_id, action_type, parent_factor_ids, generation FROM action_edges "
            "WHERE child_factor_id=? ORDER BY generation",
            (child_factor_id,),
        ).fetchall()
        out = []
        for action_id, atype, parents_json, gen in rows:
            for pid in json.loads(parents_json):
                out.append(
                    {"action_id": action_id, "action_type": atype, "parent_id": pid, "generation": gen}
                )
        return out

    # -- §75.4 attempts -------------------------------------------------------

    def record_attempt(
        self,
        *,
        attempt_id: str,
        action_id: str,
        candidate_factor_id: str | None,
        outcome: str,
        rejection_reason: str | None = None,
        delta_search_fitness: float | None = None,
        novelty_gain: float | None = None,
        pool_utility_gain: float | None = None,
        eval_cost: float = 0.0,
        llm_cost: float = 0.0,
        latency_ms: int = 0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id, action_id, candidate_factor_id, outcome, rejection_reason,
                    delta_search_fitness, novelty_gain, pool_utility_gain,
                    eval_cost, llm_cost, latency_ms, time.time(),
                ),
            )
            self._conn.commit()

    def record_evaluation(self, record: Any) -> None:
        """存 EvaluationRecord（contracts.EvaluationRecord）。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.factor_id, record.segment, record.fidelity,
                    json.dumps(record.metric_bundle), json.dumps(record.artifact_refs),
                    record.evaluator_version, record.data_snapshot_id,
                    record.universe_snapshot_id, record.label_spec_hash,
                    record.created_at.timestamp(),
                ),
            )
            self._conn.commit()

    # -- §24 ExplorationState -------------------------------------------------

    def bump_exploration(
        self,
        *,
        factor_id: str,
        action_family: str,
        success: bool,
        elite: bool = False,
        delta_fitness: float | None = None,
        novelty_gain: float | None = None,
        version_key: str = "",
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts, successes, elites, mean_delta_fitness, mean_novelty_gain"
                " FROM exploration_state WHERE factor_id=? AND action_family=?",
                (factor_id, action_family),
            ).fetchone()
            if row is None:
                attempts, successes, elites, md, mn = 0, 0, 0, 0.0, 0.0
            else:
                attempts, successes, elites, md, mn = row
            attempts += 1
            if success:
                successes += 1
            if elite:
                elites += 1
            if delta_fitness is not None:
                md = (md * (attempts - 1) + delta_fitness) / attempts
            if novelty_gain is not None:
                mn = (mn * (attempts - 1) + novelty_gain) / attempts
            saturation = 1.0 - successes / max(attempts, 1) if attempts >= 10 else 0.0
            self._conn.execute(
                "INSERT OR REPLACE INTO exploration_state VALUES (?,?,?,?,?,?,?,?,?,?)",
                (factor_id, action_family, attempts, successes, elites, md, mn,
                 saturation, 1.0, version_key),
            )
            self._conn.commit()

    def exploration_of(self, factor_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT action_family, attempts, successes, saturation FROM exploration_state"
            " WHERE factor_id=?",
            (factor_id,),
        ).fetchall()
        return [
            {"action_family": r[0], "attempts": r[1], "successes": r[2], "saturation": r[3]}
            for r in rows
        ]

    # -- §57 Retrieval Frequency Decay（P0-B：检索事件 = 被选为 parent 的真实次数） ---

    def record_retrieval(self, *, factor_id: str, action: str | None = None) -> None:
        """记录一次「factor 被检索（选为 parent / 生成）」事件。

        检索频率衰减的计数必须是「被检索了多少次」而非「有几个孩子」：
        每次 select_parents 命中该 factor 作为检索起点时由 pipeline /
        orchestrator 调用。轻量 append 表，SQLite 天然支持并发计数。
        """
        if not factor_id:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO retrieval_events(factor_id, ts, action) VALUES (?,?,?)",
                (str(factor_id), time.time(), action),
            )
            self._conn.commit()

    def retrieval_count_of(self, factor_id: str) -> int:
        """该 factor 累计被检索（选为 parent）的真实次数。"""
        if not factor_id:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM retrieval_events WHERE factor_id=?",
            (str(factor_id),),
        ).fetchone()
        return int(row[0]) if row else 0

    # -- §42 Failure Memory ----------------------------------------------------

    def record_failure(
        self,
        *,
        factor_id: str | None,
        action_family: str,
        reason: str,
        structural: bool,
        operator_version_key: str = "",
        ttl_days: float | None = None,
    ) -> None:
        now = time.time()
        expires = now + ttl_days * 86400 if (ttl_days and not structural) else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO failure_memory(factor_id, action_family, reason, structural,"
                " operator_version_key, created_at, expires_at) VALUES (?,?,?,?,?,?,?)",
                (factor_id, action_family, reason, int(structural), operator_version_key,
                 now, expires),
            )
            self._conn.commit()

    def recent_failures(self, limit: int = 3) -> list[dict[str, Any]]:
        now = time.time()
        rows = self._conn.execute(
            "SELECT factor_id, reason, structural FROM failure_memory"
            " WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [
            {"factor_id": r[0], "reason": r[1], "structural": bool(r[2])} for r in rows
        ]

    def survival_exemplars(self, k: int = 3) -> list[dict[str, Any]]:
        """§43：survival_profiles 表中置信度最高的 k 个 exemplar。

        每行带 formula（从 factor_nodes 联表取）+ survival_rate（confidence）。
        """
        rows = self._conn.execute(
            "SELECT sp.factor_id, sp.status, sp.confidence,"
            " sp.support_periods, fn.canonical_formula"
            " FROM survival_profiles sp LEFT JOIN factor_nodes fn"
            "   ON fn.factor_id = sp.factor_id"
            " ORDER BY sp.confidence DESC LIMIT ?",
            (k,),
        ).fetchall()
        return [
            {
                "factor_id": r[0],
                "status": r[1],
                "survival_rate": r[2],
                "support_periods": r[3],
                "formula": r[4],
            }
            for r in rows
        ]

    # -- §43 Survival Memory ----------------------------------------------------

    def update_survival(self, profile: Any) -> None:
        """§43：写入/更新 survival_profiles。

        接受 SurvivalProfile（dataclass，asdict 可用）或普通对象
        （asdict 会 TypeError，回退为 vars()）。
        """
        try:
            payload = json.dumps(asdict(profile))
        except (TypeError, ValueError):
            payload = json.dumps(vars(profile))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO survival_profiles VALUES (?,?,?,?,?,?)",
                (
                    profile.factor_id, profile.status, payload,
                    profile.confidence, profile.support_periods, time.time(),
                ),
            )
            self._conn.commit()

    # -- §48/§49 Direction Memory ------------------------------------------------

    def upsert_direction(
        self,
        *,
        cluster_id: str,
        member_count: int,
        alias_name: str | None = None,
        saturation: float = 0.0,
        survival_rate: float | None = None,
        version: int = 1,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO direction_clusters(cluster_id, version, member_count, alias_name,"
                " saturation, survival_rate, updated_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(cluster_id) DO UPDATE SET version=excluded.version,"
                " member_count=excluded.member_count, saturation=excluded.saturation,"
                " survival_rate=excluded.survival_rate, updated_at=excluded.updated_at",
                (cluster_id, version, member_count, alias_name, saturation, survival_rate,
                 time.time()),
            )
            self._conn.commit()

    def rare_directions(self, k: int = 5) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT cluster_id, member_count, saturation, survival_rate FROM direction_clusters"
            " ORDER BY member_count ASC LIMIT ?",
            (k,),
        ).fetchall()
        return [
            {"cluster_id": r[0], "member_count": r[1], "saturation": r[2], "survival_rate": r[3]}
            for r in rows
        ]

    def cluster_summary(self) -> dict[str, Any]:
        """§48：direction_clusters 聚合摘要（最小实现，成员数降序）。

        返回 None 值 dict（无数据时 `not cluster_summary()` 为 True）：
        {
            "total_clusters": int,
            "total_members": int,
            "top": [{"cluster_id", "member_count", "survival_rate"}, ...最多 5],
        }
        """
        count = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(member_count), 0) FROM direction_clusters"
        ).fetchone()
        total_clusters, total_members = int(count[0]), int(count[1])
        rows = self._conn.execute(
            "SELECT cluster_id, member_count, survival_rate FROM direction_clusters"
            " ORDER BY member_count DESC LIMIT 5"
        ).fetchall()
        return {
            "total_clusters": total_clusters,
            "total_members": total_members,
            "top": [
                {"cluster_id": r[0], "member_count": r[1], "survival_rate": r[2]}
                for r in rows
            ],
        }

    # -- §50 Regime Memory ---------------------------------------------------------

    def add_regime_event(
        self, *, event_id: str, start_date: str, end_date: str, description: str,
        payload: dict | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO market_regime_events VALUES (?,?,?,?,?)",
                (event_id, start_date, end_date, description, json.dumps(payload or {})),
            )
            self._conn.commit()

    def regime_event_exists(self, event_id: str) -> bool:
        """§50 幂等注册用：event_id 已存在返回 True。"""
        row = self._conn.execute(
            "SELECT 1 FROM market_regime_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return row is not None

    # -- Round 搜索轮事件（与 market regime 分离，§74） -------------------------

    def add_search_run_event(
        self, *, event_id: str, round_id: str, description: str,
        payload: dict | None = None,
    ) -> None:
        """记录一轮搜索运行的完成/汇总事件（幂等 upsert）。

        搜索轮次是引擎运行元数据，不是市场状态——round 事件不允许混进
        market_regime_events（§74：regime 表从此只放市场 regime 事件，
        由 survival.regime.register_2026_event 等写入）。
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO search_run_events VALUES (?,?,?,?,?)",
                (
                    event_id, round_id, description,
                    json.dumps(payload or {}), datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def search_run_event_exists(self, event_id: str) -> bool:
        """幂等注册用：event_id 已存在返回 True。"""
        row = self._conn.execute(
            "SELECT 1 FROM search_run_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return row is not None

    # -- §58 Export Registry ---------------------------------------------------------

    def mark_exported(self, *, factor_id: str, campaign_id: str, manifest_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO export_registry VALUES (?,?,?,?)",
                (factor_id, campaign_id, time.time(), manifest_path),
            )
            self._conn.execute(
                "UPDATE factor_nodes SET is_exported=1, status='EXPORTED' WHERE factor_id=?",
                (factor_id,),
            )
            self._conn.commit()

    def already_exported(self, factor_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM export_registry WHERE factor_id=?", (factor_id,)
        ).fetchone()
        return row is not None

    # -- §52 Seed ingestion（冷启动 10 万公式，无迭代过程） ----------------------

    def ingest_seed_library(
        self,
        entries: Iterator[tuple[str, str, str]] | list[tuple[str, str, str]],
        *,
        source_snapshot: str = "cold_start_v9",
        identity_fn: Any | None = None,
    ) -> int:
        """§35：每个 seed = root FactorNode，parents=[]，lineage_root=true，
        exportable=false。不伪造 lineage。

        entries: (formula, topic, explanation)。identity_fn(formula) ->
        (canonical, ast_hash, signal_id, family_id)。文件缺失时调用方应
        先捕获——本函数只处理已打开的迭代器。
        """
        created = 0
        for i, (formula, topic, _expl) in enumerate(entries):
            if identity_fn is not None:
                canonical, ast_hash, signal_id, family_id = identity_fn(formula)
            else:
                canonical = formula
                ast_hash = signal_id = f"raw:{i}"
                family_id = None
            ok, _ = self.upsert_factor_node(
                factor_id=f"seed_{source_snapshot}_{i}",
                canonical_formula=canonical,
                canonical_ast_hash=ast_hash,
                signal_equivalence_id=signal_id,
                parameter_family_id=family_id,
                source_system="seed_library",
                source_external_id=str(i),
                source_snapshot=source_snapshot,
                source_type="FORMULA_ONLY",
                is_seed=True,
                lineage_root=True,
                exportable=False,
            )
            if ok:
                created += 1
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO source_ingestions VALUES (?,?,?,?,?)",
                (f"seed_library:{source_snapshot}", "seed_library", source_snapshot,
                 time.time(), created),
            )
            self._conn.commit()
        return created

    def get_evaluations(self, factor_id: str) -> list[dict[str, Any]]:
        """evaluations 表中某 factor 的全部评估记录（metric_bundle 已解析）。"""
        rows = self._conn.execute(
            "SELECT segment, fidelity, metric_bundle FROM evaluations"
            " WHERE factor_id=? ORDER BY created_at DESC",
            (factor_id,),
        ).fetchall()
        return [
            {"segment": r[0], "fidelity": r[1], "metric_bundle": json.loads(r[2] or "{}")}
            for r in rows
        ]

    # -- §41 Retriever / MemoryPacket -------------------------------------------

    def build_memory_packet(
        self,
        *,
        parent_node: dict[str, Any],
        neighbors: list[dict[str, Any]] | None = None,
        max_items: int = 5,
        allowed_fields: list[str] | None = None,
        allowed_operators: list[str] | None = None,
    ) -> MemoryPacket:
        """§32/§40：Full Global Memory → Retriever → MemoryPacket → LLM。

        填充：numerical_neighbors（同 family 数值邻居，复用 retriever 的
        _fitness_of/_fitness_from_evaluations 逻辑）、survival_exemplars
        （survival_profiles top exemplars）、cluster_context（direction_clusters
        聚合摘要）、allowed_fields/allowed_operators（可选传入，缺省 []）。
        """
        fid = parent_node.get("factor_id", "")
        explored = {e["action_family"] for e in self.exploration_of(fid)}
        saturated = [
            e["action_family"] for e in self.exploration_of(fid)
            if e.get("saturation", 0) > 0.8
        ]
        all_actions = [
            "REFINE", "WINDOW_SCALE", "FIELD_SUBSTITUTION", "OPERATOR_SUBSTITUTION",
            "STATE_CONDITION", "CROSSOVER", "SCHEMA_EXPLORE", "ROBUSTIFY",
            "GENERATION_REPAIR",
        ]
        unexplored = [a for a in all_actions if a not in explored]
        return MemoryPacket(
            parent=parent_node,
            structural_neighbors=(neighbors or [])[:max_items],
            numerical_neighbors=self._numerical_neighbors_of(fid)[:max_items],
            successful_offspring=[],
            representative_failures=self.recent_failures(limit=3),
            ancestry_summary=parent_node.get("ancestry_summary", ""),
            unexplored_actions=unexplored,
            saturated_actions=saturated,
            rare_directions=self.rare_directions(k=5),
            survival_exemplars=self.survival_exemplars(k=3),
            allowed_fields=list(allowed_fields or []),
            allowed_operators=list(allowed_operators or []),
            cluster_context=self.cluster_summary(),
        )

    # -- §41 numerical neighbors（store 侧实现；retriever 复用） ----------------

    def _numerical_neighbors_of(self, factor_id: str, *, k: int = 5) -> list[dict[str, Any]]:
        """同 parameter family 的最近 k 个节点（排除自身）。

        fitness 优先取 evaluations 表（_fitness_from_evaluations），取不到再
        fallback schema_json（_fitness_of）。family_id 缺失或没有同族节点时
        返回空表。
        """
        if not factor_id:
            return []
        node = self.get_node(factor_id)
        if node is None:
            return []
        family_id = node.get("parameter_family_id")
        if not family_id:
            return []
        rows = self._conn.execute(
            "SELECT factor_id, canonical_formula, schema_json, last_seen_at"
            " FROM factor_nodes"
            " WHERE parameter_family_id=? AND factor_id<>?",
            (family_id, factor_id),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for fid, formula, schema_json, last_seen in rows:
            fitness = self._fitness_from_evaluations(fid)
            if fitness is None:
                fitness = _fitness_of(schema_json)
            out.append(
                {
                    "factor_id": fid,
                    "formula": formula,
                    "fitness": fitness,
                    "last_seen_at": last_seen,
                }
            )
        out.sort(key=lambda d: (d["fitness"] is not None, d["fitness"] or 0.0), reverse=True)
        return out[:k]

    def _fitness_from_evaluations(self, factor_id: str) -> float | None:
        """evaluations 表里优先取 single_ic / rank_ic / fitness（最新段）。

        查不到或值非有限 → None（不伪造）。
        """
        for ev in self.get_evaluations(factor_id):
            bundle = ev.get("metric_bundle") or {}
            for key in ("single_ic", "rank_ic", "fitness", "norm_ic", "search_fitness"):
                v = bundle.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != fv:  # NaN
                    continue
                return fv
        return None


# Phase 7 接线：Retriever / seed ingestion（子模块）从包外 re-export。
# 放在模块末尾避免循环导入（retriever/seed_ingestion 只依赖本模块已定义的类）。
from alphaprobe.memory.retriever import MemoryRetriever  # noqa: E402
from alphaprobe.memory.seed_ingestion import ingest_cold_start_yaml  # noqa: E402

__all__ = [
    "GlobalMemoryStore",
    "MemoryPacket",
    "MemoryRetriever",
    "ingest_cold_start_yaml",
]
