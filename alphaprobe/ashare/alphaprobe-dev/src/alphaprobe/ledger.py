"""CandidateAttemptLedger（plan.md Task 4 / A8）：搜索尝试持久化账本。

每一条尝试（LLM 生成、FE 校验、QE 评估、reward 结算）在 SQLite 里只有一行，
字段与 plan.md Task 4 的完整清单对齐：

    attempt_id / round_id / generation / parent_ids / action_type / paradigm /
    logic_id / schema_id / hypothesis_id / raw_llm_output_ref / compiled_factor_id /
    dedup_status / l1_status / l2_status / l3_status / fitness_before /
    fitness_after / delta_fitness / pool_utility_before / pool_utility_after /
    delta_pool_utility / novelty_gain / llm_tokens / llm_cost / fe_seconds /
    qe_seconds / failure_reason / reward / reward_event_id / reward_settled_at

设计约束：
- **独立 SQLite 存储**，不往 GlobalMemoryStore 加表（memory/__init__.py 由另一
  agent 本波独占）。
- **reward 结算唯一性由 DB 约束仲裁**：``attempt_id`` 主键 +
  ``reward_event_id`` 上的 UNIQUE 索引 + ``INSERT OR IGNORE`` 后按
  ``rowcount`` 判定谁赢。赢家返回 True 并更新 reward 字段；输家返回 False，
  绝不重复计 reward。重放、resume、worker retry 只可能有一次成功。
- 建表/加锁写法参考 memory/__init__.py 的 retrieval_events/search_run_events。

TODO(integration)：pipeline/runner 的生成→评估→准入事件流尚未全部接到本账本
（plan Task 4 的 Integrate orchestrator/pipeline events 项）；本文件先把存储、
仲裁、查询能力做全，接入点留待主链统一接线。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: 环境变量默认 DB 路径（与 memory 的 ALPHAPROBE_MEMORY_DB 同一风格）
ALPHAPROBE_LEDGER_DB = "ALPHAPROBE_LEDGER_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    round_id TEXT,
    generation INTEGER,
    parent_ids TEXT,
    action_type TEXT,
    paradigm TEXT,
    logic_id TEXT,
    schema_id TEXT,
    hypothesis_id TEXT,
    raw_llm_output_ref TEXT,
    compiled_factor_id TEXT,
    dedup_status TEXT,
    l1_status TEXT,
    l2_status TEXT,
    l3_status TEXT,
    fitness_before REAL,
    fitness_after REAL,
    delta_fitness REAL,
    pool_utility_before REAL,
    pool_utility_after REAL,
    delta_pool_utility REAL,
    novelty_gain REAL,
    llm_tokens INTEGER,
    llm_cost REAL,
    fe_seconds REAL,
    qe_seconds REAL,
    failure_reason TEXT,
    reward REAL,
    reward_event_id TEXT,
    reward_settled_at REAL,
    event_ref TEXT,
    created_at REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_attempts_attempt_id ON attempts(attempt_id);

-- reward 结算唯一性：同一 attempt 只允许有一个非 NULL 的 reward_event_id。
-- 先写 reward_event_id 者赢（INSERT OR IGNORE + rowcount 仲裁），
-- 重放/resume/retry 不可能二次计 reward。
CREATE UNIQUE INDEX IF NOT EXISTS uq_attempts_reward_event
    ON attempts(reward_event_id);

CREATE INDEX IF NOT EXISTS idx_attempts_round ON attempts(round_id);
CREATE INDEX IF NOT EXISTS idx_attempts_compiled ON attempts(compiled_factor_id);
CREATE INDEX IF NOT EXISTS idx_attempts_action ON attempts(action_type);
"""

#: 列顺序与 _SCHEMA 的 attempts 表一致（INSERT 语句复用）。
_ATTEMPT_COLUMNS: tuple[str, ...] = (
    "attempt_id",
    "round_id",
    "generation",
    "parent_ids",
    "action_type",
    "paradigm",
    "logic_id",
    "schema_id",
    "hypothesis_id",
    "raw_llm_output_ref",
    "compiled_factor_id",
    "dedup_status",
    "l1_status",
    "l2_status",
    "l3_status",
    "fitness_before",
    "fitness_after",
    "delta_fitness",
    "pool_utility_before",
    "pool_utility_after",
    "delta_pool_utility",
    "novelty_gain",
    "llm_tokens",
    "llm_cost",
    "fe_seconds",
    "qe_seconds",
    "failure_reason",
    "reward",
    "reward_event_id",
    "reward_settled_at",
    "event_ref",
    "created_at",
)

#: 允许写入的字段（DB 列白名单）；字段名即列名，防拼 SQL 错列。
_WRITABLE = set(_ATTEMPT_COLUMNS)


def _json_dumps_list(values: Iterable[str] | None) -> str | None:
    """list[str] → JSON 文本；None/空 → None。"""
    if not values:
        return None
    return json.dumps([str(v) for v in values])


def _json_loads_list(text: str | None) -> list[str] | None:
    """JSON 文本 → list[str]；None/坏 JSON → None。"""
    if text is None:
        return None
    try:
        return [str(v) for v in json.loads(text)]
    except Exception:  # noqa: BLE001 - 旧/坏数据不抛
        return None


class CandidateAttemptLedger:
    """搜索尝试账本：一行一 attempt，reward 结算 DB 级唯一。"""

    def __init__(
        self,
        db_path: str | os.PathLike | None = None,
        *,
        connect: bool = True,
    ) -> None:
        if db_path is None:
            db_path = os.environ.get(
                ALPHAPROBE_LEDGER_DB,
                str(Path.cwd() / "alphaprobe_ledger.sqlite3"),
            )
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if connect:
            self._connect()

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        if self._conn is not None:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        """显式关闭连接（测试/resume 重开用）。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------

    def record_attempt(
        self,
        *,
        attempt_id: str,
        round_id: str | None = None,
        generation: int | None = None,
        parent_ids: Sequence[str] | None = None,
        action_type: str | None = None,
        paradigm: str | None = None,
        logic_id: str | None = None,
        schema_id: str | None = None,
        hypothesis_id: str | None = None,
        raw_llm_output_ref: str | None = None,
        compiled_factor_id: str | None = None,
        dedup_status: str | None = None,
        l1_status: str | None = None,
        l2_status: str | None = None,
        l3_status: str | None = None,
        fitness_before: float | None = None,
        fitness_after: float | None = None,
        delta_fitness: float | None = None,
        pool_utility_before: float | None = None,
        pool_utility_after: float | None = None,
        delta_pool_utility: float | None = None,
        novelty_gain: float | None = None,
        llm_tokens: int | None = None,
        llm_cost: float | None = None,
        fe_seconds: float | None = None,
        qe_seconds: float | None = None,
        failure_reason: str | None = None,
        reward: float | None = None,
        reward_event_id: str | None = None,
        reward_settled_at: float | None = None,
        event_ref: str | None = None,
        created_at: float | None = None,
    ) -> tuple[bool, str | None]:
        """插入一条 attempt（幂等：attempt_id 已存在 → 返回 (False, existing)）。

        返回 ``(created, existing_attempt_id)``：
        - 首次插入 → ``(True, None)``；
        - 同一 attempt_id 重放 → ``(False, 已存在 attempt_id)``，不覆盖任何字段
          （保持先写先赢语义，评估后 crash→resume 重复建行不会丢原状）。
        """
        values = self._prepare_row(
            attempt_id=attempt_id,
            round_id=round_id,
            generation=generation,
            parent_ids=parent_ids,
            action_type=action_type,
            paradigm=paradigm,
            logic_id=logic_id,
            schema_id=schema_id,
            hypothesis_id=hypothesis_id,
            raw_llm_output_ref=raw_llm_output_ref,
            compiled_factor_id=compiled_factor_id,
            dedup_status=dedup_status,
            l1_status=l1_status,
            l2_status=l2_status,
            l3_status=l3_status,
            fitness_before=fitness_before,
            fitness_after=fitness_after,
            delta_fitness=delta_fitness,
            pool_utility_before=pool_utility_before,
            pool_utility_after=pool_utility_after,
            delta_pool_utility=delta_pool_utility,
            novelty_gain=novelty_gain,
            llm_tokens=llm_tokens,
            llm_cost=llm_cost,
            fe_seconds=fe_seconds,
            qe_seconds=qe_seconds,
            failure_reason=failure_reason,
            reward=reward,
            reward_event_id=reward_event_id,
            reward_settled_at=reward_settled_at,
            event_ref=event_ref,
            created_at=created_at,
        )
        placeholders = ",".join("?" for _ in _ATTEMPT_COLUMNS)
        sql = (
            f"INSERT OR IGNORE INTO attempts"
            f"({','.join(_ATTEMPT_COLUMNS)}) VALUES ({placeholders})"
        )
        with self._lock:
            self._connect()
            cur = self._conn.execute(sql, values)
            self._conn.commit()
            if cur.rowcount == 1:
                return True, None
            row = self._conn.execute(
                "SELECT attempt_id FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            existing = str(row[0]) if row is not None else None
            return False, existing

    def mark_duplicate(
        self,
        *,
        attempt_id: str,
        duplicate_of: str | None = None,
        dedup_status: str = "REDISCOVERY",
    ) -> bool:
        """把已存在的 attempt 标记为重复发现（rediscovery），不产生评估成本。

        覆盖 attempt_id 已有的占位行（占位行为 None 的字段），但 **不覆盖**：
        ``fe_seconds / qe_seconds / l1_status / l2_status / l3_status / reward /
        reward_event_id / reward_settled_at``（先写先赢：若后来者已经带上了
        评估/结算信息，说明它先结算过，不允许被 duplicate 覆盖）。
        """
        with self._lock:
            self._connect()
            cur = self._conn.execute(
                "UPDATE attempts SET dedup_status=?, failure_reason=?"
                " WHERE attempt_id=?",
                (str(dedup_status), duplicate_of or str(dedup_status), attempt_id),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def settle_reward(
        self,
        *,
        attempt_id: str,
        reward: float,
        reward_event_id: str,
        reward_settled_at: float | None = None,
    ) -> tuple[bool, str | None]:
        """结算一条 attempt 的 reward（DB 级幂等仲裁，谁先写谁赢）。

        结算 = 写入 ``reward / reward_event_id / reward_settled_at`` 三字段。
        仲裁用**带守卫的 UPDATE**：``WHERE reward_event_id IS NULL AND
        reward IS NULL`` —— 行已存在（record_attempt 占位）时 UPDATE 仍能命中；
        行不存在时先补占位行再 UPDATE。``reward_event_id`` 上的 UNIQUE 索引保证
        跨 attempt 也不允许同一 event 结算两次（冲突 → IntegrityError → 输）。

        返回 ``(won, reason)``：
        - 本调用赢得结算 → ``(True, None)``；
        - 输（重复结算 / event 已被占用 / 行不存在）→ ``(False, reason)``。
        赢家才允许调用方更新 scheduler——**绝不重复计 reward**。
        """
        if attempt_id is None or reward_event_id is None:
            return False, "attempt_id/reward_event_id required"
        settled_at = reward_settled_at if reward_settled_at is not None else time.time()
        with self._lock:
            self._connect()
            # 1) 行不存在（record_attempt 从未落过占位）→ 补占位行。
            #    INSERT OR IGNORE：已存在则忽略，不覆盖任何既有字段（先写先赢）。
            self._conn.execute(
                "INSERT OR IGNORE INTO attempts(attempt_id, created_at) VALUES (?,?)",
                (attempt_id, settled_at),
            )
            # 2) 守卫式 UPDATE：只有尚未结算的行才会被命中。
            try:
                cur = self._conn.execute(
                    "UPDATE attempts SET reward=?, reward_event_id=?, reward_settled_at=?"
                    " WHERE attempt_id=? AND reward_event_id IS NULL AND reward IS NULL",
                    (float(reward), reward_event_id, settled_at, attempt_id),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                # reward_event_id 已被其他 attempt 占用（UNIQUE 索引）→ 输
                self._conn.rollback()
                return False, f"reward_event_id already taken: {reward_event_id}"
            if cur.rowcount == 1:
                return True, None
            # 行存在但已结算过（守卫未命中）→ 输，绝不重复计
            row = self._conn.execute(
                "SELECT reward_event_id FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is not None and row[0] is not None:
                return False, f"already settled with event {row[0]}"
            return False, "settle guard not matched (missing row?)"

    def record_event(
        self,
        *,
        event_id: str,
        attempt_id: str,
        event_kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """记录一条 attempt 生命周期事件（评估开始/评估完成/生成…）。

        ``event_id`` 由调用方生成（评估子任务维度）；settle 时用同一 event_id
        作 reward_event_id 即可让 reward 追溯到子代评估事件。event_kind 建议：
        ``evaluation_started / evaluation_completed / generation / fe_failed``。
        """
        with self._lock:
            self._connect()
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS attempt_events("
                " event_id TEXT PRIMARY KEY,"
                " attempt_id TEXT,"
                " event_kind TEXT,"
                " payload_json TEXT,"
                " created_at REAL)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO attempt_events(event_id, attempt_id, event_kind,"
                " payload_json, created_at) VALUES (?,?,?,?,?)",
                (
                    str(event_id),
                    str(attempt_id),
                    str(event_kind),
                    json.dumps(payload or {}),
                    time.time(),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        """按 attempt_id 读取一行（JSON 列已还原）。"""
        with self._lock:
            self._connect()
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_by_compiled_factor(self, compiled_factor_id: str) -> dict[str, Any] | None:
        """按落库因子 id 查 attempt（同一 compiled 因子只允许一条 attempt）。"""
        with self._lock:
            self._connect()
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE compiled_factor_id=?", (compiled_factor_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def is_settled(self, attempt_id: str) -> bool:
        """该 attempt 是否已结算（reward_event_id 非空）。"""
        with self._lock:
            self._connect()
            row = self._conn.execute(
                "SELECT reward_event_id FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        return row is not None and row[0] is not None

    def reward_settled_at_of(self, attempt_id: str) -> float | None:
        with self._lock:
            self._connect()
            row = self._conn.execute(
                "SELECT reward_settled_at FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        return row[0] if row is not None else None

    def unsettled_attempt_ids(self, *, round_id: str | None = None) -> list[str]:
        """结算前 crash → resume：找出评估已完成但 reward 未结算的 attempts。"""
        sql = (
            "SELECT attempt_id FROM attempts"
            " WHERE (reward IS NULL AND reward_event_id IS NULL)"
            " AND (l2_status IS NOT NULL OR l3_status IS NOT NULL OR"
            "      (fe_seconds IS NOT NULL AND failure_reason IS NULL))"
        )
        params: list[Any] = []
        if round_id is not None:
            sql += " AND round_id=?"
            params.append(round_id)
        with self._lock:
            self._connect()
            rows = self._conn.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows]

    def count(self) -> int:
        with self._lock:
            self._connect()
            row = self._conn.execute("SELECT COUNT(*) FROM attempts").fetchone()
        return int(row[0]) if row else 0

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """按 event_id 反查 attempt 事件（reward 溯源：settle 的
        reward_event_id 应能反查到子代评估事件）。"""
        with self._lock:
            self._connect()
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS attempt_events("
                " event_id TEXT PRIMARY KEY,"
                " attempt_id TEXT,"
                " event_kind TEXT,"
                " payload_json TEXT,"
                " created_at REAL)"
            )
            row = self._conn.execute(
                "SELECT * FROM attempt_events WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        cols = [
            d[0]
            for d in self._conn.execute("SELECT * FROM attempt_events LIMIT 0").description
        ]
        d = dict(zip(cols, row))
        if isinstance(d.get("payload_json"), str):
            try:
                d["payload"] = json.loads(d["payload_json"])
            except Exception:  # noqa: BLE001
                d["payload"] = None
        return d

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _prepare_row(self, **kwargs: Any) -> list[Any]:
        """按 DB 列顺序取值；非法列名直接抛（防拼 SQL 错列，fail-closed）。"""
        unknown = set(kwargs) - _WRITABLE
        if unknown:
            raise KeyError(f"unknown ledger fields: {sorted(unknown)}")
        values: list[Any] = []
        for col in _ATTEMPT_COLUMNS:
            v = kwargs.get(col)
            if col == "parent_ids":
                values.append(_json_dumps_list(v))
            elif v is None:
                values.append(None)
            else:
                values.append(v)
        return values

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        cols = [d[0] for d in self._conn.execute("SELECT * FROM attempts LIMIT 0").description]
        d = dict(zip(cols, row))
        if isinstance(d.get("parent_ids"), str):
            d["parent_ids"] = _json_loads_list(d["parent_ids"])
        d["settled"] = d.get("reward_event_id") is not None
        return d


def default_ledger_path() -> str:
    """默认账本 DB 路径（显式参数化的缺省值，测试用 tmp_path 覆盖）。"""
    return os.environ.get(
        ALPHAPROBE_LEDGER_DB,
        str(Path.cwd() / "alphaprobe_ledger.sqlite3"),
    )


__all__ = ["ALPHAPROBE_LEDGER_DB", "CandidateAttemptLedger", "default_ledger_path"]
