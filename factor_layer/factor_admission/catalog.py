from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_BASE_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS factor_registry (
    factor_id   TEXT PRIMARY KEY,
    author      TEXT NOT NULL,
    frequency   TEXT NOT NULL,
    description TEXT,
    ast_hash    TEXT NOT NULL,
    expression  TEXT,
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
"""


_ADMISSION_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS factor_evaluation_runs (
    run_id           TEXT PRIMARY KEY,
    factor_id        TEXT NOT NULL,
    run_dir          TEXT NOT NULL,
    summary_path     TEXT NOT NULL,
    sample_start     TEXT,
    sample_end       TEXT,
    universe_id      TEXT,
    primary_horizon  INTEGER NOT NULL,
    config_hash      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id)
);

CREATE TABLE IF NOT EXISTS factor_evaluation_summary (
    run_id                      TEXT NOT NULL,
    horizon                     INTEGER NOT NULL,
    ic_mean                     REAL,
    ic_std                      REAL,
    ic_ir                       REAL,
    ic_win_rate                 REAL,
    rank_ic_mean                REAL,
    rank_ic_std                 REAL,
    rank_ic_ir                  REAL,
    rank_ic_win_rate            REAL,
    top_minus_bottom_mean       REAL,
    monotonicity_score          REAL,
    n_dates                     INTEGER,
    avg_universe_size           REAL,
    long_short_total_return     REAL,
    long_short_ann_return       REAL,
    long_short_ann_vol          REAL,
    long_short_sharpe           REAL,
    long_short_max_drawdown     REAL,
    PRIMARY KEY (run_id, horizon),
    FOREIGN KEY (run_id) REFERENCES factor_evaluation_runs(run_id)
);

CREATE TABLE IF NOT EXISTS factor_admission_status (
    factor_id               TEXT PRIMARY KEY,
    latest_run_id           TEXT,
    latest_approved_run_id  TEXT,
    status                  TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id),
    FOREIGN KEY (latest_run_id) REFERENCES factor_evaluation_runs(run_id),
    FOREIGN KEY (latest_approved_run_id) REFERENCES factor_evaluation_runs(run_id)
);

CREATE TABLE IF NOT EXISTS factor_admission_decisions (
    decision_id      TEXT PRIMARY KEY,
    factor_id        TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    decision         TEXT NOT NULL,
    decided_by       TEXT NOT NULL,
    reason           TEXT,
    policy_name      TEXT,
    policy_snapshot  TEXT,
    decided_at       TEXT NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id),
    FOREIGN KEY (run_id) REFERENCES factor_evaluation_runs(run_id)
);
"""


class AdmissionCatalog:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_BASE_SCHEMA_SQL)
        self._conn.executescript(_ADMISSION_SCHEMA_SQL)
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        self._conn.close()

    def ensure_factor_registered(self, factor_id: str) -> None:
        row = self._conn.execute(
            "SELECT factor_id FROM factor_registry WHERE factor_id = ?",
            (factor_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"因子尚未在 factor lake catalog 注册: {factor_id}")

    def upsert_evaluation_run(self, payload: dict[str, object]) -> None:
        self.ensure_factor_registered(str(payload["factor_id"]))
        self._conn.execute(
            "INSERT INTO factor_evaluation_runs "
            "(run_id, factor_id, run_dir, summary_path, sample_start, sample_end, universe_id, primary_horizon, config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "factor_id=excluded.factor_id, run_dir=excluded.run_dir, summary_path=excluded.summary_path, "
            "sample_start=excluded.sample_start, sample_end=excluded.sample_end, universe_id=excluded.universe_id, "
            "primary_horizon=excluded.primary_horizon, config_hash=excluded.config_hash, created_at=excluded.created_at",
            (
                payload["run_id"],
                payload["factor_id"],
                payload["run_dir"],
                payload["summary_path"],
                payload.get("sample_start"),
                payload.get("sample_end"),
                payload.get("universe_id"),
                int(payload["primary_horizon"]),
                payload["config_hash"],
                payload["created_at"],
            ),
        )
        self._conn.commit()

    def replace_evaluation_summary(self, run_id: str, rows: list[dict[str, object]]) -> None:
        self._conn.execute("DELETE FROM factor_evaluation_summary WHERE run_id = ?", (run_id,))
        for row in rows:
            self._conn.execute(
                "INSERT INTO factor_evaluation_summary "
                "(run_id, horizon, ic_mean, ic_std, ic_ir, ic_win_rate, rank_ic_mean, rank_ic_std, rank_ic_ir, rank_ic_win_rate, "
                "top_minus_bottom_mean, monotonicity_score, n_dates, avg_universe_size, long_short_total_return, long_short_ann_return, "
                "long_short_ann_vol, long_short_sharpe, long_short_max_drawdown) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(row["horizon"]),
                    row.get("ic_mean"),
                    row.get("ic_std"),
                    row.get("ic_ir"),
                    row.get("ic_win_rate"),
                    row.get("rank_ic_mean"),
                    row.get("rank_ic_std"),
                    row.get("rank_ic_ir"),
                    row.get("rank_ic_win_rate"),
                    row.get("top_minus_bottom_mean"),
                    row.get("monotonicity_score"),
                    row.get("n_dates"),
                    row.get("avg_universe_size"),
                    row.get("long_short_total_return"),
                    row.get("long_short_ann_return"),
                    row.get("long_short_ann_vol"),
                    row.get("long_short_sharpe"),
                    row.get("long_short_max_drawdown"),
                ),
            )
        self._conn.commit()

    def record_decision(
        self,
        *,
        factor_id: str,
        run_id: str,
        decision: str,
        decided_by: str,
        reason: str | None,
        policy_name: str,
        policy_snapshot: dict[str, object],
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        decision_id = f"{factor_id}:{run_id}:{now}"
        self._conn.execute(
            "INSERT INTO factor_admission_decisions "
            "(decision_id, factor_id, run_id, decision, decided_by, reason, policy_name, policy_snapshot, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                factor_id,
                run_id,
                decision,
                decided_by,
                reason,
                policy_name,
                json.dumps(policy_snapshot, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        self._conn.commit()
        return decision_id

    def update_factor_status(self, *, factor_id: str, latest_run_id: str, approved: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_factor_status(factor_id)
        latest_approved_run_id = existing["latest_approved_run_id"] if existing else None
        status = "approved" if approved else "evaluated"
        if approved:
            latest_approved_run_id = latest_run_id
        self._conn.execute(
            "INSERT INTO factor_admission_status (factor_id, latest_run_id, latest_approved_run_id, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(factor_id) DO UPDATE SET "
            "latest_run_id=excluded.latest_run_id, latest_approved_run_id=excluded.latest_approved_run_id, "
            "status=excluded.status, updated_at=excluded.updated_at",
            (factor_id, latest_run_id, latest_approved_run_id, status, now),
        )
        self._conn.commit()

    def get_factor_status(self, factor_id: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM factor_admission_status WHERE factor_id = ?",
            (factor_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_evaluation_run(self, run_id: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM factor_evaluation_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_factor_library(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT r.factor_id, r.author, r.frequency, r.description, r.created_at, "
            "w.start_date, w.end_date, w.row_count, "
            "s.status, s.latest_run_id, s.latest_approved_run_id, er.primary_horizon, "
            "es.ic_mean, es.rank_ic_mean, es.long_short_sharpe, es.top_minus_bottom_mean "
            "FROM factor_registry r "
            "LEFT JOIN factor_watermark w ON r.factor_id = w.factor_id "
            "LEFT JOIN factor_admission_status s ON r.factor_id = s.factor_id "
            "LEFT JOIN factor_evaluation_runs er ON s.latest_run_id = er.run_id "
            "LEFT JOIN factor_evaluation_summary es ON er.run_id = es.run_id AND er.primary_horizon = es.horizon "
            "ORDER BY r.factor_id"
        ).fetchall()
        return [dict(row) for row in rows]