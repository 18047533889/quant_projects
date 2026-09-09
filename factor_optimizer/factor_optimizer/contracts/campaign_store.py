"""Durable SQLite control-plane state for FO campaigns.

This store is deliberately small: it persists budget reservations and the
one logical sealed-test access decision. Large factor/test data remains behind
the existing opaque provider boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


class CampaignStateError(RuntimeError):
    """Raised when durable campaign state rejects an unsafe transition."""


@dataclass(frozen=True)
class Reservation:
    campaign_id: str
    attempt_id: str
    state: str
    estimated_cost: float
    lease_expires_at: float


class SQLiteCampaignStore:
    """Transactional campaign budget and sealed-access store.

    Separate instances/processes coordinate through ``BEGIN IMMEDIATE``. A
    reservation that never started may expire and release its estimate. Once
    marked STARTED it must be settled with actual resource usage; it is never
    silently refunded by lease expiry.
    """

    def __init__(self, path: str | Path):
        self.path = str(Path(path))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    max_evaluations INTEGER NOT NULL CHECK(max_evaluations > 0),
                    max_cost REAL NOT NULL CHECK(max_cost > 0),
                    evaluations_used INTEGER NOT NULL DEFAULT 0,
                    cost_used REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                    attempt_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('RESERVED','STARTED','SETTLED','RELEASED')),
                    estimated_cost REAL NOT NULL CHECK(estimated_cost >= 0),
                    actual_cost REAL,
                    lease_expires_at REAL NOT NULL,
                    PRIMARY KEY(campaign_id, attempt_id)
                );
                CREATE TABLE IF NOT EXISTS sealed_access (
                    scope_hash TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    candidate_set_hash TEXT NOT NULL,
                    dataset_identity TEXT NOT NULL,
                    split_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    profile_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('FROZEN','RUNNING','INFRA_FAILED','COMPLETED')),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    result_ref TEXT,
                    result_json TEXT,
                    exposed_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repair_knowledge (
                    effective_spec_hash TEXT NOT NULL,
                    evaluation_intent_hash TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    retry_condition TEXT,
                    payload_json TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(effective_spec_hash,evaluation_intent_hash,context_hash)
                );
                CREATE TABLE IF NOT EXISTS frozen_parameters (
                    campaign_id TEXT NOT NULL,
                    parent_factor_id TEXT NOT NULL,
                    repair_family TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY(campaign_id,parent_factor_id,repair_family)
                );
                CREATE TABLE IF NOT EXISTS hypothesis_attempts (
                    campaign_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    effective_spec_hash TEXT,
                    evaluation_intent_hash TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0,
                    has_pvalue INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(campaign_id,proposal_id)
                );
                CREATE TABLE IF NOT EXISTS hypothesis_families (
                    campaign_id TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    family_hash TEXT NOT NULL,
                    ledger_head TEXT NOT NULL,
                    ledger_count INTEGER NOT NULL,
                    artifact_json TEXT NOT NULL,
                    bound_at REAL NOT NULL,
                    PRIMARY KEY(campaign_id,family_id),
                    UNIQUE(family_hash)
                );
                """
            )
            db.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(sealed_access)").fetchall()
            }
            if "exposed_at" not in columns:
                db.execute("ALTER TABLE sealed_access ADD COLUMN exposed_at REAL")
                self._after_exposure_column_added(db)
                # The old schema could not distinguish pre-read from
                # post-read attempts. Conservatively burn every historical
                # attempted scope; only pristine FROZEN/count=0 rows remain
                # eligible for their first read.
                db.execute(
                    """UPDATE sealed_access SET exposed_at=updated_at
                       WHERE attempt_count>0 OR state!='FROZEN'"""
                )
            db.commit()

    def _after_exposure_column_added(self, db: sqlite3.Connection) -> None:
        """Test seam for proving ALTER/backfill transaction crash safety."""
        return None

    def create_campaign(self, campaign_id: str, *, max_evaluations: int, max_cost: float) -> None:
        if not campaign_id or max_evaluations < 1 or not math.isfinite(max_cost) or max_cost <= 0:
            raise ValueError("invalid durable campaign budget")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT max_evaluations,max_cost FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if existing is None:
                db.execute(
                    "INSERT INTO campaigns(campaign_id,max_evaluations,max_cost,created_at) VALUES(?,?,?,?)",
                    (campaign_id, max_evaluations, max_cost, time.time()),
                )
            elif (existing["max_evaluations"], existing["max_cost"]) != (max_evaluations, max_cost):
                raise CampaignStateError("campaign budget identity cannot change")
            db.commit()

    def reserve(self, campaign_id: str, attempt_id: str, estimated_cost: float, *, lease_seconds: float = 60) -> bool:
        if not attempt_id or not math.isfinite(estimated_cost) or estimated_cost < 0 or lease_seconds <= 0:
            raise ValueError("invalid reservation")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Only work that never started is safe to release automatically.
            db.execute(
                "UPDATE reservations SET state='RELEASED' WHERE campaign_id=? AND state='RESERVED' AND lease_expires_at<=?",
                (campaign_id, now),
            )
            prior = db.execute(
                "SELECT state FROM reservations WHERE campaign_id=? AND attempt_id=?",
                (campaign_id, attempt_id),
            ).fetchone()
            if prior is not None:
                db.commit()
                return prior["state"] in ("RESERVED", "STARTED", "SETTLED")
            campaign = db.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            if campaign is None:
                raise CampaignStateError("unknown campaign")
            active = db.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(estimated_cost),0) cost FROM reservations WHERE campaign_id=? AND state IN ('RESERVED','STARTED')",
                (campaign_id,),
            ).fetchone()
            if campaign["evaluations_used"] + active["n"] >= campaign["max_evaluations"]:
                db.commit(); return False
            if campaign["cost_used"] + active["cost"] + estimated_cost > campaign["max_cost"]:
                db.commit(); return False
            db.execute(
                "INSERT INTO reservations VALUES(?,?,?,?,?,?)",
                (campaign_id, attempt_id, "RESERVED", estimated_cost, None, now + lease_seconds),
            )
            db.commit(); return True

    def start(self, campaign_id: str, attempt_id: str) -> None:
        self._transition_reservation(campaign_id, attempt_id, "RESERVED", "STARTED")

    def release(self, campaign_id: str, attempt_id: str) -> None:
        self._transition_reservation(campaign_id, attempt_id, "RESERVED", "RELEASED")

    def settle(self, campaign_id: str, attempt_id: str, actual_cost: float) -> None:
        if not math.isfinite(actual_cost) or actual_cost < 0:
            raise ValueError("actual_cost must be finite and non-negative")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state,estimated_cost FROM reservations WHERE campaign_id=? AND attempt_id=?",
                (campaign_id, attempt_id),
            ).fetchone()
            if row is None or row["state"] != "STARTED":
                raise CampaignStateError("only a started reservation can settle")
            db.execute(
                "UPDATE reservations SET state='SETTLED',actual_cost=? WHERE campaign_id=? AND attempt_id=?",
                (actual_cost, campaign_id, attempt_id),
            )
            db.execute(
                "UPDATE campaigns SET evaluations_used=evaluations_used+1,cost_used=cost_used+? WHERE campaign_id=?",
                (actual_cost, campaign_id),
            )
            db.commit()

    def _transition_reservation(self, campaign_id: str, attempt_id: str, expected: str, target: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE reservations SET state=? WHERE campaign_id=? AND attempt_id=? AND state=?",
                (target, campaign_id, attempt_id, expected),
            ).rowcount
            if changed != 1:
                raise CampaignStateError(f"reservation must be {expected} before {target}")
            db.commit()

    @staticmethod
    def _scope_hash(dataset_identity: str, split_id: str, purpose: str) -> str:
        # Purpose is a validated attribute, not a namespace capable of
        # reopening the same physical dataset/split holdout.
        return hashlib.sha256(f"{dataset_identity}\0{split_id}".encode()).hexdigest()

    def freeze_candidate_set(self, *, campaign_id: str, candidate_set_hash: str, dataset_identity: str, split_id: str, purpose: str, profile_hash: str) -> str:
        fields = (campaign_id, candidate_set_hash, dataset_identity, split_id, purpose, profile_hash)
        if not all(isinstance(v, str) and v.strip() for v in fields):
            raise ValueError("sealed scope fields must be non-empty strings")
        scope = self._scope_hash(dataset_identity, split_id, purpose)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM sealed_access WHERE dataset_identity=? AND split_id=?",
                (dataset_identity, split_id),
            ).fetchall()
            if len(rows) > 1:
                raise CampaignStateError(
                    "multiple historical sealed scopes exist for one dataset/split"
                )
            row = rows[0] if rows else None
            if row is None:
                db.execute(
                    """INSERT INTO sealed_access(
                        scope_hash,campaign_id,candidate_set_hash,dataset_identity,
                        split_id,purpose,profile_hash,state,attempt_count,result_ref,
                        result_json,exposed_at,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (scope, campaign_id, candidate_set_hash, dataset_identity, split_id, purpose, profile_hash, "FROZEN", 0, None, None, None, now, now),
                )
            elif (
                row["candidate_set_hash"] != candidate_set_hash
                or row["profile_hash"] != profile_hash
                or row["purpose"] != purpose
            ):
                raise CampaignStateError(
                    "sealed scope is already bound to a different candidate set, profile, or purpose"
                )
            else:
                scope = row["scope_hash"]
            db.commit()
        return scope

    def begin_test_attempt(self, scope_hash: str) -> Optional[Mapping[str, Any]]:
        """Start the one logical evaluation, or return its immutable result."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sealed_access WHERE scope_hash=?", (scope_hash,)).fetchone()
            if row is None:
                raise CampaignStateError("unknown sealed scope")
            if row["state"] == "COMPLETED":
                result = json.loads(row["result_json"])
                db.commit(); return {"result_ref": row["result_ref"], "result": result}
            if row["state"] not in ("FROZEN", "INFRA_FAILED"):
                raise CampaignStateError("sealed evaluation is already running")
            if row["exposed_at"] is not None:
                raise CampaignStateError(
                    "sealed evaluation has prior or unknown test exposure"
                )
            db.execute(
                "UPDATE sealed_access SET state='RUNNING',attempt_count=attempt_count+1,updated_at=? WHERE scope_hash=?",
                (time.time(), scope_hash),
            )
            db.commit(); return None

    def mark_infrastructure_failure(self, scope_hash: str, attempt_count: int) -> None:
        """Release only an attempt that failed before physical test exposure."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """UPDATE sealed_access SET state='INFRA_FAILED',updated_at=?
                   WHERE scope_hash=? AND state='RUNNING' AND exposed_at IS NULL
                     AND attempt_count=?""",
                (time.time(), scope_hash, attempt_count),
            ).rowcount
            if changed != 1:
                raise CampaignStateError(
                    "only an unexposed running sealed evaluation can retry"
                )
            db.commit()

    def mark_test_exposed(self, scope_hash: str, attempt_count: int) -> None:
        """Atomically burn the sole physical test read before store access."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """UPDATE sealed_access SET exposed_at=?,updated_at=?
                   WHERE scope_hash=? AND state='RUNNING' AND exposed_at IS NULL
                     AND attempt_count=?""",
                (time.time(), time.time(), scope_hash, attempt_count),
            ).rowcount
            if changed != 1:
                raise CampaignStateError(
                    "sealed test data is already exposed or not running"
                )
            db.commit()

    def complete_test(self, scope_hash: str, attempt_count: int, result_ref: str, result: Mapping[str, Any]) -> None:
        if not result_ref:
            raise ValueError("result_ref is required")
        encoded = json.dumps(dict(result), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """UPDATE sealed_access SET state='COMPLETED',result_ref=?,result_json=?,updated_at=?
                   WHERE scope_hash=? AND state='RUNNING' AND exposed_at IS NOT NULL
                     AND attempt_count=?""",
                (result_ref, encoded, time.time(), scope_hash, attempt_count),
            ).rowcount
            if changed != 1:
                raise CampaignStateError("only a running sealed evaluation can complete")
            db.commit()

    def _sealed_transition(self, scope_hash: str, expected: str, target: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE sealed_access SET state=?,updated_at=? WHERE scope_hash=? AND state=?",
                (target, time.time(), scope_hash, expected),
            ).rowcount
            if changed != 1:
                raise CampaignStateError(f"sealed scope must be {expected} before {target}")
            db.commit()

    def sealed_state(self, scope_hash: str) -> Mapping[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sealed_access WHERE scope_hash=?", (scope_hash,)).fetchone()
            if row is None:
                raise CampaignStateError("unknown sealed scope")
            return dict(row)

    def budget_state(self, campaign_id: str) -> Mapping[str, Any]:
        with self._connect() as db:
            campaign = db.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            if campaign is None:
                raise CampaignStateError("unknown campaign")
            active = db.execute(
                "SELECT COUNT(*) n,COALESCE(SUM(estimated_cost),0) cost FROM reservations WHERE campaign_id=? AND state IN ('RESERVED','STARTED')",
                (campaign_id,),
            ).fetchone()
            result = dict(campaign)
            result.update(evaluations_reserved=active["n"], cost_reserved=active["cost"])
            return result

    def record_repair_failure(self, *, effective_spec_hash: str, evaluation_intent_hash: str, context_hash: str, reason: str, retry_condition: Optional[str] = None, payload: Optional[Mapping[str, Any]] = None) -> None:
        values = (effective_spec_hash, evaluation_intent_hash, context_hash, reason)
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError("repair knowledge identity and reason are required")
        encoded = json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO repair_knowledge VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(effective_spec_hash,evaluation_intent_hash,context_hash)
                DO UPDATE SET reason=excluded.reason,retry_condition=excluded.retry_condition,
                  payload_json=excluded.payload_json,attempt_count=repair_knowledge.attempt_count+1,
                  updated_at=excluded.updated_at""",
                (effective_spec_hash, evaluation_intent_hash, context_hash, "FAILED", reason, retry_condition, encoded, 1, time.time()),
            )
            db.commit()

    def repair_failure(self, *, effective_spec_hash: str, evaluation_intent_hash: str, context_hash: str) -> Optional[Mapping[str, Any]]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM repair_knowledge WHERE effective_spec_hash=? AND evaluation_intent_hash=? AND context_hash=? AND state='FAILED'",
                (effective_spec_hash, evaluation_intent_hash, context_hash),
            ).fetchone()
            if row is None: return None
            result = dict(row); result["payload"] = json.loads(result.pop("payload_json"))
            return result

    def freeze_supervised_parameter(self, campaign_id: str, state) -> None:
        from factor_optimizer.search.supervised_parameter import FrozenSupervisedParameter
        if not isinstance(state, FrozenSupervisedParameter):
            raise TypeError("state must be FrozenSupervisedParameter")
        encoded = json.dumps(state.__dict__, sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT state_hash FROM frozen_parameters WHERE campaign_id=? AND parent_factor_id=? AND repair_family=?",
                (campaign_id, state.parent_factor_id, state.repair_family),
            ).fetchone()
            if prior is not None and prior["state_hash"] != state.state_hash:
                raise CampaignStateError("supervised parameter is already frozen")
            db.execute(
                "INSERT OR IGNORE INTO frozen_parameters VALUES(?,?,?,?,?)",
                (campaign_id, state.parent_factor_id, state.repair_family, state.state_hash, encoded),
            )
            db.commit()

    def supervised_parameter(self, campaign_id: str, parent_factor_id: str, repair_family: str):
        from factor_optimizer.search.supervised_parameter import FrozenSupervisedParameter
        with self._connect() as db:
            row = db.execute(
                "SELECT state_json FROM frozen_parameters WHERE campaign_id=? AND parent_factor_id=? AND repair_family=?",
                (campaign_id, parent_factor_id, repair_family),
            ).fetchone()
            if row is None: return None
            data = json.loads(row["state_json"]); data["candidate_grid"] = tuple(data["candidate_grid"])
            return FrozenSupervisedParameter(**data)

    def record_hypothesis_attempt(self, *, campaign_id: str, proposal_id: str,
                                  evaluation_intent_hash: str, horizon: int,
                                  effective_spec_hash: Optional[str] = None,
                                  executed: bool = False, has_pvalue: bool = False) -> None:
        if not all(isinstance(v, str) and v.strip() for v in (campaign_id, proposal_id, evaluation_intent_hash)):
            raise ValueError("campaign/proposal/evaluation intent identities are required")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if has_pvalue and not executed:
            raise ValueError("an unexecuted proposal cannot have a p-value")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM hypothesis_attempts WHERE campaign_id=? AND proposal_id=?",
                (campaign_id, proposal_id),
            ).fetchone()
            payload = (effective_spec_hash, evaluation_intent_hash, horizon, int(executed), int(has_pvalue))
            if prior is None:
                db.execute("INSERT INTO hypothesis_attempts VALUES(?,?,?,?,?,?,?)",
                           (campaign_id, proposal_id, *payload))
            elif tuple(prior[k] for k in ("effective_spec_hash","evaluation_intent_hash","horizon","executed","has_pvalue")) != payload:
                raise CampaignStateError("proposal identity/outcome cannot be rewritten")
            db.commit()

    def hypothesis_family_summary(self, campaign_id: str) -> Mapping[str, int]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM hypothesis_attempts WHERE campaign_id=?", (campaign_id,)).fetchall()
            effective = {(r["effective_spec_hash"], r["evaluation_intent_hash"], r["horizon"])
                         for r in rows if r["effective_spec_hash"] is not None}
            pvalues = sum(int(r["has_pvalue"]) for r in rows)
            return {
                "proposal_count": len(rows),
                "executed_trial_count": sum(int(r["executed"]) for r in rows),
                "unique_effective_spec_count": len(effective),
                "pvalue_count": pvalues,
                "horizon_hypothesis_count": len({r["horizon"] for r in rows}),
            }

    def require_complete_fdr_family(self, campaign_id: str, submitted_pvalue_count: int) -> None:
        summary = self.hypothesis_family_summary(campaign_id)
        if submitted_pvalue_count != summary["pvalue_count"] or submitted_pvalue_count == 0:
            raise CampaignStateError("FDR input must contain the complete recorded p-value family")

    def bind_hypothesis_family(self, family, ledger) -> str:
        """Persist a QE family only when it exactly resolves a sealed FO ledger.

        The trusted family reference is returned from the verified artifact;
        callers cannot bind a free-form hash or a truncated finite-p subset.
        """
        required = (
            "family_id", "campaign_id", "content_hash", "members",
            "ledger_head", "ledger_count", "policy_hash",
            "comparison_context_hash", "correction_method", "assumptions",
        )
        missing = [name for name in required if not hasattr(family, name)]
        if missing:
            raise TypeError(f"hypothesis family missing fields: {missing}")
        verifier = getattr(family, "verify", None)
        if callable(verifier):
            verifier()
        if not getattr(ledger, "sealed", False):
            raise CampaignStateError("hypothesis family requires a sealed ledger")
        ledger.verify_chain()
        if family.ledger_head != ledger.sealed_head_hash:
            raise CampaignStateError("hypothesis family ledger head mismatch")
        if family.ledger_count != ledger.sealed_entry_count:
            raise CampaignStateError("hypothesis family ledger count mismatch")
        members = tuple(family.members)
        member_ids = [m["hypothesis_id"] for m in members]
        if len(member_ids) != len(set(member_ids)):
            raise CampaignStateError("hypothesis family has duplicate members")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM hypothesis_attempts WHERE campaign_id=?",
                (family.campaign_id,),
            ).fetchall()
            recorded = {row["proposal_id"]: row for row in rows}
            if set(member_ids) != set(recorded):
                raise CampaignStateError(
                    "hypothesis family must contain every preregistered member exactly once"
                )
            for member in members:
                row = recorded[member["hypothesis_id"]]
                if member["effective_spec_hash"] != row["effective_spec_hash"]:
                    raise CampaignStateError("family effective spec mismatch")
                if member["evaluation_intent_hash"] != row["evaluation_intent_hash"]:
                    raise CampaignStateError("family evaluation intent mismatch")
                if member["horizon"] != row["horizon"]:
                    raise CampaignStateError("family horizon mismatch")
                status = member["status"]
                if row["has_pvalue"] and status != "COMPUTED":
                    raise CampaignStateError("computed hypothesis status mismatch")
                if not row["executed"] and status not in ("PENDING", "NOT_TESTED"):
                    raise CampaignStateError("unexecuted hypothesis status mismatch")
            payload = {
                "family_id": family.family_id,
                "campaign_id": family.campaign_id,
                "policy_hash": family.policy_hash,
                "comparison_context_hash": family.comparison_context_hash,
                "members": [dict(m) for m in members],
                "ledger_head": family.ledger_head,
                "ledger_count": family.ledger_count,
                "correction_method": family.correction_method,
                "assumptions": list(family.assumptions),
                "content_hash": family.content_hash,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            prior = db.execute(
                "SELECT family_hash,artifact_json FROM hypothesis_families WHERE campaign_id=? AND family_id=?",
                (family.campaign_id, family.family_id),
            ).fetchone()
            if prior is None:
                db.execute(
                    "INSERT INTO hypothesis_families VALUES(?,?,?,?,?,?,?)",
                    (family.campaign_id, family.family_id, family.content_hash,
                     family.ledger_head, family.ledger_count, encoded, time.time()),
                )
            elif (prior["family_hash"], prior["artifact_json"]) != (family.content_hash, encoded):
                raise CampaignStateError("bound hypothesis family cannot be rewritten")
            db.commit()
        return family.content_hash


class DurableBudgetTracker:
    """BudgetTracker-compatible facade backed by SQLite transactions."""

    def __init__(self, store: SQLiteCampaignStore, campaign_id: str, budget):
        self.store, self.campaign_id, self.budget = store, campaign_id, budget
        self.trials_used = 0
        self.llm_calls_used = 0
        store.create_campaign(
            campaign_id,
            max_evaluations=budget.max_evaluations,
            max_cost=budget.max_cost_units,
        )

    @property
    def evaluations_used(self): return self.store.budget_state(self.campaign_id)["evaluations_used"]
    @property
    def cost_used(self): return self.store.budget_state(self.campaign_id)["cost_used"]
    @property
    def evaluations_reserved(self): return self.store.budget_state(self.campaign_id)["evaluations_reserved"]
    @property
    def cost_reserved(self): return self.store.budget_state(self.campaign_id)["cost_reserved"]

    def reserve_evaluation(self, cost, attempt_id=None):
        if not attempt_id: raise ValueError("durable reservations require attempt_id")
        return self.store.reserve(self.campaign_id, attempt_id, cost)

    def commit_evaluation(self, reserved_cost, actual_cost, attempt_id=None):
        if not attempt_id: raise ValueError("durable reservations require attempt_id")
        self.store.settle(self.campaign_id, attempt_id, actual_cost)

    def start_evaluation(self, attempt_id=None):
        if not attempt_id: raise ValueError("durable reservations require attempt_id")
        self.store.start(self.campaign_id, attempt_id)

    def release_evaluation(self, reserved_cost, attempt_id=None):
        if not attempt_id: raise ValueError("durable reservations require attempt_id")
        self.store.release(self.campaign_id, attempt_id)

    def has_reservation(self, attempt_id=None):
        if not attempt_id:
            return False
        with self.store._connect() as db:
            row = db.execute(
                "SELECT state FROM reservations WHERE campaign_id=? AND attempt_id=?",
                (self.campaign_id, attempt_id),
            ).fetchone()
        return row is not None and row["state"] in ("RESERVED", "STARTED")

    def remaining_cost(self):
        state = self.store.budget_state(self.campaign_id)
        return max(0.0, state["max_cost"] - state["cost_used"] - state["cost_reserved"])

    def remaining_evaluations(self):
        state = self.store.budget_state(self.campaign_id)
        return max(0, state["max_evaluations"] - state["evaluations_used"] - state["evaluations_reserved"])

    def can_propose_trial(self): return self.trials_used < self.budget.max_trials
    def record_trial(self): self.trials_used += 1
    def can_call_llm(self): return self.budget.max_llm_calls is not None and self.llm_calls_used < self.budget.max_llm_calls
    def record_llm_call(self): self.llm_calls_used += 1
    def is_exhausted(self):
        return self.trials_used >= self.budget.max_trials or self.remaining_evaluations() == 0 or self.remaining_cost() == 0
    def to_dict(self):
        state = self.store.budget_state(self.campaign_id)
        return {
            "budget": self.budget.to_dict(), "trials_used": self.trials_used,
            "evaluations_used": state["evaluations_used"], "cost_used": state["cost_used"],
            "llm_calls_used": self.llm_calls_used,
            "evaluations_reserved": state["evaluations_reserved"], "cost_reserved": state["cost_reserved"],
        }
