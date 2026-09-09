"""Maturity-gated QE observations using the platform DB/checkpoint and GC authority.

This is an incremental metric consumer, not a scheduler. A host worker calls
``drain`` with its completed market watermark. Payloads remain in the existing
artifact store and are resolved only for a bounded ready shard.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.online_moments import OnlineMoments


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Maturation times require timezone-aware UTC-convertible datetimes")
    return value.astimezone(timezone.utc).isoformat()


def _input_identity(request):
    from quant_evaluator.runtime.evaluator import authoritative_array_hash
    batch = request.batch_or_factor_ids
    return stable_content_hex(tag="MatureInput.v1", fields={
        "values": authoritative_array_hash(batch.values),
        "validity": None if batch.validity is None else authoritative_array_hash(batch.validity),
        "factors": batch.factor_ids, "labels": request.label_bundle.content_hash,
        "context": batch.context_refs,
        "time": None if batch.time_axis.values is None else tuple(batch.time_axis.values.tolist()),
        "assets": None if batch.asset_axis.values is None else tuple(batch.asset_axis.values.tolist())})


def _normalize_window(value):
    """Validate the only certified maturity-summary window specifications."""
    if value == "all_history":
        return value
    if not isinstance(value, Mapping) or set(value) != {"kind", "size"}:
        raise ValueError(
            "window must be 'all_history' or exactly "
            "{'kind': 'rolling_observations', 'size': positive_int}"
        )
    size = value["size"]
    if value["kind"] != "rolling_observations" or type(size) is not int or size < 1:
        raise ValueError("rolling_observations window size must be a positive integer")
    return {"kind": "rolling_observations", "size": size}


class LabelMaturationQueue:
    """Exact per-factor daily IC moments; explicit unsupported modes fail closed.

    State is O(factors * instances), pending inputs/output observations are
    reference-backed durable rows. Restatements use a separate versioned stream;
    they never rewrite AS_KNOWN or AS_TRADED observations.
    """
    SUPPORTED = frozenset({"rank_ic_series", "pearson_ic_series"})
    CAPABILITIES = MappingProxyType({
        "metric_ids": tuple(sorted(SUPPORTED)),
        "windows": ("all_history", "rolling_observations"),
        "rolling_unit": "last_W_observations_including_missing",
        "rolling_algorithm": "exact_bounded_database_recomputation",
        "update": True,
        "merge": False,
        "remove": False,
        "checkpoint": True,
        "restart": True,
        "revision": "separate_RESTATED_RESEARCH_stream_requires_complete_target_history_replay",
        "revision_completeness_validation": False,
        "multi_day_shard_maturity": "max_label_end_time",
        "factor_shard_bound": "configurable_positive_integer",
        "cross_section_assets": "complete_cross_section_required_not_constant_in_N",
    })

    def __init__(self, db, *, generation_coordinator=None, max_factors_per_shard=256):
        if type(max_factors_per_shard) is not int or max_factors_per_shard < 1:
            raise ValueError("max_factors_per_shard must be a positive integer")
        self.db = db
        self.coordinator = generation_coordinator
        self.max_factors_per_shard = max_factors_per_shard
        if generation_coordinator is not None and generation_coordinator.db is not db:
            raise ValueError("Maturation and GC must use the same transactional DB")
        with db.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS qe_maturity_events (event_id TEXT PRIMARY KEY, "
                         "stream_id TEXT NOT NULL, payload TEXT NOT NULL, mature_at TEXT NOT NULL, "
                         "available_at TEXT NOT NULL, status TEXT NOT NULL, result TEXT)")
            conn.execute("CREATE INDEX IF NOT EXISTS qe_maturity_ready ON qe_maturity_events(status,mature_at,available_at)")
            conn.execute("CREATE TABLE IF NOT EXISTS qe_maturity_observations (stream_id TEXT, instance_id TEXT, "
                         "factor_id TEXT, decision_at TEXT, value_json TEXT NOT NULL, "
                         "PRIMARY KEY(stream_id,instance_id,factor_id,decision_at))")
            conn.execute("CREATE TABLE IF NOT EXISTS qe_maturity_states (stream_id TEXT, instance_id TEXT, "
                         "factor_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(stream_id,instance_id,factor_id))")
            conn.execute("CREATE TABLE IF NOT EXISTS qe_maturity_roots (event_id TEXT, artifact_id TEXT, "
                         "generation_id TEXT, PRIMARY KEY(event_id,artifact_id,generation_id))")

    @property
    def capabilities(self):
        return MappingProxyType({**self.CAPABILITIES,
                                 "max_factors_per_shard": self.max_factors_per_shard})

    def _validate_shard_bounds(self, request):
        batch = request.batch_or_factor_ids
        if batch is None:
            raise ValueError("Resolved maturity request is missing its factor batch")
        if not 1 <= batch.num_times <= 256:
            raise ValueError("Maturity input must be a bounded shard of 1..256 dates")
        if not 1 <= batch.num_factors <= self.max_factors_per_shard:
            raise ValueError(
                "Maturity input factor shard exceeds max_factors_per_shard; "
                "a complete asset cross-section is still required for every factor"
            )

    def _validate_reference_factor_bound(self, request):
        factor_ref = request.factor_value_ref
        factor_ids = None if factor_ref is None else factor_ref.factor_ids
        if not factor_ids:
            raise ValueError("Pending maturity reference has no factor identities")
        if len(factor_ids) > self.max_factors_per_shard:
            raise ValueError("Pending maturity reference exceeds max_factors_per_shard")

    def enqueue(self, event_id, request, *, stream_semantics, available_at, snapshot_mode="AS_KNOWN",
                revision_id=None, generation_refs=()):
        required = {"factor_value_semantics", "universe", "clock", "window", "policy_version"}
        if set(stream_semantics) != required or not all(stream_semantics.values()):
            raise ValueError("Complete immutable stream semantics are required")
        window = _normalize_window(stream_semantics["window"])
        normalized_semantics = dict(stream_semantics)
        normalized_semantics["window"] = window
        if snapshot_mode not in {"AS_KNOWN", "AS_TRADED", "RESTATED_RESEARCH"}:
            raise ValueError("Unsupported snapshot mode")
        if (snapshot_mode == "RESTATED_RESEARCH") != bool(revision_id):
            raise ValueError("Only RESTATED_RESEARCH requires a new revision_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required")
        if not request.metric_instances or request.scenario_inputs:
            raise ValueError("Maturity updates require default-scenario metric instances")
        if any(item.metric_id not in self.SUPPORTED or item.scenario_id != "default"
               or item.output_mode == "SUMMARY_ONLY" for item in request.metric_instances):
            raise ValueError("Metric has no exact daily-IC maturity updater")
        if request.factor_value_ref is None or request.label_bundle_ref is None:
            raise ValueError("Pending labels require durable factor and label references")
        self._validate_shard_bounds(request)
        times = request.label_bundle.label_end_time
        mature_at = max(_utc(value) for value in times)
        available = _utc(available_at)
        state_schema = ("centered_moments.v2" if window == "all_history"
                        else "centered_moments.v3.rolling_observation_recompute")
        stream_id = stable_content_hex(tag="MaturityStream.v1", fields={
            **normalized_semantics, "mode": snapshot_mode, "revision_id": revision_id,
            "state_schema": state_schema})
        payload_fields = {"request": request.to_dict(), "identity": _input_identity(request),
                          "generation_refs": list(generation_refs), "snapshot_mode": snapshot_mode}
        # Preserve the existing all-history event envelope and stream identity.
        if window != "all_history":
            payload_fields["window"] = window
        payload = json.dumps(payload_fields, sort_keys=True, separators=(",", ":"))
        # Roots and pending row enter together under the existing GC epoch lock.
        with self.db.transaction() as conn:
            old = conn.execute("SELECT * FROM qe_maturity_events WHERE event_id=?", (event_id,)).fetchone()
            if old is not None:
                if (old["stream_id"], old["payload"], old["mature_at"], old["available_at"]) != (stream_id, payload, mature_at, available):
                    raise ValueError("Conflicting replay of immutable maturation event")
                return stream_id
            if generation_refs:
                if self.coordinator is None:
                    raise ValueError("Generation refs require the platform GC coordinator")
                conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")
                for artifact_id, generation_id in generation_refs:
                    if conn.execute("SELECT 1 FROM artifact_generations WHERE artifact_id=? AND generation_id=?",
                                    (artifact_id,generation_id)).fetchone() is None:
                        raise ValueError("Unknown pending-label generation")
                    if conn.execute("SELECT 1 FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                                    (artifact_id,generation_id)).fetchone() is not None:
                        raise ValueError("Pending label cannot reference a tombstoned generation")
                    conn.execute("INSERT OR IGNORE INTO artifact_gc_roots(root_kind,artifact_id,generation_id) VALUES('pending_label',?,?)",
                                 (artifact_id,generation_id))
                    conn.execute("INSERT INTO qe_maturity_roots VALUES(?,?,?)", (event_id,artifact_id,generation_id))
            conn.execute("INSERT INTO qe_maturity_events VALUES(?,?,?,?,?,'PENDING',NULL)",
                         (event_id,stream_id,payload,mature_at,available))
        return stream_id

    def drain(self, watermark, resolver, *, limit=16, backend="cpu", before_commit=None):
        """Resolve/evaluate ready shards; checkpoint observations+state+output atomically.

        ``resolver`` receives a reference-only EvaluationRequest and must return
        its verified runtime binding. Its bytes are checked against enqueue identity.
        ``before_commit`` is a failure-injection seam; exceptions roll back all state.
        """
        from quant_evaluator.runtime.evaluator import evaluate
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("drain limit must be 1..256")
        instant = _utc(watermark)
        rows = self.db.query("SELECT * FROM qe_maturity_events WHERE status='PENDING' AND mature_at<=? "
                             "AND available_at<=? ORDER BY mature_at,event_id LIMIT ?", (instant,instant,limit))
        completed = []
        for row in rows:
            payload = json.loads(row["payload"])
            window = _normalize_window(payload.get("window", "all_history"))
            reference = EvaluationRequest.from_dict(payload["request"])
            # Re-apply current worker bounds before resolving durable payloads:
            # restart under a tighter configuration cannot bypass admission.
            self._validate_reference_factor_bound(reference)
            request = resolver(reference)
            self._validate_shard_bounds(request)
            if request.to_dict() != reference.to_dict() or _input_identity(request) != payload["identity"]:
                raise ValueError("Resolved maturity bytes/semantics differ from immutable pending input")
            if any(_utc(value) > instant for value in request.label_bundle.label_end_time):
                raise ValueError("Resolver supplied immature labels")
            bundle = evaluate(request, backend=backend)
            with self.db.transaction() as conn:
                # Serializes competing consumers on the event before any state update.
                changed = conn.execute("UPDATE qe_maturity_events SET status='COMPUTING' WHERE event_id=? AND status='PENDING'",
                                       (row["event_id"],)).rowcount
                if not changed:
                    continue
                for instance_id, child in bundle.instance_results.items():
                    item = bundle.instance_specs[instance_id]
                    values = child.artifacts[item.metric_id].values
                    for f, factor_id in enumerate(bundle.factor_ids):
                        key = (row["stream_id"],instance_id,factor_id)
                        old = conn.execute("SELECT payload FROM qe_maturity_states WHERE stream_id=? AND instance_id=? AND factor_id=?", key).fetchone()
                        state = OnlineMoments(**json.loads(old[0])) if old else OnlineMoments()
                        for t, decision in enumerate(request.label_bundle.decision_time):
                            coordinate = (*key, _utc(decision))
                            value = float(values[t,f])
                            encoded = json.dumps(value if np.isfinite(value) else None)
                            prior = conn.execute("SELECT value_json FROM qe_maturity_observations WHERE stream_id=? AND instance_id=? AND factor_id=? AND decision_at=?", coordinate).fetchone()
                            if prior is not None:
                                if prior[0] != encoded:
                                    raise ValueError("AS_KNOWN observation is immutable; use a restated research stream")
                                continue
                            conn.execute("INSERT INTO qe_maturity_observations VALUES(?,?,?,?,?)", (*coordinate, encoded))
                            state.update([value])
                        if window != "all_history":
                            recent = conn.execute(
                                "SELECT value_json FROM qe_maturity_observations "
                                "WHERE stream_id=? AND instance_id=? AND factor_id=? "
                                "ORDER BY decision_at DESC LIMIT ?",
                                (*key, window["size"]),
                            ).fetchall()
                            state = OnlineMoments()
                            state.update([
                                float("nan") if item[0] == "null" else json.loads(item[0])
                                for item in reversed(recent)
                            ])
                        conn.execute("INSERT INTO qe_maturity_states VALUES(?,?,?,?) ON CONFLICT(stream_id,instance_id,factor_id) DO UPDATE SET payload=excluded.payload",
                                     (*key,json.dumps(asdict(state))))
                conn.execute("UPDATE qe_maturity_events SET status='COMPLETE',result=? WHERE event_id=?",
                             (json.dumps(bundle.to_dict()), row["event_id"]))
                for artifact_id, generation_id in payload["generation_refs"]:
                    conn.execute("DELETE FROM qe_maturity_roots WHERE event_id=? AND artifact_id=? AND generation_id=?",
                                 (row["event_id"],artifact_id,generation_id))
                    if conn.execute("SELECT 1 FROM qe_maturity_roots WHERE artifact_id=? AND generation_id=?",
                                    (artifact_id,generation_id)).fetchone() is None:
                        conn.execute("DELETE FROM artifact_gc_roots WHERE root_kind='pending_label' AND artifact_id=? AND generation_id=?",
                                     (artifact_id,generation_id))
                        conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")
                if before_commit is not None:
                    before_commit(row["event_id"])
            completed.append(row["event_id"])
        return tuple(completed)

    def summaries(self, stream_id):
        return {(row["instance_id"],row["factor_id"]): OnlineMoments(**json.loads(row["payload"])).summary()
                for row in self.db.query("SELECT * FROM qe_maturity_states WHERE stream_id=?", (stream_id,))}
