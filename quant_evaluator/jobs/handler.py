"""Adapter from platform JobSpec to the existing public QE evaluator."""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from pathlib import Path

from quant_evaluator.jobs.refs import read_manifest


_SAFE_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _stable_child_payload(child, *, manifest_digest, instance_id):
    payload = child.to_dict()
    # Runtime allocation facts must not perturb durable result identity.
    payload["timestamp"] = "1970-01-01T00:00:00+00:00"
    payload["request_id"] = "qe_" + hashlib.sha256(
        f"{manifest_digest}:{instance_id}".encode("utf-8")).hexdigest()
    payload["_qe_job"] = {
        "manifest_sha256": manifest_digest,
        "instance_id": instance_id,
    }
    encoded_without_hash = _canonical_bytes(payload)
    payload["_qe_job"]["content_sha256"] = hashlib.sha256(
        encoded_without_hash).hexdigest()
    return payload


def _atomic_replace(destination: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class QEJobHandler:
    def __init__(self, *, factor_batch_resolver, label_resolver, scenario_resolver):
        self.factor_batch_resolver = factor_batch_resolver
        self.label_resolver = label_resolver
        self.scenario_resolver = scenario_resolver

    def __call__(self, spec):
        from quant_platform.app.contracts.jobs import JobResult
        from quant_evaluator import evaluate
        from quant_evaluator.api.requests import EvaluationRequest
        from quant_evaluator.contracts.metric_instance import MetricInstance

        if spec.job_type != "qe_metric_instance_shard" or len(spec.input_artifact_refs) != 1:
            raise ValueError("QE job requires exactly one shard manifest ref")
        if not _SAFE_IDEMPOTENCY_KEY.fullmatch(spec.idempotency_key):
            raise ValueError("QE idempotency_key is unsafe for durable paths")
        manifest_path = Path(spec.input_artifact_refs[0])
        shard = read_manifest(manifest_path)
        manifest_digest = manifest_path.stem.removeprefix("qe_manifest_")
        instances = tuple(MetricInstance.from_dict(x) for x in shard.metric_instances)
        disallowed = sorted({item.output_mode for item in instances} -
                            set(shard.allowed_output_modes))
        if disallowed:
            raise ValueError(f"Metric instance output modes violate manifest policy: {disallowed}")
        labels = self.label_resolver(shard.label_ref)
        scenarios = {name: self.scenario_resolver(ref) for name, ref in shard.scenario_refs}
        request = EvaluationRequest(
            batch_or_factor_ids=self.factor_batch_resolver(shard.factor_refs),
            label_bundle=labels, metric_ids=tuple(sorted({x.metric_id for x in instances})),
            metric_instances=instances, scenario_inputs=scenarios,
            tier=shard.tier, cost_budget=shard.cost_budget,
        )
        bundle = evaluate(request)
        sink = Path(shard.result_sink_ref)
        sink.mkdir(parents=True, exist_ok=True)
        refs = []
        for instance_id, child in bundle.instance_results.items():
            payload = _stable_child_payload(
                child, manifest_digest=manifest_digest, instance_id=instance_id)
            encoded = _canonical_bytes(payload)
            content_digest = payload["_qe_job"]["content_sha256"]
            target = sink / f"qe_result_{manifest_digest}_{instance_id}_{content_digest}.json"
            if not target.exists() or target.read_bytes() != encoded:
                _atomic_replace(target, encoded)
            refs.append(str(target))
        return JobResult(tuple(sorted(refs)), summary=f"instances={len(refs)}")


__all__ = ["QEJobHandler"]
