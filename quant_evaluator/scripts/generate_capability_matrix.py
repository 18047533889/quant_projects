"""Evidence-backed QE capability matrix generator (QA-02)."""
from __future__ import annotations
import csv, os
from dataclasses import dataclass
import numpy as np
from typing import Any, Sequence

from quant_evaluator.api.requests import EvaluationBundle
from quant_evaluator.registry.metrics import list_metrics, get_metric
from quant_evaluator.backends.capability_registry import get_backend_capability_registry

NOT_RUN = "NOT_RUN"
UNSUPPORTED = "UNSUPPORTED"
DONE = "DONE"
DECLARED = "DECLARED"
NOT_DECLARED = "NOT_DECLARED"

MATRIX_COLUMNS = [
    "metric_id", "metric_version", "tier", "artifact_kind",
    "declared", "kernel", "public_cpu", "public_gpu", "adapter", "profile",
    "admission", "update", "publication", "gpu_declaration", "gpu_parity_declaration",
    "run_evidence_refs", "required_inputs", "dependencies",
]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _artifact_backend(artifact: Any) -> tuple[str | None, bool]:
    provenance = getattr(artifact, "provenance", None) or {}
    backend = provenance.get("execution_backend") or provenance.get("backend")
    return backend, provenance.get("no_fallback") is True


def _bundle_gpu_run(bundle: EvaluationBundle) -> bool:
    metadata = bundle.metadata or {}
    runtime = metadata.get("runtime") or {}
    return (metadata.get("backend_requested") == "cuda"
            and metadata.get("backend_used") == "cuda"
            and runtime.get("backend_requested") == "cuda"
            and runtime.get("backend_used") == "cuda"
            and runtime.get("gpu_device") is not None
            and int(runtime.get("h2d_bytes", 0)) > 0
            and int(runtime.get("d2h_bytes", 0)) > 0)


def _metric_artifacts(bundle: EvaluationBundle, metric_id: str) -> tuple[Any, ...]:
    found = []
    if metric_id in bundle.artifacts:
        found.append(bundle.artifacts[metric_id])
    for factor_id in bundle.factor_ids:
        artifact = bundle.factor_artifacts.get(factor_id, {}).get(metric_id)
        if artifact is not None and all(artifact is not existing for existing in found):
            found.append(artifact)
    return tuple(found)


@dataclass(frozen=True)
class CapabilityRunEvidence:
    bundle: EvaluationBundle
    metric_id: str
    backend: str
    source_ref: str
    data_identity: str

    def __post_init__(self):
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("run evidence backend must be cpu or gpu")
        if not isinstance(self.bundle, EvaluationBundle) or not self.source_ref or not self.data_identity:
            raise ValueError("run evidence requires bundle/source/data identity")
        artifacts = _metric_artifacts(self.bundle, self.metric_id)
        if self.backend == "cpu" and _bundle_gpu_run(self.bundle):
            raise ValueError("a CUDA bundle cannot be relabeled as CPU evidence")
        artifact_gpu = bool(artifacts) and all(
            _artifact_backend(a) in (("cuda", True), ("cuda_strict", True)) for a in artifacts
        )
        if self.backend == "gpu" and not (artifact_gpu or _bundle_gpu_run(self.bundle)):
            raise ValueError("GPU evidence requires authoritative CUDA run metadata")

    @classmethod
    def from_public_cpu(cls, bundle, metric_id, *, source_ref, data_identity):
        return cls._validated(bundle, metric_id, "cpu", source_ref, data_identity)

    @classmethod
    def from_public_gpu(cls, bundle, metric_id, *, source_ref, data_identity):
        return cls._validated(bundle, metric_id, "gpu", source_ref, data_identity)

    @classmethod
    def _validated(cls, bundle, metric_id, backend, source_ref, data_identity):
        if not isinstance(bundle, EvaluationBundle):
            raise TypeError("run evidence requires an actual EvaluationBundle")
        if not source_ref or not data_identity:
            raise ValueError("run evidence requires source and data identity")
        if metric_id not in bundle.metric_versions:
            raise ValueError("metric is absent from evaluated bundle")
        artifacts = _metric_artifacts(bundle, metric_id)
        if not artifacts:
            raise ValueError("evaluated metric has no artifact")
        values = [np.asarray(getattr(a, "values", ()), dtype=float) for a in artifacts]
        if any(v.size == 0 or not np.isfinite(v).any() for v in values):
            raise ValueError("metric artifact is empty or invalid")
        metrics = [bundle.get_metric(metric_id, factor_id) for factor_id in bundle.factor_ids]
        if not metrics or not all(m.valid and m.observation_count > 0 for m in metrics):
            raise ValueError("metric execution is invalid or has no observations")
        return cls(bundle, metric_id, backend, str(source_ref), str(data_identity))


def _evaluated_layers(metric_id: str, version: str,
                      evidence: Sequence[Any]) -> tuple[str, str, tuple[str, ...]]:
    cpu, gpu, refs = NOT_RUN, NOT_RUN, []
    for item in evidence:
        if not isinstance(item, CapabilityRunEvidence) or item.metric_id != metric_id:
            continue
        bundle = item.bundle
        artifact_gpu = all(
            _artifact_backend(a) in (("cuda", True), ("cuda_strict", True))
            for a in _metric_artifacts(bundle, metric_id)
        )
        if item.backend == "gpu" and not (_bundle_gpu_run(bundle) or artifact_gpu):
            continue
        if item.backend == "cpu" and _bundle_gpu_run(bundle):
            continue
        artifacts = _metric_artifacts(bundle, metric_id)
        metric_values = [bundle.get_metric(metric_id, factor_id) for factor_id in bundle.factor_ids]
        if (not artifacts or any(np.asarray(getattr(a, "values", ())).size == 0
                                 or not np.isfinite(np.asarray(a.values, dtype=float)).any()
                                 for a in artifacts)
                or not metric_values
                or not all(value.valid and value.observation_count > 0 for value in metric_values)):
            continue
        if bundle.metric_versions.get(metric_id) != version:
            continue
        evidence_ref = f"{bundle.request_id}@{item.source_ref}@{item.data_identity}"
        if item.backend == "cpu":
            cpu = DONE; refs.append(f"cpu:{evidence_ref}")
        elif item.backend == "gpu":
            gpu = DONE; refs.append(f"gpu:{evidence_ref}")
    return cpu, gpu, tuple(sorted(set(refs)))


def generate_capability_matrix(dest: str = "docs/METRIC_CAPABILITY_MATRIX.csv",
                               *, run_evidence: Sequence[Any] = ()) -> str:
    """Write declarations separately from artifact-backed run evidence."""
    registry = get_backend_capability_registry()
    rows = []
    for metric_id in list_metrics():
        spec = get_metric(metric_id)
        version = spec.metric_version
        impls = registry.implementations(metric_id)
        cuda = next((impl for impl in impls if impl.backend == "cuda"), None)
        public_cpu, public_gpu, refs = _evaluated_layers(
            metric_id, version, run_evidence
        )
        kernel_callable = callable(getattr(spec, "compute_fn", None)) or any(
            callable(impl.fn) for impl in impls
        )
        rows.append({
            "metric_id": metric_id,
            "metric_version": version,
            "tier": _enum_value(getattr(spec, "tier", "unknown")),
            "artifact_kind": getattr(spec, "artifact_kind", ""),
            "declared": DONE,
            "kernel": (DONE if kernel_callable and (public_cpu == DONE or public_gpu == DONE)
                       else DECLARED if kernel_callable else UNSUPPORTED),
            "public_cpu": public_cpu if kernel_callable else UNSUPPORTED,
            # The facade has CUDA routes beyond the auxiliary registry and
            # GPUExecutor plan.  Only a validated run can prove DONE; absent
            # evidence is conservatively NOT_RUN, never proof of no route.
            "public_gpu": public_gpu,
            "adapter": NOT_RUN, "profile": NOT_RUN, "admission": NOT_RUN,
            "update": NOT_RUN, "publication": NOT_RUN,
            "gpu_declaration": DONE if cuda is not None else NOT_DECLARED,
            "gpu_parity_declaration": (
                _enum_value(cuda.parity_status) if cuda is not None else NOT_DECLARED
            ),
            "run_evidence_refs": ",".join(refs),
            "required_inputs": ",".join(sorted(getattr(spec, "required_inputs", ()) or ())),
            "dependencies": ",".join(getattr(spec, "requires", ()) or ()),
        })
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return dest


if __name__ == "__main__":
    print("generated:", generate_capability_matrix())
