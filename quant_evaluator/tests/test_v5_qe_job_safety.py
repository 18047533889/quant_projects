import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from quant_evaluator.api.requests import EvaluationBundle
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
from quant_evaluator.jobs.handler import QEJobHandler, _atomic_replace
from quant_evaluator.jobs.refs import QEJobShard, read_manifest, write_manifest
from quant_platform.app.contracts import JobSpec


def inputs():
    rng = np.random.default_rng(19)
    t, n = 6, 20
    batch = FactorBatch(
        ("factor:f",), AxisRef("time", "int", t, np.arange(t)),
        AxisRef("asset", "str", n, np.asarray([f"a{i}" for i in range(n)])),
        np.ascontiguousarray(rng.normal(size=(t, n, 1))))
    labels = LabelBundle(
        "return", np.ascontiguousarray(rng.normal(size=(t, n))), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)))
    return batch, labels


def setup_job(tmp_path, *, metric="rank_ic_series", tier="extended",
              budget=10, modes=("FULL_DIAGNOSTIC",)):
    batch, labels = inputs()
    instance = MetricInstance(metric, scenario_id="s", horizon=1,
                              price_convention="vwap_to_vwap")
    shard = QEJobShard(
        ("factor:f",), "label:h1", (instance.to_dict(),),
        (("s", "scenario:s"),), str(tmp_path / "sink"), 0,
        tier=tier, cost_budget=budget, allowed_output_modes=modes)
    manifest = write_manifest(tmp_path / "manifests", shard)
    handler = QEJobHandler(
        factor_batch_resolver=lambda _: batch,
        label_resolver=lambda _: labels,
        scenario_resolver=lambda _: EvaluationScenario(labels))
    spec = JobSpec("qe_metric_instance_shard", "safe-key", input_artifact_refs=(manifest,))
    return handler, spec, manifest


def test_explicit_extended_policy_and_typed_roundtrip(tmp_path):
    handler, spec, manifest = setup_job(tmp_path)
    restored = read_manifest(manifest)
    assert restored.tier == "extended" and restored.cost_budget == 10
    result = handler(spec)
    payload = json.loads(Path(result.output_artifact_refs[0]).read_text())
    restored_bundle = EvaluationBundle.from_dict(payload)
    assert restored_bundle.factor_ids == ("factor:f",)
    assert payload["_qe_job"]["manifest_sha256"] in result.output_artifact_refs[0]


def test_budget_and_output_policy_fail_closed(tmp_path):
    handler, spec, _ = setup_job(tmp_path / "budget", budget=0)
    with pytest.raises(ValueError, match="cost_budget"):
        handler(spec)
    handler, spec, _ = setup_job(
        tmp_path / "mode", modes=("SUMMARY_ONLY",))
    with pytest.raises(ValueError, match="output modes"):
        handler(spec)


def test_stable_rerun_repairs_corrupt_existing_result(tmp_path):
    handler, spec, _ = setup_job(tmp_path)
    first = handler(spec).output_artifact_refs
    original = Path(first[0]).read_bytes()
    second = handler(spec).output_artifact_refs
    assert second == first and Path(first[0]).read_bytes() == original
    Path(first[0]).write_bytes(b"{broken")
    third = handler(spec).output_artifact_refs
    assert third == first and Path(first[0]).read_bytes() == original
    assert not list(Path(first[0]).parent.glob("*.tmp"))


def test_concurrent_writers_use_unique_temps_and_same_durable_ref(tmp_path):
    handler, spec, _ = setup_job(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: handler(spec), range(2)))
    assert results[0].output_artifact_refs == results[1].output_artifact_refs
    assert not list((tmp_path / "sink").glob("*.tmp"))
    json.loads(Path(results[0].output_artifact_refs[0]).read_text())


def test_manifest_hash_and_idempotency_path_are_validated(tmp_path):
    handler, spec, manifest = setup_job(tmp_path)
    tampered = Path(manifest).with_name("qe_manifest_" + "0" * 64 + ".json")
    tampered.write_bytes(Path(manifest).read_bytes())
    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(tampered)
    unsafe = JobSpec(spec.job_type, "../escape", input_artifact_refs=spec.input_artifact_refs)
    with pytest.raises(ValueError, match="unsafe"):
        handler(unsafe)


def test_atomic_fault_cleans_temp_and_retry_succeeds(tmp_path, monkeypatch):
    import quant_evaluator.jobs.handler as module
    destination = tmp_path / "sink" / "value.json"
    destination.parent.mkdir()
    real_replace = module.os.replace
    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("fault")))
    with pytest.raises(OSError, match="fault"):
        _atomic_replace(destination, b"one")
    assert not list(destination.parent.glob("*.tmp"))
    monkeypatch.setattr(module.os, "replace", real_replace)
    _atomic_replace(destination, b"two")
    assert destination.read_bytes() == b"two"
