import copy
import json

import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as harness


@pytest.fixture
def reference_report():
    return json.loads(harness.F24_REFERENCE_DEFAULT.read_text())


def test_archived_f24_reference_validates(reference_report):
    validated = harness._validate_auto_reference(reference_report)
    assert validated["config_hash"] == "0b881e778933ec39057390a7ce88178d9390a094f5aeba903220478476e019a9"
    assert validated["cuda_artifact_sha256"] == (
        "5557af86c19d06a7a17f35f78258254121b6b69ff099ab53ce4c4ce942a68841")
    binding = validated["source_identity"]
    assert binding["count"] == 24
    assert len(binding["sources"]) == 24
    assert binding["manifest_sha256"] == harness.MANIFEST_SHA256
    assert binding["total_bytes"] == sum(row["bytes"] for row in binding["sources"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report["request"].update(shape=[2586, 5460, 24]), "request.shape"),
        (lambda report: report["runs"][2].update(status="error"), "run 3 status/order"),
        (lambda report: report["runs"].reverse(), "run 1 status/order"),
        (lambda report: report["runs"][4].update(config_hash="f" * 64), "config hashes"),
        (lambda report: report["comparisons_to_first_cpu"]["cuda_strict"][0].__setitem__("pass", False), "cuda_strict parity"),
        (lambda report: report["source"]["source_binding"]["sources"][0].update(
            manifest_sha256="f" * 64), "invalid sanitized source binding row"),
    ],
)
def test_reference_rejects_tampering(reference_report, mutate, message):
    changed = copy.deepcopy(reference_report)
    mutate(changed)
    with pytest.raises(ValueError, match=message):
        harness._validate_auto_reference(changed)


def test_loaded_source_identity_ignores_loader_summaries():
    manifest = harness.MANIFEST_SHA256
    rows = []
    for index in range(24):
        digest = f"{index + 1:064x}"
        factor_id = f"fixture_{index:02d}"
        rows.append({
            "factor_id": factor_id,
            "uri": f"cos://fixture/{digest}/{factor_id}.parquet",
            "sha256": digest,
            "bytes": 1000 + index,
            "etag": f"etag-{index}",
            "manifest_sha256": manifest,
            "source_status": "evaluated",
        })
    source = {"sources": rows, "factor_finite_ratio": 0.8}
    baseline = harness._source_identity(source)
    source["factor_finite_ratio"] = 0.5
    source["load_s"] = 0.01
    assert harness._source_identity(source) == baseline


def test_reference_reports_redact_source_locators(reference_report):
    auto = json.loads((harness.F24_REFERENCE_DEFAULT.parent /
                       "real_cos_f24_mixed_three_auto_verify_20260929.json").read_text())
    for report in (reference_report, auto):
        source = report["source"]
        assert "sources" not in source
        binding = source["source_binding"]
        assert len(binding["sources"]) == 24
        assert all(set(row) == {
            "source_identity_sha256", "object_sha256", "bytes", "manifest_sha256"}
                   for row in binding["sources"])
        assert "factor_id" not in json.dumps(source)
        assert "uri" not in json.dumps(source)
        assert "etag" not in json.dumps(source)


def test_f24_mixed_three_verification_does_not_require_full_run(tmp_path):
    assert harness._validate_f24_mixed_three_request(
        factors=24, metrics=harness.DEFAULT_METRICS, run=False, output=tmp_path / "out.json",
        verify_auto_reference=True)


def test_auto_reference_requires_certified_cuda_route():
    run = {
        "backend_used": "cuda",
        "auto_backend_reason": "certified_batch_real_cos_f24_mixed_three",
        "metric_backends": {metric: "cuda" for metric in harness.DEFAULT_METRICS},
        "artifact_hash_matches_explicit_cuda": True,
        "config_hash_matches_reference": True,
        "source_identity_matches_reference": True,
    }
    assert harness._auto_reference_run_passes(run)
    for key, value in (
        ("backend_used", "cpu"),
        ("auto_backend_reason", "metric_not_certified_for_profile"),
        ("metric_backends", {metric: "cpu" for metric in harness.DEFAULT_METRICS}),
    ):
        changed = dict(run)
        changed[key] = value
        assert not harness._auto_reference_run_passes(changed)


@pytest.mark.parametrize("factor_count", (2, 8, 13, 32))
def test_other_profiles_use_safe_summary_without_f24_source_binding(factor_count):
    source = {
        "sources": [{"uri": "cos://private/location", "factor_id": "private_factor"}]
                   * factor_count,
        "days": 12, "assets": 50, "label": "safe label",
    }
    public = harness._public_source_report(source)
    assert public == {"days": 12, "assets": 50, "label": "safe label"}
    serialized = json.dumps(public)
    assert "cos://" not in serialized
    assert "private_factor" not in serialized
    assert "source_binding" not in public


def test_f24_source_binding_is_opt_in_for_safe_reports():
    source = {"sources": [{"uri": "cos://private/location", "factor_id": "private_factor"}]}
    public = harness._public_source_report(source, include_binding=False)
    assert "source_binding" not in public
    assert "cos://" not in json.dumps(public)
