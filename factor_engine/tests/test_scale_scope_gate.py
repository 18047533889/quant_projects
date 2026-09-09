import pytest
import json
from scripts.scale_scope_gate import verified_factor_execution
from scripts.scale_scope_gate import qualifies

SHA = "a" * 40


def receipt(scope="factor_scale", **overrides):
    evidence = dict(scope=scope, scale=1000, scale_unit="unique_factors",
        status="PASS", execution_verified=True, execution_ref="actual-run.json",
        execution_sha256="b" * 64, source_sha=SHA, requested=1000, completed=1000,
        failed=0, skipped=0, unique_formulas=1000, metric_instances=["identity-value-oracle"],
        backend="cpu", real_data=False)
    evidence.update(overrides)
    return dict(source_sha=SHA, collector_status="PASS", evidence=[evidence])


def test_row_scale_cannot_certify_factor_scale_or_cuda():
    data = receipt("row_scale", scale=100000, scale_unit="rows")
    assert qualifies(data, "row_scale", 100000, source_sha=SHA)
    assert not qualifies(data, "factor_scale", 1000, source_sha=SHA)
    assert not qualifies(data, "gpu_cuda", 1, source_sha=SHA)


@pytest.mark.parametrize("change", [
    {"status": "NOT_RUN"}, {"execution_verified": False}, {"execution_ref": ""},
    {"execution_sha256": ""}, {"source_sha": "c"*40}, {"scale_unit": "rows"},
    {"completed": 999}, {"skipped": 1}, {"failed": 1}, {"unique_formulas": 1},
    {"metric_instances": []}, {"scale": True},
])
def test_collector_or_incomplete_execution_cannot_grant_scope(change):
    assert not qualifies(receipt(**change), "factor_scale", 1, source_sha=SHA)


def test_scope_gate_is_capability_local_and_source_bound():
    data = receipt()
    assert qualifies(data, "factor_scale", 1000, source_sha=SHA)
    assert not qualifies(data, "factor_scale", 1000, source_sha="c"*40)
    assert not qualifies(data, "real_data_shadow", 1, source_sha=SHA)
    assert not qualifies(data, "factor_scale", 1)


@pytest.mark.parametrize("minimum", [0, -1, True])
def test_invalid_requested_scale(minimum):
    with pytest.raises(ValueError):
        qualifies(receipt(), "factor_scale", minimum, source_sha=SHA)


def test_actual_fe_report_is_required_not_job_pass(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(ValueError):
        verified_factor_execution(path, SHA)
    report = dict(status="PASS", source_sha=SHA, benchmark_sha256="b"*64,
        oracle_status="PASS", roots=1000, requested=1000, completed=1000,
        failed=0, skipped=0, unique_formulas=1000, metric_instances=["disk-oracle"],
        seconds=2, peak_rss_bytes=100, output_bytes=20, limitations="synthetic only",
        scale_dimensions=dict(F_factor_count=1000, T_timestamps=8, N_assets=32,
                              formula_family="column_plus_distinct_scalar", backend="pandas"))
    path.write_text(json.dumps(report))
    evidence = verified_factor_execution(path, SHA)
    assert qualifies(dict(source_sha=SHA, collector_status="PASS", evidence=[evidence]),
                     "factor_scale", 1000, source_sha=SHA)
    assert evidence["dimensions"]["T_timestamps"] == 8
    assert evidence["limitations"] == "synthetic only"
