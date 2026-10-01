import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_u_shape_rank_reuse_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_u_shape_rank_reuse_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_build_plan_has_four_exact_sizes_and_24_alternating_records(tmp_path):
    output_dir = tmp_path / "planned-runs"
    plan = runner.build_plan(output_dir)

    assert len(plan) == 24
    assert len({item["output_path"] for item in plan}) == 24
    assert not output_dir.exists()

    by_shape = {}
    for item in plan:
        by_shape.setdefault((item["time_points"], item["assets"]), []).append(item)

    expected = {
        (200, 640): 76_800,
        (167, 5_000): 500_000,
        (334, 5_000): 1_000_000,
        (500, 5_000): 1_500_000,
    }
    assert set(by_shape) == set(expected)
    for shape, train_rows in expected.items():
        records = by_shape[shape]
        assert len(records) == 6
        assert {item["expected_train_rows"] for item in records} == {train_rows}
        assert [(item["pair"], item["mode"]) for item in records] == [
            (1, "cached"), (1, "uncached"),
            (2, "uncached"), (2, "cached"),
            (3, "cached"), (3, "uncached"),
        ]


def test_build_sensitivity_plan_has_36_unique_alternating_records(tmp_path):
    output_dir = tmp_path / "sensitivity-runs"
    plan = runner.build_sensitivity_plan(output_dir)
    assert len(plan) == 36
    assert len({item["output_path"] for item in plan}) == 36
    assert not output_dir.exists()
    expected_profiles = {
        "u_only_max8": (("U_SHAPE_REPAIR",), 8),
        "inverted_u_only_max6": (("INVERTED_U_REPAIR",), 6),
        "u_and_inverted_max14": (("U_SHAPE_REPAIR", "INVERTED_U_REPAIR"), 14),
    }
    for profile, (families, maximum) in expected_profiles.items():
        records = [item for item in plan if item["profile"] == profile]
        assert len(records) == 12
        assert {(item["families"], item["maximum_candidates"]) for item in records} == {
            (families, maximum)
        }
        assert [(item["pair"], item["mode"]) for item in records[:6]] == [
            (1, "cached"), (1, "uncached"), (2, "uncached"),
            (2, "cached"), (3, "cached"), (3, "uncached"),
        ]


def _sensitivity_payload(*, identities=3, status="admitted", candidate_count=4,
                          required=4, evidence="same"):
    return {
        "candidate_budget": {"status": status, "required": required, "maximum": 8},
        "candidate_count": candidate_count,
        "observed_distinct_train_rank_plan_identities": identities,
        "candidate_evidence_sha256": evidence,
    }


def test_sensitivity_record_rejects_budget_overflow():
    payload = _sensitivity_payload(status="exceeded", required=9)
    with pytest.raises(RuntimeError, match="not admitted"):
        runner._validate_sensitivity_record(payload, {"maximum_candidates": 8}, "overflow.json")


def test_sensitivity_record_rejects_zero_distinct_plans():
    payload = _sensitivity_payload(identities=0)
    with pytest.raises(RuntimeError, match="distinct TRAIN rank-plan"):
        runner._validate_sensitivity_record(payload, {"maximum_candidates": 8}, "zero.json")


def test_sensitivity_pair_rejects_cached_uncached_count_mismatch():
    cached = _sensitivity_payload(identities=3)
    uncached = _sensitivity_payload(identities=2)
    with pytest.raises(RuntimeError, match="distinct rank-plan mismatch"):
        runner._validate_sensitivity_pair(cached, uncached, "mismatch")


def test_matrix_refuses_to_run_outside_project_venv_before_creating_output(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner.sys, "prefix", str(tmp_path / "wrong-env"))
    with pytest.raises(RuntimeError, match="project interpreter"):
        runner.run_matrix(output_dir)
    assert not output_dir.exists()


def test_prepare_output_dir_does_not_overwrite_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "BENCHMARK_DOCS", tmp_path)
    output_dir = tmp_path / "u_shape_rank_reuse_20261001_test_nonempty"
    output_dir.mkdir(parents=False, exist_ok=False)
    marker = output_dir / "existing.json"
    marker.write_text("preserve", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError, match="non-empty evidence directory"):
            runner._prepare_output_dir(output_dir)
        assert marker.read_text(encoding="utf-8") == "preserve"
    finally:
        marker.unlink()
        output_dir.rmdir()
