"""Regression tests for per-factor failure isolation in optimize_factors_v2."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jobs import optimize_factors_v2 as job


def test_resume_does_not_skip_outputs_with_missing_or_invalid_metadata():
    metadata = {
        "complete": {"evaluation_status": "valid", "best_treatment_token": "raw"},
        "invalid": {"evaluation_status": "invalid", "best_treatment_token": "raw"},
        "failed": {"evaluation_status": "valid", "best_treatment_token": "raw", "error": "write_failed"},
    }
    names = ["complete", "orphan", "invalid", "failed", "new"]
    assert job.pending_pages(names, {"complete", "orphan", "invalid", "failed"}, metadata) == [
        "orphan", "invalid", "failed", "new",
    ]


class _SuccessfulTrial:
    def __init__(self, page, treatment="raw"):
        self.metadata = {"params": {"page": page, "treatment": treatment}}

    def is_successful(self):
        return True


def _configure_small_batch(monkeypatch):
    index = pd.date_range("2026-01-01", periods=2)
    monkeypatch.setattr(job, "_GV", {
        "FWD": np.zeros((2, 2)), "row_index": index, "n_days": 2,
        "universe": ["A", "B"], "neutral_bundle": None,
    })
    monkeypatch.setattr(job, "apply_treatment", lambda arr, token: arr.astype(float))
    monkeypatch.setattr(job, "rankic_ir_fast", lambda arr, n_days, stride=1: (0.1, 0.2, 2))
    monkeypatch.setattr(job, "detect_dsl_preproc", lambda page: [])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: None)

    class _Runner:
        def __init__(self, *args, **kwargs): pass
        def run(self, session_id):
            return SimpleNamespace(session_id=session_id, stop_reason="budget_exhausted",
                                   trials=[_SuccessfulTrial("good")])

    monkeypatch.setattr(job, "SearchRunner", _Runner)
    return index


def test_process_chunk_records_read_failure_instead_of_dropping_factor(monkeypatch):
    """Catches the production defect where a read exception silently continued."""
    index = _configure_small_batch(monkeypatch)

    def fake_read(path):
        if path.stem == "broken":
            raise OSError("unreadable parquet")
        return pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=["A", "B"])

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    result = job.process_chunk(["good", "broken"])
    assert result["broken"]["error_phase"] == "read_factor_matrix"
    assert result["broken"]["error_type"] == "OSError"
    assert result["broken"]["fallback"] == "unavailable_raw"


def test_process_chunk_falls_back_to_raw_for_each_factor_without_successful_trial(monkeypatch):
    """Catches the production defect where one good page hid another page's total failure."""
    index = _configure_small_batch(monkeypatch)
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=index, columns=["A", "B"]))
    result = job.process_chunk(["good", "all_candidates_failed"])
    failed = result["all_candidates_failed"]
    assert failed["best_treatment_token"] == "raw"
    assert failed["fallback"] == "raw"
    assert failed["fallback_reason"] == "no_successful_candidate_trials"
    assert failed["best_rankic_ir"] == 0.2


def test_real_parquet_conversion_failure_is_isolated_from_good_factor(monkeypatch, tmp_path):
    """A bad numeric conversion must not abort a sibling backed by real parquet."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(); output_dir.mkdir()
    index = pd.date_range("2026-01-01", periods=40)
    columns = [f"S{i:02d}" for i in range(32)]
    values = np.arange(40 * 32, dtype=float).reshape(40, 32)
    pd.DataFrame(values, index=index, columns=columns).to_parquet(input_dir / "good.parquet")
    pd.DataFrame("not-a-number", index=index, columns=columns).to_parquet(
        input_dir / "bad_conversion.parquet"
    )
    monkeypatch.setattr(job, "FV_DIR", input_dir)
    monkeypatch.setattr(job, "OUT_DIR", output_dir)
    monkeypatch.setattr(job, "detect_dsl_preproc", lambda page: [])
    forward = np.vstack([
        np.arange(32, dtype=float) if day % 2 == 0 else np.arange(31, -1, -1, dtype=float)
        for day in range(40)
    ])
    monkeypatch.setattr(job, "_GV", {
        "FWD": forward,
        "row_index": index,
        "n_days": 40,
        "universe": columns,
        "neutral_bundle": None,
    })

    class _Runner:
        def __init__(self, *args, **kwargs): pass
        def run(self, session_id):
            return SimpleNamespace(session_id=session_id, stop_reason="budget_exhausted",
                                   trials=[_SuccessfulTrial("good")])

    monkeypatch.setattr(job, "SearchRunner", _Runner)
    result = job.process_chunk(["good", "bad_conversion"])
    assert result["good"]["evaluation_status"] == "valid"
    assert result["bad_conversion"]["error_phase"] == "prepare_factor_matrix"
    assert result["bad_conversion"]["error_type"] == "ValueError"


def test_output_write_failure_isolated_and_raw_without_ic_is_invalid(monkeypatch, tmp_path):
    """One failed output write and an evidence-free RAW fallback remain explicit."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(); output_dir.mkdir()
    index = pd.date_range("2026-01-01", periods=40)
    columns = [f"S{i:02d}" for i in range(32)]
    values = np.arange(40 * 32, dtype=float).reshape(40, 32)
    for page in ("good", "write_bad", "raw_invalid"):
        pd.DataFrame(values, index=index, columns=columns).to_parquet(input_dir / f"{page}.parquet")
    monkeypatch.setattr(job, "FV_DIR", input_dir)
    monkeypatch.setattr(job, "OUT_DIR", output_dir)
    monkeypatch.setattr(job, "detect_dsl_preproc", lambda page: [])
    monkeypatch.setattr(job, "_GV", {
        "FWD": values / 1000.0, "row_index": index, "n_days": 40,
        "universe": columns, "neutral_bundle": None,
    })

    class _Runner:
        def __init__(self, *args, **kwargs): pass
        def run(self, session_id):
            return SimpleNamespace(session_id=session_id, stop_reason="budget_exhausted",
                                   trials=[_SuccessfulTrial("good"), _SuccessfulTrial("write_bad")])

    monkeypatch.setattr(job, "SearchRunner", _Runner)
    real_to_parquet = pd.DataFrame.to_parquet
    def fail_one_write(frame, path, *args, **kwargs):
        if path.stem == "write_bad":
            raise OSError("disk write failed")
        return real_to_parquet(frame, path, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_one_write)
    real_rank = job.rankic_ir_fast
    def no_raw_evidence(arr, n_days, stride=1):
        if np.array_equal(arr, values.astype(np.float32), equal_nan=True):
            return 0.0, 0.0, 0
        return real_rank(arr, n_days, stride)
    monkeypatch.setattr(job, "rankic_ir_fast", no_raw_evidence)

    result = job.process_chunk(["good", "write_bad", "raw_invalid"])
    assert result["write_bad"]["error_phase"] == "write_output_matrix"
    assert result["good"]["evaluation_status"] == "invalid"
    invalid = result["raw_invalid"]
    assert invalid["evaluation_status"] == "invalid"
    assert invalid["best_rankic_ir"] is None
    assert invalid["optimization_status"] == "raw_fallback_invalid"


def test_chunk_failure_records_cover_every_submitted_page():
    records = job.chunk_failure_records(["a", "b"], RuntimeError("worker died"))
    assert set(records) == {"a", "b"}
    assert all(item["error_phase"] == "process_chunk" for item in records.values())


def test_rankic_ir_rejects_constant_cross_sections_as_no_observations(monkeypatch):
    """Constant factor ranks have undefined correlation, not zero IC evidence."""
    factor = np.ones((40, 32), dtype=float)
    forward = np.tile(np.arange(32, dtype=float), (40, 1))
    monkeypatch.setattr(job, "_GV", {"FWD": forward})
    mean, ir, count = job.rankic_ir_fast(factor, 40)
    assert count == 0
    assert np.isnan(mean)
    assert np.isnan(ir)


def test_search_callback_rejects_undefined_icir_instead_of_scoring_zero(monkeypatch, tmp_path):
    index = pd.date_range("2026-01-01", periods=40)
    columns = [f"S{i:02d}" for i in range(32)]
    pd.DataFrame(np.ones((40, 32)), index=index, columns=columns).to_parquet(tmp_path / "constant.parquet")
    monkeypatch.setattr(job, "FV_DIR", tmp_path)
    monkeypatch.setattr(job, "OUT_DIR", tmp_path)
    monkeypatch.setattr(job, "detect_dsl_preproc", lambda page: [])
    monkeypatch.setattr(job, "_GV", {
        "FWD": np.tile(np.arange(32), (40, 1)), "row_index": index,
        "n_days": 40, "universe": columns, "neutral_bundle": None,
    })

    class InspectingRunner:
        def __init__(self, **kwargs):
            self.protocol = kwargs["evaluation_fn"]

        def run(self, session_id):
            with pytest.raises(ValueError, match="undefined"):
                self.protocol.evaluator(_SuccessfulTrial("constant"), None)
            return SimpleNamespace(session_id=session_id, stop_reason="budget_exhausted", trials=[])

    monkeypatch.setattr(job, "SearchRunner", InspectingRunner)
    result = job.process_chunk(["constant"])
    assert result["constant"]["evaluation_status"] == "invalid"
    assert result["constant"]["best_rankic_ir"] is None


def test_rankic_ir_requires_two_days_and_nonconstant_ic_series(monkeypatch):
    """A single IC day and exact-zero IC variance cannot define sample ICIR."""
    factor = np.full((40, 32), np.nan)
    forward = np.full((40, 32), np.nan)
    factor[0] = np.arange(32, dtype=float)
    forward[0] = np.arange(32, dtype=float)
    monkeypatch.setattr(job, "_GV", {"FWD": forward})
    mean, ir, count = job.rankic_ir_fast(factor, 40)
    assert count == 1
    assert mean == 1.0
    assert np.isnan(ir)

    repeated_factor = np.tile(np.arange(32, dtype=float), (40, 1))
    repeated_forward = repeated_factor.copy()
    monkeypatch.setattr(job, "_GV", {"FWD": repeated_forward})
    mean, ir, count = job.rankic_ir_fast(repeated_factor, 40)
    assert count == 40
    assert mean == 1.0
    assert np.isnan(ir)
