"""Small independent parity checks for the real-COS diagnostics A/B harness."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.diagnosis.factor import diagnose_all_factors
from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import (
    legacy_diagnose_all_factors,
)


@pytest.mark.parametrize("shape", [(0, 3, 2), (4, 0, 2), (3, 5, 4)])
def test_flatten_reference_matches_strided_diagnostics(shape):
    rng = np.random.default_rng(918273)
    source = rng.normal(size=shape)
    validity = rng.random(shape) > 0.2
    if source.size:
        source.reshape(-1)[0] = np.nan
        if source.size > 1:
            source.reshape(-1)[1] = np.inf
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(shape[2])),
        AxisRef("time", "int64", shape[0], np.arange(shape[0], dtype=np.int64)),
        AxisRef("asset", "int64", shape[1], np.arange(shape[1], dtype=np.int64)),
        source, validity=validity,
    )
    assert legacy_diagnose_all_factors(batch) == diagnose_all_factors(batch)


def test_flatten_reference_matches_constant_wide_integer_semantics():
    values = np.array([2**63, 2**63 + 1], dtype=np.uint64).reshape(1, 2, 1)
    batch = FactorBatch(
        ("wide",), AxisRef("time", "int64", 1, np.array([0], dtype=np.int64)),
        AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64)), values,
    )
    assert legacy_diagnose_all_factors(batch) == diagnose_all_factors(batch)


def test_real_ab_defaults_to_preflight_and_never_loads(monkeypatch):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts import load_real_cos_factor_batch as loader
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab

    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: {"status": "ready"})
    monkeypatch.setattr(loader, "load_real_batch",
                        lambda **kwargs: pytest.fail("preflight-only must not load COS data"))
    assert run_real_cos_ab()["status"] == "preflight_only"


def test_real_ab_requires_output_before_preflight_or_load(monkeypatch):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab

    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: pytest.fail("missing output must fail before preflight"))
    with pytest.raises(ValueError, match="output is required"):
        run_real_cos_ab(run=True)


def test_real_ab_rejected_preflight_never_loads(monkeypatch, tmp_path):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts import load_real_cos_factor_batch as loader
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab

    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: {"status": "insufficient_resources"})
    monkeypatch.setattr(loader, "load_real_batch",
                        lambda **kwargs: pytest.fail("rejected preflight must not load"))
    report = run_real_cos_ab(run=True, output=tmp_path / "report.json")
    assert report["status"] == "preflight_rejected"


def _stub_real_batch():
    from types import SimpleNamespace
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import (
        EXPECTED_DTYPE, EXPECTED_SHAPE,
    )
    return SimpleNamespace(
        values=SimpleNamespace(shape=EXPECTED_SHAPE, dtype=EXPECTED_DTYPE),
        validity=None, factor_ids=tuple(f"f{i}" for i in range(EXPECTED_SHAPE[2])),
        num_factors=EXPECTED_SHAPE[2],
    )


def _stub_diagnostics():
    from quant_evaluator.api.requests import FactorDiagnosis
    return {"f0": FactorDiagnosis(
        factor_id="f0", num_valid_observations=1, num_missing=0,
        coverage=1.0, is_constant=True, has_nans=False, has_infs=False,
        min_value=2.0, max_value=2.0, mean_value=2.0,
        warnings=("Factor is constant",),
    )}


def _install_ready_run_mocks(monkeypatch, batch=None):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts import load_real_cos_factor_batch as loader
    from quant_evaluator.scripts import benchmark_factor_diagnosis_copy_ab as harness
    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: {"status": "ready"})
    monkeypatch.setattr(loader, "load_real_batch",
                        lambda **kwargs: (batch or _stub_real_batch(), None, {}))
    monkeypatch.setattr(harness, "legacy_diagnose_all_factors",
                        lambda value: _stub_diagnostics())
    monkeypatch.setattr(harness, "diagnose_all_factors",
                        lambda value: _stub_diagnostics())
    monkeypatch.setattr(harness, "_rss_bytes", lambda: 123)
    monkeypatch.setattr(harness, "_sha256", lambda path: "same")


def test_real_ab_success_writes_compact_receipt(monkeypatch, tmp_path):
    import json
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab
    _install_ready_run_mocks(monkeypatch)
    output = tmp_path / "receipt.json"
    report = run_real_cos_ab(run=True, output=output, repeats=1)
    assert report["status"] == "complete"
    assert report["shape"] == [2586, 5461, 32]
    assert report["source_sha256"]["before"] == report["source_sha256"]["after"]
    assert json.loads(output.read_text()) == report
    assert output.read_text().count("\n") == 1


def test_existing_output_fails_before_preflight_or_load(monkeypatch, tmp_path):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts import load_real_cos_factor_batch as loader
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: pytest.fail("must reject before preflight"))
    monkeypatch.setattr(loader, "load_real_batch",
                        lambda **kwargs: pytest.fail("must reject before loading"))
    with pytest.raises(FileExistsError):
        run_real_cos_ab(run=True, output=output)
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("shape,dtype", [
    ((2586, 5460, 32), np.dtype("float64")),
    ((2586, 5461, 32), np.dtype("float32")),
])
def test_real_ab_rejects_shape_or_dtype_before_timing(monkeypatch, tmp_path, shape, dtype):
    from types import SimpleNamespace
    from quant_evaluator.scripts import load_real_cos_factor_batch as loader
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab
    _install_ready_run_mocks(monkeypatch)
    bad = SimpleNamespace(values=SimpleNamespace(shape=shape, dtype=dtype))
    monkeypatch.setattr(loader, "load_real_batch", lambda **kwargs: (bad, None, {}))
    with pytest.raises(ValueError, match="differs from certified workload"):
        run_real_cos_ab(run=True, output=tmp_path / "bad.json", repeats=1)


@pytest.mark.parametrize("repeats", [True, False, 1.0, "2", 0, 11])
def test_real_ab_rejects_invalid_repeats_before_preflight(monkeypatch, repeats):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts.benchmark_factor_diagnosis_copy_ab import run_real_cos_ab
    monkeypatch.setattr(bench, "preflight_factor_count_profile",
                        lambda args: pytest.fail("must reject invalid repeats first"))
    with pytest.raises(ValueError, match="repeats"):
        run_real_cos_ab(repeats=repeats)


def test_real_ab_source_drift_before_timing_fails_closed(monkeypatch, tmp_path):
    from quant_evaluator.scripts import benchmark_factor_diagnosis_copy_ab as harness
    _install_ready_run_mocks(monkeypatch)
    calls = {"n": 0}
    def changing_hash(path):
        calls["n"] += 1
        return "before" if calls["n"] <= 2 else "changed"
    monkeypatch.setattr(harness, "_sha256", changing_hash)
    monkeypatch.setattr(harness, "legacy_diagnose_all_factors",
                        lambda value: pytest.fail("drift must fail before timing"))
    with pytest.raises(RuntimeError, match="changed during data load"):
        harness.run_real_cos_ab(run=True, output=tmp_path / "drift.json", repeats=1)


def test_real_ab_source_drift_after_pair_fails_closed(monkeypatch, tmp_path):
    from quant_evaluator.scripts import benchmark_factor_diagnosis_copy_ab as harness
    _install_ready_run_mocks(monkeypatch)
    calls = {"n": 0}
    def changing_hash(path):
        calls["n"] += 1
        return "before" if calls["n"] <= 4 else "changed"
    monkeypatch.setattr(harness, "_sha256", changing_hash)
    with pytest.raises(RuntimeError, match="changed during A/B"):
        harness.run_real_cos_ab(run=True, output=tmp_path / "drift-after.json", repeats=1)


def test_contiguous_ravel_path_matches_oracle_and_preserves_read_only_inputs():
    values = np.arange(6 * 5 * 4, dtype=np.float64).reshape(6, 5, 4)
    values[1, 2, 0] = np.nan
    values[3, 1, 2] = np.inf
    values[5, 4, 3] = -np.inf
    validity = np.ones(values.shape, dtype=bool)
    validity[2, 3, 0] = False
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(values.shape[2])),
        AxisRef("time", "int64", values.shape[0], np.arange(values.shape[0], dtype=np.int64)),
        AxisRef("asset", "int64", values.shape[1], np.arange(values.shape[1], dtype=np.int64)),
        values, validity=validity,
    )
    assert batch.values.flags.c_contiguous and not batch.values.flags.writeable
    assert batch.validity.flags.c_contiguous and not batch.validity.flags.writeable
    values_before = batch.values.copy()
    validity_before = batch.validity.copy()
    for index in range(batch.num_factors):
        plane = batch.values[:, :, index]
        valid_plane = batch.validity[:, :, index]
        contiguous_values = plane.ravel(order="C")
        contiguous_validity = valid_plane.ravel(order="C")
        np.testing.assert_array_equal(contiguous_values, plane.flatten())
        np.testing.assert_array_equal(contiguous_validity, valid_plane.flatten())
        assert contiguous_values.flags.c_contiguous
        assert contiguous_validity.flags.c_contiguous
        assert not np.shares_memory(contiguous_values, batch.values)
        assert not np.shares_memory(contiguous_validity, batch.validity)
    assert legacy_diagnose_all_factors(batch) == diagnose_all_factors(batch)
    np.testing.assert_array_equal(batch.values, values_before)
    np.testing.assert_array_equal(batch.validity, validity_before)
