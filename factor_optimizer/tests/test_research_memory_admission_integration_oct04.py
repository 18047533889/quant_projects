"""Batch memory rejection must precede chronological preparation/allocation."""
import numpy as np
import pytest

from factor_optimizer import research_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _tiny_contracts():
    times = np.arange(4, dtype=np.int64)
    assets = np.asarray(["a", "b", "c"])
    ta = AxisRef("time", str(times.dtype), 4, times)
    aa = AxisRef("asset", str(assets.dtype), 3, assets)
    batch = FactorBatch(("tiny",), ta, aa, np.ones((4, 3, 1)))
    labels = LabelBundle(
        "memory-fixture", np.ones((4, 3)), 1,
        decision_time=tuple(range(4)), label_start_time=tuple(range(1, 5)),
        label_end_time=tuple(range(2, 6)), asset_axis=aa,
    )
    return batch, labels


def test_batch_rejects_peak_budget_before_split_or_training_preparation(monkeypatch):
    batch, labels = _tiny_contracts()
    config = research_batch.BatchOptimizationConfig(max_additional_peak_bytes=1)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("memory rejection must precede split and preparation")

    monkeypatch.setattr(research_batch, "automatic_time_split", forbidden)
    monkeypatch.setattr(research_batch, "_prepare_pair_ic_split", forbidden)
    with pytest.raises(MemoryError):
        research_batch.optimize_factor_batch(batch, labels, config=config, allow_research=True)
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("max_additional_peak_bytes", True), ("max_additional_peak_bytes", 0),
    ("memory_reserve_bytes", True), ("memory_reserve_bytes", -1),
])
def test_memory_config_rejects_invalid_limits(field, value):
    with pytest.raises(ValueError):
        research_batch.BatchOptimizationConfig(**{field: value})
def test_batch_charges_configured_bootstrap_before_split(monkeypatch):
    # Missing config forwarding admits unbounded retained bootstrap draws.
    from factor_optimizer import research_memory_admission
    batch, labels = _tiny_contracts()
    config = research_batch.BatchOptimizationConfig(
        bootstrap_draws=10**12, max_additional_peak_bytes=1024**3,
        memory_reserve_bytes=0,
    )
    monkeypatch.setattr(research_memory_admission, "available_memory_bytes", lambda: 2**63)

    def forbidden(*args, **kwargs):
        raise AssertionError("bootstrap budget must reject before split")

    monkeypatch.setattr(research_batch, "automatic_time_split", forbidden)
    with pytest.raises(MemoryError):
        research_batch.optimize_factor_batch(batch, labels, config=config, allow_research=True)



@pytest.mark.parametrize("available,error", [(None, RuntimeError), (0, MemoryError)])
def test_live_headroom_rejects_before_split(monkeypatch, available, error):
    from factor_optimizer import research_memory_admission
    batch, labels = _tiny_contracts()
    monkeypatch.setattr(research_memory_admission, "available_memory_bytes", lambda: available)
    def forbidden(*args, **kwargs):
        raise AssertionError("headroom rejection must precede split")
    monkeypatch.setattr(research_batch, "automatic_time_split", forbidden)
    with pytest.raises(error):
        research_batch.optimize_factor_batch(batch, labels, allow_research=True)


def test_config_cap_rejects_without_os_reader(monkeypatch):
    from factor_optimizer import research_memory_admission
    batch, labels = _tiny_contracts()
    def forbidden():
        raise AssertionError("known cap failure must not read host memory")
    monkeypatch.setattr(research_memory_admission, "available_memory_bytes", forbidden)
    with pytest.raises(MemoryError):
        research_batch.optimize_factor_batch(
            batch, labels, allow_research=True,
            config=research_batch.BatchOptimizationConfig(max_additional_peak_bytes=1),
        )

def test_exact_headroom_with_zero_reserve_reaches_split(monkeypatch):
    from factor_optimizer import research_memory_admission
    batch, labels = _tiny_contracts()
    estimate = research_memory_admission.estimate_incremental_peak_bytes((4, 3, 1))
    config = research_batch.BatchOptimizationConfig(
        max_additional_peak_bytes=estimate, memory_reserve_bytes=0,
    )
    monkeypatch.setattr(research_memory_admission, "available_memory_bytes", lambda: estimate)
    class SplitReached(Exception):
        pass
    calls = []
    def stop_at_split(*args, **kwargs):
        calls.append(True)
        raise SplitReached
    monkeypatch.setattr(research_batch, "automatic_time_split", stop_at_split)
    with pytest.raises(SplitReached):
        research_batch.optimize_factor_batch(batch, labels, config=config, allow_research=True)
    assert calls == [True]

