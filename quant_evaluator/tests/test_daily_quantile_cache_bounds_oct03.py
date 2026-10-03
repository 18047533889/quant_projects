"""Budget and ownership controls for request-local daily quantile reuse."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.daily_quantile_cache import RequestDailyQuantilePanelCache


def _inputs():
    t, n, f = 6, 100, 2
    time = AxisRef("time", "int64", t, np.arange(t, dtype=np.int64))
    assets = AxisRef("asset", "int64", n, np.arange(n, dtype=np.int64))
    rng = np.random.default_rng(99103)
    batch = FactorBatch(("a", "b"), time, assets, rng.normal(size=(t, n, f)))
    labels = LabelBundle("bounds", rng.normal(size=(t, n)), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=assets)
    return batch, labels


def _numpy_counter(monkeypatch):
    import quant_evaluator.metrics.registry_adapters as adapters
    original = adapters.compute_quantile_returns_fast
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        kwargs["use_numba"] = False
        return original(*args, **kwargs)
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast", counted)
    return calls


def _payload_bytes(artifact):
    return (artifact.values.nbytes + artifact.counts.nbytes + artifact.valid_mask.nbytes
            + artifact.provenance["valid_period_counts_qf"].nbytes)


def test_byte_bound_counts_metadata_and_evicts_old_panel(monkeypatch):
    calls = _numpy_counter(monkeypatch)
    batch, labels = _inputs()
    probe = RequestDailyQuantilePanelCache(batch, labels).get_or_build(batch, labels, 5, 1)
    budget = _payload_bytes(probe)
    calls.clear()
    cache = RequestDailyQuantilePanelCache(batch, labels, max_retained_bytes=budget)
    first = cache.get_or_build(batch, labels, 5, 1)
    assert cache.retained_bytes == budget
    second = cache.get_or_build(batch, labels, 4, 1)
    assert cache.entry_count == 1
    assert cache.retained_bytes == _payload_bytes(second) <= budget
    assert cache.get_or_build(batch, labels, 4, 1) is second
    rebuilt = cache.get_or_build(batch, labels, 5, 1)
    assert rebuilt is not first
    np.testing.assert_array_equal(rebuilt.values, first.values)
    np.testing.assert_array_equal(rebuilt.counts, first.counts)
    assert len(calls) == 3


def test_entry_cap_and_hit_refresh_implement_lru(monkeypatch):
    calls = _numpy_counter(monkeypatch)
    batch, labels = _inputs()
    cache = RequestDailyQuantilePanelCache(batch, labels, max_entries=2)
    a = cache.get_or_build(batch, labels, 2, 1)
    b = cache.get_or_build(batch, labels, 3, 1)
    assert cache.get_or_build(batch, labels, 2, 1) is a
    cache.get_or_build(batch, labels, 4, 1)
    assert cache.entry_count == 2
    assert cache.get_or_build(batch, labels, 2, 1) is a
    assert cache.get_or_build(batch, labels, 3, 1) is not b
    assert len(calls) == 4


def test_non_root_input_ownership_is_charged_and_overbudget_does_not_retain(monkeypatch):
    _numpy_counter(monkeypatch)
    batch, labels = _inputs()
    panel = RequestDailyQuantilePanelCache(batch, labels).get_or_build(batch, labels, 5, 1)
    cache = RequestDailyQuantilePanelCache(max_retained_bytes=_payload_bytes(panel))
    actual = cache.get_or_build(batch, labels, 5, 1)
    assert cache.entry_count == 0 and cache.retained_bytes == 0
    np.testing.assert_array_equal(actual.values, panel.values)


def test_zero_budget_recomputes_without_retaining(monkeypatch):
    calls = _numpy_counter(monkeypatch)
    batch, labels = _inputs()
    cache = RequestDailyQuantilePanelCache(batch, labels, max_retained_bytes=0)
    a = cache.get_or_build(batch, labels, 5, 1)
    b = cache.get_or_build(batch, labels, 5, 1)
    assert a is not b and len(calls) == 2
    assert cache.entry_count == 0 and cache.retained_bytes == 0
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("entries", [0, -1, True, 2.5])
def test_invalid_entry_cap_rejected(entries):
    with pytest.raises(ValueError, match="max_entries"):
        RequestDailyQuantilePanelCache(max_entries=entries)
