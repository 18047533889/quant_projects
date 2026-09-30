import gc
import weakref
from dataclasses import dataclass

import pytest

from quant_evaluator.runtime import measured_auto_registry as registry


@dataclass(frozen=True)
class Policy:
    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    repetitions: int = 3
    warmups: int = 1


class Cache:
    def __init__(self):
        self.record = None

    def get(self, key):
        return self.record if key == "key" else None


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear_registry_for_tests()
    yield
    registry.clear_registry_for_tests()


def descriptor(**changes):
    fields = dict(factor_ids=("f1", "f2"), shape=(8, 4, 2), value_hash="vh",
                  label_content_hash="lh", dtype="float32",
                  axis_schema=(("time", "int64", 8), ("asset", "str", 4)),
                  metrics=("ic", "rank_ic", "turnover"))
    fields.update(changes)
    return registry.InputDescriptor.from_values(**fields)


def sample_record(winner="cuda_strict", **changes):
    d = dict(winner=winner, parity="pass", selection_basis="steady_state_median",
             cpu_median_seconds=10., cuda_median_seconds=4.,
             repetitions=3, warmups=1)
    d.update(changes)
    return d


def add(cache=None, desc=None, policy=None, gpu=None, **changes):
    cache = cache or Cache()
    record = changes.pop("calibration_record", sample_record())
    cache.record = record
    args = dict(status="calibrated", winner=record["winner"], calibration_record=record,
                source_check={"mode": "strict_full_content"}, process_source_drifted=False,
                calibration_policy=policy or Policy(), gpu_policy=gpu or {"device_ids": (0,)},
                descriptor=desc or descriptor(), setup_seconds=.5)
    args.update(changes)
    ok = registry.register_candidate(cache, "key", **args)
    return ok, cache


def test_exact_cheap_descriptor_and_policy_matching_before_validator():
    ok, cache = add()
    assert ok
    calls = []
    validate = lambda candidate: calls.append(candidate) or True
    for d in (descriptor(value_hash="other"), descriptor(label_content_hash="other"),
              descriptor(factor_ids=("f2", "f1")),
              descriptor(metrics=("rank_ic", "ic", "turnover")),
              descriptor(axis_schema=(("time", "int32", 8), ("asset", "str", 4)))):
        assert registry.lookup_candidate(descriptor=d, calibration_policy=Policy(),
            gpu_policy={"device_ids": (0,)}, static_backend="cpu", validator=validate) is None
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(repetitions=4),
        gpu_policy={"device_ids": (0,)}, static_backend="cpu", validator=validate) is None
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (1,)}, static_backend="cpu", validator=validate) is None
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cpu", validator=validate) == "cuda_strict"
    assert calls
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cuda_strict", validator=validate) is None
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [
    {"status": "parity_failed_cpu_fallback"},
    {"process_source_drifted": True},
    {"source_check": {"mode": "stat_guarded"}},
    {"calibration_policy": Policy(relative_tolerance=1e-8)},
    {"calibration_policy": Policy(absolute_tolerance=1e-6)},
    {"calibration_policy": Policy(repetitions=1)},
    {"calibration_policy": Policy(warmups=0)},
    {"calibration_record": sample_record(parity="fail")},
    {"calibration_record": sample_record(cpu_median_seconds=float("nan"))},
    {"calibration_record": sample_record(cuda_median_seconds=0)},
])
def test_untrusted_or_unhelpful_measurements_rejected(kwargs):
    # Custom record winner and policy keyword are forwarded through helper.
    assert add(**kwargs)[0] is False
    assert registry.registry_size() == 0


def test_validator_rejection_models_mutated_actual_input_and_no_replay():
    ok, cache = add()
    assert ok
    calls = []
    route = registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cpu",
        validator=lambda candidate: calls.append(candidate) or False)
    assert route is None and len(calls) == 1


def test_no_validator_on_nomatch_or_same_static_winner():
    assert add()[0]
    calls = []
    validate = lambda candidate: calls.append(candidate) or True
    assert registry.lookup_candidate(descriptor=descriptor(value_hash="changed"),
        calibration_policy=Policy(), gpu_policy={"device_ids": (0,)},
        static_backend="cpu", validator=validate) is None
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cuda_strict", validator=validate) is None
    assert calls == []


def test_policy_agnostic_default_lookup_retains_candidate_measurement_policy():
    measured_policy = Policy(repetitions=2)
    ok, cache = add(policy=measured_policy)
    assert ok
    seen = []
    route = registry.lookup_candidate(
        descriptor=descriptor(), calibration_policy=None,
        gpu_policy={"device_ids": (0,)}, static_backend="cpu",
        validator=lambda candidate: seen.append(candidate) or True,
    )
    assert route == "cuda_strict"
    assert len(seen) == 1
    assert dict(seen[0].calibration_policy)["repetitions"] == 2


def test_validator_must_leave_same_passing_cache_winner():
    ok, cache = add()
    assert ok
    cache.record = sample_record("cpu")
    assert registry.lookup_candidate(descriptor=descriptor(), calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cpu", validator=lambda _: True) is None


def test_weak_cache_reference_and_lru_bound():
    caches = []
    for i in range(33):
        ok, cache_i = add(desc=descriptor(value_hash=str(i)))
        assert ok
        caches.append(cache_i)
    assert registry.registry_size() == registry.MAX_ENTRIES
    d = descriptor(value_hash="32")
    assert registry.lookup_candidate(descriptor=d, calibration_policy=Policy(),
        gpu_policy={"device_ids": (0,)}, static_backend="cpu", validator=lambda _: True) == "cuda_strict"
    # Drop final cache owner; candidate is pruned without extending cache life.
    ok, cache = add(desc=descriptor(value_hash="gc"))
    ref = weakref.ref(cache)
    del cache
    gc.collect()
    assert ref() is None
    assert registry.lookup_candidate(descriptor=descriptor(value_hash="gc"),
        calibration_policy=Policy(), gpu_policy={"device_ids": (0,)},
        static_backend="cpu", validator=lambda _: True) is None


def test_ttl_expiry_and_fork_reset(monkeypatch):
    ok, cache = add()
    assert ok and registry.registry_size() == 1
    clock = [100.0]
    monkeypatch.setattr(registry.time, "monotonic", lambda: clock[0])
    # A newly registered item uses the controlled clock.
    registry.clear_registry_for_tests()
    assert add(cache=cache)[0]
    clock[0] += registry.TTL_SECONDS
    assert registry.registry_size() == 0
    assert add(cache=cache)[0]
    registry._after_fork_child()
    assert registry.registry_size() == 0
