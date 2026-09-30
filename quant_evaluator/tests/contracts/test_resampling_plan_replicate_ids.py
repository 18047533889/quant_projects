from quant_evaluator.contracts import resampling
from quant_evaluator.contracts.resampling import ResamplingPlan


def test_replicate_ids_reuse_one_digest_per_property_access(monkeypatch):
    plan = ResamplingPlan(
        time_ids=tuple(range(20)),
        clock_ref="replicate-id-test",
        block_length=5,
        num_replicates=200,
        seed=17,
    )
    expected_hash = plan.content_hash
    original = resampling.stable_content_hex
    calls = []

    def count_digest(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(resampling, "stable_content_hex", count_digest)

    expected_ids = tuple(f"{expected_hash}:{i}" for i in range(200))
    assert plan.replicate_ids == expected_ids
    assert len(calls) == 1

    # Re-access recomputes from the frozen plan instead of caching across calls.
    assert plan.replicate_ids == expected_ids
    assert len(calls) == 2

