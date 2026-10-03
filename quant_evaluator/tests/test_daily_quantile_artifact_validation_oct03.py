"""Fail-closed typed daily quantile artifact reuse validation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _inputs():
    rng = np.random.default_rng(82106)
    t, n, f = 24, 100, 2
    times = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = rng.integers(0, 17, size=(t, n, f)).astype(float)
    values[rng.random(values.shape) < .02] = np.nan
    labels = rng.normal(size=(t, n))
    labels[rng.random(labels.shape) < .03] = np.nan
    validity = np.ones(values.shape, dtype=bool)
    validity[3::8, ::11, :] = False
    batch = FactorBatch(("f0", "f1"), times, assets, values, validity=validity)
    bundle = LabelBundle(
        "artifact-reuse", labels, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=assets,
    )
    return batch, bundle


def _artifact(batch, labels, *, n_quantiles=5, min_assets=10):
    from quant_evaluator.metrics.registry_adapters import build_daily_quantile_return_artifact

    return build_daily_quantile_return_artifact(
        batch, labels, n_quantiles=n_quantiles, min_assets=min_assets,
        min_periods=1,
    )


def _cached_artifact(batch, labels, *, n_quantiles=5, min_assets=10):
    from quant_evaluator.runtime.daily_quantile_cache import RequestDailyQuantilePanelCache

    return RequestDailyQuantilePanelCache().get_or_build(
        batch, labels, n_quantiles, min_assets,
    )


def _mutate_provenance(artifact, **updates):
    provenance = dict(artifact.provenance)
    provenance.update(updates)
    return replace(artifact, provenance=provenance)


def _assert_same_metadata(left, right):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize("adapter_name", ["spread", "full"])
@pytest.mark.parametrize("mismatch", [
    "label_content_hash", "factor_value_hash", "factor_axis", "quantile_count",
    "min_assets", "producer_version", "tie_method", "valid_mask_contents",
])
def test_profile_adapters_reject_daily_artifacts_from_different_contracts(
        adapter_name, mismatch):
    import quant_evaluator.metrics.registry_adapters as adapters

    batch, labels = _inputs()
    artifact = _cached_artifact(batch, labels)
    if mismatch == "label_content_hash":
        artifact = _mutate_provenance(artifact, label_content_hash="stale-label-hash")
    elif mismatch == "factor_value_hash":
        artifact = _mutate_provenance(artifact, factor_value_hash="stale-factor-hash")
    elif mismatch == "factor_axis":
        artifact = replace(artifact, factor_axis=("wrong0", "wrong1"))
    elif mismatch == "quantile_count":
        artifact = _mutate_provenance(artifact, n_quantiles=4)
    elif mismatch == "min_assets":
        artifact = _cached_artifact(batch, labels, min_assets=2)
    elif mismatch == "producer_version":
        artifact = replace(artifact, producer_version="0.9.0")
    elif mismatch == "tie_method":
        artifact = _mutate_provenance(artifact, tie_method="average")
    elif mismatch == "valid_mask_contents":
        artifact = replace(artifact, valid_mask=np.zeros_like(artifact.valid_mask))

    adapter = (adapters.compute_quantile_spread_value if adapter_name == "spread"
               else adapters.compute_quantile_returns_full_value)
    with pytest.raises(ValueError):
        adapter(batch, labels, min_periods=1, n_quantiles=5,
                daily_quantile_artifact=artifact)


def test_daily_artifact_contract_rejects_mismatched_valid_mask_shape():
    batch, labels = _inputs()
    artifact = _cached_artifact(batch, labels)
    with pytest.raises(ValueError, match="values/counts/valid_mask"):
        replace(artifact, valid_mask=artifact.valid_mask[:, :-1, :])


def test_public_metric_parameters_cannot_inject_daily_quantile_artifact():
    from quant_evaluator.contracts.errors import InvalidContractError
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs()
    artifact = _artifact(batch, labels)
    with pytest.raises(InvalidContractError, match="daily_quantile_artifact"):
        evaluate(
            batch, labels, backend="cpu", metrics=("quantile_spread",),
            metric_parameters={"quantile_spread": {
                "daily_quantile_artifact": artifact,
            }},
        )


@pytest.mark.parametrize("adapter_name", ["spread", "full"])
@pytest.mark.parametrize("input_change", ["factor_values", "factor_validity"])
def test_same_axes_and_ids_do_not_authorize_artifact_for_different_factor_input(
        adapter_name, input_change):
    from dataclasses import replace
    import quant_evaluator.metrics.registry_adapters as adapters

    batch, labels = _inputs()
    artifact = _cached_artifact(batch, labels)
    assert batch.value_hash is None

    if input_change == "factor_values":
        changed = replace(batch, values=batch.values + 0.125)
    else:
        validity = batch.validity.copy()
        validity[0, 0, 0] = ~validity[0, 0, 0]
        changed = replace(batch, validity=validity)
    assert changed.factor_ids == batch.factor_ids
    assert changed.time_axis.values.tolist() == batch.time_axis.values.tolist()
    assert changed.asset_axis.values.tolist() == batch.asset_axis.values.tolist()
    assert changed.value_hash is None
    if input_change == "factor_validity":
        np.testing.assert_array_equal(changed.values, batch.values)
        assert not np.array_equal(changed.validity, batch.validity)
    else:
        np.testing.assert_array_equal(changed.validity, batch.validity)
        assert not np.array_equal(changed.values, batch.values)

    adapter = (adapters.compute_quantile_spread_value if adapter_name == "spread"
               else adapters.compute_quantile_returns_full_value)
    with pytest.raises(ValueError):
        adapter(changed, labels, min_periods=1, n_quantiles=5,
                daily_quantile_artifact=artifact)


def test_cached_artifact_direct_adapter_reuse_matches_recompute_without_extra_kernel(
        monkeypatch):
    import quant_evaluator.metrics.quantile as quantile
    import quant_evaluator.metrics.registry_adapters as adapters

    batch, labels = _inputs()
    artifact = _cached_artifact(batch, labels)
    original = quantile.compute_quantile_returns_fast
    calls = []

    def counted(*args, **kwargs):
        kwargs["use_numba"] = False
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(quantile, "compute_quantile_returns_fast", counted)
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast", counted)
    cached_spread = adapters.compute_quantile_spread_value(
        batch, labels, min_periods=3, n_quantiles=5,
        daily_quantile_artifact=artifact,
    )
    cached_full = adapters.compute_quantile_returns_full_value(
        batch, labels, min_periods=2, n_quantiles=5,
        daily_quantile_artifact=artifact,
    )
    assert calls == []

    recomputed_spread = adapters.compute_quantile_spread_value(
        batch, labels, min_periods=3, n_quantiles=5,
    )
    recomputed_full = adapters.compute_quantile_returns_full_value(
        batch, labels, min_periods=2, n_quantiles=5,
    )
    assert len(calls) == 2
    np.testing.assert_allclose(cached_spread, recomputed_spread,
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(cached_full, recomputed_full,
                               rtol=1e-9, atol=1e-12, equal_nan=True)


def test_public_builder_metadata_and_contents_are_stable_without_request_binding():
    batch, labels = _inputs()
    first = _artifact(batch, labels)
    second = _artifact(batch, labels)

    assert "input_binding" not in first.provenance
    _assert_same_metadata(first.provenance, second.provenance)
    assert first.factor_axis == second.factor_axis
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)


@pytest.mark.parametrize("key_args", [
    (5.5, 10), (True, 10), (5, 10.5), (5, True),
])
def test_cache_rejects_fractional_or_boolean_key_parameters_even_after_warm_hit(key_args):
    from quant_evaluator.runtime.daily_quantile_cache import RequestDailyQuantilePanelCache

    batch, labels = _inputs()
    cache = RequestDailyQuantilePanelCache()
    cache.get_or_build(batch, labels, 5, 10)
    with pytest.raises((TypeError, ValueError)):
        cache.get_or_build(batch, labels, *key_args)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_cache_rejects_invalid_retained_byte_budgets(budget):
    from quant_evaluator.runtime.daily_quantile_cache import RequestDailyQuantilePanelCache

    with pytest.raises((TypeError, ValueError)):
        RequestDailyQuantilePanelCache(max_retained_bytes=budget)


def test_zero_retained_byte_budget_disables_retention_but_still_builds_artifacts():
    from quant_evaluator.runtime.daily_quantile_cache import RequestDailyQuantilePanelCache

    batch, labels = _inputs()
    cache = RequestDailyQuantilePanelCache(max_retained_bytes=0)
    first = cache.get_or_build(batch, labels, 5, 10)
    second = cache.get_or_build(batch, labels, 5, 10)
    assert first is not None and second is not None
    assert first is not second
    assert "input_binding" in first.provenance
