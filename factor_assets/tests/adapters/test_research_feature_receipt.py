"""Fail-closed tests for research-only, array-bound feature receipts.

This module is intentionally loaded inside the test rather than imported at
collection time: before the adapter exists, RED is an assertion about the
missing API, not a pytest collection/import error.
"""

import importlib
import importlib.util
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.array_identity import authoritative_array_hash


ADAPTER_MODULE = "factor_assets.adapters.research_feature_receipt"
CONTRACT_MODULE = "factor_assets.contracts.research_feature_receipt"


def _api():
    spec = importlib.util.find_spec(ADAPTER_MODULE)
    assert spec is not None, "research receipt builder must be implemented"
    module = importlib.import_module(ADAPTER_MODULE)
    assert callable(getattr(module, "build_research_feature_receipt", None)), (
        "research receipt builder must be callable"
    )
    contract_spec = importlib.util.find_spec(CONTRACT_MODULE)
    assert contract_spec is not None, "research receipt DTO must be implemented"
    contract = importlib.import_module(CONTRACT_MODULE)
    assert callable(getattr(contract, "ResearchSourceBinding", None)), (
        "ResearchSourceBinding DTO must be callable"
    )
    return module.build_research_feature_receipt, contract.ResearchSourceBinding


def _batch(*, values=None, validity=None, times=None, assets=None, factor_id="factor-a"):
    times = np.arange(3, dtype=np.int64) if times is None else np.asarray(times)
    assets = np.asarray(["A", "B", "C", "D"], dtype=object) if assets is None else np.asarray(assets)
    if values is None:
        values = np.arange(12, dtype=np.float64).reshape(3, 4, 1) / 10
    return FactorBatch(
        (factor_id,),
        AxisRef("time", str(times.dtype), len(times), times),
        AxisRef("asset", str(assets.dtype), len(assets), assets),
        np.asarray(values),
        validity=None if validity is None else np.asarray(validity),
    )


def _binding(batch, *, factor_id=None, **overrides):
    _, Binding = _api()
    defaults = {
        "factor_id": batch.factor_ids[0] if factor_id is None else factor_id,
        "manifest_uri": "cos://research/manifest.json",
        "manifest_sha256": hashlib.sha256(b"manifest fixture").hexdigest(),
        "source_uri": "cos://research/factor-a.parquet",
        "source_sha256": hashlib.sha256(b"source fixture").hexdigest(),
        "time_axis_hash": authoritative_array_hash(batch.time_axis.values),
        "asset_axis_hash": authoritative_array_hash(batch.asset_axis.values),
        "values_hash": authoritative_array_hash(batch.values),
        "validity_hash": None if batch.validity is None else authoritative_array_hash(batch.validity),
    }
    defaults.update(overrides)
    return Binding(**defaults)


def _receipt(
    batch, *, binding=None, embedding=(0.125, -0.25, 0.5),
    embedding_spec="research-embedding-v1",
    embedding_model_version="fixture-model-1", **kwargs,
):
    build, _ = _api()
    options = {
        "embedding_spec": embedding_spec,
        "embedding_model_version": embedding_model_version,
    }
    options.update(kwargs)
    return build(
        batch,
        source_binding=_binding(batch) if binding is None else binding,
        embedding=embedding,
        **options,
    )


def test_builder_is_available_and_receipt_is_research_only_payload_free_and_immutable():
    batch = _batch()
    receipt = _receipt(batch)
    assert receipt.scope == "research_only"
    assert receipt.schema_version == 1
    assert receipt.source_binding_status == "caller_supplied"
    assert receipt.source_binding == _binding(batch)
    assert receipt.values_dtype == str(batch.values.dtype)
    assert receipt.validity_dtype is None
    assert receipt.shape == batch.values.shape
    assert receipt.time_axis_name == "time"
    assert receipt.time_axis_dtype == str(batch.time_axis.values.dtype)
    assert receipt.asset_axis_name == "asset"
    assert receipt.asset_axis_dtype == str(batch.asset_axis.values.dtype)
    assert receipt.embedding == (0.125, -0.25, 0.5)
    assert receipt.embedding_spec == "research-embedding-v1"
    assert receipt.embedding_model_version == "fixture-model-1"
    assert isinstance(receipt.content_hash, str) and receipt.content_hash
    with pytest.raises((AttributeError, TypeError)):
        receipt.scope = "production"
    rendered = repr(receipt)
    assert repr(batch.values) not in rendered and "array(" not in rendered
    assert not any(isinstance(value, np.ndarray) for value in vars(receipt).values())


def test_independent_equal_batches_have_equal_content_hash_and_fresh_allocation():
    original = _batch()
    first = _receipt(original)
    second_batch = _batch(values=np.array(original.values, copy=True))
    second = _receipt(second_batch)
    assert first is not second
    assert first.content_hash == second.content_hash


@pytest.mark.parametrize("field,value", [
    ("factor_id", "factor-b"),
    ("time_axis_hash", hashlib.sha256(b"wrong time").hexdigest()),
    ("asset_axis_hash", hashlib.sha256(b"wrong assets").hexdigest()),
    ("values_hash", hashlib.sha256(b"wrong values").hexdigest()),
    ("validity_hash", hashlib.sha256(b"wrong validity").hexdigest()),
])
def test_modified_source_binding_fields_fail_closed(field, value):
    batch = _batch(validity=np.ones((3, 4, 1), dtype=bool))
    binding = _binding(batch, **{field: value})
    with pytest.raises((TypeError, ValueError)):
        _receipt(batch, binding=binding)


def test_absent_vs_explicit_all_valid_mask_are_distinct_and_exactly_bound():
    absent = _batch()
    present = _batch(validity=np.ones((3, 4, 1), dtype=bool))
    absent_receipt = _receipt(absent)
    present_receipt = _receipt(present)
    assert absent_receipt.content_hash != present_receipt.content_hash
    assert present_receipt.validity_dtype == "bool"
    wrong_mask_binding = _binding(present, validity_hash=None)
    with pytest.raises((TypeError, ValueError)):
        _receipt(present, binding=wrong_mask_binding)


def test_caller_supplied_source_fields_are_recorded_but_not_claimed_verified():
    batch = _batch()
    binding = _binding(batch)
    original = _receipt(batch, binding=binding)
    altered = _receipt(batch, binding=replace(
        binding,
        manifest_uri="cos://research/alternate-manifest.json",
        manifest_sha256=hashlib.sha256(b"alternate manifest").hexdigest(),
        source_uri="cos://research/alternate-factor.parquet",
        source_sha256=hashlib.sha256(b"alternate source").hexdigest(),
    ))
    assert original.source_binding_status == "caller_supplied"
    assert altered.source_binding_status == "caller_supplied"
    assert altered.content_hash != original.content_hash


def test_builder_rejects_non_single_factor_and_non_qe_factorbatch():
    build, _ = _api()
    one = _batch()
    multi = FactorBatch(
        ("factor-a", "factor-b"), one.time_axis, one.asset_axis,
        np.concatenate([one.values, one.values], axis=2),
    )
    with pytest.raises((TypeError, ValueError)):
        _receipt(multi)
    with pytest.raises(TypeError):
        build(
            object(), source_binding=_binding(one), embedding=(0.1,),
            embedding_spec="research-embedding-v1", embedding_model_version="fixture-model-1",
        )


def test_changed_dtype_values_validity_and_coordinates_do_not_match_binding():
    base = _batch(validity=np.ones((3, 4, 1), dtype=bool))
    binding = _binding(base)
    changed_values = np.array(base.values, dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        _receipt(_batch(values=changed_values, validity=base.validity), binding=binding)
    changed_mask = np.array(base.validity, copy=True)
    changed_mask[0, 0, 0] = False
    with pytest.raises((TypeError, ValueError)):
        _receipt(_batch(validity=changed_mask), binding=binding)
    changed_times = np.array([10, 11, 13], dtype=np.int64)
    with pytest.raises((TypeError, ValueError)):
        _receipt(_batch(times=changed_times, validity=base.validity), binding=binding)
    changed_assets = np.asarray(["A", "B", "C", "E"], dtype=object)
    with pytest.raises((TypeError, ValueError)):
        _receipt(_batch(assets=changed_assets, validity=base.validity), binding=binding)


def test_unsupported_structured_coordinate_encoding_fails_closed():
    structured_assets = np.array([(i, i + 10) for i in range(4)], dtype=[("id", "i4"), ("venue", "i4")])
    batch = _batch(assets=structured_assets)
    _, Binding = _api()
    binding = Binding(
        factor_id="factor-a", manifest_uri="cos://research/manifest.json",
        manifest_sha256=hashlib.sha256(b"manifest fixture").hexdigest(),
        source_uri="cos://research/factor-a.parquet",
        source_sha256=hashlib.sha256(b"source fixture").hexdigest(),
        time_axis_hash=authoritative_array_hash(batch.time_axis.values),
        asset_axis_hash=hashlib.sha256(b"unsupported structured axis").hexdigest(),
        values_hash=authoritative_array_hash(batch.values), validity_hash=None,
    )
    with pytest.raises((TypeError, ValueError)):
        _receipt(batch, binding=binding)


def test_decimal_object_coordinates_use_qe_lossless_array_codec():
    from decimal import Decimal

    batch = _batch(assets=np.asarray([Decimal(i) for i in range(4)], dtype=object))
    receipt = _receipt(batch)
    assert receipt.source_binding.asset_axis_hash == authoritative_array_hash(batch.asset_axis.values)


@pytest.mark.parametrize("embedding", [(), (float("nan"),), (float("inf"),), (1.0,) * 4097])
def test_embedding_must_be_finite_nonempty_and_bounded(embedding):
    batch = _batch()
    with pytest.raises((TypeError, ValueError)):
        _receipt(batch, embedding=embedding)


def test_binding_hashes_are_authoritative_array_hashes_not_byte_or_value_hash_claims():
    batch = _batch(validity=np.ones((3, 4, 1), dtype=bool))
    binding = _binding(batch)
    assert binding.values_hash == authoritative_array_hash(batch.values)
    assert binding.validity_hash == authoritative_array_hash(batch.validity)
    assert binding.time_axis_hash == authoritative_array_hash(batch.time_axis.values)
    assert binding.asset_axis_hash == authoritative_array_hash(batch.asset_axis.values)
    receipt = _receipt(batch, binding=binding)
    assert receipt.source_binding.source_sha256 == binding.source_sha256
    assert receipt.source_binding.manifest_sha256 == binding.manifest_sha256


def test_numpy_real_embedding_scalars_normalize_and_bool_or_complex_are_rejected():
    batch = _batch()
    receipt = _receipt(batch, embedding=(np.float32(0.125), np.float64(-0.25), np.int64(2)))
    assert receipt.embedding == (0.125, -0.25, 2.0)
    assert all(type(value) is float for value in receipt.embedding)
    for bad in (np.bool_(True), np.complex64(1 + 2j)):
        with pytest.raises((TypeError, ValueError)):
            _receipt(batch, embedding=(bad,))


@pytest.mark.parametrize("field,changed", [
    ("embedding_spec", "research-embedding-v2"),
    ("embedding_model_version", "fixture-model-2"),
])
def test_embedding_spec_and_model_version_are_bound_into_content_hash(field, changed):
    batch = _batch()
    baseline = _receipt(batch)
    alternate = _receipt(batch, **{field: changed})
    assert alternate.content_hash != baseline.content_hash


@pytest.mark.parametrize("with_mask", [False, True])
def test_nan_and_infinity_factor_cells_are_preserved_and_exactly_hash_bound(with_mask):
    values = np.asarray([[[np.nan], [np.inf]], [[-np.inf], [1.0]]], dtype=np.float64)
    validity = np.asarray([[[True], [False]], [[True], [True]]], dtype=bool) if with_mask else None
    batch = _batch(values=values, validity=validity, times=np.asarray([0, 1]), assets=np.asarray(["A", "B"], dtype=object))
    receipt = _receipt(batch)
    assert receipt.source_binding.values_hash == authoritative_array_hash(values)
    assert receipt.validity_dtype == ("bool" if with_mask else None)


def test_research_receipt_is_not_accepted_as_production_fingerprint_input():
    from factor_assets.adapters.fingerprint import build_similarity_fingerprint

    class Model:
        version = "fixture-model-1"

        def embed(self, values, validity_mask):
            return (0.1,)

    receipt = _receipt(_batch())
    with pytest.raises(TypeError, match="feature_bundle lacks required preprocessing evidence"):
        build_similarity_fingerprint(
            receipt, object(), embedding_model=Model(),
            embedding_spec="research-embedding-v1", mask_policy="none",
            direction="signed", aggregation_method="mean",
            embedding_model_version="fixture-model-1", window="research-window",
        )


def test_dto_rejects_malformed_sha256_and_production_scope_claims():
    batch = _batch()
    _, Binding = _api()
    with pytest.raises(ValueError, match="SHA-256"):
        _binding(batch, source_sha256="not-a-digest")

    receipt = _receipt(batch)
    with pytest.raises(ValueError, match="scope"):
        replace(receipt, scope="production")
