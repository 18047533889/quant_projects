"""Strict, bounded bridge from loader cohorts to research feature receipts.

The production defect this file guards against is treating a multi-factor
cohort or caller-authored source rows as if they were independently bound,
verified single-factor receipts.
"""

import copy
import hashlib
import importlib
import importlib.util

import numpy as np
import pytest

from quant_evaluator.contracts.array_identity import authoritative_array_hash
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch


ADAPTER_MODULE = "factor_assets.adapters.research_cohort_receipts"
RECEIPT_MODULE = "factor_assets.adapters.research_feature_receipt"
CONTRACT_MODULE = "factor_assets.contracts.research_feature_receipt"
EMBEDDING_SPEC = "cohort-fixture-v1"
EMBEDDING_MODEL_VERSION = "fixture-model-1"


def _api():
    try:
        spec = importlib.util.find_spec(ADAPTER_MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "cohort research receipt adapter must be implemented"
    module = importlib.import_module(ADAPTER_MODULE)
    builder = getattr(module, "build_cohort_research_feature_receipts", None)
    assert callable(builder), "cohort research receipt adapter must be callable"
    assert callable(getattr(module, "_build_one", None)), (
        "adapter must expose its one-factor construction seam"
    )
    return module, builder


def _cohort(*, include_validity=True, all_valid=False):
    factor_ids = ("factor-a", "factor-b", "factor-c")
    times = np.array([10, 11], dtype=np.int64)
    assets = np.array(["A", "B", "C"], dtype=object)
    time_axis = AxisRef("time", str(times.dtype), len(times), times)
    asset_axis = AxisRef("asset", str(assets.dtype), len(assets), assets)
    values = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 10
    validity = np.ones((2, 3, 3), dtype=bool)
    if not all_valid:
        validity[1, 2, 1] = False
    batch = FactorBatch(
        factor_ids, time_axis, asset_axis, values,
        validity=validity if include_validity else None,
    )

    manifest_sha256 = hashlib.sha256(b"loader manifest fixture").hexdigest()
    manifest_uri = f"cos://research/{manifest_sha256}/landing_manifest.json"
    sources = []
    for i, factor_id in enumerate(factor_ids):
        source_sha256 = hashlib.sha256(f"source-{factor_id}".encode()).hexdigest()
        sources.append({
            "factor": factor_id,
            "uri": f"cos://research/factors/{source_sha256}.parquet",
            "sha256": source_sha256,
            "manifest_sha256": manifest_sha256,
            "etag": f"fixture-etag-{i}",
            "downloaded_bytes": 4096 + i,
        })
    provenance = {
        "coverage_policy": "strict",
        "manifest_uri": manifest_uri,
        "retained_factor_ids": list(factor_ids),
        "quarantined_factors": [],
        "sources": sources,
    }
    embeddings = {
        "factor-a": (0.1, 0.2),
        "factor-b": (-0.3, 0.4),
        "factor-c": (0.5, -0.6),
    }
    return batch, provenance, embeddings


def _direct_receipt(batch, provenance, factor_id, embedding):
    receipt_module = importlib.import_module(RECEIPT_MODULE)
    contract = importlib.import_module(CONTRACT_MODULE)
    factor_index = batch.factor_ids.index(factor_id)
    child = FactorBatch(
        (factor_id,), batch.time_axis, batch.asset_axis,
        np.ascontiguousarray(batch.values[:, :, factor_index:factor_index + 1]),
        validity=(
            None if batch.validity is None else np.ascontiguousarray(
                batch.validity[:, :, factor_index:factor_index + 1]
            )
        ),
    )
    row = next(source for source in provenance["sources"] if source["factor"] == factor_id)
    binding = contract.ResearchSourceBinding(
        factor_id=factor_id,
        manifest_uri=provenance["manifest_uri"],
        manifest_sha256=row["manifest_sha256"],
        source_uri=row["uri"],
        source_sha256=row["sha256"],
        time_axis_hash=authoritative_array_hash(child.time_axis.values),
        asset_axis_hash=authoritative_array_hash(child.asset_axis.values),
        values_hash=authoritative_array_hash(child.values),
        validity_hash=(
            None if child.validity is None
            else authoritative_array_hash(child.validity)
        ),
    )
    return receipt_module.build_research_feature_receipt(
        child,
        source_binding=binding,
        embedding=embedding,
        embedding_spec=EMBEDDING_SPEC,
        embedding_model_version=EMBEDDING_MODEL_VERSION,
    )


def _build(batch, provenance, embeddings, *, max_slice_bytes=128 * 1024**2):
    _, builder = _api()
    return builder(
        batch,
        provenance,
        embeddings=embeddings,
        embedding_spec=EMBEDDING_SPEC,
        embedding_model_version=EMBEDDING_MODEL_VERSION,
        max_slice_bytes=max_slice_bytes,
    )


def test_cohort_adapter_returns_individually_bound_receipts_in_batch_order():
    batch, provenance, embeddings = _cohort()
    receipts = _build(batch, provenance, embeddings)

    assert tuple(receipt.source_binding.factor_id for receipt in receipts) == batch.factor_ids
    direct = tuple(
        _direct_receipt(batch, provenance, factor_id, embeddings[factor_id])
        for factor_id in batch.factor_ids
    )
    assert receipts == direct
    assert len({receipt.content_hash for receipt in receipts}) == len(batch.factor_ids)
    assert all(receipt.source_binding_status == "caller_supplied" for receipt in receipts)
    assert all(
        receipt.source_binding.values_hash != authoritative_array_hash(batch.values)
        for receipt in receipts
    )


def test_absent_validity_stays_absent_and_differs_from_explicit_all_valid_mask():
    absent_batch, provenance, embeddings = _cohort(include_validity=False)
    absent = _build(absent_batch, provenance, embeddings)

    all_valid_batch, same_provenance, same_embeddings = _cohort(all_valid=True)
    present = _build(all_valid_batch, same_provenance, same_embeddings)

    assert all(receipt.validity_dtype is None for receipt in absent)
    assert all(receipt.validity_dtype == "bool" for receipt in present)
    assert all(
        absent_receipt.content_hash != present_receipt.content_hash
        for absent_receipt, present_receipt in zip(absent, present)
    )
    assert absent == tuple(
        _direct_receipt(absent_batch, provenance, factor_id, embeddings[factor_id])
        for factor_id in absent_batch.factor_ids
    )


def test_strided_parent_buffers_produce_exact_per_factor_receipts():
    batch, provenance, embeddings = _cohort()
    value_store = np.empty((2, 3, 6), dtype=batch.values.dtype)
    value_store[:, :, ::2] = batch.values
    strided_values = value_store[:, :, ::2]
    validity_store = np.empty((2, 3, 6), dtype=bool)
    validity_store[:, :, ::2] = batch.validity
    strided_validity = validity_store[:, :, ::2]
    assert not strided_values.flags.c_contiguous
    assert not strided_validity.flags.c_contiguous
    strided_values.flags.writeable = False
    strided_validity.flags.writeable = False
    object.__setattr__(batch, "values", strided_values)
    object.__setattr__(batch, "validity", strided_validity)

    receipts = _build(batch, provenance, embeddings)
    direct = tuple(
        _direct_receipt(batch, provenance, factor_id, embeddings[factor_id])
        for factor_id in batch.factor_ids
    )
    assert receipts == direct
    assert all(
        receipt.source_binding.values_hash != authoritative_array_hash(batch.values)
        for receipt in receipts
    )


@pytest.mark.parametrize("damage", [
    "coverage_policy", "quarantined", "retained_missing", "retained_duplicate",
    "retained_unknown", "retained_reordered", "source_missing", "source_duplicate",
    "source_unknown", "source_reordered", "manifest_uri_digest",
    "manifest_sha256", "source_sha256", "downloaded_bytes_zero", "embedding_missing",
    "embedding_extra",
])
def test_invalid_cohort_binding_rejected_before_single_factor_construction(monkeypatch, damage):
    adapter, _ = _api()
    batch, provenance, embeddings = _cohort()
    bad = copy.deepcopy(provenance)
    bad_embeddings = dict(embeddings)
    row = bad["sources"][0]
    if damage == "coverage_policy":
        bad["coverage_policy"] = "best_effort"
    elif damage == "quarantined":
        bad["quarantined_factors"] = ["factor-z"]
    elif damage == "retained_missing":
        bad["retained_factor_ids"].pop()
    elif damage == "retained_duplicate":
        bad["retained_factor_ids"][1] = bad["retained_factor_ids"][0]
    elif damage == "retained_unknown":
        bad["retained_factor_ids"][1] = "factor-z"
    elif damage == "retained_reordered":
        bad["retained_factor_ids"].reverse()
    elif damage == "source_missing":
        bad["sources"].pop()
    elif damage == "source_duplicate":
        bad["sources"][1]["factor"] = bad["sources"][0]["factor"]
    elif damage == "source_unknown":
        bad["sources"][1]["factor"] = "factor-z"
    elif damage == "source_reordered":
        bad["sources"].reverse()
    elif damage == "manifest_uri_digest":
        bad["manifest_uri"] = "cos://research/" + "f" * 64 + "/landing_manifest.json"
    elif damage == "manifest_sha256":
        row["manifest_sha256"] = "f" * 64
    elif damage == "source_sha256":
        row["sha256"] = "not-a-sha256"
    elif damage == "downloaded_bytes_zero":
        row["downloaded_bytes"] = 0
    elif damage == "embedding_missing":
        del bad_embeddings["factor-b"]
    else:
        bad_embeddings["factor-z"] = (0.7,)

    calls = []
    monkeypatch.setattr(adapter, "_build_one", lambda *args, **kwargs: calls.append(args))
    with pytest.raises((TypeError, ValueError)):
        _build(batch, bad, bad_embeddings)
    assert calls == []


def test_whitespace_padded_factor_id_rejected_during_preflight(monkeypatch):
    adapter, _ = _api()
    batch, provenance, embeddings = _cohort()
    invalid_ids = ("factor-a", " factor-b ", "factor-c")
    padded = FactorBatch(
        invalid_ids, batch.time_axis, batch.asset_axis, batch.values,
        validity=batch.validity,
    )
    bad = copy.deepcopy(provenance)
    bad["retained_factor_ids"] = list(invalid_ids)
    bad["sources"][1]["factor"] = invalid_ids[1]
    bad_embeddings = dict(embeddings)
    bad_embeddings[invalid_ids[1]] = bad_embeddings.pop("factor-b")
    assert set(bad_embeddings) == set(invalid_ids)

    calls = []

    def forbidden_build_one(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("_build_one ran before identifier validation")

    monkeypatch.setattr(adapter, "_build_one", forbidden_build_one)
    with pytest.raises(ValueError):
        _build(padded, bad, bad_embeddings)
    assert calls == []


def test_infinite_embedding_generator_is_bounded_and_rejected_before_construction(monkeypatch):
    adapter, _ = _api()
    batch, provenance, embeddings = _cohort()
    consumed = 0

    def infinite_embedding():
        nonlocal consumed
        while True:
            consumed += 1
            yield 0.25

    bad_embeddings = dict(embeddings)
    bad_embeddings["factor-a"] = infinite_embedding()
    calls = []
    monkeypatch.setattr(
        adapter, "_build_one",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises((TypeError, ValueError)):
        _build(batch, provenance, bad_embeddings)
    assert consumed <= 4097
    assert calls == []


def test_freeze_copies_parent_slices_and_child_buffers_die_after_receipt_build(monkeypatch):
    import gc
    import weakref

    adapter, _ = _api()
    batch, provenance, embeddings = _cohort()
    freeze_module = importlib.import_module(
        "quant_evaluator.contracts.metric_artifacts"
    )
    original_freeze = freeze_module._freeze_array
    freeze_inputs = []
    child_buffer_refs = []

    def observed_freeze(value, name, **kwargs):
        if name in ("factor values", "factor validity"):
            matches = [
                index for index in range(batch.num_factors)
                if np.shares_memory(
                    value,
                    (batch.values if name == "factor values" else batch.validity)
                    [:, :, index:index + 1],
                )
            ]
            freeze_inputs.append((name, matches))
        frozen = original_freeze(value, name, **kwargs)
        if name in ("factor values", "factor validity"):
            child_buffer_refs.append(weakref.ref(frozen))
        return frozen

    monkeypatch.setattr(freeze_module, "_freeze_array", observed_freeze)
    receipts = _build(batch, provenance, embeddings)
    assert len(receipts) == batch.num_factors
    assert len(freeze_inputs) == 2 * batch.num_factors
    assert all(len(matches) == 1 for _, matches in freeze_inputs)
    assert sorted(matches[0] for _, matches in freeze_inputs) == [
        index for index in range(batch.num_factors)
        for _ in range(2)
    ]
    assert len(child_buffer_refs) == 2 * batch.num_factors
    gc.collect()
    assert all(reference() is None for reference in child_buffer_refs)


def test_slice_budget_is_admitted_before_single_factor_construction(monkeypatch):
    adapter, _ = _api()
    batch, provenance, embeddings = _cohort()
    one_slice_bytes = (
        batch.values[:, :, :1].size * batch.values.dtype.itemsize
        + batch.validity[:, :, :1].size * batch.validity.dtype.itemsize
    )
    assert one_slice_bytes == 54
    calls = []
    monkeypatch.setattr(adapter, "_build_one", lambda *args, **kwargs: calls.append(args))

    with pytest.raises((TypeError, ValueError), match="(?i)(budget|limit|bytes)"):
        _build(batch, provenance, embeddings, max_slice_bytes=one_slice_bytes - 1)
    assert calls == []
