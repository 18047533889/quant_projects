"""Contracts for canonical evaluation identity fields and residency probes."""
from types import SimpleNamespace

import numpy as np

from quant_evaluator.contracts._array_hash_cache import ArrayHashStateCache
from quant_evaluator.contracts.array_identity import authoritative_array_hash
from quant_evaluator.contracts._hashutil import stable_content_hex_streamed_arrays
from quant_evaluator.runtime.evaluation_identity import (
    build_evaluation_config_hash_fields,
    evaluation_identity_cache_ready,
)


def _readonly(values):
    source = np.asarray(values, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _fields(batch, labels):
    return build_evaluation_config_hash_fields(
        batch, labels, ("coverage",), metric_versions={"coverage": "test.v1"},
        metric_parameters={}, context=None, quantile_builder_parameters={},
        portfolio_returns=None, holding_returns=None, portfolio_spec=None,
        trade_eligibility=None, calendar_snapshot=None, exposure_panel=None,
        generalization_evidence=None, split_ref=None, request_fields={},
    )


def test_builder_preserves_evaluation_config_v2_hash_bytes():
    values = _readonly(np.arange(24).reshape(4, 3, 2))
    batch = SimpleNamespace(factor_ids=("f0", "f1"), values=values, validity=None)
    labels = SimpleNamespace(content_hash="label-content-hash")
    actual = _fields(batch, labels)
    reference = {
        "metrics": ("coverage",), "versions": {"coverage": "test.v1"},
        "parameters": {}, "label_hash": "label-content-hash",
        "factor_ids": ("f0", "f1"), "factor_values": values,
        "factor_validity": None, "context": None,
        "quantile_builder_parameters": {}, "calendar_snapshot": None,
        "portfolio_returns": None, "holding_returns": None,
        "portfolio_spec": None, "trade_eligibility": None,
        "exposure_panel": None, "generalization_evidence": None,
        "split_ref": None,
    }
    assert stable_content_hex_streamed_arrays(
        tag="EvaluationConfig.v2", fields=actual,
        array_keys=("factor_values", "factor_validity"),
    ) == stable_content_hex_streamed_arrays(
        tag="EvaluationConfig.v2", fields=reference,
        array_keys=("factor_values", "factor_validity"),
    )


def test_residency_probe_requires_raw_and_json_states_without_promoting(monkeypatch):
    from quant_evaluator.contracts import array_identity
    from quant_evaluator.contracts import _hashutil as hashutil

    raw_cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    json_cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", raw_cache)
    monkeypatch.setattr(hashutil, "_ARRAY_HASH_STATE_CACHE", json_cache)
    array = _readonly(np.arange(16))
    fields = {"factor_values": array, "factor_validity": None,
              "label_hash": "precomputed-label-hash"}
    probe = dict(raw_arrays=(array,), fields=fields,
                 tag="EvaluationConfig.v2",
                 array_keys=("factor_values", "factor_validity"))

    assert evaluation_identity_cache_ready(**probe) is False
    authoritative_array_hash(array)
    raw_order = tuple(raw_cache._entries)
    assert evaluation_identity_cache_ready(**probe) is False
    assert tuple(raw_cache._entries) == raw_order

    stable_content_hex_streamed_arrays(**{
        "tag": probe["tag"], "fields": fields,
        "array_keys": probe["array_keys"],
    })
    json_order = tuple(json_cache._entries)
    raw_order = tuple(raw_cache._entries)
    assert evaluation_identity_cache_ready(**probe) is True
    assert tuple(raw_cache._entries) == raw_order
    assert tuple(json_cache._entries) == json_order

    with raw_cache._lock:
        raw_cache._entries.clear()
    assert evaluation_identity_cache_ready(**probe) is False
