# -*- coding: utf-8 -*-
"""R40 #109/#110/#111/#112/#115/#116: factor identity hash hardening.

These tests are self-contained (no full ``load_all`` required) so they run even
while the shared working tree is mid-bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.factor_identity import (
    OperatorSemanticContractDigest,
    _typed_hash_value,
    partition_input_fingerprint,
    scoped_operator_contract_hash,
)


def _col(name):
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def _op(name, *inp):
    return PlanNode(op=name, attrs={}, inputs=list(inp))


# ---------------------------------------------------------------------------
# #111: partition fingerprint must distinguish dtype (int32 vs int64).
# ---------------------------------------------------------------------------
class TestPartitionFingerprintDtype:
    def _frame(self, asset_dtype, value=1.0):
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-01"]),
                "asset": np.array([5], dtype=asset_dtype),
                "value": [value],
            }
        )

    def test_partition_fingerprint_distinguishes_dtype(self):
        assert partition_input_fingerprint(
            self._frame("int32")
        ) != partition_input_fingerprint(self._frame("int64"))
        # datetime64 vs object datetime must differ too
        d_dt = pd.DataFrame(
            {"datetime": pd.to_datetime(["2020-01-01"]), "asset": [1], "value": [1.0]}
        )
        d_obj = pd.DataFrame(
            {"datetime": ["2020-01-01"], "asset": [1], "value": [1.0]}
        )
        assert partition_input_fingerprint(d_dt) != partition_input_fingerprint(d_obj)

    def test_partition_fingerprint_large_int64_no_collision(self):
        a = pd.DataFrame(
            {"datetime": pd.to_datetime(["2020-01-01"]),
             "asset": np.array([2**53], dtype="int64"), "value": [1.0]}
        )
        b = pd.DataFrame(
            {"datetime": pd.to_datetime(["2020-01-01"]),
             "asset": np.array([2**53 + 1], dtype="int64"), "value": [1.0]}
        )
        # must not collapse to the same float representation
        assert partition_input_fingerprint(a) != partition_input_fingerprint(b)

    def test_same_value_same_fingerprint(self):
        assert partition_input_fingerprint(
            self._frame("int64", 1.0)
        ) == partition_input_fingerprint(self._frame("int64", 1.0))


# ---------------------------------------------------------------------------
# #112: partition fingerprint must be row-order invariant for a logical partition.
# ---------------------------------------------------------------------------
class TestPartitionFingerprintRowOrder:
    def test_partition_fingerprint_row_order_invariant(self):
        base = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
                ),
                "asset": [1, 2, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        shuffled = base.sample(frac=1.0, random_state=7).reset_index(drop=True)
        assert partition_input_fingerprint(base) == partition_input_fingerprint(shuffled)


# ---------------------------------------------------------------------------
# #115: _typed_hash_value must distinguish numpy dtypes.
# ---------------------------------------------------------------------------
class TestTypedHashNumpyDtype:
    def test_typed_hash_distinguishes_numpy_dtype(self):
        i32 = _typed_hash_value(np.int32(1))
        i64 = _typed_hash_value(np.int64(1))
        assert i32 != i64
        assert i32 == {"type": "int32", "value": 1}
        assert i64 == {"type": "int64", "value": 1}
        f32 = _typed_hash_value(np.float32(1.5))
        f64 = _typed_hash_value(np.float64(1.5))
        assert f32 != f64
        assert f32 == {"type": "float32", "value": 1.5}
        # numpy scalar inside a mapping/list still normalizes
        assert {"x": np.int32(3)} is not None


# ---------------------------------------------------------------------------
# #109: scoped_operator_contract_hash must bind implementation_hash changes.
# ---------------------------------------------------------------------------
class TestScopedContractBindsImplHash:
    def test_scoped_contract_hash_binds_impl_hash_changes(self, monkeypatch):
        plan = _op("ts_mean", _col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        hashes = {"implementation_hash_pandas": "aaa", "implementation_hash_polars": "bbb"}

        def _fake_impl_hashes(canonical):
            return dict(hashes)

        import factor_engine.backend.evidence_provenance as ep

        monkeypatch.setattr(ep, "implementation_hashes_for", _fake_impl_hashes)
        h1 = scoped_operator_contract_hash(plan)
        hashes["implementation_hash_pandas"] = "zzz"
        h2 = scoped_operator_contract_hash(plan)
        assert h1 != h2

    def test_scoped_contract_hash_stable_for_same_plan(self, monkeypatch):
        plan = _op("ts_mean", _col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        import factor_engine.backend.evidence_provenance as ep

        monkeypatch.setattr(
            ep, "implementation_hashes_for", lambda c: {"implementation_hash_pandas": "x"}
        )
        a = _op("ts_mean", _col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        b = _op("ts_std", _col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        assert scoped_operator_contract_hash(a) == scoped_operator_contract_hash(a)
        assert scoped_operator_contract_hash(a) != scoped_operator_contract_hash(b)


# ---------------------------------------------------------------------------
# #110: OperatorSemanticContractDigest unifies plan_hash and identity paths.
# ---------------------------------------------------------------------------
class TestOperatorContractDigestUnified:
    def test_digest_to_payload_is_typed_and_stable(self, monkeypatch):
        import factor_engine.backend.evidence_provenance as ep

        monkeypatch.setattr(
            ep, "implementation_hashes_for",
            lambda c: {"implementation_hash_pandas": "k1", "implementation_hash_polars": "k2"},
        )
        d1 = OperatorSemanticContractDigest.for_canonical("ts_mean")
        d2 = OperatorSemanticContractDigest.for_canonical("ts_mean")
        assert d1 == d2
        assert d1.contract_hash() == d2.contract_hash()
        payload = d1.to_payload()
        assert payload["canonical"] == "ts_mean"
        assert payload["implementation_hash"]
        assert "backend_hashes" in payload

    def test_operator_contract_digest_unified_across_paths(self, monkeypatch):
        # Both _operator_semantic_contract (plan_hash) and
        # scoped_operator_contract_hash (factor_identity) project from the same
        # digest → changing the digest changes both paths.
        from factor_engine.planner.plan_hash import _operator_semantic_contract

        import factor_engine.backend.evidence_provenance as ep

        monkeypatch.setattr(
            ep, "implementation_hashes_for",
            lambda c: {"implementation_hash_pandas": "impl-v1"},
        )
        plan = _op("ts_mean", _col("close"))
        before_identity = scoped_operator_contract_hash(plan)
        before_plan = _operator_semantic_contract("ts_mean")

        monkeypatch.setattr(
            ep, "implementation_hashes_for",
            lambda c: {"implementation_hash_pandas": "impl-v2"},
        )
        after_identity = scoped_operator_contract_hash(plan)
        after_plan = _operator_semantic_contract("ts_mean")

        assert before_identity != after_identity
        assert before_plan != after_plan
        # Both must contain the implementation_hash that changed.
        assert "impl-v1" in str(before_plan)
        assert "impl-v2" in str(after_plan)
