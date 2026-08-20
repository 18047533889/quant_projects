# -*- coding: utf-8 -*-
"""R14 #2 factor_matrix semantic version gate 自动绑定测试。

旧实现 ``engine.materialize_matrix()`` / ``matrix_service`` 把调用方手工传入的
``factor_versions``（缺省 None）原样转发给 materializer，production 主链**不自动
绑定 semantic digest**——普通调用可能完全没有 factor-version gate，把不同语义的
因子混进同一 matrix。

R14 #2 改动：
  * matrix 声明的 ``universe/frequency`` 必须先与每个 factor 的实际执行 scope
    （``_scope_from_factor``，优先 ``factor.semantic_identity``）一致，否则拒绝；
  * 按真实执行的 factor scope + analysis + source contract 自动算 ``semantic_digest``
    （与 ``execute_materialize`` 同源）；production 下任何 factor 缺 version
    直接拒绝，调用方手工版本与自动算出的不一致也拒绝。

本文件尽量用 tmp_path + InMemorySeriesSource 隔离，不污染真实 factor_matrix。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime import matrix_service
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {"close": pd.Series(np.arange(6, dtype=float) + 1.0, index=idx)}


def _matrix_base(tmp_path):
    return tmp_path / "matrix" / "universe=test_u" / "freq=1d"


# ---------------------------------------------------------------------------
# scope 校验：matrix 声明与 factor 实际执行 scope 一致
# ---------------------------------------------------------------------------


def test_r14_frequency_mismatch_rejected(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f = Factor(name="a", expr=rank(col("close")), freq="5m")
    with pytest.raises(ValueError, match="frequency"):
        eng.materialize_matrix(
            [f], universe="test_u", frequency="1d", matrix_root=tmp_path / "matrix"
        )


def test_r14_universe_mismatch_rejected(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f = Factor(name="a", expr=rank(col("close")), universe="CSI500")
    with pytest.raises(ValueError, match="universe"):
        eng.materialize_matrix(
            [f], universe="CSI300", frequency="1d", matrix_root=tmp_path / "matrix"
        )


def test_r14_scope_match_passes(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f = Factor(name="a", expr=rank(col("close")))  # freq 缺省 1d / universe 缺省 ALL
    summary = eng.materialize_matrix(
        [f], universe="test_u", frequency="1d", matrix_root=tmp_path / "matrix"
    )
    assert summary["rows_written"] > 0


# ---------------------------------------------------------------------------
# 自动绑定 semantic digest（research 也绑定；production fail-closed）
# ---------------------------------------------------------------------------


def test_r14_research_auto_binds_versions(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f1 = Factor(name="a", expr=rank(col("close")))
    f2 = Factor(name="b", expr=ts_mean(col("close"), 2))
    eng.materialize_matrix(
        [f1, f2],
        factor_ids=["alpha_001", "alpha_002"],
        universe="test_u",
        matrix_root=tmp_path / "matrix",
    )
    manifest = json.loads((_matrix_base(tmp_path) / "manifest.json").read_text())
    # research 下也自动绑定了 digest（不再只有空 {}）
    assert "alpha_001" in manifest["factors"]
    assert "alpha_002" in manifest["factors"]
    d1 = manifest["factors"]["alpha_001"]["semantic_digest"]
    d2 = manifest["factors"]["alpha_002"]["semantic_digest"]
    assert isinstance(d1, str) and len(d1) == 16
    assert d1 != d2  # 不同公式 → 不同 digest


def test_r14_semantic_identity_frequency_drives_digest(tmp_path):
    """factor.semantic_identity.frequency 是执行权威：同公式 1d vs 5m digest 不同。

    R14 #2 + #3：matrix 自动绑定必须消费 ``_scope_from_factor`` 的 canonical scope
    （优先 semantic_identity），否则会绑到 factor.freq 的旧值上。
    """
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    from runtime.factor_identity import FactorSemanticIdentity

    # 语义身份频率 5m（模拟 rebuild 场景：执行按 5m，但 factor.freq=1d）。
    # ``_scope_from_factor`` 必须优先 semantic_identity.frequency → scope=5m →
    # 与 matrix 声明的 1d 冲突。
    f2 = Factor(
        name="b",
        expr=rank(col("close")),
        freq="1d",
        semantic_identity=FactorSemanticIdentity(
            ir_hash="h" * 64,
            operator_contract_hash="o" * 64,
            field_contract_hash="f" * 64,
            source_contract_hash="s" * 64,
            source_dependency_hash="d" * 64,
            frequency="5m",
        ),
    )
    with pytest.raises(ValueError, match="frequency"):
        eng.materialize_matrix(
            [f2], universe="test_u", frequency="1d", matrix_root=tmp_path / "matrix"
        )


def test_r14_production_missing_digest_rejected(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    eng.run_mode = "production"
    f1 = Factor(name="a", expr=rank(col("close")))
    stub_out = {"results": {f1.name: _data()["close"]}, "analyses": {}}
    with mock.patch.object(eng, "run_many", return_value=stub_out):
        with pytest.raises(RuntimeError, match="拒绝无版本发布"):
            eng.materialize_matrix(
                [f1], universe="test_u", frequency="1d", matrix_root=tmp_path / "matrix"
            )


def test_r14_production_conflicting_caller_version_rejected(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    eng.run_mode = "production"
    f1 = Factor(name="a", expr=rank(col("close")))
    stub_out = {
        "results": {f1.name: _data()["close"]},
        "analyses": {f1.name: SimpleNamespace(ir=None)},
    }
    with mock.patch.object(eng, "run_many", return_value=stub_out), mock.patch.object(
        matrix_service, "_matrix_factor_digest", return_value="auto-digest-16"
    ):
        with pytest.raises(ValueError, match="不一致"):
            eng.materialize_matrix(
                [f1],
                universe="test_u",
                frequency="1d",
                matrix_root=tmp_path / "matrix",
                factor_versions={f1.name: "manual-digest-16"},
            )


def test_r14_production_auto_bind_passes(tmp_path):
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    eng.run_mode = "production"
    f1 = Factor(name="a", expr=rank(col("close")))
    stub_out = {
        "results": {f1.name: _data()["close"]},
        "analyses": {f1.name: SimpleNamespace(ir=None)},
    }
    with mock.patch.object(eng, "run_many", return_value=stub_out), mock.patch.object(
        matrix_service, "_matrix_factor_digest", return_value="auto-digest-16"
    ):
        summary = eng.materialize_matrix(
            [f1], universe="test_u", frequency="1d", matrix_root=tmp_path / "matrix"
        )
    manifest = json.loads((_matrix_base(tmp_path) / "manifest.json").read_text())
    assert manifest["factors"]["a"]["semantic_digest"] == "auto-digest-16"
    assert summary["manifest_version"] == 1
