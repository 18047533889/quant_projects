# -*- coding: utf-8
"""intraday minute parity evidence artifact 一致性门禁。

``evidence/intraday_minute_parity.json`` 由 ``scripts/certify_intraday_parity.py``
生成，记录每个分钟算子在哪些后端上通过了真实分钟形状 parity。本测试只验证
artifact 与当前 harness/算子面一致（覆盖完整、无越权声明），不做运行时重算。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("polars")
pytest.importorskip("duckdb")

from cleaned_operators import load_all

load_all()

FE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = FE_ROOT / "evidence/intraday_minute_parity.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_intraday_parity_artifact_covers_all_minute_ops():
    from cleaned_operators.microstructure.intraday_agg import __all__ as minute_ops

    backends = _artifact()["backends"]
    assert set(backends) == set(minute_ops), (
        f"artifact coverage mismatch: missing {set(minute_ops) - set(backends)}"
    )


def test_intraday_parity_artifact_claims_match_harness():
    from tests.backend_parity import intraday_minute_parity as h

    backends = _artifact()["backends"]
    for name, meta in backends.items():
        if meta["status"] == "certified":
            assert name in h._POLARS_OPS and name in h._SQL_OPS
            assert meta["polars"] is True and meta["duckdb_sql"] is True
        elif meta["status"] == "polars_only":
            assert name in h._POLARS_OPS or name in h._POLARS_LIMIT_OPS
            assert meta["polars"] is True
            assert meta["duckdb_sql"] is False
        else:
            assert meta["status"] == "reference_only"
            assert name in h._REFERENCE_ONLY


def test_intraday_parity_artifact_reference_only_is_explicit():
    from tests.backend_parity.intraday_minute_parity import _REFERENCE_ONLY

    backends = _artifact()["backends"]
    ref_only = {n for n, m in backends.items() if m["status"] == "reference_only"}
    assert ref_only == _REFERENCE_ONLY


def test_intraday_parity_artifact_schema():
    data = _artifact()
    assert data["schema_version"] == "intraday_minute_parity.v1"
    assert data["session_tz"] == "Asia/Shanghai"
    assert data["operator_source_hash"]
    assert data["test_file_hash"]
    assert data["helper_file_hash"]
