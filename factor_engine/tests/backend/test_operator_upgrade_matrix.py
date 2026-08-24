# -*- coding: utf-8
"""升级矩阵 freshness 与 Batch 分层不变量。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.cleaned_operators import load_all

    load_all()


def test_operator_upgrade_matrix_yaml_fresh():
    script = FE_ROOT / "scripts" / "export_operator_upgrade_matrix.py"
    out = FE_ROOT.parent / "evidence" / "operator_upgrade_matrix.yaml"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=str(FE_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_batch_a_pending_has_block_reason():
    from factor_engine.backend.operator_upgrade_matrix import build_operator_upgrade_row, infer_upgrade_batch
    from tests.backend_parity.operator_case_registry import batch_a_pending

    pending = batch_a_pending()
    assert len(pending) > 0
    for canon in pending[:5]:
        row = build_operator_upgrade_row(canon)
        assert infer_upgrade_batch(canon) == "A"
        assert row.block_reason is not None
        assert row.certification_gaps


def test_ts_argmax_is_python_rolling_not_native():
    from factor_engine.backend.operator_upgrade_matrix import build_operator_upgrade_row
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE, POLARS_LONG_PYTHON_ROLLING

    for op in ("ts_argmax", "ts_argmin"):
        assert op in POLARS_LONG_PYTHON_ROLLING
        assert op not in POLARS_LONG_NATIVE
        row = build_operator_upgrade_row(op)
        assert row.production_status == "python_rolling"
        assert row.upgrade_batch == "C"


def test_composite_requires_lowered_primitives_certified():
    from factor_engine.backend.composite_evidence import composite_production_safe
    from factor_engine.backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from factor_engine.planner.composite_lowering import lowered_primitives

    for name in ("MOM", "ROC", "OBV"):
        prims = lowered_primitives(name)
        assert prims
        if composite_production_safe(name):
            assert all(p in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE for p in prims)
