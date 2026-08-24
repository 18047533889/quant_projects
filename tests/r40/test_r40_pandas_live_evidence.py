# -*- coding: utf-8 -*-
"""R40 #62 测试：PandasBackendLiveEvidence 实跑验证输出语义正确性。"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def test_pandas_backend_live_evidence_validates_actual_behavior():
    from factor_engine.backend.pandas_backend import PandasBackendLiveEvidence

    ev = PandasBackendLiveEvidence()
    summary = ev.validate_known_set()
    assert summary["failed"] == 0
    assert summary["passed"] >= 3
    # 每个已知参考 canonical 都有版本化证据记录。
    for canon, rec in ev.evidence().items():
        assert rec["version_sha"]
        assert rec["passed"] is True, f"{canon}: {rec.get('actual')}"


def test_pandas_backend_live_evidence_detects_wrong_reference():
    from factor_engine.backend.pandas_backend import PandasBackendLiveEvidence, _REFERENCE_CANONICALS

    wrong = dict(_REFERENCE_CANONICALS)
    wrong["abs"] = (wrong["abs"][0], [999.0] * 6)
    ev = PandasBackendLiveEvidence(reference_canonicals=wrong)
    ev.validate_known_set()
    assert ev.evidence()["abs"]["passed"] is False
    assert ev.summary()["failed"] >= 1


def test_pandas_backend_live_evidence_singleton():
    from factor_engine.backend.pandas_backend import (
        get_pandas_live_evidence,
        reset_pandas_live_evidence,
    )

    reset_pandas_live_evidence()
    try:
        a = get_pandas_live_evidence()
        b = get_pandas_live_evidence()
        assert a is b
    finally:
        reset_pandas_live_evidence()
