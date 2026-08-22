# -*- coding: utf-8 -*-
"""R9-P0-035 / §55 regression tests: every SEARCHABLE parameter needs three
behavior-based proofs — feasibility_verified, sensitivity_verified and
non_equivalence_verified — produced by ``audit_searchable_params``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.operator_audits import (
    _output_signature,
    audit_searchable_params,
)


def _panel(n: int = 80, cols: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, (n, cols)), axis=0),
        index=idx, columns=[chr(ord("A") + i) for i in range(cols)],
    )


def test_output_signature_has_three_levels():
    # Two vectors with identical mean/std/count but different value patterns
    # must NOT collide (R9-P1-039 multi-level signature).
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 1.0, 2.0, 2.0, 6.0, 6.0])
    assert _output_signature(a) != _output_signature(b)
    # And a NaN-mask difference is caught too.
    c = a.copy()
    c[3] = np.nan
    assert _output_signature(a) != _output_signature(c)
    # Same data -> identical signature.
    assert _output_signature(a) == _output_signature(a.copy())


def test_audit_searchable_params_produces_three_proofs():
    """A real searchable-param operator (ts_qn_scale window) yields
    feasibility / sensitivity / non-equivalence proofs, each pass."""
    import cleaned_operators  # noqa: F401
    cleaned_operators.load_all()
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_qn_scale")
    assert op is not None
    res = audit_searchable_params(op, _panel())
    metrics = {r.metric for r in res}
    assert "window.feasibility_verified" in metrics
    assert "window.sensitivity_verified" in metrics
    assert "window.non_equivalence_verified" in metrics
    for r in res:
        if "window." in r.metric:
            assert r.status == "pass", f"{r.metric}: {r.status} ({r.observed})"
            assert r.expected != r.observed or r.status == "pass"


def test_audit_searchable_params_result_shape():
    """Every audit result is a structured AuditResult with the 6 fields."""
    import cleaned_operators  # noqa: F401
    cleaned_operators.load_all()
    from cleaned_operators.operator_audits import AuditResult
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_ema")
    assert op is not None
    res = audit_searchable_params(op, _panel())
    assert isinstance(res, list) and res
    for r in res:
        assert isinstance(r, AuditResult)
        assert r.status in {"pass", "fail", "info"}
        assert r.test_name == "audit_searchable_params"
        assert r.canonical
        assert r.metric
