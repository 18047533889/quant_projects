# -*- coding: utf-8 -*-
"""Model-audit remediation M-190 / M-192 / M-233 / M-240 for path-signature.

M-190 — ``ts_signature_mahalanobis_anomaly`` (research_transform.py) is
PRIOR_REFERENCE_CURRENT_QUERY: the current row builds the QUERY signature while
the strided prior history rows form the REFERENCE distribution.  Verified, no
code change needed.

M-192 — ``ts_path_signature_depth2_norm`` is FIXED depth 2 (no ``depth`` search
parameter on the surface; the kernel computes the level-2 signature norm).
Verified, no code change needed.

M-233 — the three path-signature operators document that the current row must
be the path endpoint (current row participates in the signature; history is a
trailing window) — SELF_FIT_DESCRIPTIVE timing.

M-240 — ``ts_path_leadlag_area`` lag is strictly validated via
``strict_int(lag, "lag", lower=1)``: bool / non-integer / NaN / Inf / ``lag<1``
raise instead of the old silent ``max(1, int(lag))`` clamp.  Valid values keep
identical behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Importing the modules registers their canonicals directly (no load_all needed,
# so the tests are robust to concurrent-session load blockers elsewhere).
import factor_engine.cleaned_operators.research_transform  # noqa: E402, F401  (mahalanobis anomaly)
import factor_engine.cleaned_operators.ts_model.path_signature as ps  # noqa: E402  (3 canonicals)
from factor_engine.cleaned_operators.model_timing import TimingKind, timing_kind_for  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _get(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    assert op is not None, name
    return op


def _xypanels(n: int = 40):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = pd.DataFrame({"C0": np.arange(n, dtype=float)}, index=idx)
    y = pd.DataFrame({"C0": np.arange(n, dtype=float)[::-1]}, index=idx)
    return x, y


def _fin(x: pd.DataFrame) -> int:
    return int(np.isfinite(x.to_numpy(dtype=float)).sum())


# ---------------------------------------------------------------------------
# M-240 — strict lag validation (no silent clamp)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.5, np.nan, np.inf, "3", None])
def test_leadlag_rejects_silent_clamp_values(bad):
    """M-240: bool / non-positive / fractional / non-finite / string lag must
    raise — the old ``max(1, int(lag))`` quietly clamped True/0/-3 to 1."""
    x, y = _xypanels()
    op = _get("ts_path_leadlag_area")
    with pytest.raises((ValueError, TypeError)):
        op.calculate(x, y, window=20, lag=bad)


def test_leadlag_lag_zero_no_longer_clamps_to_one():
    """M-240 core regression: ``lag=0`` used to be silently clamped to 1 by
    ``max(1, int(lag))``; it is now a contract violation."""
    x, y = _xypanels()
    op = _get("ts_path_leadlag_area")
    with pytest.raises(ValueError):
        op.calculate(x, y, window=20, lag=0)


def test_leadlag_valid_lag_behavior_preserved():
    """M-240: valid lags still work and an integral float (2.0) is accepted
    value-equal to 2 — identical output to before."""
    x, y = _xypanels()
    op = _get("ts_path_leadlag_area")
    out1 = op.calculate(x, y, window=20, lag=1)
    out2 = op.calculate(x, y, window=20, lag=2)
    out2f = op.calculate(x, y, window=20, lag=2.0)
    assert _fin(out1) > 0
    assert _fin(out2) > 0
    assert np.allclose(out2.to_numpy(), out2f.to_numpy(), equal_nan=True)


def test_leadlag_kernel_uses_strict_int_not_max_clamp():
    """M-240: the kernel source must use the strict-int gate and no longer
    contain the silent ``l = max(1, int(lag))`` clamp as executable code."""
    src = Path(ps.__file__).read_text(encoding="utf-8")
    assert "strict_int(lag, \"lag\", lower=1)" in src
    assert "l = max(1, int(lag))" not in src
    # direct kernel call path accepts valid lag and rejects invalid
    a = np.arange(30.0)
    b = np.arange(30.0)[::-1]
    area = ps._leadlag_area(a, b, 60, 1)
    assert area == area  # not NaN for a long enough finite path
    with pytest.raises((ValueError, TypeError)):
        ps._leadlag_area(a, b, 60, 0)
    with pytest.raises((ValueError, TypeError)):
        ps._leadlag_area(a, b, 60, True)


# ---------------------------------------------------------------------------
# M-233 — current-row-as-path-endpoint documentation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ts_path_signature_area",
    "ts_path_signature_depth2_norm",
    "ts_path_leadlag_area",
])
def test_path_signature_descriptions_document_current_row_endpoint(name):
    """M-233: each description documents that the current row is the path
    endpoint (current row participates) and the history is a trailing window."""
    desc = _get(name).metadata.description
    assert "当前行必须是路径端点" in desc, (name, desc)
    assert "trailing window" in desc, (name, desc)


# ---------------------------------------------------------------------------
# M-192 — fixed depth 2 (verify only)
# ---------------------------------------------------------------------------

def test_depth2_norm_is_fixed_depth_no_search_param():
    """M-192: depth is not a search parameter — surface carries only x/y/window
    and the kernel computes the level-2 signature norm."""
    op = _get("ts_path_signature_depth2_norm")
    assert list(op.metadata.param_names) == ["x", "y", "window"]
    assert "depth" not in op.metadata.param_names
    x, y = _xypanels(30)
    out = op.calculate(x, y, window=20)
    assert _fin(out) > 0


# ---------------------------------------------------------------------------
# M-190 — prior-reference / current-query (verify only)
# ---------------------------------------------------------------------------

def test_mahalanobis_anomaly_is_prior_reference_current_query():
    """M-190: classified as PRIOR_REFERENCE_CURRENT_QUERY — the current row is
    the QUERY, the strided prior history is the REFERENCE distribution."""
    assert timing_kind_for("ts_signature_mahalanobis_anomaly") is TimingKind.PRIOR_REFERENCE_CURRENT_QUERY


def test_mahalanobis_anomaly_fixes_depth_two():
    """M-190/192: the operator hard-rejects depth != 2 and carries no depth
    search parameter."""
    op = _get("ts_signature_mahalanobis_anomaly")
    assert "depth" not in op.metadata.param_names
    x, y = _xypanels(40)
    z = pd.DataFrame({"C0": np.sin(np.arange(40.0))}, index=x.index)
    with pytest.raises(ValueError):
        op.calculate(x, y, z, path_window=8, history_window=48, depth=3)
