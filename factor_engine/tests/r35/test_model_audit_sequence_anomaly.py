# -*- coding: utf-8 -*-
"""Model-audit remediation M-101 / M-102 / M-140 / M-142.

M-101 — ``sequence_anomaly._mp_stats`` must validate ``m`` strictly (integer >=
        3); ``m < 3``, fractional, NaN/Inf or bool raises instead of the legacy
        ``max(3, int(m))`` clamp/truncate.

M-102 — ``history_window`` must be a finite positive integer; ``<= 0`` raises
        instead of silently expanding to full history.

M-140 — the Mahalanobis / local-density operators (candle_state_space.py)
        document their prior-reference/current-query split in the metadata
        description ("历史参考严格 ≤t-1、当前查询=t，查询行排除在参考之外").

M-142 — ``candle_state_space.last_mahalanobis_telemetry()`` exposes per-fit
        ``p_effective`` / ``N_effective`` / ``condition_number`` /
        ``dropped_constant_dims`` / ``failure_reason``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators.base import OperatorParameterError  # noqa: E402
from cleaned_operators.candle_state_space import (  # noqa: E402
    TsVectorStateLocalDensity,
    TsVectorStateMahalanobis,
    _mahalanobis_series,
    last_mahalanobis_telemetry,
)
from cleaned_operators.ts_model.sequence_anomaly import _mp_stats  # noqa: E402


def _series(n: int = 60, seed: int = 0) -> np.ndarray:
    return np.arange(1.0, n + 1.0)


def _panel(vals: np.ndarray, instruments: int = 1) -> pd.DataFrame:
    if vals.ndim == 1:
        vals = vals[:, None]
    idx = pd.date_range("2024-01-01", periods=vals.shape[0], freq="D")
    return pd.DataFrame(vals, index=idx, columns=[f"C{k}" for k in range(vals.shape[1])])


# ---------------------------------------------------------------------------
# M-101: m is strictly validated as an integer >= 3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_m", [2, 1, 0, -5])
def test_m101_m_below_3_raises(bad_m: int):
    with pytest.raises(OperatorParameterError):
        _mp_stats(_series(), bad_m, "discord", 20)


@pytest.mark.parametrize("bad_m", [20.5, np.nan, np.inf, -np.inf, True, np.bool_(True), "20"])
def test_m101_m_non_integer_nan_inf_bool_raises(bad_m):
    with pytest.raises((OperatorParameterError, TypeError)):
        _mp_stats(_series(), bad_m, "discord", 20)


def test_m101_valid_m_accepted_and_deterministic():
    v = _series(60)
    a = _mp_stats(v, 5, "discord", 20)
    b = _mp_stats(v, 5.0, "discord", 20.0)  # value-equal float is accepted
    assert np.isnan(a) or np.isfinite(a)
    assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# M-102: history_window is strictly validated as a positive integer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_hist", [0, -1, -100])
def test_m102_history_window_nonpositive_raises(bad_hist: int):
    with pytest.raises(OperatorParameterError):
        _mp_stats(_series(), 5, "discord", bad_hist)


@pytest.mark.parametrize("bad_hist", [20.5, np.nan, np.inf, True, "20"])
def test_m102_history_window_fractional_nan_bool_raises(bad_hist):
    with pytest.raises((OperatorParameterError, TypeError)):
        _mp_stats(_series(), 5, "discord", bad_hist)


def test_m102_valid_positive_history_accepted():
    v = _series(60)
    out = _mp_stats(v, 5, "discord", 20)
    assert np.isnan(out) or np.isfinite(out)


# ---------------------------------------------------------------------------
# M-142: Mahalanobis fit-quality telemetry
# ---------------------------------------------------------------------------

def test_m142_telemetry_accessor_returns_expected_keys():
    tel = last_mahalanobis_telemetry()
    assert set(tel.keys()) == {
        "p_effective", "N_effective", "condition_number",
        "dropped_constant_dims", "failure_reason",
    }
    # defensive copy
    tel["failure_reason"] = "mutated"
    assert last_mahalanobis_telemetry()["failure_reason"] != "mutated"


def test_m142_telemetry_ok_on_healthy_fit():
    rng = np.random.default_rng(5)
    n = 80
    arrs = [rng.normal(0.0, 1.0, (n, 1)) for _ in range(4)]
    out = _mahalanobis_series(*arrs, window=40, shrinkage=0.5)
    assert np.isfinite(out).mean() > 0.5
    tel = last_mahalanobis_telemetry()
    assert tel["failure_reason"] == "ok"
    assert tel["p_effective"] == 4
    assert tel["N_effective"] is not None and tel["N_effective"] >= 20
    assert tel["condition_number"] is not None and np.isfinite(tel["condition_number"])
    assert tel["condition_number"] >= 1.0
    assert tel["dropped_constant_dims"] == 0


def test_m142_telemetry_insufficient_sample_reveals_floor():
    rng = np.random.default_rng(2)
    small = np.full((30, 1), np.nan)
    small[:6] = rng.normal(0.0, 1.0, (6, 1))
    args = [small.copy() for _ in range(4)]
    out = _mahalanobis_series(*args, window=10, shrinkage=0.5)
    assert np.all(np.isnan(out)), "N < 5p must be NaN for mahalanobis"
    tel = last_mahalanobis_telemetry()
    assert tel["failure_reason"] == "insufficient_sample"
    assert tel["N_effective"] is not None
    assert tel["p_effective"] is not None
    assert tel["N_effective"] < 5 * tel["p_effective"]
    assert tel["condition_number"] is None


def test_m142_telemetry_all_constant_dims_dropped():
    const = np.ones((20, 1))
    args = [const.copy() for _ in range(4)]
    _mahalanobis_series(*args, window=10, shrinkage=0.5)
    tel = last_mahalanobis_telemetry()
    assert tel["failure_reason"] == "all_constant"
    assert tel["p_effective"] == 0
    assert tel["dropped_constant_dims"] == 4


def test_m142_telemetry_trailing_nan_preserves_last_fit():
    rng = np.random.default_rng(9)
    n = 80
    arrs = [rng.normal(0.0, 1.0, (n, 1)) for _ in range(4)]
    for a in arrs:
        a[-1] = np.nan
    _mahalanobis_series(*arrs, window=40, shrinkage=0.5)
    tel = last_mahalanobis_telemetry()
    # a NaN current row is a data skip, NOT a fit — the last genuine fit
    # decision must survive (M-142).
    assert tel["failure_reason"] == "ok"
    assert tel["p_effective"] == 4


# ---------------------------------------------------------------------------
# M-140: Reference/Query split documented on the operator metadata
# ---------------------------------------------------------------------------

def test_m140_mahalanobis_metadata_documents_reference_query():
    desc = TsVectorStateMahalanobis.metadata.description
    assert "历史参考严格 ≤t-1、当前查询=t" in desc
    assert "查询行排除在参考之外" in desc


def test_m140_local_density_metadata_documents_reference_query():
    desc = TsVectorStateLocalDensity.metadata.description
    assert "历史参考严格 ≤t-1、当前查询=t" in desc
    assert "查询行排除在参考之外" in desc


def test_m140_mahalanobis_query_excluded_from_own_reference():
    """Behavioral check of the documented split: the trailing anomaly must be
    the distance of the query (today) to a reference built ONLY on prior rows,
    so a large current value is not dragged back toward its own reference."""
    rng = np.random.default_rng(11)
    n = 80
    base = rng.normal(0.0, 1.0, (n, 1))
    # Spike the current row (the query) to an extreme; since it is excluded
    # from the reference, the anomaly stays large instead of self-cancelling.
    spiked = base.copy()
    spiked[-1] = 20.0
    normal = _mahalanobis_series(*[base.copy() for _ in range(4)], window=40, shrinkage=0.5)[-1, 0]
    spiked_out = _mahalanobis_series(*[spiked.copy() for _ in range(4)], window=40, shrinkage=0.5)[-1, 0]
    assert np.isfinite(spiked_out)
    assert spiked_out > normal, "query appears to be part of its own reference"
