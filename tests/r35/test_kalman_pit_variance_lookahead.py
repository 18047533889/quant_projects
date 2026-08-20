# -*- coding: utf-8 -*-
"""PIT audit: Kalman dimensionless noise scaling must use only the observed prefix.

CONCRETE BUG (2026-08-13):
``_finite_variance(vals)`` computes variance over the ENTIRE series (all finite
observations including future ones) BEFORE the recursive filter loop. This causes
a PIT violation for outputs that are NOT scale-invariant:

- ``ts_kalman_level``: NO BUG. The level is scale-invariant because when both q
  and r are scaled by var_x, the Kalman gain K = P/(P+r) remains unchanged (P also
  scales by var_x), so mu(t) is identical regardless of global variance.

- ``ts_kalman_innovation_z``: **BUG**. The standardized innovation divides by
  sqrt(P+r), which scales with var_x. So innov_z = innov / sqrt(k*(P+r)) where
  k=var_x, making it depend on the global variance (including future data).

- ``ts_kalman_beta_uncertainty``: **BUG**. The covariance P scales with var_x,
  so P(t) depends on the global variance of x (including future observations).

The causal contract: at row t the filter has observed only vals[0..t]; the noise
scale must derive from that observed prefix, never from future data.

Fix: compute variance incrementally as finite observations arrive (maintain running
sum/sum-of-squares, update var_t with each finite row), so var_t uses only vals[:t+1].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cleaned_operators.registry import OperatorRegistry


def _get(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    assert op is not None, name
    return op


def _panel(vals: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D")
    return pd.DataFrame(vals[:, None], index=idx, columns=["C0"])


def _load():
    import cleaned_operators.ts_model.state_space  # noqa: F401


# ---------------------------------------------------------------------------
# Regression: level is scale-invariant (no bug)
# ---------------------------------------------------------------------------

def test_kalman_level_dimensionless_is_scale_invariant():
    """REGRESSION: ``ts_kalman_level`` with ``scale_mode="dimensionless"`` is
    scale-invariant by design.

    When both q and r are scaled by var_x, the Kalman gain K = P/(P+r) remains
    unchanged because P also scales by var_x. Therefore, the filtered level mu(t)
    is identical regardless of the global variance. This is the INTENDED behavior
    — the level output does NOT have a PIT bug."""
    _load()
    rng = np.random.default_rng(100)
    prefix = rng.standard_normal(30) * 0.02  # common prefix
    suffix_a = rng.standard_normal(20) * 0.02
    suffix_b = rng.standard_normal(20) * 0.10  # 5x larger variance in the tail
    series_a = np.concatenate([prefix, suffix_a])
    series_b = np.concatenate([prefix, suffix_b])
    level = _get("ts_kalman_level")
    out_a = level.calculate(_panel(series_a), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    out_b = level.calculate(_panel(series_b), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    # Scale invariance: the outputs should be identical even though var_b = 12*var_a
    assert np.allclose(out_a[:30], out_b[:30], equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Poison tests: innovation_z and uncertainty have PIT bugs
# ---------------------------------------------------------------------------

def test_kalman_innovation_z_dimensionless_pit_violation():
    """**PIT BUG**: ``ts_kalman_innovation_z`` with ``scale_mode="dimensionless"``
    depends on the global variance (including future data).

    innov_z = innov / sqrt(P + r). When both P and r scale by var_x, innov_z scales
    by 1/sqrt(var_x). Two series with identical prefix but different global variance
    produce DIFFERENT innov_z values in the prefix (the filter at time t 'sees' the
    future variance)."""
    _load()
    rng = np.random.default_rng(100)
    prefix = rng.standard_normal(30) * 0.02
    suffix_a = rng.standard_normal(20) * 0.02
    suffix_b = rng.standard_normal(20) * 0.10  # 5x larger variance
    series_a = np.concatenate([prefix, suffix_a])
    series_b = np.concatenate([prefix, suffix_b])

    innov = _get("ts_kalman_innovation_z")
    out_a = innov.calculate(_panel(series_a), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()
    out_b = innov.calculate(_panel(series_b), q=0.01, r=1.0, scale_mode="dimensionless")["C0"].to_numpy()

    # var_b / var_a ≈ 12, so innov_z should scale by sqrt(12) ≈ 3.46
    prefix_a = out_a[:30]
    prefix_b = out_b[:30]
    max_diff = np.nanmax(np.abs(prefix_a - prefix_b))

    # The causal contract says they must be identical (at t=29, no suffix observed yet).
    # But they differ significantly because var_x includes the suffix.
    if max_diff > 0.1:  # threshold chosen to detect the ~sqrt(12) scaling
        pytest.fail(
            f"PIT VIOLATION in innovation_z: future data altered the past output. "
            f"Max diff in prefix: {max_diff:.3f}. The filter at row t saw future variance. "
            f"Expected ratio sqrt(var_b/var_a) = sqrt(12) ≈ 3.46, observed diff confirms this."
        )


def test_kalman_beta_uncertainty_dimensionless_pit_violation():
    """**PIT BUG**: ``ts_kalman_beta_uncertainty`` with ``scale_mode="dimensionless"``
    outputs P (the covariance), which scales by var_x. Two series with identical prefix
    but different global x variance produce DIFFERENT P values in the prefix."""
    _load()
    rng = np.random.default_rng(101)
    n_prefix = 30
    n_suffix = 20
    x_prefix = rng.standard_normal(n_prefix) * 0.02
    y_prefix = 1.2 * x_prefix + rng.standard_normal(n_prefix) * 0.005
    x_suffix_a = rng.standard_normal(n_suffix) * 0.02
    x_suffix_b = rng.standard_normal(n_suffix) * 0.10  # 5x larger x variance
    y_suffix_a = 1.2 * x_suffix_a + rng.standard_normal(n_suffix) * 0.005
    y_suffix_b = 1.2 * x_suffix_b + rng.standard_normal(n_suffix) * 0.005

    y_a = np.concatenate([y_prefix, y_suffix_a])
    x_a = np.concatenate([x_prefix, x_suffix_a])
    y_b = np.concatenate([y_prefix, y_suffix_b])
    x_b = np.concatenate([x_prefix, x_suffix_b])

    unc = _get("ts_kalman_beta_uncertainty")
    out_a = unc.calculate(_panel(y_a), _panel(x_a), q=0.01, r=1.0,
                          scale_mode="dimensionless", min_warmup=10)["C0"].to_numpy()
    out_b = unc.calculate(_panel(y_b), _panel(x_b), q=0.01, r=1.0,
                          scale_mode="dimensionless", min_warmup=10)["C0"].to_numpy()

    prefix_a = out_a[:n_prefix]
    prefix_b = out_b[:n_prefix]

    # P scales by var_x, so we expect P_b ≈ 12 * P_a (non-NaN values)
    finite_mask = np.isfinite(prefix_a) & np.isfinite(prefix_b)
    if finite_mask.sum() > 0:
        ratio = np.median(prefix_b[finite_mask] / prefix_a[finite_mask])
        # Causal contract: P(t) should depend only on x[:t+1], so ratio should be ~1.
        # But it's ~12 because var_x includes the future suffix.
        if ratio > 2.0:  # threshold to detect the ~12x scaling
            pytest.fail(
                f"PIT VIOLATION in beta_uncertainty: future x variance altered P. "
                f"Median P_b/P_a ratio in prefix: {ratio:.2f} (expected ~1, got ~12). "
                f"This means P(t) depends on var(x[:end]), not var(x[:t+1])."
            )


def test_kalman_trend_dimensionless_is_scale_invariant():
    """REGRESSION: ``ts_kalman_trend`` (the slope) is scale-invariant like level.
    The trend state is invariant to uniform (q, r, P) scaling, so no PIT bug."""
    _load()
    rng = np.random.default_rng(102)
    prefix = rng.standard_normal(30) * 0.02
    suffix_a = rng.standard_normal(20) * 0.02
    suffix_b = rng.standard_normal(20) * 0.10
    series_a = np.concatenate([prefix, suffix_a])
    series_b = np.concatenate([prefix, suffix_b])

    trend = _get("ts_kalman_trend")
    out_a = trend.calculate(_panel(series_a), q_level=1e-5, q_trend=1e-5, r=1.0,
                            scale_mode="dimensionless")["C0"].to_numpy()
    out_b = trend.calculate(_panel(series_b), q_level=1e-5, q_trend=1e-5, r=1.0,
                            scale_mode="dimensionless")["C0"].to_numpy()

    assert np.allclose(out_a[:30], out_b[:30], equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Regression: absolute mode must remain unaffected (no variance computation)
# ---------------------------------------------------------------------------

def test_absolute_mode_unaffected_by_future():
    """Regression: ``scale_mode="absolute"`` (default) bypasses variance
    computation entirely, so it has always been causal (no bug)."""
    _load()
    rng = np.random.default_rng(103)
    prefix = rng.standard_normal(30) * 0.02
    suffix_a = rng.standard_normal(20) * 0.02
    suffix_b = rng.standard_normal(20) * 0.10
    series_a = np.concatenate([prefix, suffix_a])
    series_b = np.concatenate([prefix, suffix_b])

    level = _get("ts_kalman_level")
    innov = _get("ts_kalman_innovation_z")

    # absolute mode (default): q/r are used as-is, no variance rescaling
    level_a = level.calculate(_panel(series_a), q=1e-4, r=1.0, scale_mode="absolute")["C0"].to_numpy()
    level_b = level.calculate(_panel(series_b), q=1e-4, r=1.0, scale_mode="absolute")["C0"].to_numpy()
    innov_a = innov.calculate(_panel(series_a), q=1e-4, r=1.0, scale_mode="absolute")["C0"].to_numpy()
    innov_b = innov.calculate(_panel(series_b), q=1e-4, r=1.0, scale_mode="absolute")["C0"].to_numpy()

    # Prefix outputs must be identical (and they are — absolute mode is clean)
    assert np.allclose(level_a[:30], level_b[:30], equal_nan=True, atol=1e-12)
    assert np.allclose(innov_a[:30], innov_b[:30], equal_nan=True, atol=1e-12)
