# -*- coding: utf-8 -*-
"""R11 round-2 operator-definition fixes for the relation package.

Review §16 — kurtosis naming honesty:
  * ``relation_distribution_pearson_kurtosis`` = E[(X-μ)⁴]/σ⁴ (normal ≈ 3).
  * ``relation_distribution_excess_kurtosis`` = Pearson kurtosis − 3 (normal ≈ 0).
  * the historical ``relation_distribution_kurtosis`` spelling resolves as a
    deprecated alias of the Pearson canonical.

Review §17 — integer parameters must not truncate:
  * ``relation_hhi_change`` / ``relation_entropy_change`` /
    ``relation_rank_mobility`` / ``relation_share_mobility`` (and the event /
    index siblings in the relation package) declare strict-integer ParamSpecs;
    a non-integer finite window such as ``5.1`` is REJECTED, never silently
    ``int(window)``-truncated to ``5``.
  * change-family operators require ``window >= 2``; event operators require
    ``event_effective_lag < window``.

NOTE: this module registers the relation operators directly instead of calling
``ensure_cleaned_loaded()`` so the regression tests stay runnable even while a
concurrent session is mid-edit on unrelated governance surfaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.relation.distribution  # noqa: F401  (registers relation ops)
import cleaned_operators.relation.ops  # noqa: F401  (registers relation/index/event ops)
from cleaned_operators.registry import OperatorRegistry


def _op(canonical: str):
    op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    assert op is not None, f"{canonical} missing pandas_numpy backend"
    return op


def _normal_panels(n_rows: int = 1, n_slots: int = 2000, cols=("A",)) -> list[pd.DataFrame]:
    """``n_slots`` relation-rank panels, each a standard-normal cell."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n_rows)
    return [
        pd.DataFrame(
            rng.normal(0.0, 1.0, (n_rows, len(cols))),
            index=idx,
            columns=list(cols),
        )
        for _ in range(n_slots)
    ]


# ---------------------------------------------------------------------------
# review §16 — kurtosis naming
# ---------------------------------------------------------------------------


def test_pearson_kurtosis_approx_3_for_normal_sample():
    panels = _normal_panels()
    out = _op("relation_distribution_pearson_kurtosis").calculate(*panels)
    value = float(out.iloc[0, 0])
    # Pearson kurtosis of a standard normal ≈ 3 (excess kurtosis ≈ 0).  With
    # 2000 iid normal draws the estimator has std ~ sqrt(24/2000) ≈ 0.11, so a
    # [2.5, 3.5] window is a decisive separation from the excess-kurtosis family.
    assert 2.5 <= value <= 3.5, f"Pearson kurtosis of a normal should be ~3, got {value}"


def test_excess_kurtosis_approx_0_for_normal_sample():
    panels = _normal_panels()
    out = _op("relation_distribution_excess_kurtosis").calculate(*panels)
    value = float(out.iloc[0, 0])
    assert -0.6 <= value <= 0.6, f"excess kurtosis of a normal should be ~0, got {value}"


def test_pearson_minus_excess_is_three():
    """excess = pearson − 3 exactly (same inputs)."""
    panels = _normal_panels(n_slots=600)
    pearson = _op("relation_distribution_pearson_kurtosis").calculate(*panels).iloc[0, 0]
    excess = _op("relation_distribution_excess_kurtosis").calculate(*panels).iloc[0, 0]
    assert float(pearson) - float(excess) == pytest.approx(3.0)


def test_old_kurtosis_name_resolves_via_alias():
    # The historical spelling is a deprecated alias -> the Pearson canonical.
    assert (
        OperatorRegistry.resolve_canonical("relation_distribution_kurtosis")
        == "relation_distribution_pearson_kurtosis"
    )
    panels = _normal_panels(n_slots=800)
    legacy = _op("relation_distribution_kurtosis").calculate(*panels)
    pearson = _op("relation_distribution_pearson_kurtosis").calculate(*panels)
    pd.testing.assert_frame_equal(legacy, pearson, check_dtype=False)


def test_both_kurtosis_canonicals_registered():
    for canon in (
        "relation_distribution_pearson_kurtosis",
        "relation_distribution_excess_kurtosis",
    ):
        meta = _op(canon).metadata
        assert meta.output_unit == "dimensionless", canon
        assert any("unit:dimensionless" == t for t in (meta.tags or [])), canon


# ---------------------------------------------------------------------------
# review §17 — integer parameters must not truncate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canonical,panel_count",
    [
        ("relation_hhi_change", 1),
        ("relation_entropy_change", 1),
        ("relation_rank_mobility", 3),
        ("relation_share_mobility", 3),
    ],
)
def test_fractional_window_rejected_not_truncated(canonical, panel_count):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=10)
    panels = [
        pd.DataFrame(rng.random((10, 2)) + 0.1, index=idx, columns=["A", "B"])
        for _ in range(panel_count)
    ]
    # 5.1 must be rejected — never silently truncated to 5.
    with pytest.raises(Exception, match="integer"):
        _op(canonical).calculate(*panels, window=5.1)


@pytest.mark.parametrize(
    "canonical,panel_count",
    [
        ("relation_hhi_change", 1),
        ("relation_entropy_change", 1),
        ("relation_rank_mobility", 3),
        ("relation_share_mobility", 3),
    ],
)
def test_valid_integer_window_still_computes(canonical, panel_count):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=12)
    panels = [
        pd.DataFrame(rng.random((12, 2)) + 0.1, index=idx, columns=["A", "B"])
        for _ in range(panel_count)
    ]
    out = _op(canonical).calculate(*panels, window=3)
    assert out.shape == (12, 2)
    assert out.index.equals(idx)


def test_change_operators_require_window_at_least_two():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=8)
    panel = pd.DataFrame(rng.random((8, 2)) + 0.1, index=idx, columns=["A", "B"])
    for canon in ("relation_hhi_change", "relation_entropy_change",
                  "relation_concentration_acceleration"):
        with pytest.raises(Exception, match=">= 2"):
            _op(canon).calculate(panel, window=1)


def test_topk_k_strict_integer():
    idx = pd.date_range("2024-01-01", periods=1)
    panels = [pd.DataFrame([[0.2]], index=idx, columns=["A"]),
              pd.DataFrame([[0.1]], index=idx, columns=["A"]),
              pd.DataFrame([[0.1]], index=idx, columns=["A"])]
    with pytest.raises(Exception, match="integer"):
        _op("relation_topk_concentration").calculate(*panels, k=2.1)
    out = _op("relation_topk_concentration").calculate(*panels, k=2)
    assert float(out.iloc[0, 0]) == pytest.approx((0.2 + 0.1) / 0.4)


def test_event_operators_strict_integer_window_and_lag():
    idx = pd.date_range("2024-01-01", periods=10)
    ret = pd.DataFrame(np.linspace(0.01, 0.1, 10), index=idx, columns=["A"])
    event = pd.DataFrame([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         index=idx, columns=["A"])
    for canon in (
        "event_cumulative_return_past",
        "event_abnormal_return_past",
        "event_return_since_last",
        "event_arithmetic_return_sum",
        "event_log_return_sum",
        "event_active_count",
    ):
        if canon == "event_abnormal_return_past":
            args = (ret, ret * 0.5, event)
        else:
            args = (ret, event)
        op = _op(canon)
        with pytest.raises(Exception, match="integer"):
            op.calculate(*args, window=5.1)
        with pytest.raises(Exception, match=">= 0"):
            op.calculate(*args, window=5, event_effective_lag=-1)
        # relational feasibility: lag < window
        with pytest.raises(Exception, match="event_effective_lag < window"):
            op.calculate(*args, window=3, event_effective_lag=3)
        # valid integer values still compute
        out = op.calculate(*args, window=5, event_effective_lag=1)
        assert out.shape == (10, 1)


def test_index_ops_strict_integer():
    idx = pd.date_range("2024-01-01", periods=10)
    weight = pd.DataFrame(np.linspace(0.1, 1.0, 10), index=idx, columns=["A"])
    member = pd.DataFrame([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          index=idx, columns=["A"])
    with pytest.raises(Exception, match="integer"):
        _op("index_weight_change").calculate(weight, window=5.1)
    with pytest.raises(Exception, match="integer"):
        _op("index_membership_age").calculate(member, max_lookback=5.1)
    assert _op("index_weight_change").calculate(weight, window=4).shape == (10, 1)
    assert _op("index_membership_age").calculate(member, max_lookback=4).shape == (10, 1)
    # None (the declared default) still means uncapped.
    assert _op("index_membership_age").calculate(member).shape == (10, 1)
