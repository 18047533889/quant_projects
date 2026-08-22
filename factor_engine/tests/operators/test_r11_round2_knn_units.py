# -*- coding: utf-8 -*-
"""R11 round-2 unit-algebra regression for the dynamic KNN operators.

Two typed-factor-composition pollution bugs were fixed in
``cleaned_operators/dynamic_knn.py``:

* ``cs_knn_peer_mean_ex_self``  — output is the *mean of peer target values*,
  so the output unit MUST be ``same_as:target`` (it was mislabelled ``ratio``).
  A ``ratio`` tag would let downstream typed algebra treat a same-unit mean as
  dimensionless.
* ``cs_knn_graph_dirichlet_energy`` — output is ``mean_{j∈NN(i)} (x_i - x_j)²``,
  so the output unit MUST be ``unit(target)²`` (it was mislabelled ``ratio``).
  The tag algebra does not express a superscript square; the closest supported
  algebraic form is the product ``unit(target)*unit(target)`` (same product
  syntax as ``relation_weighted_change``'s ``unit(value)*unit(weight)``).

Each metadata assertion is a pinned reference; the compute assertions verify a
hand-computed 5-stock line panel (k=2 neighbor sets are fixed and deterministic)
and that scaling the target by 10 scales the peer-mean by 10 (linear, unit-covariant
``same_as:target``) and the Dirichlet energy by 100 (quadratic, ``unit(target)²``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _meta(name: str):
    return getattr(_op(name), "metadata", None)


_COLS = ["A", "B", "C", "D", "E"]
_ONE = pd.date_range("2024-01-01", periods=1, freq="B")
_TWO = pd.date_range("2024-01-01", periods=2, freq="B")


def _line_feats(values=None):
    """Three identical feature panels => stock i's k=2 neighbors are its two
    nearest neighbours on the value line (ties at the kth radius are inclusive,
    so the middle names keep exactly 2 neighbours and the ends keep 2 too)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0] if values is None else values
    return [
        pd.DataFrame([values], index=_ONE, columns=_COLS),
        pd.DataFrame([values], index=_ONE, columns=_COLS),
        pd.DataFrame([values], index=_ONE, columns=_COLS),
    ]


# ---------------------------------------------------------------------------
# metadata: output_unit + unit: tag
# ---------------------------------------------------------------------------
def test_knn_peer_mean_output_unit_same_as_target() -> None:
    meta = _meta("cs_knn_peer_mean_ex_self")
    assert meta is not None
    # A mean of peer *target* values keeps the target's unit.
    assert meta.output_unit == "same_as:target"
    assert any(t.startswith("unit:same_as:target") for t in (meta.tags or []))
    # The old mislabelled dimensionless ratio must be gone.
    assert not any(t.startswith("unit:ratio") for t in (meta.tags or []))


def test_knn_graph_dirichlet_energy_output_unit_squared_target() -> None:
    meta = _meta("cs_knn_graph_dirichlet_energy")
    assert meta is not None
    # mean((x_i - x_j)^2) carries unit(target)^2; expressed as the product
    # unit(target)*unit(target) (the tag algebra has no superscript square).
    assert meta.output_unit == "unit(target)*unit(target)"
    assert any("unit:unit(target)*unit(target)" == t for t in (meta.tags or []))
    assert not any(t.startswith("unit:ratio") for t in (meta.tags or []))


def test_knn_neighbor_retention_remains_ratio() -> None:
    # Jaccard overlap of two neighbor sets is genuinely dimensionless -> the
    # untouched ``ratio`` tag is correct and must not be swept into a unit.
    meta = _meta("cs_knn_neighbor_retention")
    assert meta is not None
    assert meta.output_unit is None
    assert any(t.startswith("unit:ratio") for t in (meta.tags or []))


# ---------------------------------------------------------------------------
# compute correctness on a hand-computed 5-stock line panel (k=2)
# ---------------------------------------------------------------------------
def test_knn_peer_mean_line_panel_hand_computed() -> None:
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=_ONE, columns=_COLS)
    f1, f2, f3 = _line_feats()
    out = _op("cs_knn_peer_mean_ex_self").calculate(target, f1, f2, f3, k=2)
    # k=2 neighbors on the line: A->{B,C} B->{A,C} C->{B,D} D->{C,E} E->{C,D}
    expected = [25.0, 20.0, 30.0, 40.0, 35.0]  # mean of peer target values
    np.testing.assert_allclose(out.to_numpy(), np.array([expected]), rtol=1e-9, equal_nan=True)


def test_knn_graph_dirichlet_energy_line_panel_hand_computed() -> None:
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=_ONE, columns=_COLS)
    f1, f2, f3 = _line_feats()
    out = _op("cs_knn_graph_dirichlet_energy").calculate(target, f1, f2, f3, k=2)
    # mean((x_i - x_j)^2) over the k=2 neighbors listed above.
    expected = [250.0, 100.0, 100.0, 100.0, 250.0]
    np.testing.assert_allclose(out.to_numpy(), np.array([expected]), rtol=1e-9, equal_nan=True)


def test_knn_peer_mean_two_dates_translation_equivariant() -> None:
    # Each date is an independent cross-section: shifting every target by +5
    # shifts the peer mean by +5 on each date (same_as:target, additive).
    t2 = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [15.0, 25.0, 35.0, 45.0, 55.0]],
        index=_TWO, columns=_COLS,
    )
    feats2 = [pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2, index=_TWO, columns=_COLS) for _ in range(3)]
    out = _op("cs_knn_peer_mean_ex_self").calculate(t2, *feats2, k=2)
    expected = np.array([[25.0, 20.0, 30.0, 40.0, 35.0],
                         [30.0, 25.0, 35.0, 45.0, 40.0]])
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-9, equal_nan=True)


def test_knn_graph_dirichlet_energy_two_dates_translation_invariant() -> None:
    # Shifting the target by a constant leaves the squared gaps unchanged, so
    # the Dirichlet energy is identical on both dates (unit(target)^2).
    t2 = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [15.0, 25.0, 35.0, 45.0, 55.0]],
        index=_TWO, columns=_COLS,
    )
    feats2 = [pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2, index=_TWO, columns=_COLS) for _ in range(3)]
    out = _op("cs_knn_graph_dirichlet_energy").calculate(t2, *feats2, k=2)
    expected = np.array([[250.0, 100.0, 100.0, 100.0, 250.0],
                         [250.0, 100.0, 100.0, 100.0, 250.0]])
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# unit-covariant scaling (dimensional consistency of the declared units)
# ---------------------------------------------------------------------------
def test_knn_units_covariant_scaling() -> None:
    base = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=_ONE, columns=_COLS)
    f1, f2, f3 = _line_feats()
    scaled = base * 10.0  # features unchanged -> identical neighbor sets

    pm1 = _op("cs_knn_peer_mean_ex_self").calculate(base, f1, f2, f3, k=2)
    pm10 = _op("cs_knn_peer_mean_ex_self").calculate(scaled, f1, f2, f3, k=2)
    # same_as:target -> linear in target.
    np.testing.assert_allclose(pm10.to_numpy(), 10.0 * pm1.to_numpy(), rtol=1e-9, equal_nan=True)

    en1 = _op("cs_knn_graph_dirichlet_energy").calculate(base, f1, f2, f3, k=2)
    en10 = _op("cs_knn_graph_dirichlet_energy").calculate(scaled, f1, f2, f3, k=2)
    # unit(target)^2 -> quadratic in target.
    np.testing.assert_allclose(en10.to_numpy(), 100.0 * en1.to_numpy(), rtol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# fail-closed edge cases (unchanged behaviour guard)
# ---------------------------------------------------------------------------
def test_knn_fail_closed_insufficient_neighbor_targets() -> None:
    # 5-stock line, k=2.  Endpoint E's target is missing, so E itself skips
    # (target missing) and D's neighbour set {C, E} loses its only other finite
    # member (E) -> D has < k finite peer values and fails closed to NaN.
    # A/B/C are untouched and still compute (R4-82: never call a <k-peer mean /
    # energy a k-NN mean).
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, np.nan]], index=_ONE, columns=_COLS)
    f1, f2, f3 = _line_feats()
    pm = _op("cs_knn_peer_mean_ex_self").calculate(target, f1, f2, f3, k=2)
    np.testing.assert_allclose(pm.to_numpy(), np.array([[25.0, 20.0, 30.0, np.nan, np.nan]]),
                               rtol=1e-9, equal_nan=True)
    en = _op("cs_knn_graph_dirichlet_energy").calculate(target, f1, f2, f3, k=2)
    np.testing.assert_allclose(en.to_numpy(), np.array([[250.0, 100.0, 100.0, np.nan, np.nan]]),
                               rtol=1e-9, equal_nan=True)


def test_knn_fail_closed_too_few_stocks() -> None:
    # Only 3 stocks but k=3: no name has k genuine neighbours (count-1=2<k),
    # so the whole panel fails closed instead of silently averaging 2 peers.
    target = pd.DataFrame([[10.0, 20.0, 30.0]], index=_ONE, columns=["A", "B", "C"])
    f1 = pd.DataFrame([[1.0, 2.0, 3.0]], index=_ONE, columns=["A", "B", "C"])
    f2 = f1.copy()
    f3 = f1.copy()
    pm = _op("cs_knn_peer_mean_ex_self").calculate(target, f1, f2, f3, k=3)
    assert pm.isna().all().all()
    en = _op("cs_knn_graph_dirichlet_energy").calculate(target, f1, f2, f3, k=3)
    assert en.isna().all().all()
