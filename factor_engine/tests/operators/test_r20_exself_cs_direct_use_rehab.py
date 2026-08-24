# -*- coding: utf-8 -*-
"""R20 intra ex-self (leave-one-out cross-sectional) DirectUse oracles.

Slice: R20-EXSELF-CS-DIRECTUSE.

Duplicate-audit outcome (survey BEFORE implementing — documented neighbors,
none is the same estimator):
* ``group_ex_self_mean`` (group_ext.py; polars twins in
  common/polars_group.py / common/polars_group_advanced.py /
  polars_native/group_batch1.py / auto_polars_all.py) — LOO group mean
  LEVEL (unit of x, not terminal-usable; R11 #135 comment says exactly
  this).  The x-minus-LOO-mean DIRECT-USE gap and the z/rank/mad_z
  standardized variants did NOT exist.
* ``group_ex_self_std`` / ``group_ex_self_mad`` / ``group_ex_self_quantile``
  (alpha_language_cross.py ~L316/347/388; polars twins incl. approximate
  full-group-proxy variants in group_batch1.py / polars_group_advanced.py)
  — dispersion/quantile LEVELS over the LOO peer set, not standardized
  against x.
* ``group_ex_self_weighted_mean`` (group_ext.py L119, relation/ops.py
  L540) — LOO weighted mean level.
* ``relation_weighted_std_ex_self`` (alpha_language_cross.py L431) — LOO
  node-weighted std level.
* ``group_zscore`` (polars_native/group_batch1.py L929) — FULL-GROUP
  z-score INCLUDING self, zero-fills degenerate std (otherwise(0.0)) —
  the exact self-inclusion bias and silent zero this family removes.
* ``cs_zscore`` / ``cs_rank`` / ``cs_demean`` (common/polars_cs_basic.py
  L90/L59/L123) — whole-cross-section, self included, no group dimension,
  no min_peers guard.
* ``group_neutralize`` (polars_native/group_batch1.py L418) — subtract
  full-group mean (self included).
* ``group_peer_deviation_index`` (common/polars_group_advanced.py L815) —
  (x - FULL-GROUP median)/FULL-GROUP MAD, self included on both sides.
  Nearest neighbor to ex_self_mad_z; the LOO variant did not exist.
* ``group_peer_beta_deviation`` (cross_section/peer_ops.py L108) —
  x - LOO WEIGHTED mean for beta panels (weighted, beta-specific).
* ``cs_knn_peer_mean_ex_self`` (dynamic_knn.py L247) — kNN-similarity
  peer mean (not a group label).
* intra ``*_ex_self`` intraday canonicals (intraday/realized_beta.py etc.)
  — intraday market-model estimators, different domain.

Landed canonicals (all purely cross-sectional per row t; NaN/±Inf peers
excluded from stats, NaN/±Inf self -> NaN; < min_peers finite in-group
peers EXCLUDING self -> NaN; degenerate LOO std/MAD -> NaN never 0;
min_peers ParamSpec(int, min=3) == runtime guard):
* ``ex_self_mean_gap`` — x - LOO mean.
* ``ex_self_zscore``   — (x - LOO mean) / LOO std (ddof=0 over peers).
* ``ex_self_rank_pct`` — LOO average-rank percentile in [0,1].
* ``ex_self_mad_z``    — (x - LOO median) / LOO MAD (raw MAD units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_exself_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("ex_self_zscore", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import exself_cs_v1  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_exself_chain()


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


ALL_NAMES = ("ex_self_mean_gap", "ex_self_zscore", "ex_self_rank_pct", "ex_self_mad_z")

_AUDITED_NEIGHBORS = (
    "group_ex_self_mean",
    "group_ex_self_std",
    "group_ex_self_mad",
    "group_ex_self_quantile",
    "group_ex_self_weighted_mean",
    "relation_weighted_std_ex_self",
    "group_zscore",
    "cs_zscore",
    "cs_rank",
    "cs_demean",
    "group_neutralize",
    "group_peer_deviation_index",
)


def _panel(rows=30, cols=9, seed=42):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(rows, cols)), columns=[f"s{i}" for i in range(cols)])
    # group layout: 5 x A, 3 x B, 1 x C  ->  C has zero peers (NaN),
    # B has 2 peers (< min_peers=3 -> NaN)
    labels = np.array(["A"] * 5 + ["B"] * 3 + ["C"], dtype=object)
    g = pd.DataFrame(
        np.tile(labels, (rows, 1)), index=x.index, columns=x.columns
    )
    return x, g


# ---------------------------------------------------------------------------
# manual oracles — independent reimplementations (plain pandas/python loops,
# sharing no code with cleaned_operators/technical/exself_cs_v1.py)
# ---------------------------------------------------------------------------
def _oracle_stat(x: pd.DataFrame, g: pd.DataFrame, min_peers: int, stat: str) -> pd.DataFrame:
    out = np.full(x.shape, np.nan, dtype=float)
    xv = x.to_numpy(dtype=float)
    gv = g.to_numpy()
    for r in range(x.shape[0]):
        row, grow = xv[r], gv[r]
        for j in range(x.shape[1]):
            v = row[j]
            if not np.isfinite(v):
                continue
            peers = [row[k] for k in range(x.shape[1])
                     if k != j and grow[k] == grow[j] and np.isfinite(row[k])]
            if len(peers) < min_peers:
                continue
            peers = np.asarray(peers, dtype=float)
            if stat == "mean_gap":
                out[r, j] = v - float(np.mean(peers))
            elif stat == "zscore":
                mu = float(np.mean(peers))
                sd = float(np.std(peers))  # ddof=0
                if sd <= 0.0:
                    continue
                out[r, j] = (v - mu) / sd
            elif stat == "rank_pct":
                below = sum(1.0 for p in peers if p < v)
                ties = sum(1.0 for p in peers if p == v)
                out[r, j] = (below + 0.5 * ties) / float(len(peers))
            elif stat == "mad_z":
                med = float(np.median(peers))
                mad = float(np.median(np.abs(peers - med)))
                if mad <= 0.0:
                    continue
                out[r, j] = (v - med) / mad
    return pd.DataFrame(out, index=x.index, columns=x.columns)


@pytest.mark.parametrize(
    "name,stat",
    [
        ("ex_self_mean_gap", "mean_gap"),
        ("ex_self_zscore", "zscore"),
        ("ex_self_rank_pct", "rank_pct"),
        ("ex_self_mad_z", "mad_z"),
    ],
)
def test_matches_manual_oracle(name, stat):
    x, g = _panel(rows=40, cols=9, seed=101)
    out = _op(name).calculate(x, g, min_peers=3)
    exp = _oracle_stat(x, g, 3, stat)
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_matches_manual_oracle_with_nans_and_inf(name):
    # NaN and ±Inf peers must be EXCLUDED from stats; a NaN self -> NaN out
    rng = np.random.default_rng(55)
    vals = rng.normal(size=(18, 12))
    vals[3, 2] = np.nan
    vals[5, 8] = np.inf
    vals[7, 1] = -np.inf
    vals[10, 4] = np.nan
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(12)])
    labels = np.array(["A"] * 7 + ["B"] * 5, dtype=object)
    g = pd.DataFrame(np.tile(labels, (18, 1)), index=x.index, columns=x.columns)
    stat = {
        "ex_self_mean_gap": "mean_gap",
        "ex_self_zscore": "zscore",
        "ex_self_rank_pct": "rank_pct",
        "ex_self_mad_z": "mad_z",
    }[name]
    out = _op(name).calculate(x, g, min_peers=3)
    exp = _oracle_stat(x, g, 3, stat)
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # NaN / ±Inf self cells are NaN in the output
    assert np.isnan(out.to_numpy()[3, 2])
    assert np.isnan(out.to_numpy()[5, 8])
    assert np.isnan(out.to_numpy()[7, 1])
    assert np.isnan(out.to_numpy()[10, 4])


@pytest.mark.parametrize("name", ALL_NAMES)
def test_self_nan_self_inf_fail_closed(name):
    rng = np.random.default_rng(77)
    x = pd.DataFrame(rng.normal(size=(6, 6)), columns=[f"s{i}" for i in range(6)])
    g = pd.DataFrame(np.tile(np.array(["A"] * 6, dtype=object), (6, 1)),
                     index=x.index, columns=x.columns)
    for bad in (np.nan, np.inf, -np.inf):
        y = x.copy()
        y.iloc[2, 3] = bad
        out = _op(name).calculate(y, g, min_peers=3)
        assert np.isnan(out.to_numpy()[2, 3]), (name, bad)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_min_peers_gate(name):
    # group of exactly 3 finite members -> 2 peers each -> NaN
    rng = np.random.default_rng(88)
    x = pd.DataFrame(rng.normal(size=(5, 3)), columns=["a", "b", "c"])
    g = pd.DataFrame(np.tile(np.array(["A"] * 3, dtype=object), (5, 1)),
                     index=x.index, columns=x.columns)
    out = _op(name).calculate(x, g, min_peers=3)
    assert np.isnan(out.to_numpy()).all()
    # group of 4 -> 3 peers each -> finite everywhere
    x4 = pd.DataFrame(rng.normal(size=(5, 4)), columns=["a", "b", "c", "d"])
    g4 = pd.DataFrame(np.tile(np.array(["A"] * 4, dtype=object), (5, 1)),
                      index=x4.index, columns=x4.columns)
    out4 = _op(name).calculate(x4, g4, min_peers=3)
    assert np.isfinite(out4.to_numpy()).all()
    # one NaN member in a group of 4: at that row EVERY member now has at
    # most 2 finite peers (the NaN member is excluded from all peer sets,
    # and its own output is NaN) -> whole row NaN
    x5 = x4.copy()
    x5.iloc[2, 1] = np.nan
    out5 = _op(name).calculate(x5, g4, min_peers=3)
    assert np.isnan(out5.to_numpy()[2]).all()
    # rows without the NaN are unaffected
    assert np.isfinite(out5.to_numpy()[0]).all()


def _nan_aware_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_min_peers_below_three_rejected(name):
    x, g = _panel(rows=5, cols=9, seed=90)
    for bad in (2, 1, 0, -1, 2.5, 3.5):
        with pytest.raises(ValueError):
            _op(name).calculate(x, g, min_peers=bad)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_degenerate_dispersion_nan_never_zero(name):
    # identical values in a group: LOO std == 0 AND LOO MAD == 0 -> NaN for
    # the z-score and mad-z canonicals; mean_gap is exactly 0 (a legitimate
    # finite gap); rank_pct with all-equal peers is exactly 0.5.
    vals = np.tile(np.array([1.5] * 5 + [2.0, 4.0, 6.0, 8.0]), (4, 1))
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(9)])
    _, g = _panel(rows=4, cols=9, seed=91)
    out = _op(name).calculate(x, g, min_peers=3).to_numpy()
    if name == "ex_self_zscore":
        assert np.isnan(out[:, :5]).all()
        assert np.isnan(out[:, 5:8]).all()   # group B: 2 peers < 3 -> NaN
        assert np.isnan(out[:, 8]).all()     # group C: 0 peers -> NaN
    elif name == "ex_self_mad_z":
        assert np.isnan(out[:, :5]).all()
        assert np.isnan(out[:, 5:8]).all()
        assert np.isnan(out[:, 8]).all()
    elif name == "ex_self_mean_gap":
        np.testing.assert_allclose(out[:, :5], 0.0, rtol=1e-12)
        assert np.isnan(out[:, 5:]).all()    # B/C too small for min_peers=3
    else:  # rank_pct: all-equal peers -> ties -> 0.5
        np.testing.assert_allclose(out[:, :5], 0.5, rtol=1e-12)
        assert np.isnan(out[:, 5:]).all()


def test_rank_pct_bounded_and_boundary():
    # one dominant / one dominated member in a group of 4
    vals = np.tile(np.array([10.0, 1.0, 2.0, 3.0]), (3, 1))
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(4)])
    g = pd.DataFrame(np.tile(np.array(["A"] * 4, dtype=object), (3, 1)),
                     index=x.index, columns=x.columns)
    out = _op("ex_self_rank_pct").calculate(x, g, min_peers=3).to_numpy()
    assert out[0, 0] == 1.0   # strictly above all 3 peers
    assert out[0, 1] == 0.0   # strictly below all 3 peers
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_rank_pct_ties_split_halfway():
    # group of 4: [1, 1, 3, 2]
    # self=1 (cols 0/1): peers {1,3,2} -> below=0, ties=1 -> 0.5/3 = 1/6
    # self=3 (col 2):    peers {1,1,2} -> below=3           -> 1.0
    # self=2 (col 3):    peers {1,1,3} -> below=2           -> 2/3
    vals = np.tile(np.array([1.0, 1.0, 3.0, 2.0]), (2, 1))
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(4)])
    g = pd.DataFrame(np.tile(np.array(["A"] * 4, dtype=object), (2, 1)),
                     index=x.index, columns=x.columns)
    out = _op("ex_self_rank_pct").calculate(x, g, min_peers=3).to_numpy()
    np.testing.assert_allclose(out[0, 0], 1.0 / 6.0, rtol=1e-12)
    np.testing.assert_allclose(out[0, 1], 1.0 / 6.0, rtol=1e-12)
    np.testing.assert_allclose(out[0, 2], 1.0, rtol=1e-12)
    np.testing.assert_allclose(out[0, 3], 2.0 / 3.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# causality: purely cross-sectional per row (pinned semantics)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_causality_mutation_semantics(name):
    """Mutating cell (t, col) must not change ANY other row t' != t.

    Same-row OTHER columns: the pinned semantics allow a change ONLY for
    columns sharing the mutated cell's group label at row t (the LOO peer
    set of a same-group peer loses/gains the mutated value); different-group
    columns at row t must be bit-identical.
    """
    rng = np.random.default_rng(33)
    x = pd.DataFrame(rng.normal(size=(15, 9)), columns=[f"s{i}" for i in range(9)])
    _, g = _panel(rows=15, cols=9, seed=34)
    x_mut = x.copy()
    x_mut.iloc[7, 0] = x.iloc[7, 0] + 5.0  # same group A at row 7
    a = _op(name).calculate(x, g, min_peers=3)
    b = _op(name).calculate(x_mut, g, min_peers=3)
    # rows != 7 are bit-identical (no time leakage — trivially true by
    # construction, asserted anyway per the family contract)
    pd.testing.assert_frame_equal(a.drop(index=a.index[7]), b.drop(index=b.index[7]))
    # row 7: group-B/C columns (>= 5) unchanged (NaN-aware); group-A columns
    # may change
    assert _nan_aware_equal(a.to_numpy()[7, 5:], b.to_numpy()[7, 5:]), \
        "different-group same-row cells must not move"
    # and at least one same-group cell DID move (non-vacuous peer effect)
    assert not _nan_aware_equal(a.to_numpy()[7, :5], b.to_numpy()[7, :5])


@pytest.mark.parametrize("name", ALL_NAMES)
def test_prefix_invariance(name):
    # prefix invariance across time is trivially true for a per-row
    # cross-sectional operator — asserted anyway per the family contract
    x, g = _panel(rows=25, cols=9, seed=35)
    full = _op(name).calculate(x, g, min_peers=3)
    head = _op(name).calculate(x.iloc[:15], g.iloc[:15], min_peers=3)
    pd.testing.assert_frame_equal(full.iloc[:15], head)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_column_independence(name):
    # no cross-column leak THROUGH the group labels of other columns: moving
    # column 3 to its own group changes column 3 AND the same-group (A)
    # members' LOO stats (their peer set genuinely loses col 3's value),
    # but group-B/C columns (>= 5) must be bit-identical.
    rng = np.random.default_rng(36)
    x = pd.DataFrame(rng.normal(size=(8, 9)), columns=[f"s{i}" for i in range(9)])
    _, g = _panel(rows=8, cols=9, seed=37)
    g2 = g.copy()
    g2.iloc[:, 3] = "Z"  # column 3 becomes its own group -> zero peers
    a = _op(name).calculate(x, g, min_peers=3)
    b = _op(name).calculate(x, g2, min_peers=3)
    for j in range(5, 9):
        assert _nan_aware_equal(a.to_numpy()[:, j], b.to_numpy()[:, j]), (name, j)
    assert not _nan_aware_equal(a.to_numpy()[:, 3], b.to_numpy()[:, 3]), name


# ---------------------------------------------------------------------------
# governance: ParamSpec / runtime guard / promotion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_spec_and_runtime_guard_equality(name):
    specs = _op(name).metadata.param_specs
    assert "min_peers" in specs, name
    spec = specs["min_peers"]
    assert spec.dtype is int, name
    assert spec.min == 3, name
    assert spec.param_role is not None, name
    tags = _op(name).metadata.tags
    assert "causal" in tags and "pit_safe" in tags, name
    # runtime guard EQUALS spec min: min_peers=3 passes, 2 raises
    x, g = _panel(rows=6, cols=9, seed=38)
    _op(name).calculate(x, g, min_peers=3)  # no raise
    with pytest.raises(ValueError):
        _op(name).calculate(x, g, min_peers=2)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_neighbor_canonicals_untouched(name):
    # duplicate-skip assertions naming the audited neighbors: each neighbor
    # keeps its own registration; the landed names never shadow them and
    # the landed names were NOT any neighbor's canonical
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for nb in _AUDITED_NEIGHBORS:
        # neighbor remains registered under some backend (registry-wide,
        # production-or-any mode)
        found = any(
            OperatorRegistry.get(nb, b) is not None for b in ("pandas_numpy", "polars")
        ) or OperatorRegistry.get(nb, mode="any") is not None
        assert found, f"audited neighbor {nb} disappeared"
    assert name not in _AUDITED_NEIGHBORS


def test_group_zscore_zero_fill_separated():
    # the nearest self-INCLUSIVE neighbor zero-fills degenerate std; our
    # LOO z refuses to — documented differentiation, pinned here so a
    # future refactor cannot silently converge the two contracts.
    # Review P2: also invoke the neighbor itself on the same degenerate
    # input, so the test pins BOTH sides of the separated contracts (the
    # old version asserted only our NaN side and was vacuous w.r.t. the
    # neighbor).
    vals = np.tile(np.array([2.0] * 5 + [1.0, 2.0, 3.0, 4.0]), (3, 1))
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(9)])
    _, g = _panel(rows=3, cols=9, seed=39)
    out = _op("ex_self_zscore").calculate(x, g, min_peers=3).to_numpy()
    assert np.isnan(out[:, :5]).all()  # identical peers -> LOO std 0 -> NaN
    # neighbor side: group_zscore (polars native) zero-fills the same
    # degenerate group (group_batch1.py otherwise(0.0)) — genuinely a
    # different contract, not an untested assumption.
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.panel_polars import panel_to_polars

    nb = OperatorRegistry.get("group_zscore", "polars")
    if nb is None:  # pragma: no cover - registry wiring regression guard
        pytest.fail("group_zscore polars native not registered")
    # whichever polars registration answers (native group_batch1 or the
    # common/group_polars row wrapper), the group panel must be numeric
    # for the wrapper path — factorize the labels.
    codes, _ = pd.factorize(g.to_numpy().ravel())
    g_num = pd.DataFrame(codes.reshape(g.shape).astype(float),
                         index=g.index, columns=g.columns)
    nb_out = nb.calculate(panel_to_polars(x), panel_to_polars(g_num))
    nb_arr = np.asarray(nb_out.to_pandas(), dtype=float)
    np.testing.assert_allclose(nb_arr[:, :5], 0.0, rtol=1e-12)
    assert np.isfinite(nb_arr[:, :5]).all()  # zero-FILLED, not NaN


def test_zscore_high_level_low_dispersion():
    # Review P1: the running-sum ss identity (S2 - S^2/m) was
    # catastrophically cancellative at |mean|/sigma ~ 1e9+ — it emitted a
    # finite-but-wrong z (or NaN by rounding sign).  The exact two-pass
    # peer variance must recover the oracle z at high level / low
    # dispersion.
    level, spread, n = 1e9, 1e-6, 6
    vals = np.tile(level + spread * np.arange(n, dtype=float), (3, 1))
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(n)])
    g = pd.DataFrame(np.tile(np.array(["A"] * n, dtype=object), (3, 1)),
                     index=x.index, columns=x.columns)
    out = _op("ex_self_zscore").calculate(x, g, min_peers=3).to_numpy()
    exp = _oracle_stat(x, g, 3, "zscore").to_numpy()
    finite = np.isfinite(exp)
    assert finite.any()
    np.testing.assert_allclose(
        out[finite], exp[finite], rtol=1e-6, atol=1e-9
    )


def test_zscore_overflow_peers_fail_closed_never_zero():
    # Review P2: peers near the float max made var overflow.  With the
    # running-sum kernel var=inf gave z = x/inf = exactly 0.0 — a finite
    # zero emission disclaimed by the never-0 contract.  The two-pass
    # kernel detects the non-finite variance and fails closed to NaN.
    vals = np.tile(
        np.array([1e200, -1e200, 1e200, -1e200, 1.0, 1e200]), (2, 1)
    )
    x = pd.DataFrame(vals, columns=[f"s{i}" for i in range(6)])
    g = pd.DataFrame(np.tile(np.array(["A"] * 6, dtype=object), (2, 1)),
                     index=x.index, columns=x.columns)
    out = _op("ex_self_zscore").calculate(x, g, min_peers=3).to_numpy()
    # the 1.0 self among huge peers must be NaN (var overflows), never 0.0
    assert np.isnan(out[:, 4]).all()
