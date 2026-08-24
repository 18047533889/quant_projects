# -*- coding: utf-8 -*-
"""R20-EXSELF-CS-DIRECTUSE: intra ex-self (leave-one-out cross-sectional)
causal dimensionless Direct Alpha canonicals.

Four canonicals, all purely cross-sectional per trading row (no time
leakage by construction — row t reads row t only):

* ``ex_self_mean_gap``  — x minus the leave-one-out group mean
                         (unit-bearing gap, kept for the gap family; the
                         peers-excluded mean is the CAUSAL benchmark: the
                         standard ``cs_demean`` includes x itself).
* ``ex_self_zscore``    — (x - LOO mean) / LOO std (ddof=0), dimensionless.
                         Exact two-pass peer variance (review P1) — the
                         running-sum identity is not used (catastrophic
                         cancellation on high-level / low-dispersion
                         groups).
* ``ex_self_rank_pct``  — LOO rank percentile of self among its finite
                         in-group peers (average rank / peer count,
                         excluding self from both numerator and
                         denominator), bounded in [0, 1].
* ``ex_self_mad_z``     — (x - LOO median) / LOO MAD, dimensionless robust z
                         (NO 1.4826 consistency constant — raw MAD units,
                         matching ``group_peer_deviation_index``).

Duplicate audit (survey BEFORE implementing — documented skips + neighbors):

SURVEY of existing ex-self family (grep across cleaned_operators/):
* ``group_ex_self_mean`` (group_ext.py; polars twins in
  common/polars_group.py, common/polars_group_advanced.py,
  polars_native/group_batch1.py, auto_polars_all.py) — the LOO group MEAN
  LEVEL.  Not a direct alpha: output carries x's unit (R11 #135 comment in
  group_ext.py says exactly this).  ``ex_self_mean_gap`` below is the
  x-minus-LOO-mean DIRECT-USE gap (a different estimator: level vs gap,
  and the level is NOT terminal-usable); z/rank/mad_z variants do not exist.
* ``group_ex_self_std`` (alpha_language_cross.py ~L316;
  polars_native/group_batch1.py L207; common/polars_group_advanced.py L111)
  — the LOO group STD LEVEL (pandas variant is exact LOO; the polars
  variants use the full-group std as an approximate proxy — documented in
  their source).  Again a dispersion LEVEL, not a standardized z.
* ``group_ex_self_mad`` (alpha_language_cross.py ~L347; same polars twins)
  — the LOO group MAD LEVEL.  Not (x-median)/MAD.
* ``group_ex_self_quantile`` (alpha_language_cross.py ~L388) — LOO
  quantile level.
* ``group_ex_self_weighted_mean`` (group_ext.py L119, relation/ops.py L540,
  polars twins) — LOO weighted mean level.
* ``relation_weighted_std_ex_self`` (alpha_language_cross.py L431) — LOO
  node-weighted std level.
* ``group_zscore`` (polars_native/group_batch1.py L929) — FULL-GROUP
  z-score INCLUDING self in mean and std, and it zero-fills a degenerate
  std (``otherwise(0.0)``) — exactly the self-inclusion bias this family
  removes and the silent-zero this family refuses.
* ``cs_zscore`` (common/polars_cs_basic.py L90) — whole-cross-section
  z-score including self, no group dimension, no min_peers guard.
* ``cs_rank`` (common/polars_cs_basic.py L59) — whole-cross-section rank
  INCLUDING self in the rank and the denominator; no group dimension, no
  min_peers guard, no LOO.
* ``cs_demean`` (common/polars_cs_basic.py L123) — whole-cross-section
  demean including self.
* ``group_neutralize`` (polars_native/group_batch1.py L418) — subtract
  full-group mean (self included).
* ``group_peer_deviation_index`` (common/polars_group_advanced.py L815) —
  (x - FULL-GROUP median) / FULL-GROUP MAD, self included on both sides.
  Nearest neighbor to ``ex_self_mad_z``; the LOO variant differs and
  existed nowhere.
* ``group_peer_beta_deviation`` (cross_section/peer_ops.py L108) —
  x - LOO WEIGHTED mean for beta panels (weighted, beta-specific, unit
  gap; not a dimensionless z/rank/pct, no LOO std/mad/rank statistics).
* ``cs_knn_peer_mean_ex_self`` (dynamic_knn.py L247) — kNN-similarity
  peer mean (similarity-weighted neighborhood, not a group label; level,
  not standardized).
* intra ``*_ex_self`` intraday canonicals (intraday/realized_beta.py,
  intra_state_space.py, time_structure_v2.py: intra_realized_beta_ex_self,
  intra_idiosyncratic_variance_ex_self etc.) — intraday market-model
  estimators, different domain and math.

DUPLICATE-AUDIT VERDICT: none of ex_self_mean_gap / ex_self_zscore /
ex_self_rank_pct / ex_self_mad_z existed as an exact semantic equivalent;
all four landed.  Nearest neighbors ``group_zscore`` / ``cs_zscore`` /
``cs_rank`` / ``group_peer_deviation_index`` all INCLUDE self (and
``group_zscore`` zero-fills degenerate std) — the leave-one-out variants
are the genuinely missing "intra ex-self" P0 family.

Family contract (all four):
* Purely cross-sectional per row t: output row t depends ONLY on input
  row t (pinned by a mutation test — a mutation at (t, col) leaves every
  other row t' and — per the shared cross-section semantics — affects
  only column `col` at row t; columns are independent cross-sections).
* Fail-closed peers: NaN / ±Inf peer values are EXCLUDED from the LOO
  statistics, but a NaN/±Inf SELF value propagates to NaN output.
* min_peers: require at least min_peers (>= 3 by contract) FINITE
  in-group peers EXCLUDING self, else NaN — a group too small to
  standardize against is undefined, never zero.
* Degenerate denominators (LOO std == 0, LOO MAD == 0) -> NaN, never 0.
* LOO mean computed stably as (n*mean - x)/(n-1) via running group sums
  (never a re-scan per member).
* LOO std computed as an EXACT TWO-PASS over the peer set (review P1):
  peer mean first, then ss = sum((p - mean)^2) over the n-1 peers,
  var = ss/(n-1), ddof=0 (population std of the peer set, matching
  np.std default and the cs_zscore convention).  The running-sum identity
  ss = S2 - S^2/m is algebraically equal but catastrophically
  cancellative on high-level / low-dispersion groups (error ~eps*S2 can
  exceed the true ss — finite-but-wrong z or NaN by rounding sign); it is
  deliberately NOT used.  Two-pass ss is non-negative by construction
  (no clamp needed) and an overflowed ss (|peer| ~ 1e200+) fails closed
  to NaN instead of emitting z = x/inf = 0.0.

Known numeric limitations (documented, deliberate):
* The absolute dispersion floor _EPS=1e-12 on var (and on LOO MAD) is
  scale-dependent: a group with peer std < ~1e-6 (or MAD < 1e-12)
  yields NaN even when well-posed — panels should be pre-scaled to O(1)
  (returns / standardized factors), which is the intended DirectUse
  input domain.
* LOO rank percentile: average rank of x among the n-1 finite peers
  (strictly-lower count + 0.5*ties, self excluded), divided by (n-1).
  Ties with self's own value among peers count as half — the standard
  average-rank convention.  With zero peers it is NaN (min_peers guard).
* All outputs dimensionless except ``ex_self_mean_gap`` (unit of x — a
  deliberate gap canonical; registered under the gap family, NOT claimed
  as a bounded ratio).
* ``ex_self_rank_pct`` output in [0, 1].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.alignment import align_panel_inputs  # noqa: F401  (used via _aligned)

_EPS = 1e-12


def _pi(v: Any, name: str = "min_peers") -> int:
    """Runtime guard for min_peers: strict integer >= 3 (fail loudly)."""
    if isinstance(v, bool):
        raise ValueError(f"{name} must be an integer >= 3")
    try:
        iv = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 3") from exc
    if iv != v:
        # reject 3.5-style silent truncation (ParamSpec dtype=int mirrors this)
        raise ValueError(f"{name} must be an integer >= 3")
    if iv < 3:
        raise ValueError(f"{name} must be >= 3")
    return iv


def _aligned(x: pd.DataFrame, group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return align_panel_inputs(x, group, strict_axes=True, names=["x", "group"])


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _group_sums_row(x_row: np.ndarray, g_row: np.ndarray, finite: np.ndarray):
    """Per-label (sum, sumsq, count) over FINITE members only.

    Returns {label: (S, S2, n)} for a single cross-sectional row.
    """
    sums: dict[Any, tuple[float, float, int]] = {}
    for j in np.flatnonzero(finite):
        label = g_row[j]
        v = float(x_row[j])
        if label in sums:
            S, S2, n = sums[label]
            sums[label] = (S + v, S2 + v * v, n + 1)
        else:
            sums[label] = (v, v * v, 1)
    return sums


def _finite_indices(g_row: np.ndarray, label: Any, finite: np.ndarray) -> np.ndarray:
    return np.flatnonzero((g_row == label) & finite)


_MIN_PEERS_SPEC = ParamSpec(
    dtype=int, min=3, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY
)


def _meta(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="cross_section",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_section", "group_neutralization", "daily", "pit_safe", "causal",
            "typed_v2", f"signature:{','.join(params)}->series",
            f"unit:{unit}", "cost:1",
        ],
        param_specs={"min_peers": _MIN_PEERS_SPEC},
    )


def _register(name: str, description: str, params: list[str], fn, *, unit: str) -> None:
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"ExSelfCsV1_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="cross_section",
        business_category="group_neutralization",
        canonical=name,
        source="technical_exself_cs_v1",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    # same extended-surface contract as cross_section/peer_ops.py ``_mk``:
    # the module owns its EXTENDED_ONLY_CANONICALS entries at import time.
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def ex_self_mean_gap(x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    """x minus the leave-one-out group mean (peer-mean gap).

    gap_i = x_i - (S_g - x_i) / (n_g - 1), where S_g / n_g are the finite
    in-group sum / count at row t.  Self NaN/±Inf -> NaN; fewer than
    min_peers finite in-group peers EXCLUDING self -> NaN.  Output carries
    x's unit (documented; the dimensionless standardized forms are the
    zscore/rank/mad_z canonicals below)."""
    minp = _pi(min_peers)
    x, group = _aligned(x, group)
    xv = x.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        sums = _group_sums_row(xv[row], gv[row], finite[row])
        g_row = gv[row]
        for j in range(cols):
            if not finite[row, j]:
                continue  # self NaN/±Inf -> NaN (fail-closed)
            label = g_row[j]
            if label not in sums:
                continue
            S, _S2, n = sums[label]
            m = n - 1  # finite peers excluding self
            if m < minp:
                continue
            loo_mean = (S - float(xv[row, j])) / float(m)
            out[row, j] = float(xv[row, j]) - loo_mean
    return _frame_like(x, out)


def ex_self_zscore(x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    """(x - LOO mean) / LOO std — leave-one-out cross-sectional z-score.

    LOO mean = (S - x)/(n-1).  LOO std (ddof=0 over the n-1 peers) via an
    EXACT TWO-PASS over the peer set (review P1): peer mean first, then
    ss = sum((p - mean)^2), var = ss/(n-1).  The running-sum identity
    ss = S2 - S^2/m is algebraically equal but catastrophically
    cancellative on high-level / low-dispersion groups (error ~eps*S2 can
    exceed the true ss — emitting a finite-but-wrong z, or NaN by
    rounding sign) and is NOT used.  Two-pass ss is non-negative by
    construction; an overflowed ss (~|peer|^2 > float max) fails closed
    to NaN rather than emitting z = x/inf = 0.0.
    Self NaN/±Inf -> NaN; peers < min_peers -> NaN; LOO std == 0
    (degenerate dispersion) -> NaN, never 0."""
    minp = _pi(min_peers)
    x, group = _aligned(x, group)
    xv = x.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        g_row = gv[row]
        for j in range(cols):
            if not finite[row, j]:
                continue
            label = g_row[j]
            members = _finite_indices(g_row, label, finite[row])
            others = members[members != j]
            if others.size < minp:
                continue
            v = float(xv[row, j])
            peers = xv[row, others]
            # pass 1: peer mean; pass 2: centered sum of squares —
            # non-negative by construction (no cancellation, no clamp).
            # |peer| ~ 1e200+ overflows ss to inf -> fail closed below
            # rather than emitting z = x/inf = 0.0 (review P2).
            loo_mean = float(peers.mean())
            # overflow-to-inf is the DETECTED fail-closed case — compute
            # under errstate so the deliberate probe is not a warning.
            with np.errstate(over="ignore"):
                var = float(np.sum((peers - loo_mean) ** 2)) / float(others.size)
            if not np.isfinite(var):
                continue  # overflowed dispersion — fail closed, never 0.0
            if var <= _EPS:
                continue  # degenerate dispersion -> NaN, never 0
            out[row, j] = (v - loo_mean) / float(np.sqrt(var))
    return _frame_like(x, out)


def ex_self_rank_pct(x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    """Leave-one-out rank percentile of self among its finite in-group peers.

    With peers P = finite in-group values excluding self (|P| = m):
        rank = |{p in P : p < x}| + 0.5 * |{p in P : p == x}|
        pct = rank / m   (in [0, 1]; 0 = strictly below all peers,
                          1 = strictly above all peers, 0.5 = at the peer
                          median, ties split halfway — average-rank
                          convention with SELF EXCLUDED from both the
                          numerator and the denominator).
    Self NaN/±Inf -> NaN; m < min_peers -> NaN.  Output in [0, 1]."""
    minp = _pi(min_peers)
    x, group = _aligned(x, group)
    xv = x.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        g_row = gv[row]
        for j in range(cols):
            if not finite[row, j]:
                continue
            label = g_row[j]
            peers = _finite_indices(g_row, label, finite[row])
            m = peers.size - 1  # excluding self
            if m < minp:
                continue
            v = float(xv[row, j])
            peer_vals = xv[row, peers[peers != j]]
            below = float(np.count_nonzero(peer_vals < v))
            ties = float(np.count_nonzero(peer_vals == v))
            out[row, j] = (below + 0.5 * ties) / float(m)
    return _frame_like(x, out)


def ex_self_mad_z(x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    """(x - LOO median) / LOO MAD — robust leave-one-out z.

    LOO median / MAD computed over the finite in-group peers EXCLUDING
    self (exact O(m log m) per member — the peer set genuinely changes
    per member, no running-sum shortcut exists for medians).  MAD in RAW
    units (NO 1.4826 consistency constant — same convention as
    ``group_peer_deviation_index``).  Self NaN/±Inf -> NaN; peers <
    min_peers -> NaN; LOO MAD == 0 -> NaN, never 0."""
    minp = _pi(min_peers)
    x, group = _aligned(x, group)
    xv = x.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        g_row = gv[row]
        for j in range(cols):
            if not finite[row, j]:
                continue
            label = g_row[j]
            peers = _finite_indices(g_row, label, finite[row])
            others = peers[peers != j]
            if others.size < minp:
                continue
            v = float(xv[row, j])
            vals = xv[row, others]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            if mad <= _EPS:
                continue  # degenerate robust scale -> NaN, never 0
            out[row, j] = (v - med) / mad
    return _frame_like(x, out)


# ---------------------------------------------------------------------------
# registration (event_state_v2 spec-driven table + single loop idiom)
# ---------------------------------------------------------------------------

_SPECS = [
    (
        "ex_self_mean_gap",
        ["x", "group", "min_peers"],
        ex_self_mean_gap,
        "x - leave-one-out group mean (peer-mean gap); unit of x; self NaN/Inf -> NaN; < min_peers finite peers -> NaN.",
        "same_as:x",
    ),
    (
        "ex_self_zscore",
        ["x", "group", "min_peers"],
        ex_self_zscore,
        "(x - LOO mean) / LOO std (ddof=0 over the n-1 peers, running-sum stable); dimensionless; zero LOO std -> NaN.",
        "ratio",
    ),
    (
        "ex_self_rank_pct",
        ["x", "group", "min_peers"],
        ex_self_rank_pct,
        "LOO average-rank percentile of self among finite in-group peers (self excluded from numerator and denominator); bounded [0,1].",
        "ratio",
    ),
    (
        "ex_self_mad_z",
        ["x", "group", "min_peers"],
        ex_self_mad_z,
        "(x - LOO median) / LOO MAD (raw MAD units, no 1.4826 constant); robust dimensionless z; zero LOO MAD -> NaN.",
        "ratio",
    ),
]

for _name, _params, _fn, _desc, _unit in _SPECS:
    _register(_name, _desc, _params, _fn, unit=_unit)
