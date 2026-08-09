# -*- coding: utf-8 -*-
"""Factor semantic deduplication signatures (round-7 P0, review §56/57 + WS-G).

AlphaMiner / AlphaProbe deduplication by string AST alone is too weak: two
expressions that parse differently can be numerically identical after a
monotone / rank transform.  This module provides the dedup layers the review
asked for:

1. **AST hash**         — structural hash of the parsed expression tree
   (kept for cheap exact filtering; the caller supplies it because only the
   caller holds the AST).
2. **Numeric signature**   — hash of the exact output values on a fixed probe
   panel: two factors that emit bit-identical outputs are the same factor.
3. **Rank signature**      — hash of the cross-sectional (axis=1) rank:
   ``rank(x) == rank(10*x) == rank(log(x))`` for positive ``x``, so scale /
   monotone transforms are treated as one *search* factor.
4. **Per-kind signatures** — ``signature_for(kind, compute)`` dispatches
   ALPHA / CONDITION / EVENT / GLOBAL_STATE to the signature family that
   matches the factor's semantics.

The **algebraic canonical layer is deliberately ABSENT** (review #304): there
is no proven algebraic-equivalence canonicalization in this module, so the
dedup contract is conservative and relies on the numeric and rank signatures.
Empty / absent algebraic claims are honest — the module never merges two
factors it cannot prove equivalent.

Signatures are deterministic across machines:
* little-endian float64 byte serialization for every payload;
* fixed synthetic probe panel (deterministic RNG);
* NaN handled as an explicit validity mask, never substituted with a sentinel
  (review #303).

The dedup policy is **conservative**: a match at ANY layer marks the pair as a
candidate duplicate for search purposes, but the pair is never hard-merged —
downstream fitness may still prefer one over the other (complexity, tradability).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

_PROBE_ROWS = 240
# review #301: at least 128 instruments so cross-sectional rank (axis=1) is a
# meaningful A-share cross-section, not a handful of columns.
_PROBE_COLS = 128
_PROBE_SEED = 20260808


def probe_panel(seed: int = _PROBE_SEED) -> pd.DataFrame:
    """Deterministic synthetic panel for numeric / rank signatures."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=_PROBE_ROWS, freq="D")
    return pd.DataFrame(
        rng.normal(size=(_PROBE_ROWS, _PROBE_COLS)),
        index=idx,
        columns=[f"C{i}" for i in range(_PROBE_COLS)],
    )


# --------------------------------------------------------------------------- #
# Typed fixture regimes (review #299 / #301)
# --------------------------------------------------------------------------- #
def build_typed_fixtures(
    *, rows: int = _PROBE_ROWS, cols: int = _PROBE_COLS
) -> dict[str, pd.DataFrame]:
    """Typed fixture regimes for audit probes.

    Returns a dict ``name -> (rows x cols)`` panel.  Each fixture models a
    declared semantic kind / regime rather than a single generic normal panel:

    * ``gaussian``      — plain standard-normal returns (base case);
    * ``heavy_tail``    — Student-t(3) returns (fat tails);
    * ``trend``         — strong per-column time trend + noise;
    * ``mean_revert``   — AR(1) with rho = -0.6 (mean reversion);
    * ``ties``          — discrete values (rank ties everywhere);
    * ``gaps``          — ~10% missing observations (NaN gaps);
    * ``positive_only`` — strictly positive values (log / ratio domains);
    * ``event_mask``    — sparse event firings (~2% non-zero);
    * ``group``         — categorical group ids (cross-sectional groups);
    * ``ohlc``          — open/high/low/close blocks (price geometry).

    All fixtures share the same deterministic seed and index/column layout so
    signatures stay stable across machines.
    """
    rng = np.random.default_rng(_PROBE_SEED)
    idx = pd.date_range("2023-01-02", periods=rows, freq="D")
    cols_list = [f"C{i}" for i in range(cols)]

    def frame(mat: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(mat, dtype=float), index=idx, columns=cols_list)

    t = np.arange(rows, dtype=float)[:, None]
    fixtures: dict[str, pd.DataFrame] = {}

    fixtures["gaussian"] = frame(rng.normal(size=(rows, cols)))
    fixtures["heavy_tail"] = frame(rng.standard_t(df=3.0, size=(rows, cols)))
    fixtures["trend"] = frame((t / rows) * 10.0 + rng.normal(scale=0.5, size=(rows, cols)))

    ar = np.zeros((rows, cols))
    for i in range(1, rows):
        ar[i] = -0.6 * ar[i - 1] + rng.normal(size=(1, cols))
    fixtures["mean_revert"] = frame(ar)

    fixtures["ties"] = frame(np.round(rng.normal(size=(rows, cols)), 0))

    gaps = rng.normal(size=(rows, cols))
    gaps[rng.random(size=(rows, cols)) < 0.10] = np.nan
    # R10 #21: "gaps" also models MISSING ROWS — whole no-trading days (suspension /
    # limit-across-the-board), so the per-date path must SKIP them, never invent data.
    if rows > 3:
        gaps[2, :] = np.nan
        gaps[rows - 2, :] = np.nan
    fixtures["gaps"] = frame(gaps)

    fixtures["positive_only"] = frame(np.abs(rng.normal(size=(rows, cols))) + 1e-3)

    # R10-P0-033: an event fixture is NOT a single semantics.  Split it into
    # (a) a boolean event mask, (b) signed event intensity and (c) positive
    # event intensity — an EventBool consumer must not misinterpret Gaussian
    # magnitudes as 0/1, and a signed-intensity consumer must not see only 0/1.
    ev_bool = (rng.random(size=(rows, cols)) < 0.02).astype(float)
    fixtures["event_mask"] = frame(ev_bool)
    fixtures["event_bool"] = frame(ev_bool.copy())
    ev_signed = np.zeros((rows, cols))
    ev_signed[ev_bool == 1.0] = rng.standard_normal(size=int(ev_bool.sum()))
    fixtures["event_signed_intensity"] = frame(ev_signed)
    ev_pos = np.zeros((rows, cols))
    ev_pos[ev_bool == 1.0] = rng.uniform(0.5, 2.0, size=int(ev_bool.sum()))
    fixtures["event_positive_intensity"] = frame(ev_pos)

    fixtures["group"] = frame(np.tile(np.arange(cols) % 10, (rows, 1)).astype(float))

    n_blocks = max(1, cols // 4)
    close = np.exp(np.cumsum(rng.normal(scale=0.01, size=(rows, n_blocks)), axis=0))
    open_ = np.vstack([close[0:1] * 0.99, close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=(rows, n_blocks)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=(rows, n_blocks)))
    # R10-P0-032: initialize with NaN, never np.empty — when ``cols % 4 != 0``
    # the leftover columns used to hold uninitialized garbage.
    ohlc = np.full((rows, cols), np.nan)
    for b in range(n_blocks):
        ohlc[:, 4 * b] = open_[:, b]
        ohlc[:, 4 * b + 1] = high[:, b]
        ohlc[:, 4 * b + 2] = low[:, b]
        ohlc[:, 4 * b + 3] = close[:, b]
    fixtures["ohlc"] = frame(ohlc)

    # R10 #21: A-share realistic trading fixture — random walk with jumps and
    # vol clustering (GARCH(1,1)-style).  Jump-heavy / high-vol days stress the
    # extreme-day behavior that a flattened Gaussian rho hides entirely.
    returns = np.zeros((rows, cols))
    vol = np.full((1, cols), 0.01)
    for i in range(1, rows):
        vol = 0.94 * vol + 0.06 * 0.01 * np.abs(rng.normal(size=(1, cols)))
        shocks = rng.normal(size=(1, cols)) * vol
        jumps = rng.random(size=(1, cols)) < 0.012
        shocks += jumps * rng.standard_normal(size=(1, cols)) * 0.06
        returns[i] = shocks
    price = np.exp(np.cumsum(returns, axis=0))
    fixtures["ashare"] = frame(price)

    return fixtures


# Legacy numeric-control names — scalar knobs for operators that do not declare
# ``panel_params`` / ``input_fields`` and whose kernels are opaque
# (``*args, **kwargs``).  These are treated as scalar parameters; everything
# else is assumed to be a panel/data input.
NUMERIC_CONTROL_PARAMS = frozenset({
    "window", "min_periods", "order", "lag", "periods", "bins",
    "coefficient_index", "max_q", "block", "level", "band", "d", "k", "q",
    "short_periods", "long_periods", "span", "alpha", "lower", "upper",
    "threshold", "min_bin_count", "n_quantiles", "cap", "floor",
    "body_window", "shadow_window", "penetration", "ratio", "pct", "shift",
    "max_iter", "n_iter", "segments", "min_segments", "max_segments",
    "include_current", "left", "right", "up_count", "down_count",
    "n_levels", "n_buckets", "preset", "side",
})


def split_scalar_panel_params(
    all_params: list[str] | None,
    meta: Any,
    canon: str,
) -> tuple[list[str], list[str]]:
    """Split ``all_params`` into ``(scalar_params, panel_params)`` (review #295).

    Panel parameters are the data fields fed as DataFrames (``close``,
    ``volume``, ...); scalar parameters are the numeric / string knobs
    (``window``, ``lag``, ``mode``, ...).  A panel parameter must NEVER be
    sampled as a scalar and a scalar knob must never be fed a DataFrame.

    Resolution order: declared ``panel_params`` -> declared ``input_fields``
    -> OperatorSpec inference (``build_operator_spec``) -> legacy
    numeric-control-name heuristic for opaque (``*args``) kernels.
    """
    all_params = list(all_params or [])
    if not all_params:
        return [], []
    declared_panel = set(getattr(meta, "panel_params", None) or ())
    declared_inputs = set(getattr(meta, "input_fields", None) or ())
    inferred_panel: set[str] = set()
    try:
        from cleaned_operators.operator_spec import build_operator_spec

        spec = build_operator_spec(canon, backend="pandas_numpy")
        if spec is not None and spec.panel_params:
            inferred_panel = set(spec.panel_params)
    except Exception:
        inferred_panel = set()
    panel = declared_panel | declared_inputs | inferred_panel
    if panel:
        # A numeric-control knob (window/lag/...) can never be a PANEL data
        # field, even if a legacy metadata block (mis)lists it in
        # ``input_fields`` — see ts_average_volume / ts_regression_slope.
        panel = {p for p in panel if p in all_params and p not in NUMERIC_CONTROL_PARAMS}
        return [p for p in all_params if p not in panel], [p for p in all_params if p in panel]
    scalar = [p for p in all_params if p in NUMERIC_CONTROL_PARAMS]
    return scalar, [p for p in all_params if p not in scalar]


# --------------------------------------------------------------------------- #
# Deterministic byte hashing
# --------------------------------------------------------------------------- #
def _matrix_bytes(arr: np.ndarray) -> bytes:
    """Little-endian float64 byte serialization (deterministic across machines)."""
    return np.asarray(arr, dtype="<f8").tobytes()


def _hash_matrix(arr: np.ndarray) -> str:
    """Exact, NaN-validity-aware matrix hash (review #302 / #303).

    * NaN placement is hashed through a separate NaN mask — moving a NaN to a
      different cell changes the signature (no ``-1e300`` substitution).
    * Finite values are hashed at full float64 precision — NO ``round(9)``:
      two factors that differ in the 10th decimal hash differently.
    * Deterministic across machines (little-endian float64 serialization).
    """
    arr = np.asarray(arr, dtype="<f8")
    nan_mask = np.isnan(arr)
    h = hashlib.sha256()
    h.update(nan_mask.astype(np.uint8).tobytes())
    h.update(_matrix_bytes(np.where(nan_mask, 0.0, arr)))
    return h.hexdigest()[:20]


def _hash_columns(frame: pd.DataFrame) -> str:
    """Exact numeric hash of a factor output panel."""
    return _hash_matrix(frame.to_numpy(dtype=float))


def _hash_columns_rank(frame: pd.DataFrame, *, axis: int = 1) -> str:
    """Rank hash of a factor output panel.

    ``axis=1`` ranks each ROW's columns — the cross-sectional rank across
    instruments per day (A-share convention).  ``axis=0`` ranks each column's
    time series (per-instrument time rank), used only by the GLOBAL_STATE /
    CONDITION signatures.
    """
    ranks = frame.rank(axis=axis, method="average", pct=True)
    return _hash_matrix(ranks.to_numpy(dtype=float))


# --------------------------------------------------------------------------- #
# Signature families
# --------------------------------------------------------------------------- #
def numeric_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Exact-output signature of ``compute`` on the (fixed by default) probe panel."""
    return _hash_columns(compute(probe_panel() if panel is None else panel))


def rank_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Cross-sectional rank signature (scale / monotone invariant).

    ``panel`` may be supplied for transforms that are only defined on a
    restricted domain (e.g. ``log`` needs strictly positive values).
    """
    return _hash_columns_rank(
        compute(probe_panel() if panel is None else panel), axis=1
    )


def time_series_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Time-series (per-instrument) rank signature — GLOBAL_STATE broadcasts."""
    return _hash_columns_rank(
        compute(probe_panel() if panel is None else panel), axis=0
    )


def condition_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Time-state signature for CONDITION factors.

    A condition is a per-period state; the signature combines the binarised
    state mask (``finite & > 0``) with the time-series rank of the raw output.
    """
    probe = probe_panel() if panel is None else panel
    out = np.asarray(compute(probe).to_numpy(dtype=float))
    state = np.isfinite(out) & (out > 0)
    h = hashlib.sha256()
    h.update(state.astype(np.uint8).tobytes())
    ranks = pd.DataFrame(out).rank(axis=0, method="average", pct=True).to_numpy(dtype=float)
    h.update(_matrix_bytes(ranks))
    return h.hexdigest()[:20]


def event_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Event-timing signature for EVENT factors.

    The signature hashes where (time, instrument) an event fires (non-zero)
    plus the magnitudes at full precision, so both the timing and the size of
    the event matter.
    """
    probe = probe_panel() if panel is None else panel
    out = np.asarray(compute(probe).to_numpy(dtype=float))
    events = np.isfinite(out) & (out != 0)
    h = hashlib.sha256()
    h.update(events.astype(np.uint8).tobytes())
    h.update(_matrix_bytes(np.where(events, out, 0.0)))
    return h.hexdigest()[:20]


class FactorKind:
    """Factor-kind enum for :func:`signature_for` (review #305)."""

    ALPHA = "ALPHA"                # cross-sectional alpha (rank across stocks)
    CONDITION = "CONDITION"        # time-state gate / condition factor
    EVENT = "EVENT"                # event-timing factor
    GLOBAL_STATE = "GLOBAL_STATE"  # time-series broadcast (benchmark, regime)


def signature_for(
    kind: str,
    compute: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    panel: pd.DataFrame | None = None,
) -> str:
    """Per-factor-kind dedup signature dispatch (review #305).

    * ``ALPHA``        → cross-sectional rank signature (axis=1);
    * ``GLOBAL_STATE`` → time-series rank signature (axis=0);
    * ``CONDITION``    → time-state signature;
    * ``EVENT``        → event-timing signature.
    """
    if kind == FactorKind.ALPHA:
        return rank_signature(compute, panel=panel)
    if kind == FactorKind.GLOBAL_STATE:
        return time_series_signature(compute, panel=panel)
    if kind == FactorKind.CONDITION:
        return condition_signature(compute, panel=panel)
    if kind == FactorKind.EVENT:
        return event_signature(compute, panel=panel)
    raise ValueError(f"unknown factor kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Dedup policy (review #306)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DedupPolicy:
    """Explicit dedup policy.

    ``sign_invariant=True`` treats ``x`` and ``-x`` as the same search factor
    (they land in the same dedup bucket); ``False`` (the default) keeps them in
    different buckets.

    ``rank_equivalence=True`` (default) is the *terminal-rank-equivalence*
    policy: two factors are duplicates when they order instruments identically
    each day, so ``x`` and ``2*x`` (identical cross-sectional ranks) MERGE.
    ``rank_equivalence=False`` is the *compositional* policy: duplicates must
    emit the same output VALUES, so ``x`` and ``2*x`` are KEPT DISTINCT
    (R10 #21).
    """

    sign_invariant: bool = False
    rank_equivalence: bool = True

    def in_sign_bucket(
        self,
        compute: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        panel: pd.DataFrame | None = None,
    ) -> str:
        """Dedup bucket key for ``compute`` under this policy.

        Sign-invariant policies evaluate both ``x`` and ``-x`` and return the
        lexicographically smaller rank signature, so sign-flipped factors
        collide.  Sign-sensitive policies return the plain rank signature.
        """
        probe = probe_panel() if panel is None else panel
        sig = rank_signature(compute, panel=probe)
        if not self.sign_invariant:
            return sig
        sig_neg = rank_signature(lambda p: -compute(p), panel=probe)
        return min(sig, sig_neg)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def factor_signatures(
    compute: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    ast_hash: str,
) -> dict[str, str]:
    """Numeric + rank signatures for one factor compute function.

    The algebraic-canonical layer is intentionally ABSENT (review #304): there
    is no proven algebraic-equivalence canonicalization here, so the dedup
    contract is conservative and relies on the numeric and rank layers.

    R10-P0-028: ``ast_hash`` is a REQUIRED, non-empty argument.  The old API
    returned ``"ast_hash": ""`` and expected the caller to fill it in later —
    a caller holding that dict carried an incomplete signature.  An empty or
    missing hash fails at the call boundary instead.
    """
    if not isinstance(ast_hash, str) or not ast_hash.strip():
        raise ValueError(
            "factor_signatures requires a non-empty ast_hash (the caller owns "
            "the expression AST)"
        )
    return {
        "ast_hash": ast_hash,
        "numeric_signature": numeric_signature(compute),
        "rank_signature": rank_signature(compute),
    }


def dedup_bucket(
    compute: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    factor_kind: str,
    panel: pd.DataFrame | None = None,
) -> str:
    """Primary dedup key for search.

    R10-P0-031: ``factor_kind`` is MANDATORY — ALPHA / CONDITION / EVENT /
    GLOBAL_STATE have different equivalence semantics and must never be deduped
    through a single cross-sectional rank signature.

    R10-P0-022: without an explicit ``panel`` the fingerprint spans MULTIPLE
    market regimes (``FactorBehaviorSignature``), so two factors that collide
    on a Gaussian probe but diverge in trend / heavy-tail / ties / missing
    states do NOT share a bucket.  A caller that explicitly pins a ``panel``
    scopes the signature to that single regime.
    """
    if panel is not None:
        key = signature_for(factor_kind, compute, panel=panel)
    else:
        sig = multi_regime_signature(compute, kind=factor_kind)
        payload = json.dumps(sig.to_dict(), sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    # R10 #22: the KIND is part of the dedup GROUP identity.  ALPHA / CONDITION /
    # EVENT / GLOBAL_STATE have different equivalence semantics and must NEVER
    # share a bucket, even when two signatures happen to collide.
    return f"{factor_kind}:{key}"


#: regimes used by the multi-regime fingerprint (R10-P0-022) and the default
#: multi-regime duplicate decision.  ``positive_only`` is excluded by default:
#: it is domain-restricted (log / ratio transforms) and would bias every ALPHA
#: factor to collide on it.
_DEFAULT_MULTI_REGIMES = ("gaussian", "heavy_tail", "trend", "mean_revert", "ties", "gaps")

#: R10 #21: the full multi-regime DUPLICATE DECISION spans every typed fixture —
#: including the domain-restricted (positive_only), event (bool / signed
#: intensity), group / categorical, OHLC and A-share trading regimes — so a pair
#: must agree on the vast majority of regimes (AND days) before being merged.
_RICH_MULTI_REGIMES = (
    "gaussian", "heavy_tail", "trend", "mean_revert", "ties", "gaps",
    "positive_only", "event_bool", "event_signed_intensity", "group", "ohlc",
    "ashare",
)


@dataclass(frozen=True)
class FactorBehaviorSignature:
    """Structured multi-regime dedup fingerprint (R10-P0-022).

    Each field is the per-kind signature of ``compute`` on that typed regime.
    Two factors are dedup candidates only when MOST regimes agree — a pair that
    is indistinguishable under a Gaussian draw but reacts differently to an
    extreme / trend / missing-data regime is genuinely different.
    """

    gaussian: str = ""
    heavy_tail: str = ""
    trend: str = ""
    mean_revert: str = ""
    ties: str = ""
    gaps: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "gaussian": self.gaussian,
            "heavy_tail": self.heavy_tail,
            "trend": self.trend,
            "mean_revert": self.mean_revert,
            "ties": self.ties,
            "gaps": self.gaps,
        }

    def matched_regimes(self, other: "FactorBehaviorSignature") -> tuple[int, int]:
        """Count of regimes whose per-kind signature equals ``other``'s."""
        mine = self.to_dict()
        theirs = other.to_dict()
        matched = sum(1 for key in mine if mine[key] and mine[key] == theirs[key])
        return matched, len([k for k in mine if mine[k]])


def multi_regime_signature(
    compute: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    kind: str = FactorKind.ALPHA,
) -> FactorBehaviorSignature:
    """Build the multi-regime :class:`FactorBehaviorSignature` for ``compute``."""
    fixtures = build_typed_fixtures()
    return FactorBehaviorSignature(
        gaussian=signature_for(kind, compute, panel=fixtures["gaussian"]),
        heavy_tail=signature_for(kind, compute, panel=fixtures["heavy_tail"]),
        trend=signature_for(kind, compute, panel=fixtures["trend"]),
        mean_revert=signature_for(kind, compute, panel=fixtures["mean_revert"]),
        ties=signature_for(kind, compute, panel=fixtures["ties"]),
        gaps=signature_for(kind, compute, panel=fixtures["gaps"]),
    )


def _date_rhos(ra: np.ndarray, rb: np.ndarray, *, min_peers: int) -> list[float]:
    """Per-date Spearman (rank) correlations, NOT one flattened correlation.

    R10-P0-023: flattening every date×stock pair into one rho hides a regime
    where 90% of dates are identical and 10% (the alpha-bearing ones) are the
    opposite.  Returning a per-date series lets the caller require that
    essentially EVERY date be equivalent before deleting a factor.
    """
    rhos: list[float] = []
    for t in range(ra.shape[0]):
        a, b = ra[t], rb[t]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_peers:
            continue
        r = np.corrcoef(a[mask], b[mask])[0, 1]
        if np.isfinite(r):
            rhos.append(float(r))
    return rhos


def _per_date_duplicates(
    ra: np.ndarray,
    rb: np.ndarray,
    *,
    rho_threshold: float,
    min_peers: int,
) -> bool:
    """True only when essentially EVERY date ranks the two factors identically.

    The review's criterion: median rho must be near-exact, the bottom-decile
    rho must still be highly similar, and the large majority of dates must be
    at ``rho > 0.999``.  A pair that is identical on 90% of dates but opposite
    on the 10% extreme dates fails the bottom-decile / near-exact-fraction
    conditions — that 10% is precisely the alpha a cold-start library must keep.
    """
    rhos = _date_rhos(ra, rb, min_peers=min_peers)
    if not rhos:
        return False
    arr = np.asarray(rhos, dtype=float)
    if float(np.median(arr)) < rho_threshold:
        return False
    if float(np.percentile(arr, 10)) < 0.9:
        return False
    if float((arr > 0.999).mean()) < 0.9:
        return False
    return True


# --------------------------------------------------------------------------- #
# R10 #21: per-day metrics + regime-specific similarity
# --------------------------------------------------------------------------- #
def _state_labels(out: np.ndarray, *, scheme: str) -> np.ndarray:
    """Discretise an output panel into state labels for CONDITION / EVENT kinds.

    * ``boolean``     — ``(finite & > 0)`` active mask (0/1);
    * ``sign``        — per-cell sign ``{-1, 0, 1}`` (signed event intensity);
    * ``categorical`` — nearest-integer label (discrete / group state factors).
    """
    out = np.asarray(out, dtype=float)
    if scheme == "boolean":
        return (np.isfinite(out) & (out > 0)).astype(np.int8)
    if scheme == "sign":
        return np.where(np.isnan(out), 0, np.sign(out)).astype(np.int8)
    if scheme == "categorical":
        return np.where(np.isnan(out), 0, np.rint(out)).astype(np.int8)
    raise ValueError(f"unknown state-label scheme: {scheme!r}")


def _position_bucket(v: np.ndarray, k: int) -> np.ndarray:
    """Map a 1-D value vector into ``k`` ordinal position buckets (0..k-1)."""
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return (order * k // len(v)).astype(int)


def _state_agreement_day(av: np.ndarray, bv: np.ndarray) -> float:
    """Jaccard-style state-label agreement for one day (R10 #23).

    The plain per-cell equality rate is dominated by the ubiquitous 0/0 cells
    (an event factor fires on ~2% of cells), so two factors firing on entirely
    different instruments would still report ~98% agreement.  Restricting to the
    union of ACTIVE cells — and requiring equal labels there — scores disjoint
    event sets 0 and identical sets 1.
    """
    active_a = av != 0
    active_b = bv != 0
    union = active_a | active_b
    if not union.any():
        return 1.0
    inter = active_a & active_b
    jac = float(inter.sum()) / float(union.sum())
    if jac == 0.0:
        return 0.0
    agree = float((av[inter] == bv[inter]).mean())
    return float(jac * agree)


@dataclass(frozen=True)
class PerDateMetrics:
    """Per-day similarity of two factor outputs on a single regime.

    * ``spearman``              — Spearman_t: rank correlation across
                                  instruments on that date;
    * ``top_decile_overlap``    — fraction of top-decile members shared;
    * ``bottom_decile_overlap`` — fraction of bottom-decile members shared;
    * ``position_overlap``      — fraction of instruments in the same position
                                  decile bucket;
    * ``state_agreement``       — Jaccard-style state-label agreement (only for
                                  CONDITION / EVENT kinds).
    """

    spearman: float | None
    top_decile_overlap: float
    bottom_decile_overlap: float
    position_overlap: float
    state_agreement: float | None


def per_date_metrics(
    ra: np.ndarray,
    rb: np.ndarray,
    *,
    min_peers: int,
    decile: float = 0.1,
    state_labels: bool = False,
) -> list[PerDateMetrics]:
    """PER-DAY similarity metrics between two output matrices (R10 #21).

    ``ra`` / ``rb`` are date x instrument matrices (ranks for ALPHA /
    GLOBAL_STATE, state labels for CONDITION / EVENT).  One
    :class:`PerDateMetrics` is emitted per day (row); days with fewer than
    ``min_peers`` finite peers are skipped.
    """
    out: list[PerDateMetrics] = []
    for t in range(ra.shape[0]):
        a = np.asarray(ra[t], dtype=float)
        b = np.asarray(rb[t], dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n < min_peers:
            continue
        av = a[mask]
        bv = b[mask]
        state = _state_agreement_day(av, bv) if state_labels else None
        if len(av) > 1:
            sp = float(np.corrcoef(av, bv)[0, 1])
        else:
            sp = 1.0
        if not np.isfinite(sp):
            # constant rows (all-flat / ties days): fall back to equality rate.
            sp = float((av == bv).mean())
        n_top = max(1, int(np.ceil(decile * n)))
        a_top = set(np.argpartition(av, -n_top)[-n_top:].tolist())
        b_top = set(np.argpartition(bv, -n_top)[-n_top:].tolist())
        top_ov = float(len(a_top & b_top) / n_top)
        a_bot = set(np.argpartition(av, n_top)[:n_top].tolist())
        b_bot = set(np.argpartition(bv, n_top)[:n_top].tolist())
        bot_ov = float(len(a_bot & b_bot) / n_top)
        k = max(2, int(round(1.0 / decile)))
        pos_ov = float((_position_bucket(av, k) == _position_bucket(bv, k)).mean())
        out.append(
            PerDateMetrics(
                spearman=sp,
                top_decile_overlap=top_ov,
                bottom_decile_overlap=bot_ov,
                position_overlap=pos_ov,
                state_agreement=state,
            )
        )
    return out


def day_is_duplicate(
    m: PerDateMetrics,
    *,
    rho_threshold: float,
    overlap_threshold: float,
    require_state: bool = False,
) -> bool:
    """Whether a single day's metrics count as a *duplicate day*.

    Rank kinds (ALPHA / GLOBAL_STATE) require near-exact rank correlation AND
    top/bottom/position overlap.  State kinds (CONDITION / EVENT) are gated by
    state-label agreement — value correlation is NOT the deciding signal
    (R10 #23).
    """
    if require_state:
        if m.state_agreement is None or m.state_agreement < overlap_threshold:
            return False
        return True
    if m.spearman is None or m.spearman < rho_threshold:
        return False
    if m.top_decile_overlap < overlap_threshold:
        return False
    if m.bottom_decile_overlap < overlap_threshold:
        return False
    if m.position_overlap < overlap_threshold:
        return False
    return True


@dataclass(frozen=True)
class DayMetricsSummary:
    """Aggregated per-day metrics across one regime.

    Reports median / p10 / p90 Spearman, median overlaps, the duplicate-day
    RATIO and — for state kinds — worst-day state agreement.  The ratio is the
    headline signal: it replaces the old single flattened date×stock rho.
    """

    n_days: int
    n_duplicate_days: int
    duplicate_day_ratio: float
    spearman_median: float | None
    spearman_p10: float | None
    spearman_p90: float | None
    spearman_min: float | None
    top_decile_overlap_median: float
    bottom_decile_overlap_median: float
    position_overlap_median: float
    state_agreement_median: float | None
    state_agreement_min: float | None

    def to_dict(self) -> dict:
        return {
            "n_days": self.n_days,
            "n_duplicate_days": self.n_duplicate_days,
            "duplicate_day_ratio": self.duplicate_day_ratio,
            "spearman_median": self.spearman_median,
            "spearman_p10": self.spearman_p10,
            "spearman_p90": self.spearman_p90,
            "spearman_min": self.spearman_min,
            "top_decile_overlap_median": self.top_decile_overlap_median,
            "bottom_decile_overlap_median": self.bottom_decile_overlap_median,
            "position_overlap_median": self.position_overlap_median,
            "state_agreement_median": self.state_agreement_median,
            "state_agreement_min": self.state_agreement_min,
        }


def aggregate_day_metrics(
    metrics: list[PerDateMetrics],
    *,
    rho_threshold: float,
    overlap_threshold: float,
    require_state: bool = False,
) -> DayMetricsSummary:
    """Aggregate :func:`per_date_metrics` into a :class:`DayMetricsSummary`."""
    n_days = len(metrics)
    if n_days == 0:
        return DayMetricsSummary(
            n_days=0, n_duplicate_days=0, duplicate_day_ratio=0.0,
            spearman_median=None, spearman_p10=None, spearman_p90=None,
            spearman_min=None,
            top_decile_overlap_median=float("nan"),
            bottom_decile_overlap_median=float("nan"),
            position_overlap_median=float("nan"),
            state_agreement_median=None, state_agreement_min=None,
        )
    dup = sum(
        1
        for m in metrics
        if day_is_duplicate(
            m, rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
            require_state=require_state,
        )
    )
    sp = [m.spearman for m in metrics if m.spearman is not None]
    state = [m.state_agreement for m in metrics if m.state_agreement is not None]

    def med(v):
        return float(np.median(v)) if v else None

    def pctl(v, q):
        return float(np.percentile(v, q)) if v else None

    return DayMetricsSummary(
        n_days=n_days,
        n_duplicate_days=dup,
        duplicate_day_ratio=dup / n_days,
        spearman_median=med(sp),
        spearman_p10=pctl(sp, 10),
        spearman_p90=pctl(sp, 90),
        spearman_min=float(np.min(sp)) if sp else None,
        top_decile_overlap_median=med([m.top_decile_overlap for m in metrics]),
        bottom_decile_overlap_median=med([m.bottom_decile_overlap for m in metrics]),
        position_overlap_median=med([m.position_overlap for m in metrics]),
        state_agreement_median=med(state),
        state_agreement_min=float(np.min(state)) if state else None,
    )


def _exact_value_days(ra: np.ndarray, rb: np.ndarray, *, min_peers: int) -> float:
    """Fraction of days whose raw OUTPUTS are bit-identical (compositional policy).

    Both the NaN placement and every finite value must match exactly — the
    compositional policy's definition of "same factor".
    """
    n = 0
    match = 0
    for t in range(ra.shape[0]):
        a = np.asarray(ra[t], dtype=float)
        b = np.asarray(rb[t], dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_peers:
            continue
        n += 1
        if bool((np.isnan(a) == np.isnan(b)).all()) and np.array_equal(a[mask], b[mask]):
            match += 1
    return float(match / n) if n else 0.0


def _rich_per_date_duplicate(
    ra: np.ndarray,
    rb: np.ndarray,
    *,
    rho_threshold: float,
    overlap_threshold: float,
    min_duplicate_day_ratio: float,
    min_peers: int,
    require_state: bool,
) -> bool:
    """Per-date duplicate decision for one regime (R10 #21).

    Duplicate only when (a) the vast majority of days are duplicate days AND
    (b) the WORST day is still near-identical.  Criterion (b) is what rejects a
    pair that is 99% identical on normal days but opposite on the 1% extreme
    days — the flattened rho would pass, the worst-day check does not.
    """
    metrics = per_date_metrics(
        ra, rb, min_peers=min_peers, state_labels=require_state
    )
    summary = aggregate_day_metrics(
        metrics, rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
        require_state=require_state,
    )
    if summary.n_days == 0:
        return False
    if summary.duplicate_day_ratio < min_duplicate_day_ratio:
        return False
    if require_state:
        if summary.state_agreement_min is None or summary.state_agreement_min < overlap_threshold:
            return False
    else:
        if summary.spearman_min is None or summary.spearman_min < overlap_threshold:
            return False
    return True


#: kinds whose equivalence is defined by STATE labels (R10 #22 / #23).
_STATE_KINDS = frozenset({FactorKind.CONDITION, FactorKind.EVENT})


def _compare_matrix(
    compute: Callable[[pd.DataFrame], pd.DataFrame],
    probe: pd.DataFrame,
    *,
    factor_kind: str,
    rank: bool,
) -> np.ndarray:
    """Matrix compared per-date for one factor kind (R10 #22).

    * ALPHA        → cross-sectional rank (axis=1, dates x instruments);
    * GLOBAL_STATE → time-series rank, transposed (instruments x dates);
    * CONDITION    → boolean state labels;
    * EVENT        → sign state labels.
    """
    out = np.asarray(compute(probe).to_numpy(dtype=float))
    if factor_kind == FactorKind.GLOBAL_STATE:
        if rank:
            return pd.DataFrame(out).rank(axis=0, method="average").to_numpy(dtype=float).T
        return out.T
    if factor_kind == FactorKind.CONDITION:
        return _state_labels(out, scheme="boolean")
    if factor_kind == FactorKind.EVENT:
        return _state_labels(out, scheme="sign")
    if rank:
        return pd.DataFrame(out).rank(axis=1, method="average").to_numpy(dtype=float)
    return out


def _regime_duplicate(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    probe: pd.DataFrame,
    *,
    factor_kind: str,
    rho_threshold: float,
    overlap_threshold: float,
    min_duplicate_day_ratio: float,
    policy: DedupPolicy,
) -> bool:
    require_state = factor_kind in _STATE_KINDS
    if policy.rank_equivalence:
        ra = _compare_matrix(fa, probe, factor_kind=factor_kind, rank=True)
        rb = _compare_matrix(fb, probe, factor_kind=factor_kind, rank=True)
        min_peers = min(20, ra.shape[1])
        dup = _rich_per_date_duplicate(
            ra, rb, rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
            min_duplicate_day_ratio=min_duplicate_day_ratio, min_peers=min_peers,
            require_state=require_state,
        )
        if not dup and policy.sign_invariant and factor_kind not in _STATE_KINDS:
            rb_neg = _compare_matrix(
                lambda p: -fb(p), probe, factor_kind=factor_kind, rank=True
            )
            dup = _rich_per_date_duplicate(
                ra, rb_neg, rho_threshold=rho_threshold,
                overlap_threshold=overlap_threshold,
                min_duplicate_day_ratio=min_duplicate_day_ratio,
                min_peers=min_peers, require_state=require_state,
            )
        return dup
    # Compositional policy (R10 #21): same output VALUES, not just same ranks.
    ra = _compare_matrix(fa, probe, factor_kind=factor_kind, rank=False)
    rb = _compare_matrix(fb, probe, factor_kind=factor_kind, rank=False)
    min_peers = min(20, ra.shape[1])
    exact = _exact_value_days(ra, rb, min_peers=min_peers)
    if exact < min_duplicate_day_ratio and policy.sign_invariant and factor_kind not in _STATE_KINDS:
        exact = max(exact, _exact_value_days(ra, -rb, min_peers=min_peers))
    return exact >= min_duplicate_day_ratio


def _regime_summary(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    probe: pd.DataFrame,
    *,
    factor_kind: str,
    policy: DedupPolicy,
    rho_threshold: float,
    overlap_threshold: float,
    decile: float,
) -> DayMetricsSummary:
    require_state = factor_kind in _STATE_KINDS
    if policy.rank_equivalence:
        ra = _compare_matrix(fa, probe, factor_kind=factor_kind, rank=True)
        rb = _compare_matrix(fb, probe, factor_kind=factor_kind, rank=True)
    else:
        ra = _compare_matrix(fa, probe, factor_kind=factor_kind, rank=False)
        rb = _compare_matrix(fb, probe, factor_kind=factor_kind, rank=False)
    min_peers = min(20, ra.shape[1])
    metrics = per_date_metrics(
        ra, rb, min_peers=min_peers, decile=decile, state_labels=require_state
    )
    return aggregate_day_metrics(
        metrics, rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
        require_state=require_state,
    )


def regime_similarity_report(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    factor_kind: str = FactorKind.ALPHA,
    regimes: tuple[str, ...] | None = None,
    policy: DedupPolicy = DedupPolicy(),
    rho_threshold: float = 0.99999,
    overlap_threshold: float = 0.9,
    decile: float = 0.1,
) -> dict[str, DayMetricsSummary]:
    """Regime-specific per-day similarity for a factor pair (R10 #21).

    Returns ``regime -> DayMetricsSummary`` for every fixture regime, so the
    caller sees WHERE the pair diverges (heavy-tail? gaps? events?) instead of a
    single flattened rho.
    """
    fixtures = build_typed_fixtures()
    return {
        name: _regime_summary(
            fa, fb, fixtures[name], factor_kind=factor_kind, policy=policy,
            rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
            decile=decile,
        )
        for name in (regimes or _RICH_MULTI_REGIMES)
        if name in fixtures
    }


def multi_regime_duplicate_decision(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    factor_kind: str = FactorKind.ALPHA,
    panel: pd.DataFrame | None = None,
    regimes: tuple[str, ...] | None = None,
    policy: DedupPolicy = DedupPolicy(sign_invariant=False),
    min_regime_agreement: float = 0.66,
    rho_threshold: float = 0.99999,
    overlap_threshold: float = 0.9,
    min_duplicate_day_ratio: float = 0.9,
) -> tuple[bool, dict[str, DayMetricsSummary]]:
    """Multi-regime duplicate decision with a human-readable report (R10 #21).

    The pair is declared a duplicate only when a supermajority of regimes each
    declare it a duplicate under :func:`_rich_per_date_duplicate` (majority of
    DAYS AND worst-day still near-identical).  Returns ``(is_duplicate, report)``.
    """
    if panel is not None:
        probes = (("gaussian", panel),)
    else:
        fixtures = build_typed_fixtures()
        probes = tuple(
            (name, fixtures[name])
            for name in (regimes or _RICH_MULTI_REGIMES)
            if name in fixtures
        )
    votes = 0
    total = 0
    report: dict[str, DayMetricsSummary] = {}
    for name, probe in probes:
        total += 1
        dup = _regime_duplicate(
            fa, fb, probe, factor_kind=factor_kind, rho_threshold=rho_threshold,
            overlap_threshold=overlap_threshold,
            min_duplicate_day_ratio=min_duplicate_day_ratio, policy=policy,
        )
        if dup:
            votes += 1
        report[name] = _regime_summary(
            fa, fb, probe, factor_kind=factor_kind, policy=policy,
            rho_threshold=rho_threshold, overlap_threshold=overlap_threshold,
            decile=0.1,
        )
    if total == 0:
        return False, report
    return (votes / total) >= min_regime_agreement, report


def are_rank_duplicates(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    panel: pd.DataFrame | None = None,
    rho_threshold: float = 0.99999,
    policy: DedupPolicy = DedupPolicy(sign_invariant=False),
    min_regime_agreement: float = 0.66,
    regimes: tuple[str, ...] | None = None,
) -> bool:
    """Multi-regime, per-date dedup decision for a pair of factor computes.

    Ranks are computed **cross-sectionally** (``rank(axis=1)`` — rank each
    row's columns, i.e. across instruments per day), the A-share convention
    (review #300).

    R10-P0-022/023: the default spans the typed market regimes and requires a
    supermajority of regimes to agree PER DATE (never a single flattened
    correlation on one Gaussian panel).  ``x`` vs ``-x`` remains NOT a
    duplicate under ``sign_invariant=False`` and IS one under
    ``sign_invariant=True`` (review #306).

    An explicit ``panel`` scopes the decision to that single regime.
    """
    if panel is not None:
        probes = (("gaussian", panel),)
    else:
        fixtures = build_typed_fixtures()
        probes = tuple(
            (name, fixtures[name])
            for name in (regimes or _DEFAULT_MULTI_REGIMES)
            if name in fixtures
        )
    votes = 0
    total = 0
    for _name, probe in probes:
        total += 1
        ra = fa(probe).rank(axis=1, method="average").to_numpy(dtype=float)
        rb = fb(probe).rank(axis=1, method="average").to_numpy(dtype=float)
        min_peers = min(20, ra.shape[1])
        dup = _per_date_duplicates(ra, rb, rho_threshold=rho_threshold, min_peers=min_peers)
        if not dup and policy.sign_invariant:
            rb_neg = (-fb(probe)).rank(axis=1, method="average").to_numpy(dtype=float)
            dup = _per_date_duplicates(
                ra, rb_neg, rho_threshold=rho_threshold, min_peers=min_peers
            )
        if dup:
            votes += 1
    if total == 0:
        return False
    return (votes / total) >= min_regime_agreement


# Backward-compatible import surface.
dedup_bucket_key = dedup_bucket
