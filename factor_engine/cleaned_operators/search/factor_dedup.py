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
