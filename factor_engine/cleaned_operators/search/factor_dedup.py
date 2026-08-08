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
    fixtures["gaps"] = frame(gaps)

    fixtures["positive_only"] = frame(np.abs(rng.normal(size=(rows, cols))) + 1e-3)

    ev_mask = rng.random(size=(rows, cols)) < 0.02
    ev = np.zeros((rows, cols))
    ev[ev_mask] = rng.normal(size=int(ev_mask.sum()))
    fixtures["event_mask"] = frame(ev)

    fixtures["group"] = frame(np.tile(np.arange(cols) % 10, (rows, 1)).astype(float))

    n_blocks = max(1, cols // 4)
    close = np.exp(np.cumsum(rng.normal(scale=0.01, size=(rows, n_blocks)), axis=0))
    open_ = np.vstack([close[0:1] * 0.99, close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=(rows, n_blocks)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=(rows, n_blocks)))
    ohlc = np.empty((rows, cols))
    for b in range(n_blocks):
        ohlc[:, 4 * b] = open_[:, b]
        ohlc[:, 4 * b + 1] = high[:, b]
        ohlc[:, 4 * b + 2] = low[:, b]
        ohlc[:, 4 * b + 3] = close[:, b]
    fixtures["ohlc"] = frame(ohlc)

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
    """

    sign_invariant: bool = False

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
def factor_signatures(compute: Callable[[pd.DataFrame], pd.DataFrame]) -> dict[str, str]:
    """Numeric + rank signatures for one factor compute function.

    The algebraic-canonical layer is intentionally ABSENT (review #304): there
    is no proven algebraic-equivalence canonicalization here, so the dedup
    contract is conservative and relies on the numeric and rank layers.
    ``ast_hash`` is left to the caller (it needs the expression AST).
    """
    return {
        "ast_hash": "",
        "numeric_signature": numeric_signature(compute),
        "rank_signature": rank_signature(compute),
    }


def dedup_bucket(
    compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None
) -> str:
    """Primary dedup key for search: the cross-sectional rank signature.

    Two factors landing in the same bucket are indistinguishable by
    cross-sectional ordering on the probe panel — treating them as separate
    alpha hypotheses only wastes search budget on identical hypotheses.
    """
    return rank_signature(compute, panel=panel)


def _rank_corr(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    probe: pd.DataFrame,
    rho_threshold: float,
) -> bool:
    """True when the cross-sectional ranks of ``fa``/``fb`` correlate >= threshold.

    Only **positive** correlation counts (``r >= threshold``): a perfect
    anti-correlation (``r == -1``, e.g. ``x`` vs ``-x``) is NOT a duplicate
    under the default sign-sensitive policy.  Sign-invariant dedup is handled
    by :func:`are_rank_duplicates` re-checking the negated pair.
    """
    ra = fa(probe).rank(axis=1, method="average").to_numpy(dtype=float)
    rb = fb(probe).rank(axis=1, method="average").to_numpy(dtype=float)
    common = np.isfinite(ra) & np.isfinite(rb)
    if int(common.sum()) < 50:
        return False
    r = np.corrcoef(ra[common], rb[common])[0, 1]
    return bool(np.isfinite(r) and r >= rho_threshold)


def are_rank_duplicates(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    panel: pd.DataFrame | None = None,
    rho_threshold: float = 0.99999,
    policy: DedupPolicy = DedupPolicy(sign_invariant=False),
) -> bool:
    """High-rank-correlation dedup decision for a pair of factor computes.

    Ranks are computed **cross-sectionally** (``rank(axis=1)`` — rank each
    row's columns, i.e. across instruments per day), the A-share convention
    (review #300).

    With ``policy.sign_invariant=True``, ``x`` and ``-x`` are considered
    duplicates; with the default ``DedupPolicy(sign_invariant=False)`` they
    are not (review #306).
    """
    probe = probe_panel() if panel is None else panel
    if policy.sign_invariant:
        return any(
            _rank_corr(ga, gb, probe, rho_threshold)
            for ga, gb in (
                (fa, fb),
                (fa, lambda p: -fb(p)),
                (lambda p: -fa(p), fb),
            )
        )
    return _rank_corr(fa, fb, probe, rho_threshold)


# Backward-compatible import surface.
dedup_bucket_key = dedup_bucket
