# -*- coding: utf-8 -*-
"""Four-layer factor semantic deduplication signatures (round-7 P0).

AlphaMiner / AlphaProbe deduplication by string AST alone is too weak: two
expressions that parse differently can be numerically identical after algebra or
rank.  This module provides the four signatures the review asked for:

1. **AST hash**         — structural hash of the parsed expression tree
   (the historical string-level dedup, kept for cheap exact filtering).
2. **Algebraic canonical** — the operator-expression DAG canonicalised by
   commutative reordering / numeric folding / sign rewriting where the backend
   can prove equivalence (best effort; conservative = never merge two factors
   that are actually distinct).
3. **Numeric signature**   — hash of the exact output values on a fixed probe
   panel: two factors that emit bit-identical outputs are the same factor.
4. **Rank signature**      — hash of the per-column cross-sectional rank:
   ``rank(x) == rank(10*x) == rank(log(x))`` for positive ``x``, so scale /
   monotone transforms are treated as one *search* factor.

The rank signature is the strongest practical signal for "does this count as a
new factor for search" and is deliberately computed on a fixed synthetic probe
panel (deterministic RNG) so the signature is stable across machines.

The dedup policy is **conservative**: a match at ANY layer marks the pair as a
candidate duplicate for search purposes, but the pair is never hard-merged —
downstream fitness may still prefer one over the other (complexity, tradability).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

_PROBE_ROWS = 240
_PROBE_COLS = 8
_PROBE_SEED = 20260808


def probe_panel(seed: int = _PROBE_SEED) -> pd.DataFrame:
    """Deterministic synthetic panel for numeric / rank signatures."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=_PROBE_ROWS, freq="D")
    return pd.DataFrame(rng.normal(size=(_PROBE_ROWS, _PROBE_COLS)), index=idx, columns=[f"C{i}" for i in range(_PROBE_COLS)])


def _hash_columns(frame: pd.DataFrame) -> str:
    import hashlib

    # NaN-invariant, order-stable column hash of the numeric payload.
    arr = np.nan_to_num(frame.to_numpy(dtype=float), nan=-1e300)
    payload = arr.round(9).tobytes()
    return hashlib.sha256(payload).hexdigest()[:20]


def _hash_columns_rank(frame: pd.DataFrame) -> str:
    import hashlib

    ranks = frame.rank(axis=0, method="average", pct=True)
    arr = np.nan_to_num(ranks.to_numpy(dtype=float), nan=-1.0)
    return hashlib.sha256(arr.round(6).tobytes()).hexdigest()[:20]


def numeric_signature(compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None) -> str:
    """Exact-output signature of ``compute`` on the (fixed by default) probe panel."""
    return _hash_columns(compute(probe_panel() if panel is None else panel))


def rank_signature(compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None) -> str:
    """Cross-sectional rank signature (scale / monotone invariant).

    ``panel`` may be supplied for transforms that are only defined on a
    restricted domain (e.g. ``log`` needs strictly positive values); the default
    probe panel is standard-normal (mixed sign).
    """
    return _hash_columns_rank(compute(probe_panel() if panel is None else panel))


def factor_signatures(compute: Callable[[pd.DataFrame], pd.DataFrame]) -> dict[str, str]:
    """All four signatures for one factor compute function.

    ``ast_hash`` is left to the caller (it needs the expression AST); this
    helper fills the numeric and rank layers plus an algebraic placeholder
    that is conservative (empty string = "no proven algebraic equivalence").
    """
    return {
        "ast_hash": "",
        "algebraic_canonical": "",
        "numeric_signature": numeric_signature(compute),
        "rank_signature": rank_signature(compute),
    }


def dedup_bucket(compute: Callable[[pd.DataFrame], pd.DataFrame], *, panel: pd.DataFrame | None = None) -> str:
    """Primary dedup key for search: the rank signature.

    Two factors landing in the same bucket are indistinguishable by
    cross-sectional ordering on the probe panel — treating them as separate
    alpha hypotheses only wastes search budget on identical hypotheses.
    """
    return rank_signature(compute, panel=panel)


def are_rank_duplicates(
    fa: Callable[[pd.DataFrame], pd.DataFrame],
    fb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    panel: pd.DataFrame | None = None,
    rho_threshold: float = 0.99999,
) -> bool:
    """High-rank-correlation dedup decision for a pair of factor computes."""
    probe = probe_panel() if panel is None else panel
    ra = fa(probe).rank(axis=0, method="average")
    rb = fb(probe).rank(axis=0, method="average")
    common = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(rb.to_numpy(dtype=float))
    if int(common.sum()) < 50:
        return False
    r = np.corrcoef(ra.to_numpy(dtype=float)[common], rb.to_numpy(dtype=float)[common])[0, 1]
    return bool(np.isfinite(r) and abs(r) >= rho_threshold)


# Backward-compatible import surface.
dedup_bucket_key = dedup_bucket
