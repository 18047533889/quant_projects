# -*- coding: utf-8 -*-
"""Compositional Data (CoDa / Aitchison) primitives (2026-08-08 Gemini V2 round).

A *composition* is a vector of positive parts that sum to (or describe) a
whole.  The operators operate on aligned panels (date x instrument), one part
per panel, and are all elementwise in the (date, instrument) cell — they never
touch the time axis, so PIT is trivially preserved.

* ``composition_clr_component``      — the centred log-ratio of one component:
  ``clr_i = ln x_i - mean(ln x)`` (P0).
* ``composition_aitchison_distance`` — ``‖clr(A) - clr(B)‖₂`` between two
  same-dimension compositions (P0).
* ``composition_ilr_balance``        — the normalised isometric balance between
  two groups of components ``√(rs/(r+s))·ln(gmean(A)/gmean(B))`` (P0).
* ``composition_entropy``            — Shannon entropy of the closed
  composition ``H = -Σ p_i ln p_i`` (P1).
* ``composition_normalized_entropy`` — ``H / ln(P)``, entropy normalised into
  ``[0, 1]`` by the number of parts ``P`` (3-part max ``ln 3``, 8-part max
  ``ln 8``) (P1-18).
* ``composition_js_divergence``      — Jensen-Shannon divergence between two
  closed compositions (P1).

Contract:

* ``zero_policy`` is fixed to ``"reject"`` — a component that is missing (NaN)
  or non-positive (a structural zero) invalidates the whole composition cell
  and fails closed.  No arbitrary ``1e-6`` imputation is ever applied (a fixed
  additive constant would be meaningless across money magnitudes), and no
  negative flow is silently fed into a log-ratio.
* At least three components are required (the family's 3-8 component surface).
  Aitchison distance / ILR balance / JS divergence take two same-dimension
  compositions (up to 6 components each side).
* ``composition_id`` (optional kernel keyword): all parts of one composition
  must belong to the SAME family — a known field from a different family (or
  mixed known families) is rejected at the call boundary (P0-69).
* Named two-composition pairing: when ``x``/``y`` parts carry explicit field
  names, ``x_i`` must correspond to the SAME named part ``y_i`` before the
  positional pairing is trusted (P0-70); anonymous positional panels keep the
  positional pairing.
* CompositionSchema (P0-74..77): the family is NOT a single broad
  "fundamental" bag.  A composition part must have the same unit, be strictly
  positive, describe one economic whole, and carry an explicit PartId.  Known
  misclassified parts — EPS (per-share unit), MarketCap / Turnover (whole-firm
  valuations, not parts of a whole), raw Volume / Amount (shares vs currency;
  only composable once converted to share-of-total activity), raw OHLC (four
  price statistics, not four parts of a total) — are gated with a clear
  ``ValueError`` at the call boundary.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.gemini_v2_common import (
    _align,
    frame_like,
    register_dual,
    union_extended,
)

_EPS = 1e-12


def _check_zero_policy(zero_policy: str) -> None:
    if str(zero_policy) != "reject":
        raise ValueError("zero_policy must be 'reject' (structural zeros invalidate the cell)")


def _parts(*frames: pd.DataFrame | None) -> list[pd.DataFrame]:
    return [f for f in frames if f is not None]


# ---------------------------------------------------------------------------
# P0-69 CompositionId / P0-70 PartId / P0-74..77 CompositionSchema —
# composition-family identity.
# ---------------------------------------------------------------------------
# A composition is a vector of parts that (a) have the SAME unit, (b) are
# strictly positive, (c) describe ONE economic whole, and (d) carry an explicit
# PartId.  ``_COMPOSITION_SCHEMA_FIELDS`` is the allow-list of *known* fields
# that satisfy that schema (financial-statement money flows — same currency,
# same firm).  Mixing parts from different families (e.g.
# ``Revenue + MarketCap + Volume``) produces an Aitchison geometry that is
# meaningless across money/volume magnitudes, so the operators fail closed at
# the call boundary.
#
# P0-74/75/76/77 — ``_MISCLASSIFIED_PARTS`` is the explicit gate for known
# fields that do NOT satisfy the schema and must never enter a composition as a
# part:
#   * ``eps`` is per-share — a DIFFERENT unit from the money-flow whole, so
#     Revenue/NetIncome/TotalAssets/Liabilities/Equity are NOT one single
#     compositional whole with EPS.
#   * ``market_cap`` / ``turnover`` are valuations / whole-firm flows, not
#     positive parts of one part-whole decomposition.
#   * raw ``volume`` / ``amount`` are shares vs currency — NOT composable with
#     each other (or with money flows); only usable once converted to
#     share-of-total activity (in which case the part carries a *share* field
#     name and is no longer raw ``volume``).
#   * raw OHLC (open/high/low/close/pre_close/vwap) are four price STATISTICS,
#     not four positive parts of a total.
# A named part that is neither in the schema allow-list nor in the
# misclassified gate is an *unknown* field and passes (a caller declaring a
# volume-by-bucket share ``share_of_volume_bucket_1`` or a holder share
# ``holder_share_3`` supplies its own CompositionSchema).
# R26-096..099: the previous ``financial_statement`` "family" is NOT a valid
# part-whole composition schema.  "Same currency + same firm" is not "the same
# whole's mutually-exclusive parts": Revenue and NetIncome are flows of DIFFERENT
# economic magnitudes (not exclusive parts of one whole), and
# ``Assets = Liabilities + Equity`` makes Assets the WHOLE — listing A, L, E as
# three parts double-counts (R26-097).  None of these fields is auto-certified
# as a composition part; a caller with a real part-whole relationship (e.g.
# liability sub-items summing to total liabilities) must supply its own
# explicit CompositionSchema (R26-098/102).
_COMPOSITION_SCHEMA_FIELDS: dict[str, str] = {}

# Known fields that are NOT valid composition parts (P0-74/75/76/77 + R26-096..099).
_MISCLASSIFIED_PARTS: dict[str, str] = {
    # R26-096..099: same-currency same-firm does NOT make mutually-exclusive
    # parts of one whole.  Revenue / NetIncome are different economic flows;
    # ``Assets = Liabilities + Equity`` makes Assets the whole (listing A,L,E as
    # three parts double-counts); net_profit / net_income are aliases of the
    # same economic concept (a duplicate PartId).  None may be auto-certified as
    # a composition part; a genuine part-whole requires an explicit
    # CompositionSchema.
    "revenue": "income-statement flow, not a mutually-exclusive part of one balance-sheet whole (R26-096/097)",
    "operating_revenue": "income-statement flow, not a mutually-exclusive part of one whole (R26-096/097)",
    "net_profit": "income-statement flow; alias of net_income (duplicate economic part, R26-099)",
    "net_income": "income-statement flow; alias of net_profit (duplicate economic part, R26-099)",
    "total_assets": "balance-sheet WHOLE (A = L + E) — listing it as a part double-counts (R26-097)",
    "total_liabilities": "balance-sheet side; not composable with assets/equity as three generic parts (R26-097)",
    "equity": "balance-sheet side; not composable with assets/liabilities as three generic parts (R26-097)",
    "shareholders_equity": "balance-sheet side; not composable with assets/liabilities as three generic parts (R26-097)",
    "eps": "per-share value (different unit from a money-flow whole; P0-74)",
    "market_cap": "market valuation (not a positive part of one economic whole; P0-75)",
    "mkt_cap": "market valuation (not a positive part of one economic whole; P0-75)",
    "free_market_cap": "market valuation (not a positive part of one economic whole; P0-75)",
    "turnover_ratio": "turnover (not a positive part of one economic whole; P0-75)",
    "turnover": "turnover (not a positive part of one economic whole; P0-75)",
    "volume": "raw shares/count — convert to a share-of-total first (P0-76)",
    "vol": "raw shares/count — convert to a share-of-total first (P0-76)",
    "amount": "raw currency turnover — convert to a share-of-total first (P0-76)",
    "amt": "raw currency turnover — convert to a share-of-total first (P0-76)",
    "open": "OHLC price statistic, not a positive part of a total (P0-77)",
    "close": "OHLC price statistic, not a positive part of a total (P0-77)",
    "high": "OHLC price statistic, not a positive part of a total (P0-77)",
    "low": "OHLC price statistic, not a positive part of a total (P0-77)",
    "pre_close": "price statistic, not a positive part of a total (P0-77)",
    "prev_close": "price statistic, not a positive part of a total (P0-77)",
    "vwap": "price statistic, not a positive part of a total (P0-77)",
}


# Recognised field-concept names (schema allow-list ∪ misclassified gate) — the
# only single-column names trusted as PartIds (R26-101).
_PART_CONCEPT_NAMES = frozenset(
    (set(_COMPOSITION_SCHEMA_FIELDS.keys()) | set(_MISCLASSIFIED_PARTS.keys()))
)


def _part_field_name(frame: pd.DataFrame) -> str | None:
    """Best-effort part-field name of one composition panel.

    R26-100/101: a field name is ONLY trusted when it comes from an explicit
    ``.name`` attribute or a single-column frame whose column name is a
    RECOGNISED field concept (in the schema allow-list or the misclassified
    gate).  A standard FE wide panel (rows=date, columns=instrument codes like
    ``000001.SZ`` / ``s0``) must NEVER be misread as a field id — the stock
    column name is not a PartId.  Unrecognised single-column names -> ``None``
    (anonymous, positionally paired, and NOT schema-certified).
    """
    name = getattr(frame, "name", None)
    if isinstance(name, str) and name:
        return name
    if getattr(frame, "shape", (2, 0))[1] == 1:
        col = frame.columns[0]
        if isinstance(col, str) and col and col in _PART_CONCEPT_NAMES:
            return col
    return None


def _same_composition(parts: Sequence[pd.DataFrame], composition_id: str | None = None) -> None:
    """P0-69 + P0-74/75/76/77: all parts of one composition must satisfy the
    CompositionSchema AND belong to the SAME family.

    A named part is checked against the schema FIRST: a *known* field that is
    not composable (EPS, market_cap, turnover, raw volume/amount, raw OHLC) is
    rejected with a clear error regardless of ``composition_id``.  Then the
    P0-69 family consistency is enforced:

    * ``composition_id`` provided -> every part whose field is known must
      belong to that exact family; anonymous / unknown parts are not checkable
      and pass.
    * ``composition_id`` is None -> known-family parts must all agree; a
      disagreement means the caller smuggled parts from different families into
      one composition and the call is rejected at the boundary.
    * no composition_id and no known-family parts -> allowed as before
      (single-family default behaviour).
    """
    seen: set[str] = set()
    for frame in parts:
        field = _part_field_name(frame)
        if field is None:
            continue
        key = str(field).lower()
        if key in _MISCLASSIFIED_PARTS:
            raise ValueError(
                f"composition part {field!r} is not a valid composition part: "
                f"{_MISCLASSIFIED_PARTS[key]}; composition operators accept "
                "ONLY a CompositionSchema — same units, strictly positive, one "
                "economic whole, explicit PartId (P0-74/75/76/77)"
            )
        family = _COMPOSITION_SCHEMA_FIELDS.get(key)
        if family is None:
            # R15-INC-203: a named part that claims an EXPLICIT schema family
            # (``composition_id``) but is not a known member of it is a schema
            # lie — fail closed.  Without a schema (``composition_id=None``) the
            # part is checked positionally and passes (backward compat).  The
            # caller who wants a custom part (e.g. ``share_of_volume_bucket_1``)
            # must supply an explicit CompositionSchema object; the operator
            # never guesses that an unknown name belongs to a declared family.
            if composition_id is not None:
                raise ValueError(
                    f"composition part {field!r} is not a known member of "
                    f"family {composition_id!r}; supply an explicit "
                    "CompositionSchema declaring it, or omit composition_id for "
                    "positional pairing (R15-INC-203)"
                )
            continue
        seen.add(family)
        if composition_id is not None and family != composition_id:
            raise ValueError(
                f"composition part {field!r} belongs to family {family!r} but "
                f"composition_id declares family {composition_id!r}; all parts "
                "of a composition must come from the SAME family (P0-69)"
            )
    if composition_id is None and len(seen) > 1:
        raise ValueError(
            "composition parts come from different families "
            f"({sorted(seen)}) — e.g. Revenue + MarketCap + Volume is not a "
            "valid composition; all parts must share one family (P0-69)"
        )


def _check_part_identity(a: Sequence[pd.DataFrame], b: Sequence[pd.DataFrame]) -> None:
    """P0-70: when parts are NAMED, ``x_i`` must pair with the SAME named part ``y_i``.

    Aitchison distance / JS divergence pair ``x_i <-> y_i`` positionally.  If
    the parts carry explicit field names, trust the names — not the position —
    and require ``PartId(x_i) == PartId(y_i)`` for every ``i``.  Anonymous
    positional panels keep the positional pairing (documented).
    """
    fields_a = [_part_field_name(f) for f in a]
    fields_b = [_part_field_name(f) for f in b]
    if any(f is None for f in fields_a) or any(f is None for f in fields_b):
        return  # anonymous positional panels — positional pairing is the contract
    for i, (fa, fb) in enumerate(zip(fields_a, fields_b)):
        if str(fa).lower() != str(fb).lower():
            raise ValueError(
                f"composition part identity mismatch at position {i}: x part "
                f"{fa!r} pairs with y part {fb!r} — a named x_i must "
                "correspond to the SAME named part in y (P0-70 PartId check)"
            )


def _comp_logs(frames: Sequence[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Stacked logs of aligned parts and the all-finite cell mask."""
    aligned = _align(*frames)
    arrs = [a.to_numpy(dtype=float) for a in aligned]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(a) for a in arrs])  # (P, R, C) — non-positive -> -inf/nan
    valid = np.all(np.isfinite(logs), axis=0)
    return logs, valid


def _close(logs: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Closure of parts [i0, i1) -> probability vector (P, R, C)."""
    sub = logs[i0:i1]
    w = np.exp(sub - sub.max(axis=0, keepdims=True))
    return w / w.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _composition_clr_component(
    target: pd.DataFrame,
    x1: pd.DataFrame | None = None,
    x2: pd.DataFrame | None = None,
    x3: pd.DataFrame | None = None,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    x7: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    parts = _parts(target, x1, x2, x3, x4, x5, x6, x7)
    if len(parts) < 3:
        raise ValueError("composition_clr_component requires >= 3 components")
    _same_composition(parts, composition_id)
    logs, valid = _comp_logs(parts)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    clr = logs[0] - logs.mean(axis=0)
    out[valid] = clr[valid]
    return frame_like(target, out)


def _composition_entropy(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    x7: pd.DataFrame | None = None,
    x8: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    parts = _parts(x1, x2, x3, x4, x5, x6, x7, x8)
    if len(parts) < 3:
        raise ValueError("composition_entropy requires >= 3 components")
    _same_composition(parts, composition_id)
    logs, valid = _comp_logs(parts)
    p = _close(logs, 0, len(parts))
    ent = -np.sum(p * np.log(p + _EPS), axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = ent[valid]
    return frame_like(x1, out)


def _composition_normalized_entropy(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    x7: pd.DataFrame | None = None,
    x8: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    """P1-18: normalised Shannon entropy ``H / ln(P)`` of the closed composition.

    ``H`` is the raw Shannon entropy (as in :func:`_composition_entropy`) and
    ``P`` the number of parts, so the result lies in ``[0, 1]`` (``ln(3)``
    normalises a 3-part composition, ``ln(8)`` an 8-part one).  A degenerate
    ``P <= 1`` composition has ``ln(P) <= 0`` and maps to NaN (the shared
    ``>= 3`` component gate already guards this, kept defensively).
    """
    _check_zero_policy(zero_policy)
    parts = _parts(x1, x2, x3, x4, x5, x6, x7, x8)
    if len(parts) < 3:
        raise ValueError("composition_normalized_entropy requires >= 3 components")
    _same_composition(parts, composition_id)
    logs, valid = _comp_logs(parts)
    p = _close(logs, 0, len(parts))
    ent = -np.sum(p * np.log(p + _EPS), axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    n_parts = len(parts)
    if n_parts > 1:
        norm = ent / np.log(n_parts)
        out[valid] = norm[valid]
    # n_parts <= 1 -> ln(P) <= 0 -> output stays NaN (P1-18 degenerate gate)
    return frame_like(x1, out)


def _composition_aitchison_distance(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        raise ValueError("composition_aitchison_distance requires equal-size compositions (2..6 parts each)")
    _same_composition([*a, *b], composition_id)
    _check_part_identity(a, b)
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    la = logs[:na]
    lb = logs[na:]
    clr_a = la - la.mean(axis=0, keepdims=True)
    clr_b = lb - lb.mean(axis=0, keepdims=True)
    dist = np.sqrt(np.sum((clr_a - clr_b) ** 2, axis=0))
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = dist[valid]
    return frame_like(x1, out)


def _composition_ilr_balance(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("composition_ilr_balance requires >= 1 component per side")
    _same_composition([*a, *b], composition_id)
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    gmean_a = np.exp(logs[:na].mean(axis=0))
    gmean_b = np.exp(logs[na:].mean(axis=0))
    r = float(na)
    s = float(len(b))
    # P1-32: the parts are strictly positive (zero_policy="reject" fails closed on
    # non-positive parts), so ``gmean_a/gmean_b > 0`` and the log is well-defined
    # WITHOUT an arbitrary ``+ EPS`` floor.
    balance = np.sqrt(r * s / (r + s)) * np.log(gmean_a / gmean_b)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = balance[valid]
    return frame_like(x1, out)


def _composition_js_divergence(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
    composition_id: str | None = None,
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        raise ValueError("composition_js_divergence requires equal-size compositions (2..6 parts each)")
    _same_composition([*a, *b], composition_id)
    _check_part_identity(a, b)
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    p = _close(logs, 0, na)
    q = _close(logs, na, na + len(b))
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)), axis=0) + 0.5 * np.sum(
        q * (np.log(q + _EPS) - np.log(m + _EPS)), axis=0
    )
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = js[valid]
    return frame_like(x1, out)


_SPECS: dict[str, dict[str, Any]] = {
    "composition_clr_component": {
        "fn": _composition_clr_component,
        # P1-32: ``zero_policy`` is fixed to "reject" (the only legal value) and
        # is NOT a search parameter — the kernel keeps the internal default.
        "params": ["target", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "log_ratio",
        "cost": 2,
        "tags_extra": ["fundamental"],
        "output_unit": "log_ratio",
    },
    "composition_entropy": {
        "fn": _composition_entropy,
        # P1-32: ``zero_policy`` fixed to "reject" — not a search parameter.
        "params": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "entropy",
        "cost": 2,
        "tags_extra": ["fundamental"],
        "output_unit": "entropy",
    },
    "composition_normalized_entropy": {
        "fn": _composition_normalized_entropy,
        # P1-18: normalised Shannon entropy H/ln(P) in [0, 1]; same input
        # contract as composition_entropy (3..8 components, zero_policy=reject).
        "params": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "dimensionless_score",
        "cost": 2,
        "tags_extra": ["fundamental"],
        "output_unit": "dimensionless_score",
    },
    "composition_aitchison_distance": {
        "fn": _composition_aitchison_distance,
        # P1-32: ``zero_policy`` fixed to "reject" — not a search parameter.
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "log_ratio",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "log_ratio",
    },
    "composition_ilr_balance": {
        "fn": _composition_ilr_balance,
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "log_ratio",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "log_ratio",
    },
    "composition_js_divergence": {
        "fn": _composition_js_divergence,
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "dimensionless_score",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "dimensionless_score",
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="composition",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
            param_specs=spec.get("param_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
