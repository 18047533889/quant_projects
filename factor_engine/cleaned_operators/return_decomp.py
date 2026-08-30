# -*- coding: utf-8 -*-
"""Daily return decomposition operators.

Split the close-to-close return into overnight / intraday / VWAP segments.
All operators are elementwise safe-division daily-panel transforms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str, price_params: tuple[str, ...], available_at: str = "session_close", same_session_usable: bool = False) -> OperatorMetadata:
    # R11 P0-60: ``price_basis`` is a declared keyword-only scalar — the typed-IR
    # / caller asserts the price basis (RAW vs CONTINUOUS vs ...) from semantic
    # metadata; the runtime gate verifies it against concept columns.  Appended
    # as a trailing slot so 2-panel positional calls bind price_basis to its
    # kernel default (None).
    declared = [*params, "price_basis"]
    return OperatorMetadata(
        name=name,
        category="return_decomposition",
        description=description,
        param_names=declared,
        return_type="series",
        tags=[
            "return_decomposition", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(declared)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
        # R11 #164: every price input must share the same PriceBasis — the typed
        # search/compile layer reads these declarations so a raw open and an
        # adjusted (continuous) close can never be combined into a return.
        input_units={p: "price" for p in price_params},
        compatible_units={p: ("price",) for p in price_params},
        # NEW-158: machine-readable availability.  overnight_return is usable as
        # soon as the open is known (after open); open_to_vwap (full-day VWAP),
        # open_close_return and vwap_to_close_return are all session-close.
        # ``same_session_usable=False`` tells the execution layer these outputs
        # must never feed an intra-session decision.
        available_at=available_at,
        same_session_usable=same_session_usable,
    )


def _assert_same_axes(*frames: pd.DataFrame) -> None:
    """R11 #163: pandas ``/`` auto-aligns on labels; differing open/close/VWAP
    axes would be silently unioned/reindexed, breaking shape preservation.
    Require exact index/columns first and fail loudly on any mismatch."""
    if not frames:
        return
    ref = frames[0]
    for other in frames[1:]:
        if not ref.index.equals(other.index) or not ref.columns.equals(other.columns):
            raise ValueError(
                "all price inputs of a return decomposition must share the exact "
                f"same index/columns; found {ref.shape} vs {other.shape} — a "
                "silent label reindex would corrupt the return"
            )


def _price_basis_of_column(name: str) -> str | None:
    """Resolve a column's canonical price basis from the field-concept registry.

    ``None`` means the column carries no resolvable basis (e.g. a bare symbol
    name) — permissive, matching the analyzer's untyped-column stance.
    """
    from factor_engine.fields.concepts import concept_alias_map, get_concept

    try:
        alias_map = concept_alias_map()
    except Exception:
        return None
    concept_id = alias_map.get(str(name).lower())
    if not concept_id:
        return None
    try:
        concept = get_concept(concept_id)
    except Exception:
        return None
    if concept is not None and concept.price_basis:
        return concept.price_basis
    return None


def _assert_shared_price_basis(*frames: pd.DataFrame, price_basis: str | None = None) -> None:
    """R11 #164 + P0-60: reject a mixed — or *unverifiable* — price basis.

    A raw open and an adjusted (continuous) close must NEVER be combined into a
    return — the ratio would mix split-unadjusted and split-adjusted levels.
    The typed-IR layer enforces this at compile time; this is the runtime
    defense-in-depth.

    ``price_basis`` is the authoritative basis declared by the caller from the
    AST child semantic type / panel metadata (P0-60 — the basis must come from
    the type system, never inferred from instrument column names).  When
    provided, every column that DOES resolve must agree with it, and unresolvable
    instrument-code columns (``000001.SZ``) are trusted (the caller asserted the
    basis).  When omitted, the gate FAILS CLOSED on any non-concept-named column
    — a return decomposition requires every price column to positively resolve
    to a single shared PriceBasis, otherwise the basis cannot be verified.
    """
    resolved: set[str] = set()
    unverifiable = False
    for frame in frames:
        for col in frame.columns:
            basis = _price_basis_of_column(str(col))
            if basis is None:
                unverifiable = True
                continue
            resolved.add(basis)
    if price_basis is not None:
        if resolved and any(b != price_basis for b in resolved):
            raise ValueError(
                "mixed price basis: caller declared "
                f"{price_basis!r} but concept columns resolve to "
                f"{sorted(resolved)} — a return decomposition cannot combine "
                "raw and adjusted levels (R11 #164 / P0-60)"
            )
        return  # caller asserted the basis; unresolvable columns are trusted
    if unverifiable:
        raise ValueError(
            "cannot verify shared price basis: some price columns are not "
            "concept-named (their PriceBasis — raw vs continuous vs ... — is "
            "unknown).  A return decomposition must positively establish that "
            "every price column shares one basis; an unverifiable "
            "instrument-code column is rejected (P0-60 fail-closed) instead of "
            "silently combining prices."
        )
    if len(resolved) > 1:
        raise ValueError(
            "mixed price basis across inputs "
            f"({sorted(resolved)}) — a return decomposition requires all price "
            "inputs to share the same PriceBasis (raw vs continuous must never "
            "combine into a return)"
        )


def _data_quality_error_type() -> type[Exception] | None:
    """Resolve the strict-mode error type for the P0-61 PositivePrice gate.

    Prefers the runtime governance ``DataQualityError``, then the backend
    ``OperatorDomainError``; returns ``None`` when neither is importable (the
    gate then degrades to NaN-only, documented in ``_safe_ratio``).
    """
    try:
        from factor_engine.runtime.resource_errors import DataQualityError
        return DataQualityError
    except Exception:  # pragma: no cover - leaf module, always importable in-tree
        pass
    try:
        from factor_engine.backend.operator_errors import OperatorDomainError
        return OperatorDomainError
    except Exception:  # pragma: no cover - leaf module, always importable in-tree
        return None


def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Safe price ratio with the P0-61 PositivePrice gate.

    A return is only defined on *positive* prices.  Any price input
    (open / close / preclose / vwap) that is finite and <= 0 must NOT produce a
    normal return: the cell is mapped to NaN (fail closed).  Zero denominators
    and non-finite ratios are NaN as before.

    ``strict=True`` raises the governance ``DataQualityError`` on the first
    non-positive price input instead of returning NaN (when neither
    ``DataQualityError`` nor ``OperatorDomainError`` is importable the strict
    mode degrades to NaN — the gate's core "no normal return from a bad price"
    invariant is preserved either way).
    """
    num = numerator.to_numpy(dtype=float) if hasattr(numerator, "to_numpy") else np.asarray(numerator, dtype=float)
    den = denominator.to_numpy(dtype=float) if hasattr(denominator, "to_numpy") else np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    # P0-61: a finite price <= 0 is not a price.  A zero denominator was already
    # NaN; a negative denominator (or a zero/negative numerator) would otherwise
    # emit a normal-looking return — reject the cell.
    bad = (np.isfinite(num) & (num <= 0)) | (np.isfinite(den) & (den <= 0))
    if strict and bool(np.any(bad)):
        error_type = _data_quality_error_type()
        if error_type is not None:
            raise error_type(
                "return decomposition received a non-positive price input "
                "(finite price <= 0); a return must never be produced from a "
                "zero/negative price (P0-61 PositivePrice gate)"
            )
        # error type not importable -> documented NaN fallback (fall through)
    out[bad] = np.nan
    out[~np.isfinite(out)] = np.nan
    if isinstance(numerator, pd.DataFrame):
        return pd.DataFrame(out, index=numerator.index, columns=numerator.columns)
    return out


@register_operator(
    name="overnight_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="overnight_return",
    source="return_decomp",
    status="experimental",
)
class OvernightReturn(SeriesOperator):
    """隔夜收益：open / pre_close - 1。"""

    metadata = _metadata(
        "overnight_return",
        "隔夜收益 open_px/pre_close - 1。",
        ["open_px", "pre_close"],
        domain="return",
        unit="return",
        price_params=("open_px", "pre_close"),
        # NEW-158: known as soon as the opening price prints (not session close).
        available_at="after_open",
        same_session_usable=False,
    )

    def _calculate_series(self, open_px: pd.DataFrame, pre_close: pd.DataFrame, price_basis: str | None = None, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, pre_close)
        _assert_shared_price_basis(open_px, pre_close, price_basis=price_basis)
        return _safe_ratio(open_px, pre_close) - 1.0

    def __call__(self, *args, **kwargs):
        # R55 platform-audit P0: the platform formula pack writes
        # ``overnight_return`` / ``overnight_return()`` (zero-arg).  Resolve the
        # implicit ``(open, pre_close)`` from the engine's daily columns so the
        # bare name is a first-class callable on the compat surface.
        if len(args) == 0 and not kwargs:
            from factor_engine.api.columns import col

            return self._calculate_series(col("open"), col("pre_close"))
        if len(args) == 2:
            return self._calculate_series(args[0], args[1], **kwargs)
        return self._calculate_series(*args, **kwargs)


@register_operator(
    name="open_close_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="open_close_return",
    source="return_decomp",
    status="experimental",
)
class OpenCloseReturn(SeriesOperator):
    """日内收益（开盘→收盘）：close / open - 1。"""

    metadata = _metadata(
        "open_close_return",
        "开盘到收盘收益 close/open_px - 1。",
        ["open_px", "close"],
        domain="return",
        unit="return",
        price_params=("open_px", "close"),
    )

    def _calculate_series(self, open_px: pd.DataFrame, close: pd.DataFrame, price_basis: str | None = None, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, close)
        _assert_shared_price_basis(open_px, close, price_basis=price_basis)
        return _safe_ratio(close, open_px) - 1.0


@register_operator(
    name="open_to_vwap_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="open_to_vwap_return",
    source="return_decomp",
    status="experimental",
)
class OpenToVwapReturn(SeriesOperator):
    """开盘到 VWAP 收益：vwap / open - 1。"""

    metadata = _metadata(
        "open_to_vwap_return",
        "开盘到 VWAP 收益 vwap/open_px - 1。",
        ["open_px", "vwap"],
        domain="return",
        unit="return",
        price_params=("open_px", "vwap"),
    )

    def _calculate_series(self, open_px: pd.DataFrame, vwap: pd.DataFrame, price_basis: str | None = None, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, vwap)
        _assert_shared_price_basis(open_px, vwap, price_basis=price_basis)
        return _safe_ratio(vwap, open_px) - 1.0


@register_operator(
    name="vwap_to_close_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="vwap_to_close_return",
    source="return_decomp",
    status="experimental",
)
class VwapToCloseReturn(SeriesOperator):
    """VWAP 到收盘收益：close / vwap - 1。"""

    metadata = _metadata(
        "vwap_to_close_return",
        "VWAP 到收盘收益 close/vwap - 1。",
        ["vwap", "close"],
        domain="return",
        unit="return",
        price_params=("vwap", "close"),
    )

    def _calculate_series(self, vwap: pd.DataFrame, close: pd.DataFrame, price_basis: str | None = None, **_: Any) -> pd.DataFrame:
        _assert_same_axes(vwap, close)
        _assert_shared_price_basis(vwap, close, price_basis=price_basis)
        return _safe_ratio(close, vwap) - 1.0
