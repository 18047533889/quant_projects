# -*- coding: utf-8 -*-
"""Stable LQTP neutralization surface: three public names only.

Public API
----------
- ``industry_neutralize`` — industry group demean
- ``size_neutralize`` — OLS residual vs ``log(max(MarketCap, 1))``
- ``industry_size_neutralize`` — industry demean, then size residual

Pinned one-arg sources (aligned with cogalpha eval_extensions):
- industry: ``IndustryDaily.IndustryCode``
- size: ``SizeDaily.MarketCap`` (logical alias of StockValuationDaily)
"""
from __future__ import annotations

from typing import Any, Callable

_DIALECT = "lqtp"
_DIALECT_VERSION = "2026-07-19"

# Stable public names. Everything else redirects here.
STABLE_NEUTRALIZE_NAMES: tuple[str, ...] = (
    "industry_neutralize",
    "size_neutralize",
    "industry_size_neutralize",
)


def _factory(canonical: str) -> Callable[..., Any]:
    from api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def _industry_source():
    from api.source_ref import source_col
    return source_col(
        "IndustryDaily",
        "IndustryCode",
        dialect=_DIALECT,
        dialect_version=_DIALECT_VERSION,
    )


def _size_source():
    from api.source_ref import source_col
    return source_col(
        "SizeDaily",
        "MarketCap",
        dialect=_DIALECT,
        dialect_version=_DIALECT_VERSION,
    )


def industry_neutralize(*args: Any):
    """Industry-group demean: ``group_neutralize(x, industry)``."""
    if len(args) == 1:
        return _factory("group_neutralize")(args[0], _industry_source())
    if len(args) == 2:
        return _factory("group_neutralize")(args[0], args[1])
    raise ValueError(
        "industry_neutralize accepts (x) or explicit (x, industry); "
        f"stable neutralize names are {STABLE_NEUTRALIZE_NAMES}"
    )


def size_neutralize(*args: Any):
    """Size neutralize: residual vs log market cap (SizeNeutralize semantics)."""
    if len(args) == 1:
        return _factory("size_neutralize")(args[0], _size_source())
    if len(args) == 2:
        return _factory("size_neutralize")(args[0], args[1])
    raise ValueError(
        "size_neutralize accepts (x) or explicit (x, market_cap); "
        f"stable neutralize names are {STABLE_NEUTRALIZE_NAMES}"
    )


def industry_size_neutralize(*args: Any):
    """Industry then size (matches eval_extensions dual-neutral RankIC)."""
    if len(args) == 1:
        return _factory("industry_size_neutralize")(
            args[0], _industry_source(), _size_source()
        )
    if len(args) == 3:
        return _factory("industry_size_neutralize")(args[0], args[1], args[2])
    raise ValueError(
        "industry_size_neutralize accepts (x) or explicit (x, industry, market_cap); "
        f"stable neutralize names are {STABLE_NEUTRALIZE_NAMES}"
    )


def neutralize(*args: Any):
    """Retired public name — redirect callers to the three stable APIs."""
    if len(args) == 2:
        # Soft compat: two-arg demean historically meant industry/group neutralize.
        return industry_neutralize(args[0], args[1])
    raise ValueError(
        "neutralize is not a stable public name; use one of "
        f"{STABLE_NEUTRALIZE_NAMES}"
    )


def augment_neutralization(allow: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    """Install the three stable neutralize entry points and legacy redirects."""
    out = dict(allow)
    out["industry_neutralize"] = industry_neutralize
    out["size_neutralize"] = size_neutralize
    out["industry_size_neutralize"] = industry_size_neutralize

    # Legacy aliases → stable names
    out["industry_neutral"] = industry_neutralize
    out["ind_neutralize"] = industry_neutralize
    out["market_cap_neutralize"] = size_neutralize
    out["cap_neutralize"] = size_neutralize
    out["size_industry_neutralize"] = industry_size_neutralize
    out["neutralize"] = neutralize
    return out
