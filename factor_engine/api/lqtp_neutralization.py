# -*- coding: utf-8 -*-
"""Source-aware neutralization rules for the LQTP dialect."""
from __future__ import annotations

from typing import Any, Callable


def _factory(canonical: str) -> Callable[..., Any]:
    from api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def industry_neutralize(*args: Any):
    """LQTP industry_neutralize(x) uses the contemporaneous IndustryDaily code."""
    if len(args) == 1:
        from api.source_ref import source_col
        group = source_col(
            "IndustryDaily",
            "IndustryCode",
            dialect="lqtp",
            dialect_version="2026-07-19",
        )
        return _factory("neutralize")(args[0], group)
    if len(args) == 2:
        return _factory("neutralize")(args[0], args[1])
    raise ValueError("industry_neutralize accepts (x) or explicit (x, group)")


def neutralize(*args: Any):
    """Require explicit exposures until the one-arg size+industry contract is sourced."""
    if len(args) == 2:
        return _factory("neutralize")(args[0], args[1])
    if len(args) == 1:
        raise ValueError(
            "neutralize(x) is recognized as an LQTP source-aware size/industry operation, "
            "but its exact exposure contract is not installed; use industry_neutralize(x) "
            "or explicit neutralize(x, exposure)"
        )
    raise ValueError("neutralize accepts explicit (x, exposure)")


def _blocked_size(name: str):
    def dispatch(*args: Any, **kwargs: Any):
        raise ValueError(
            f"{name} is recognized, but the supplied LQTP material does not pin the exact "
            "SizeDaily exposure transform (raw market cap vs log/standardized size); "
            "execution remains fail-closed"
        )
    dispatch.__name__ = name
    return dispatch


def augment_neutralization(allow: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    out = dict(allow)
    out["industry_neutralize"] = industry_neutralize
    out["industry_neutral"] = industry_neutralize
    out["ind_neutralize"] = industry_neutralize
    out["neutralize"] = neutralize
    for name in (
        "size_neutralize",
        "market_cap_neutralize",
        "cap_neutralize",
        "size_industry_neutralize",
        "industry_size_neutralize",
    ):
        out[name] = _blocked_size(name)
    return out
