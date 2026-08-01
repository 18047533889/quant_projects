# -*- coding: utf-8 -*-
"""Complete parameter-domain contracts for strict fiscal primitives.

The original signature table predates revision-aware fiscal kernels and treated
several TTM/YOY transforms as parameterless.  This module upgrades the in-memory
signature registry before evidence or routing consumes it.  Evidence stores a
hash of the resulting dataclass payload, so future changes remain fail-closed.
"""
from __future__ import annotations

_APPLIED = False


def apply_production_signature_v2() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from backend import production_signature as ps

    c = ps._c
    signature = ps.OperatorProductionSignature

    fiscal = {
        "period_average": signature(
            "period_average",
            (
                c("periods", "positive_integer", 2),
                c("require_consecutive", "boolean", 3),
                c(
                    "revision_policy",
                    "enum",
                    4,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "period_change": signature(
            "period_change",
            (
                c("periods", "positive_integer", 2),
                c("mode", "enum", 3, choices=("absolute", "ratio", "log")),
                c("require_consecutive", "boolean", 4),
                c(
                    "revision_policy",
                    "enum",
                    5,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "period_cagr": signature(
            "period_cagr",
            (
                c("periods", "positive_integer", 2),
                c("periods_per_year", "positive_integer", 3),
                c("sign_policy", "enum", 4, choices=("strict", "absolute")),
                c("require_consecutive", "boolean", 5),
                c(
                    "revision_policy",
                    "enum",
                    6,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "quarter_from_cumulative": signature(
            "quarter_from_cumulative",
            (
                c(
                    "revision_policy",
                    "enum",
                    3,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "ttm_from_quarterly": signature(
            "ttm_from_quarterly",
            (
                c("periods", "positive_integer", 2),
                c("require_consecutive", "boolean", 3),
                c(
                    "revision_policy",
                    "enum",
                    4,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "ttm_from_cumulative": signature(
            "ttm_from_cumulative",
            (
                c(
                    "revision_policy",
                    "enum",
                    3,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
        "yoy_by_period": signature(
            "yoy_by_period",
            (
                c("periods", "positive_integer", 2),
                c(
                    "denominator",
                    "enum",
                    3,
                    choices=("signed", "absolute"),
                ),
                c("require_consecutive", "boolean", 4),
                c(
                    "revision_policy",
                    "enum",
                    5,
                    choices=("latest_available", "first_available"),
                ),
            ),
            default_status="production",
        ),
    }
    ps.PRODUCTION_SIGNATURES.update(fiscal)
    ps._COMPATIBILITY_SIGNATURES["period_lag"] = signature(
        "period_lag",
        (
            c("periods", "nonnegative_integer", 2),
            c(
                "revision_policy",
                "enum",
                3,
                choices=("latest_available", "first_available"),
            ),
        ),
        default_status="production",
    )
    _APPLIED = True
