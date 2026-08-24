# -*- coding: utf-8 -*-
"""Audited factor-operator overhaul package."""
from __future__ import annotations

_REGISTERED = False


def register_all() -> None:
    """Register audited non-fiscal primitives.

    Fiscal-period canonicals intentionally have one runtime owner:
    ``cleaned_operators.fiscal_strict``.  The historical
    ``overhaul.fundamental`` module is retained only as migration source code
    and is no longer imported into the active registry; this removes the old
    registration-order dependency where period_change/average/CAGR briefly
    pointed at period_lag before a later layer replaced them.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from cleaned_operators.overhaul import (
        compat,
        daily,
        polars_fixes,
        regression,
        technical,
    )

    daily.register()
    regression.register()
    technical.register()
    # Targeted backend fixes and compatibility wrappers must load after the
    # audited canonical implementations so they replace only the affected slot.
    polars_fixes.register()
    compat.register()
    _REGISTERED = True


def finalize() -> None:
    from cleaned_operators.overhaul.cleanup import finalize as _finalize

    _finalize()
