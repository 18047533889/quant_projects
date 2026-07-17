# -*- coding: utf-8 -*-
"""Audited factor-operator overhaul package."""
from __future__ import annotations

_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from cleaned_operators.overhaul import daily, fundamental, regression, technical

    daily.register()
    regression.register()
    fundamental.register()
    technical.register()
    _REGISTERED = True


def finalize() -> None:
    from cleaned_operators.overhaul.cleanup import finalize as _finalize

    _finalize()
