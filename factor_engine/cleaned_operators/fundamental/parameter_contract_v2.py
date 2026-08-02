# -*- coding: utf-8 -*-
"""Repair the shared positive-integer validator calling convention.

Historical code declared ``_pos_int(value, minimum, name)`` while the reviewed
fundamental-v2 family consistently calls ``_pos_int(value, name, minimum)``.
This compatibility shim accepts both conventions and is installed immediately
after ``transforms_v2`` and before dependent modules import the helper.
"""
from __future__ import annotations

from typing import Any

from cleaned_operators.fundamental import transforms_v2


def _pos_int(
    value: Any,
    name_or_minimum: str | int = "window",
    minimum_or_name: int | str = 1,
) -> int:
    if isinstance(name_or_minimum, str):
        name = name_or_minimum
        minimum = int(minimum_or_name)
    else:
        minimum = int(name_or_minimum)
        name = (
            str(minimum_or_name)
            if isinstance(minimum_or_name, str)
            else "window"
        )
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be integer >= {minimum}") from error
    if parsed != value or parsed < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return parsed


transforms_v2._pos_int = _pos_int
