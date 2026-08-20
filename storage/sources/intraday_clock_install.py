# -*- coding: utf-8 -*-
"""Backward-compatible intraday runtime installer.

The runtime is now dispatched explicitly by ``LQTPLogicalDataSource``.  Keeping
this no-op function avoids breaking older import-time bootstrap code without
mutating class methods globally.
"""
from __future__ import annotations


def install_intraday_clock_runtime() -> None:
    return None
