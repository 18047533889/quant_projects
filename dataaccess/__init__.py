"""Compatibility import for the canonical :mod:`data_access` package.

The repository implementation directory is named ``dataaccess``.  The stable
Python package name remains ``data_access``; keep this spelling as a thin alias
so applications using either top-level name reach the same hardened API.
"""
from __future__ import annotations

import sys

_alias_name = __name__
import data_access as _canonical

# Re-export the canonical module and replace this module entry so callers do not
# get a second top-level package object with different singleton state.  Capture
# the alias name before updating globals because the update also replaces
# ``__name__`` with ``data_access``.
globals().update(_canonical.__dict__)
sys.modules[_alias_name] = _canonical
