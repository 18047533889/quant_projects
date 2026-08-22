# -*- coding: utf-8 -*-
"""Compatibility wrapper for the source repository's factor runtime import."""
from __future__ import annotations

from .cos_runtime import install_cos_runtime

install_cos_factor_runtime = install_cos_runtime

__all__ = ["install_cos_factor_runtime"]
