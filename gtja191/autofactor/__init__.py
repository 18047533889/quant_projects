"""GTJA191 factor-pack provider for external AutoFactorEvaluation.

Importing ``load_pack`` requires the external ``evaluation`` package.
"""
from __future__ import annotations

__all__ = ["load_pack"]


def __getattr__(name: str):
    if name == "load_pack":
        from .provider import load_pack

        return load_pack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
