"""Read-only compatibility shim for canonical build metadata."""
from __future__ import annotations
from functools import lru_cache
from data_access.core.build_metadata import load_build_info

@lru_cache(maxsize=1)
def _info():
    return load_build_info()

def build_sha() -> str | None:
    return _info().build_sha

def package_version() -> str | None:
    return _info().version

def build_id() -> str | None:
    return _info().build_id

def build_time():
    return _info().build_time

def full_version(base: str | None = None) -> str:
    return _info().version or (base or "")

__all__ = ["build_sha", "package_version", "build_id", "build_time", "full_version"]
