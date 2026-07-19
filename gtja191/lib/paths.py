from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_factor_engine_root() -> Path | None:
    """可选：本机重建时若旁边有 factor_engine，则用于完整 parse_expr。"""
    env = os.environ.get("FACTOR_ENGINE_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "api" / "dsl_parser.py").exists():
            return root
    sibling = PACKAGE_ROOT.parent / "factor_engine"
    if (sibling / "api" / "dsl_parser.py").exists():
        return sibling.resolve()
    return None


def resolve_data_access_root() -> Path | None:
    """可选：本机旁边有 dataaccess/（包名 data_access）时返回根目录。"""
    env = os.environ.get("DATA_ACCESS_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "store.py").exists():
            return root
    parent = PACKAGE_ROOT.parent
    for name in ("dataaccess", "data_access"):
        sibling = parent / name
        if (sibling / "store.py").exists():
            return sibling.resolve()
    return None
