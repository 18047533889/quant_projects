from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_factor_engine_root() -> Path | None:
    env = os.environ.get("FACTOR_ENGINE_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "api" / "dsl_parser.py").exists():
            return root
    sibling = PACKAGE_ROOT.parent / "factor_engine"
    if (sibling / "api" / "dsl_parser.py").exists():
        return sibling.resolve()
    return None
