"""factor_engine 路径与 PYTHONPATH 引导。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def factor_engine_root() -> Path:
    override = os.getenv("FACTOR_ENGINE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "factor_engine"
        if (candidate / "api").is_dir():
            return candidate
    raise FileNotFoundError("无法在路径上定位 quant_projects/factor_engine")


def ensure_factor_engine_importable() -> Path:
    root = factor_engine_root()
    if not (root / "api").is_dir():
        raise FileNotFoundError(f"factor_engine 未找到: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
