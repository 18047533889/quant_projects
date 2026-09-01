"""factor_engine 路径与 PYTHONPATH 引导。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def factor_engine_root() -> Path:
    override = os.environ.get("FACTOR_ENGINE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "factor_engine"
        if (candidate / "api").is_dir():
            return candidate
    raise FileNotFoundError("无法在路径上定位 quant_projects/factor_engine")


def ensure_factor_engine_importable() -> Path:
    root = factor_engine_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if "api" not in sys.modules:
        try:
            import api  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(f"factor_engine 未找到: {root}") from exc
    return root
