"""冷启动库 — 从 quant_projects/cold_start_library 导入（与 factor_engine 平行）。"""

from __future__ import annotations

import sys
from pathlib import Path


def _locate_cold_start_library_src() -> Path:
    """定位与 factor_engine 同级的 cold_start_library/src。"""
    for parent in Path(__file__).resolve().parents:
        fe = parent / "factor_engine"
        src = parent / "cold_start_library" / "src"
        if fe.is_dir() and (src / "cold_start_library").is_dir():
            return src
    raise FileNotFoundError(
        "未找到 cold_start_library。期望布局：\n"
        "  quant_projects/factor_engine/\n"
        "  quant_projects/cold_start_library/\n"
        "或设置 PYTHONPATH 指向 cold_start_library/src，或 pip install -e cold_start_library"
    )


def _ensure_cold_start_library_path() -> Path:
    src = _locate_cold_start_library_src()
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


COLD_START_LIBRARY_SRC = _ensure_cold_start_library_path()
COLD_START_LIBRARY_ROOT = COLD_START_LIBRARY_SRC.parent

from cold_start_library import (  # noqa: E402
    ColdStartEntry,
    default_yaml_path,
    load_cold_start_for_training,
    load_cold_start_mixed_for_training,
    load_cold_start_multi_library_for_training,
    load_cold_start_yaml,
    package_root,
    sample_cold_start_entries,
)
from cold_start_library.runtime.backtrack import (  # noqa: E402
    MAX_COLD_START_WINDOW,
    recommended_max_backtrack_days,
)

DEFAULT_YAML = default_yaml_path("ashare", "backend_v9_core.yaml")
ALPHA101_DEFAULT_YAML = default_yaml_path("alpha101", "wq101.yaml")
ALPHA158_DEFAULT_YAML = default_yaml_path("alpha158", "qlib158.yaml")
ALPHA191_DEFAULT_YAML = default_yaml_path("alpha191", "gtja191.yaml")

__all__ = [
    "ALPHA101_DEFAULT_YAML",
    "ALPHA158_DEFAULT_YAML",
    "ALPHA191_DEFAULT_YAML",
    "COLD_START_LIBRARY_ROOT",
    "COLD_START_LIBRARY_SRC",
    "ColdStartEntry",
    "DEFAULT_YAML",
    "MAX_COLD_START_WINDOW",
    "default_yaml_path",
    "load_cold_start_for_training",
    "load_cold_start_mixed_for_training",
    "load_cold_start_multi_library_for_training",
    "load_cold_start_yaml",
    "package_root",
    "recommended_max_backtrack_days",
    "sample_cold_start_entries",
]
