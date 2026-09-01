"""continuous 包：7×24 连续挖掘（§56 / Phase 8）。"""

from __future__ import annotations

from alphaprobe.continuous.round_manager import (
    RoundManager,
    run_continuous_v2,
    STATE_FILENAME,
    STOP_QUOTA_FILENAME,
)

__all__ = [
    "RoundManager",
    "run_continuous_v2",
    "STATE_FILENAME",
    "STOP_QUOTA_FILENAME",
]
