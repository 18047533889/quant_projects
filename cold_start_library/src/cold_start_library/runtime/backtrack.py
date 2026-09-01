"""冷启动窗口回溯天数建议。"""

from __future__ import annotations

# V9 库窗口上沿保守估计（覆盖长窗技术指标）
MAX_COLD_START_WINDOW = 1000


def recommended_max_backtrack_days(*, label_days: int = 20, extra_margin: int = 50) -> int:
    return MAX_COLD_START_WINDOW + int(label_days) + int(extra_margin)
