"""alphaprobe.legacy — 生产主链淘汰的 legacy 命名空间（Task 1）。

仅 OFFLINE_TEST 使用；PRODUCTION / RESEARCH_DEGRADED 评估只认 QE 权威。
"""

from alphaprobe.legacy.evaluators import (  # noqa: F401
    _all_none,
    _bundle_from_plane,
    make_fe_evaluate_fn,
)

__all__ = [
    "make_fe_evaluate_fn",
    "_bundle_from_plane",
    "_all_none",
]
