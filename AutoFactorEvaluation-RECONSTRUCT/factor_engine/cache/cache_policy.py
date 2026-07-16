"""缓存层策略：L0–L3 默认开关与 production 偏好。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePolicy:
    """执行期缓存分层策略。"""

    enable_l0_cse: bool = True
    enable_l1_panel: bool = True
    enable_l2_subplan: bool = True
    enable_l3_disk: bool = False
    dedupe_column_io: bool = True

    @classmethod
    def for_mode(cls, mode: str) -> "CachePolicy":
        """按运行模式返回策略；``production`` 会额外开启 L3 磁盘缓存。"""
        prod = str(mode).lower() == "production"
        return cls(
            enable_l0_cse=True,
            enable_l1_panel=True,
            enable_l2_subplan=True,
            enable_l3_disk=prod,
            dedupe_column_io=True,
        )
