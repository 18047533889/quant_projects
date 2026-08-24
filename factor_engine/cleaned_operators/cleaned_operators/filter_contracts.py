# -*- coding: utf-8 -*-
"""Filter Layer契约定义（2026-08-12信号滤波层专项）。

FilterRole 描述算子在滤波管线中的职责；JumpPreservationPolicy 声明对真实跳变的
处理方式；FilterContract 是完整契约（因果性/状态/时间分片安全/预热/滞后类型）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class FilterRole(Enum):
    """Filter算子的管线职责分类。"""
    DESPIKE = "despike"                      # 去毛刺（单点/短时异常）
    LOW_PASS = "low_pass"                    # 低通滤波（固定衰减）
    ADAPTIVE_LOW_PASS = "adaptive_low_pass"  # 自适应平滑（ER驱动）
    HYSTERESIS = "hysteresis"                # 迟滞/死区
    RATE_LIMIT = "rate_limit"                # 变化率限制
    STATE_HOLD = "state_hold"                # 状态保持/锁存
    JUMP_DETECTOR = "jump_detector"          # 跳变检测
    SIGNAL_NORMALIZER = "normalizer"         # 信号归一化


class JumpPreservationPolicy(Enum):
    """真实跳变（非毛刺）的保留策略。"""
    PRESERVE_ALL_FINITE_JUMPS = "preserve_all"         # 保留全部有限跳变
    ROBUST_CLIP_INNOVATION = "clip_innovation"         # 稳健截断innovation
    DELAYED_CONFIRMATION = "delayed_confirmation"      # 延迟确认真实性


@dataclass(frozen=True)
class FilterContract:
    """滤波算子的完整执行契约。

    Attributes:
        role: 在滤波管线中的职责
        causal: 严格因果（当前观测不参与自己的阈值估计）
        uses_current_observation: 当前观测是否参与输出计算（≠参与阈值）
        stateful: 递归依赖历史状态（非滑动窗口）
        checkpointable: 状态可序列化/恢复
        time_shard_safe: 时间切片安全（without checkpoint）
        warmup: 预热期（行数）
        lag_class: 滞后类别（zero/one/variable/unknown）
        jump_policy: 对真实跳变的保留策略
        turnover_control: 是否显式控制换手
    """
    role: FilterRole
    causal: bool
    uses_current_observation: bool
    stateful: bool
    checkpointable: bool
    time_shard_safe: bool
    warmup: int
    lag_class: Literal["zero", "one", "variable", "unknown"]
    jump_policy: JumpPreservationPolicy
    turnover_control: bool
