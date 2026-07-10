# -*- coding: utf-8
"""从 YAML 加载 DQ Profile，供 research / production 模式切换阈值。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from runtime.quality.dq_gates import DQThresholds
from runtime.quality.input_dq import InputDQThresholds

_PROFILES_PATH = Path(__file__).resolve().parent / "dq_profiles.yaml"


@lru_cache(maxsize=1)
def _load_profiles_payload() -> dict[str, Any]:
    if not _PROFILES_PATH.is_file():
        return {"profiles": {}}
    payload = yaml.safe_load(_PROFILES_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"DQ profiles 必须是 mapping: {_PROFILES_PATH}")
    return payload


def list_dq_profiles() -> list[str]:
    """列出 ``dq_profiles.yaml`` 中已定义的 profile 名称。"""
    profiles = _load_profiles_payload().get("profiles") or {}
    return sorted(profiles.keys())


def resolve_output_dq_thresholds(profile: str | None) -> DQThresholds | None:
    """按 profile 名返回产出 DQ 阈值；None 表示使用 DQThresholds 默认值。"""
    if not profile:
        return None
    profiles = _load_profiles_payload().get("profiles") or {}
    section = profiles.get(profile)
    if section is None:
        raise KeyError(f"未知 DQ profile: {profile!r}，可选: {sorted(profiles.keys())}")
    output = section.get("output") or {}
    return DQThresholds(
        min_coverage=float(output.get("min_coverage", 0.05)),
        max_nan_ratio=float(output.get("max_nan_ratio", 0.95)),
        max_inf_ratio=float(output.get("max_inf_ratio", 0.0)),
        max_abs_value=float(output.get("max_abs_value", 1e8)),
        min_rows=int(output.get("min_rows", 1)),
        min_instruments_per_day=int(output.get("min_instruments_per_day", 1)),
    )


def resolve_input_dq_thresholds(profile: str | None) -> InputDQThresholds | None:
    """按 profile 名返回输入 DQ 阈值；None 表示使用 ``InputDQThresholds`` 默认值。"""
    if not profile:
        return None
    profiles = _load_profiles_payload().get("profiles") or {}
    section = profiles.get(profile)
    if section is None:
        raise KeyError(f"未知 DQ profile: {profile!r}")
    inp = section.get("input") or {}
    return InputDQThresholds(
        min_rows=int(inp.get("min_rows", 1)),
        min_non_null_ratio=float(inp.get("min_non_null_ratio", 0.01)),
        min_instruments=int(inp.get("min_instruments", 1)),
        max_inf_ratio=float(inp.get("max_inf_ratio", 0.0)),
    )
