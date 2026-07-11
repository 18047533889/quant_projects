from __future__ import annotations

import hashlib
import json
import uuid

_COORDINATE_KEYS = ("signal_structure", "asset_class", "frequency_bucket", "domain_root")


def generate_factor_id(coordinates: dict[str, str], seed: object | None = None) -> str:
    """生成全局唯一 factor_id。

    格式: {asset_class}_{frequency_bucket}_{signal_structure}_{suffix8}
    示例: equity_1d_cross_sectional_224d2523

    seed 为空时 suffix8 为随机 UUID 前缀；seed 非空时 suffix8 为
    seed 的稳定哈希前缀，用于 pipeline 重跑幂等。
    """
    asset_class = coordinates.get("asset_class", "unknown")
    frequency = coordinates.get("frequency_bucket", "unknown")
    signal = coordinates.get("signal_structure", "unknown")
    if seed is None:
        suffix = uuid.uuid4().hex[:8]
    else:
        payload = json.dumps(seed, ensure_ascii=False, sort_keys=True, default=str)
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{asset_class}_{frequency}_{signal}_{suffix}"


def extract_coordinates(config: dict[str, object]) -> dict[str, str]:
    """从 candidate Config 中提取库坐标。"""
    missing = [k for k in _COORDINATE_KEYS if k not in config]
    if missing:
        raise ValueError(f"Config 缺少必要坐标字段: {missing}")
    return {k: str(config[k]) for k in _COORDINATE_KEYS}
