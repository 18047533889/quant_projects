"""因子资产化模块（Assetization）"""
from .scripts.compute import run_assetization
from .scripts.registry import generate_factor_id, extract_coordinates

__all__ = [
    "run_assetization",
    "generate_factor_id",
    "extract_coordinates",
]
