"""Assetization 实现代码。"""
from .compute import run_assetization
from .registry import generate_factor_id, extract_coordinates

__all__ = [
    "run_assetization",
    "generate_factor_id",
    "extract_coordinates",
]
