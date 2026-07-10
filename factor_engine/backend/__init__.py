# -*- coding: utf-8 -*-
"""Backend 包公共导出：Backend 基类、调试后端与工厂函数。"""
from .base import Backend
from .debug_backend import DebugBackend
from .factory import build_backend

__all__ = ["Backend", "DebugBackend", "build_backend"]

