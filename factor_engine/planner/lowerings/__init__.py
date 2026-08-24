# -*- coding: utf-8
"""Composite lowering 注册包（import 触发 @register_lowering）。"""
from __future__ import annotations

from factor_engine.planner.lowerings import ashare as _ashare  # noqa: F401
from factor_engine.planner.lowerings import fundamental as _fundamental  # noqa: F401
from factor_engine.planner.lowerings import microstructure as _microstructure  # noqa: F401
from factor_engine.planner.lowerings import technical as _technical  # noqa: F401
from factor_engine.planner.lowerings import timeseries as _timeseries  # noqa: F401
from factor_engine.planner.lowerings import next_stage as _next_stage  # noqa: F401

__all__ = ["ashare", "fundamental", "microstructure", "technical", "timeseries", "next_stage"]
