"""边界修复：production 开关统一、run_many 门禁。"""

from __future__ import annotations

import os

import pytest

from runtime.production_policy import (
    ProductionPolicyViolation,
    assert_production_run_flags,
    resolve_run_mode,
)


def test_resolve_run_mode_from_quant_production_mode(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE", raising=False)
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    assert resolve_run_mode() == "production"


def test_explicit_run_mode_overrides_quant_production_mode(monkeypatch):
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    assert resolve_run_mode("research") == "research"


def test_run_many_production_requires_all_flags():
    with pytest.raises(ProductionPolicyViolation, match="input_dq_check"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=False,
            auto_warmup=True,
            pit_enforce=True,
            context="run_many",
        )


def test_run_many_production_fast_path_blocked_without_flags():
    """production 下快路径也必须满足 warmup/PIT/input_dq。"""
    with pytest.raises(ProductionPolicyViolation):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=False,
            context="run_many",
        )
