# -*- coding: utf-8 -*-
"""R45: STRICT_REMOTE 审计守卫 —— 过程级证明零本地持久写。

``local_disk_policy.strict_remote_audit_guard`` 不依赖调用方主动声明字节，而是
在 STRICT_REMOTE 下**直接拦截** ``Path.open(w)/write_bytes/write_text``、
``os.replace``、``DataFrame.to_parquet``、``pyarrow.parquet.write_table`` 的本地
持久写 —— 因此"零本地磁盘"由过程本身证明（fail-closed），而非信任调用方。

本测试按文件独立加载 ``local_disk_policy.py``，绕过 ``dataaccess/__init__.py``
既有循环导入（见 R45 注记），只验证守卫语义。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# 独立加载 local_disk_policy（不触发 dataaccess 包 __init__ 循环导入）。
_spec = importlib.util.spec_from_file_location(
    "_r45_ldp", str(Path(__file__).resolve().parents[3] / "data_access" / "read" / "local_disk_policy.py")
)
_ldp = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_ldp)

LocalDiskPolicyViolation = _ldp.LocalDiskPolicyViolation
strict_remote_audit_guard = _ldp.strict_remote_audit_guard
spill_policy_for = _ldp.spill_policy_for
LocalDiskPolicy = _ldp.LocalDiskPolicy


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_PRODUCTION", raising=False)
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)


def test_guard_blocks_direct_path_write_bytes_under_strict(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    p = Path("/tmp/r45_audit_write_bytes.bin")
    p.unlink(missing_ok=True)
    try:
        with strict_remote_audit_guard():
            with pytest.raises(LocalDiskPolicyViolation):
                p.write_bytes(b"x")
    finally:
        p.unlink(missing_ok=True)


def test_guard_blocks_direct_path_open_write_under_strict(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    p = Path("/tmp/r45_audit_open.bin")
    p.unlink(missing_ok=True)
    try:
        with strict_remote_audit_guard():
            with pytest.raises(LocalDiskPolicyViolation):
                with p.open("wb") as fh:
                    fh.write(b"x")
    finally:
        p.unlink(missing_ok=True)


def test_guard_blocks_os_replace_to_persistent_under_strict(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    src = Path("/tmp/r45_audit_src.bin")
    dst = Path("/tmp/r45_audit_dst.bin")
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    src.write_bytes(b"x")
    try:
        with strict_remote_audit_guard():
            with pytest.raises(LocalDiskPolicyViolation):
                os.replace(str(src), str(dst))
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def test_guard_blocks_df_to_parquet_under_strict(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    p = Path("/tmp/r45_audit_df.parquet")
    p.unlink(missing_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2]})
        with strict_remote_audit_guard():
            with pytest.raises(LocalDiskPolicyViolation):
                df.to_parquet(str(p))
    finally:
        p.unlink(missing_ok=True)


def test_research_allows_direct_local_write(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "HIGH_PERFORMANCE")
    p = Path("/tmp/r45_research_write.bin")
    p.unlink(missing_ok=True)
    try:
        with strict_remote_audit_guard():
            p.write_bytes(b"x")  # research 放行
        assert p.read_bytes() == b"x"
    finally:
        p.unlink(missing_ok=True)


def test_no_guard_no_interference_default_policy(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)
    p = Path("/tmp/r45_no_guard.bin")
    p.unlink(missing_ok=True)
    try:
        p.write_bytes(b"x")  # 未进守卫，不受影响
        assert p.read_bytes() == b"x"
    finally:
        p.unlink(missing_ok=True)


def test_guard_nested_blocks_then_releases(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    p = Path("/tmp/r45_nested.bin")
    p.unlink(missing_ok=True)
    try:
        with strict_remote_audit_guard():
            with pytest.raises(LocalDiskPolicyViolation):
                p.write_bytes(b"x")
        # 退出守卫后，write_bytes 恢复为原生（不再拦截）。
        p.write_bytes(b"y")
        assert p.read_bytes() == b"y"
    finally:
        p.unlink(missing_ok=True)


def test_guard_spill_policy_still_respected(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)
    monkeypatch.delenv("FACTOR_ENGINE_PRODUCTION", raising=False)
    assert spill_policy_for() is LocalDiskPolicy.HIGH_PERFORMANCE
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION", "1")
    assert spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE
