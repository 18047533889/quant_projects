"""data_access.r30.versioning —— R30-P1-028 版本治理。

把 R30 层共享的版本常量（来自 ``data_access.r30._shared``）与 SCM/commit 驱动
的 ``data_access._build_meta.build_sha`` 折叠成一份版本清单，并给出兼容性判定：

    - ``data_access_version_manifest()``：当前进程的全部版本标识（package_version /
      build_sha / api / contract_schema / registry_schema / semantic_schema /
      storage_format / identity_schema）。
    - ``assert_version_compat(required)``：按 dict 逐项比对，返回不匹配的版本名
      列表（空 = 兼容）。

纯 additive：不修改任何既有文件。
"""
from __future__ import annotations

from typing import Any

# R30-P1-028：从 _shared 复用版本常量（单一权威，不重复定义）。
from data_access.r30._shared import (
    API_VERSION,
    CONTRACT_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    SEMANTIC_SCHEMA_VERSION,
    STORAGE_FORMAT_VERSION,
    version_gate,
)

__all__ = [
    "API_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "SEMANTIC_SCHEMA_VERSION",
    "STORAGE_FORMAT_VERSION",
    "version_gate",
    "data_access_version_manifest",
    "assert_version_compat",
]


def _package_version() -> str | None:
    """当前 data_access 包版本（best-effort）。"""
    try:
        from data_access import __version__

        return str(__version__)
    except Exception:
        return None


def _build_sha() -> str | None:
    """SCM/commit 驱动的 build SHA（best-effort，失败 → None）。"""
    try:
        from data_access._build_meta import build_sha as _build_sha_fn

        return _build_sha_fn()
    except Exception:
        return None


def data_access_version_manifest() -> dict[str, str | None]:
    """当前进程的版本清单。"""
    return {
        "package_version": _package_version(),
        "build_sha": _build_sha(),
        "api": API_VERSION,
        "contract_schema": CONTRACT_SCHEMA_VERSION,
        "registry_schema": REGISTRY_SCHEMA_VERSION,
        "semantic_schema": SEMANTIC_SCHEMA_VERSION,
        "storage_format": STORAGE_FORMAT_VERSION,
        "identity_schema": IDENTITY_SCHEMA_VERSION,
    }


def assert_version_compat(required: dict[str, Any]) -> list[str]:
    """比对 ``required`` 与当前清单，返回不匹配的版本名列表（空 = 兼容）。

    每个 required 键在清单里存在且 ``str`` 相等才算匹配；清单缺键/值不同都记
    为不匹配。required 里出现未知键也视为不匹配（调用方拼错键会暴露）。
    """
    manifest = data_access_version_manifest()
    mismatched: list[str] = []
    for key, required_val in required.items():
        current = manifest.get(key)
        if current is None or str(current) != str(required_val):
            mismatched.append(key)
    return mismatched
