"""R29-P0 #207 —— SCM/commit 驱动的构建版本元数据。

两台机器都显示 ``0.8.0`` 但运行不同代码 = 无法复现。这里把 build SHA 固化进
版本号与数据身份：

    - ``build_sha()``：优先 ``DATA_ACCESS_BUILD_SHA`` / ``GIT_COMMIT``（CI/CD 显式
      注入），否则运行时 ``git rev-parse HEAD``（本仓库）；都不行 → None。
    - ``full_version(base)``：``0.10.2+build.<sha>``。
    - snapshot_id / lineage 身份并入 build_sha——同一份数据在不同代码构建下
      snapshot 身份不同（换代码即换缓存，杜绝「代码改了、缓存还用旧的」）。
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache

_GIT_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def build_sha() -> str | None:
    """当前构建的 SCM commit SHA（env 优先，回退 git HEAD；都无 → None）。"""
    env_sha = (
        os.environ.get("DATA_ACCESS_BUILD_SHA")
        or os.environ.get("GIT_COMMIT")
        or os.environ.get("SOURCE_VERSION")
    )
    if env_sha and str(env_sha).strip():
        return str(env_sha).strip()[:40]
    return _git_head_sha()


def _git_head_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=_GIT_DIR,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:40]
    except Exception:
        pass
    return None


def full_version(base: str) -> str:
    """``base`` + ``+build.<sha>``（无 sha 时原样返回 base）。"""
    sha = build_sha()
    if not sha:
        return base
    return f"{base}+build.{sha}"


__all__ = ["build_sha", "full_version"]
