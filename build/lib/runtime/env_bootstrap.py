# -*- coding: utf-8 -*-
"""运行时环境引导：只在开发环境加载 monorepo ``.env``。

R21-141..143: production 默认禁止自动读取 monorepo ``.env`` —— 生产 secrets
必须来自 secret manager / deployment environment。开发 profile
（``FACTOR_ENGINE_DEV_PROFILE=1``）才允许自动加载。启动时输出 non-secret
config digest，便于核对部署环境与预期一致。
"""

from __future__ import annotations

import hashlib
import json
import os

from workspace_paths import load_env_file


def _is_dev_profile() -> bool:
    value = os.environ.get("FACTOR_ENGINE_DEV_PROFILE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    # QUANT_PRODUCTION_MODE=on is an explicit production signal: never auto-load
    # the monorepo .env in that case.
    production = os.environ.get("QUANT_PRODUCTION_MODE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return not production


def bootstrap_runtime_env() -> None:
    """进程级一次性环境引导。

    开发 profile（默认）加载 ``quant_projects/.env``；production 显式跳过。
    多次调用安全（``load_env_file`` 幂等）。
    """
    if _is_dev_profile():
        load_env_file()


def non_secret_env_digest(*prefixes: str) -> str:
    """R21-143: 输出 non-secret 环境配置 digest（不包含密钥键）。"""
    import re

    secret_hint = re.compile(
        r"(PASSWORD|TOKEN|SECRET|API_KEY|AUTHORIZATION|DSN|PRIVATE_KEY)", re.IGNORECASE
    )
    filtered: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if prefixes and not key.startswith(prefixes):
            continue
        if secret_hint.search(key):
            continue
        filtered[key] = str(value)[:120]
    return hashlib.sha256(
        json.dumps(filtered, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
