# -*- coding: utf-8 -*-
"""CredentialProvider 契约 + CredentialMaterial（R24 P0-S1）。

## 为什么需要

``data_access.cos.remote.resolve_s3_credentials`` 之前会：

- 解析 ``~/.cos.yaml`` 的 ``cos.base.secretid / secretkey``；
- ``os.environ.setdefault("COS_SECRET_ID", ...)`` 把 secret 写进全局环境。

这会削弱部署侧依赖 ``clean-cos-ro``（受限身份 wrapper）做的服务器权限分级：
DataAccess httpfs 自己拿了 base credential，遇到 403 还能换更高权限重试，
绕过服务器原本的受限 wrapper。

## 新契约

production 下凭证只能来自**明确 CredentialProvider**：

- A. CVM / 容器 / 主机绑定角色
- B. STS 临时凭证（access key + secret + session_token + expires_at）
- C. 部署系统显式注入的 server-scoped env credential
- D. 显式配置的 CredentialProvider

**不允许** production 自动解析 ``~/.cos.yaml -> cos.base.secret*``；research
显式 ``DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1`` 才放行（§3.4）。
任何 provider 都不允许把 secret 写回 ``os.environ``（§3.3：子进程继承 /
crash dump / 长驻 worker 其他模块可读 / 测试进程污染）。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from data_access.core.exceptions import ValidationError
from data_access.security.redaction import redact_secret


@dataclass(frozen=True)
class CredentialMaterial:
    """一次凭证解析的产物（不可变，secret 字段 repr 脱敏）。

    ``source`` 标注来源（env / sts / provider / coscli-research），供审计
    lineage 记录（§29）。``credential_scope_id`` 标识这份凭证对应的权限范围
    （如 IAM policy hash），cache manifest 用它做 scope 绑定（§5.5）。
    """

    access_key_id: str
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)
    expires_at: datetime | None = None
    principal_id: str | None = None
    credential_scope_id: str | None = None
    source: str = "env"

    def __post_init__(self) -> None:
        if not self.access_key_id or not self.secret_access_key:
            raise ValidationError("CredentialMaterial 必须同时提供 access_key_id 与 secret")

    @property
    def has_session_token(self) -> bool:
        return bool(self.session_token)

    def to_safe_dict(self) -> dict[str, object]:
        """非 secret 审计视图（§29 lineage / cache manifest scope 绑定用）。"""
        return {
            "access_key_id": redact_secret(self.access_key_id),
            "principal_id": self.principal_id,
            "credential_scope_id": self.credential_scope_id,
            "source": self.source,
            "has_session_token": self.has_session_token,
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at is not None else None
            ),
        }


@runtime_checkable
class CredentialProvider(Protocol):
    """明确凭证来源（§3.1）。实现类返回临时/受限凭证，绝不应写 os.environ。"""

    def resolve(self) -> CredentialMaterial:
        """解析当前 server 的 COS/S3 凭证；无法解析抛 ValidationError。"""
        ...


_ENV_SOURCE = ("COS_SECRET_ID", "AWS_ACCESS_KEY_ID", "S3_ACCESS_KEY_ID")
_ENV_SECRET = ("COS_SECRET_KEY", "AWS_SECRET_ACCESS_KEY", "S3_SECRET_ACCESS_KEY")
_ENV_TOKEN = ("COS_SESSION_TOKEN", "AWS_SESSION_TOKEN", "S3_SESSION_TOKEN")


class EnvCredentialProvider:
    """Option C：部署系统显式注入的 server-scoped env credential。

    session token 可选（STS/CAM 临时身份）。**只读 env，绝不写 env。**
    """

    source = "env"

    def resolve(self) -> CredentialMaterial:
        access = next(
            (os.environ.get(k, "").strip() for k in _ENV_SOURCE if os.environ.get(k, "").strip()),
            "",
        )
        secret = next(
            (os.environ.get(k, "").strip() for k in _ENV_SECRET if os.environ.get(k, "").strip()),
            "",
        )
        if not access or not secret:
            raise ValidationError(
                "无可用 COS/S3 凭证：请通过部署注入 COS_SECRET_ID + COS_SECRET_KEY "
                "（或 AWS_/S3_ 前缀），或显式配置 CredentialProvider。"
            )
        token = next(
            (os.environ.get(k, "").strip() for k in _ENV_TOKEN if os.environ.get(k, "").strip()),
            None,
        )
        scope = (
            os.environ.get("DATA_ACCESS_CREDENTIAL_SCOPE_ID", "").strip() or None
        )
        principal = (
            os.environ.get("DATA_ACCESS_PRINCIPAL_ID", "").strip() or None
        )
        return CredentialMaterial(
            access_key_id=access,
            secret_access_key=secret,
            session_token=token or None,
            principal_id=principal,
            credential_scope_id=scope,
            source="env",
        )


def allow_coscli_config_parse() -> bool:
    """research 显式 opt-in 读 ``~/.cos.yaml``（§3.4 / §3.5）。

    production 永远 False（见 ``data_access.security.policy`` 的 fail-closed
    判定）；即使 research 也要显式 ``DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1``。
    """
    from data_access.read.query_budget import is_strict_semantics

    if is_strict_semantics():
        return False
    return os.environ.get("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_provider_lock = threading.Lock()
_global_provider: CredentialProvider | None = None


def set_credential_provider(provider: CredentialProvider | None) -> None:
    """显式注入 CredentialProvider（Option D，§3.1）。

    测试 / 部署系统注入用；process-global。production 未注入时由
    ``resolve_s3_credentials`` 走 env provider（§3.1 C）。
    """
    global _global_provider
    with _provider_lock:
        _global_provider = provider


def _global_credential_provider() -> CredentialProvider | None:
    return _global_provider


class CosCliConfigProvider:
    """research-only 来源：读 coscli/clean-cos-ro 配置（``~/.cos.yaml``）。

    production / strict 下绝不能被解析（见 ``allow_coscli_config_parse``）。
    DataAccess 不猜 profile、不换更高权限 profile、遇到 403 不换 base credential
    重试——这里只是「research 开发机显式 opt-in」的最后手段。
    """

    source = "coscli-research"

    def resolve(self) -> CredentialMaterial:
        if not allow_coscli_config_parse():
            raise ValidationError(
                "production/strict 禁止解析 ~/.cos.yaml 获取 COS 凭证"
                "（DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE 仅在 research 显式开启）。"
                "请走部署注入的 env credential 或显式 CredentialProvider。"
            )
        explicit = (
            os.environ.get("DATA_ACCESS_COS_YAML")
            or os.environ.get("COS_CONFIG_FILE")
            or ""
        ).strip()
        candidates = [explicit] if explicit else [os.path.expanduser("~/.cos.yaml")]
        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            try:
                import yaml

                with open(path, "r", encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh) or {}
                base = (payload.get("cos") or {}).get("base") or {}
                sid = str(base.get("secretid") or "").strip()
                skey = str(base.get("secretkey") or "").strip()
                if sid and skey:
                    return CredentialMaterial(
                        access_key_id=sid,
                        secret_access_key=skey,
                        principal_id=None,
                        credential_scope_id=None,
                        source="coscli-research",
                    )
            except Exception:
                continue
        raise ValidationError("~/.cos.yaml 不存在或缺少 cos.base.secretid/secretkey")


__all__ = [
    "CredentialMaterial",
    "CredentialProvider",
    "EnvCredentialProvider",
    "CosCliConfigProvider",
    "allow_coscli_config_parse",
    "set_credential_provider",
    "_global_credential_provider",
]
