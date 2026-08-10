# -*- coding: utf-8 -*-
"""Secret 脱敏工具（R24 P0-S1 §3.5 / P1-S6 / P1-S8 §35）。

约束：日志 / telemetry / exception / repr 绝不能打印：

- SecretId 全量（必要时只显示 prefix + hash）
- SecretKey
- session token
- Authorization header
- signed URL

本模块是唯一事实源；``S3Credentials.__repr__``、HTTP 异常序列化、
``cos/s3_duckdb.py`` 日志、``read/telemetry.py`` 都消费它。
"""
from __future__ import annotations

import hashlib

# 保留的 access key 前缀字符数（Tencent COS SecretId 以 AKID 开头，展示前缀便于
# 运维区分身份，但不泄露可用的完整凭证）。
_KEEP_PREFIX = 8


def redact_secret(value: str | None, *, keep_prefix: int = _KEEP_PREFIX) -> str:
    """把 secret/token/key 抹成 ``<prefix>...<sha256[:12]>`` 形式。

    None / 空串返回 ``"<redacted:empty>"``；非法类型返回 ``"<redacted>"``。
    """
    if value is None:
        return "<redacted:empty>"
    if not isinstance(value, str) or not value:
        return "<redacted:empty>"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix = value[:_KEEP_PREFIX]
    return f"{prefix}...{digest}"


def redact_uri(uri: str | None) -> str:
    """把可能携带 signed query 的 URI 抹成只有 scheme://host/path 的形式。

    对象 key 本身不是 secret，但签名参数（X-Cos-Signature / X-Qq-Acl /
    Signature=）会泄露签名所用凭证范围；HTTP 错误里出现的完整 URI 一律先过这里。
    """
    if not uri:
        return "<redacted:empty>"
    text = str(uri)
    if "?" in text:
        text = text.split("?", 1)[0]
    return text.rstrip("/")


def redact_authorization_header(value: str | None) -> str:
    if not value:
        return "<redacted:empty>"
    text = str(value)
    marker = text[:32]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{marker}...{digest}"


# 触发整值脱敏的 key 名（递归匹配，大小写不敏感）。
_SECRET_KEYS = frozenset(
    {
        "secret",
        "token",
        "authorization",
        "password",
        "credential",
        "session_token",
        "signed",
        "signature",
        "secret_key",
        "secret_id",
        "access_key",
        "private_key",
        "api_key",
        "password",
    }
)


def sanitize_audit_payload(value: object, *, _path: tuple[str, ...] = ()) -> object:
    """R26-P1-012：审计 payload 递归脱敏（paths/params/error/extra）。

    - key 命中 secret 特征（secret/token/authorization/password/credential/
      session_token/signature/...) → 整值 ``redact_secret``；
    - string 值形如 ``s3://...?...signature=...`` 或含 ``X-Cos-Signature`` →
      ``redact_uri``（去掉 query）；
    - 嵌套 list/dict/tuple 递归。
    """
    if isinstance(value, dict):
        return {
            str(k): sanitize_audit_payload(v, _path=_path + (str(k),))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_audit_payload(v, _path=_path) for v in value]
    if isinstance(value, str):
        low_key = " ".join(_path).lower()
        if any(k in low_key for k in _SECRET_KEYS):
            return redact_secret(value)
        if "signature" in low_key:
            return redact_uri(value)
        # signed URL 兜底：query 里带签名参数。
        if value.startswith(("s3://", "cos://", "http://", "https://")) and (
            "signature=" in value.lower()
            or "x-cos-signature" in value.lower()
            or "x-qq-acl" in value.lower()
        ):
            return redact_uri(value)
        return value
    return value


__all__ = [
    "redact_secret",
    "redact_uri",
    "redact_authorization_header",
    "sanitize_audit_payload",
]
