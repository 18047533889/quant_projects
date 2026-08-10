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


__all__ = [
    "redact_secret",
    "redact_uri",
    "redact_authorization_header",
]
