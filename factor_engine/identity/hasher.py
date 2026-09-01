# -*- coding: utf-8 -*-
"""SHA-256 哈希工具（§12/§13/§45）。

所有 hash 输入必须含命名空间 ``"factor_identity_v1"``：
    hash = sha256(namespace + stable_serialization)

提供 ``factor_id`` 派生（如 ``"F_" + base32/hex 截短 11 位``）。

禁 BLAKE3 新依赖，禁 python ``hash()``。
"""

from __future__ import annotations

import hashlib
import base64

_NAMESPACE = "factor_identity_v1"


def compute_hash(data: str) -> str:
    """计算含命名空间的 SHA-256 哈希。

    Parameters
    ----------
    data : str
        稳定的序列化文本（如 canonical_ast_text 输出）。

    Returns
    -------
    str
        64 字符全长的十六进制哈希串。
    """
    return hashlib.sha256(
        (_NAMESPACE + data).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def compute_hash_bytes(data: str) -> bytes:
    """返回原始 32 字节哈希值。"""
    return hashlib.sha256(
        (_NAMESPACE + data).encode("utf-8"), usedforsecurity=False
    ).digest()


def derive_factor_id(hex_hash: str, length: int = 11) -> str:
    """从十六进制哈希派生因子 ID。

    Parameters
    ----------
    hex_hash : str
        64 字符十六进制哈希。
    length : int
        base32 编码截断长度（默认 11 字符，提供约 55 位熵）。

    Returns
    -------
    str
        ``"F_" + base32 编码前 ``length`` 字符``。

    Notes
    -----
    SHA-256 提供 256 位。base32(sha256) 约 51 字符。
    截断前 11 字符提供 11 * 5 = 55 位熵，碰撞概率约 2^-28 ≈ 3.7e-9
    （百万级因子池）。如需更高碰撞保证，使用完整哈希。
    """
    raw = bytes.fromhex(hex_hash)
    b32 = base64.b32encode(raw).decode("ascii").rstrip("=")
    # base32 每字符 5 位；截断到 11 字符 = 55 位（进位到字节边界 7 字节）。
    # 为稳定边界，按字节截断（``(length * 5 + 7) // 8`` 字节）后再 b32 编码，
    # 但为保持对外契约 ``F_ + 前 length 字符`` 不变，直接截字符串。
    return "F_" + b32[:length]


def compute_namespaced_hash(namespace: str, data: str) -> str:
    """使用自定义命名空间的哈希计算。"""
    return hashlib.sha256(
        (namespace + data).encode("utf-8"), usedforsecurity=False
    ).hexdigest()