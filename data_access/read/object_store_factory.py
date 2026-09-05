# -*- coding: utf-8 -*-
"""data_access.read.object_store_factory —— 统一对象存储工厂。

对应 UPSTREAM_FIX_PLAN 问题二「建议增加统一对象存储工厂」：

    def get_object_store(*, backend, bucket=None, prefix=None, root=None) -> ObjectStore

- ``backend="local"``  → :class:`LocalObjectStore`（``root`` 或
  ``DATA_ACCESS_OBJECT_STORE_ROOT`` 环境变量注入；禁止默认写不存在的用户目录）。
- ``backend="cos"`` / ``"s3"`` → :class:`COSObjectStore`（bucket 必填；prefix 仅
  记录边界，key 仍由调用方相对化）。
- 其它 backend → ``ValidationError`` fail-closed（不猜、不静默降级 local）。

生产凭据只走既有 credential provider / env / STS（``COSObjectStore._s3``
内部 ``resolve_s3_credentials``）；本工厂绝不读 admin-cos 配置当 runtime 凭据，
绝不写日志。
"""
from __future__ import annotations

import os
from pathlib import Path

from data_access.core.exceptions import ValidationError
from data_access.read.object_store import (
    COSObjectStore,
    LocalObjectStore,
    ObjectStore,
)

__all__ = ["get_object_store"]


def get_object_store(
    *,
    backend: str,
    bucket: str | None = None,
    prefix: str | None = None,
    root: str | Path | None = None,
    **kwargs,
) -> ObjectStore:
    """按声明 backend 构造 ObjectStore（UPSTREAM_FIX_PLAN 问题二）。

    参数：
        backend: "local" | "cos" | "s3"（大小写不敏感；其余 fail-closed）
        bucket:  cos/s3 必填
        prefix:  远端逻辑前缀（记录用，不改变 ObjectStore 的 key 语义）
        root:    local 必填（或 env ``DATA_ACCESS_OBJECT_STORE_ROOT``）
        **kwargs: 透传给对应实现的构造参数（part_size/inflight/...）

    返回：
        ObjectStore 实例（LocalObjectStore / COSObjectStore）。

    拒绝规则：
        - cos/s3 缺 bucket → ValidationError；
        - local 缺 root 且 env 未设 → ValidationError（禁止默认用户目录）；
        - 未知 backend → ValidationError（不静默降级）。
    """
    b = str(backend or "").strip().lower()
    if b in {"cos", "s3"}:
        if not bucket:
            raise ValidationError(
                f"get_object_store(backend={b!r}) 缺 bucket（fail-closed，"
                "禁止构造无 bucket 的 COS 对象存储）"
            )
        # prefix 只做边界记录；COSObjectStore 的 key 本就相对 bucket 根。
        del prefix
        return COSObjectStore(str(bucket), **kwargs)
    if b == "local":
        root_val = root or os.environ.get("DATA_ACCESS_OBJECT_STORE_ROOT")
        if not root_val:
            raise ValidationError(
                "get_object_store(backend='local') 缺 root：请显式传 root 或设 "
                "DATA_ACCESS_OBJECT_STORE_ROOT（禁止默认写不存在的用户目录）"
            )
        return LocalObjectStore(Path(root_val), **kwargs)
    raise ValidationError(
        f"get_object_store 未知 storage backend {backend!r}；"
        "支持 local / cos / s3（fail-closed，不静默降级）"
    )
