"""
data_access.write.object_storage_boundary —— 对象存储授权边界（P0-8）。

本地 Path 授权（authorization_boundary.DatasetPathBoundary / WriteAuthorizationGuard）
只覆盖本地文件系统。对象存储（COS）的授权必须按对象资源建模，而不是模糊的
``resource="factor_lake", action="factor:write"``。

对象资源：
    provider=COS, bucket, key_prefix, dataset, namespace, market, environment, generation

动作：
    object:read / object:list / object:write / object:delete /
    multipart:create / multipart:upload / multipart:complete / metadata:commit

``authorize_factor_write(factor_id)`` 必须解析因子的实际目标 ``bucket + canonical prefix``
并对照允许的资源 scope 检查，而不是只查一个模糊的 factor:write。

注意：应用层路径 allowlist 是 defense-in-depth，**不替代** COS CAM/STS 强制。
本模块只提供边界 + ``resolve_factor_write_target(factor_id)``（给定小配置返回
``(bucket, key_prefix)``）。canonical layout 权威本身是另一个 agent（后续 wave）；
这里提供可插拔的 resolver，layout 权威后续可喂入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from data_access.core.exceptions import AccessDeniedError, ValidationError

# ---- 对象存储动作 ----------------------------------------------------------

ACTION_OBJECT_READ = "object:read"
ACTION_OBJECT_LIST = "object:list"
ACTION_OBJECT_WRITE = "object:write"
ACTION_OBJECT_DELETE = "object:delete"
ACTION_MULTIPART_CREATE = "multipart:create"
ACTION_MULTIPART_UPLOAD = "multipart:upload"
ACTION_MULTIPART_COMPLETE = "multipart:complete"
ACTION_METADATA_COMMIT = "metadata:commit"

ALL_OBJECT_ACTIONS = frozenset([
    ACTION_OBJECT_READ,
    ACTION_OBJECT_LIST,
    ACTION_OBJECT_WRITE,
    ACTION_OBJECT_DELETE,
    ACTION_MULTIPART_CREATE,
    ACTION_MULTIPART_UPLOAD,
    ACTION_MULTIPART_COMPLETE,
    ACTION_METADATA_COMMIT,
])

# 写类动作（用于 require_any_object_write_action）。
ALL_OBJECT_WRITE_ACTIONS = frozenset([
    ACTION_OBJECT_WRITE,
    ACTION_OBJECT_DELETE,
    ACTION_MULTIPART_CREATE,
    ACTION_MULTIPART_UPLOAD,
    ACTION_MULTIPART_COMPLETE,
    ACTION_METADATA_COMMIT,
])


@dataclass(frozen=True)
class ObjectResource:
    """对象存储资源标识。"""

    provider: str = "COS"
    bucket: str = ""
    key_prefix: str = ""
    dataset: str = ""
    namespace: str = ""
    market: str = ""
    environment: str = ""
    generation: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "key_prefix": self.key_prefix,
            "dataset": self.dataset,
            "namespace": self.namespace,
            "market": self.market,
            "environment": self.environment,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ObjectResource":
        return cls(
            provider=str(d.get("provider", "COS")),
            bucket=str(d.get("bucket", "")),
            key_prefix=str(d.get("key_prefix", "")),
            dataset=str(d.get("dataset", "")),
            namespace=str(d.get("namespace", "")),
            market=str(d.get("market", "")),
            environment=str(d.get("environment", "")),
            generation=str(d.get("generation", "")),
        )


@dataclass(frozen=True)
class ObjectResourceScope:
    """允许的对象资源 scope（可含通配）。

    bucket / key_prefix / dataset / namespace / market / environment 支持 ``*``
    通配（前缀匹配）。generation 通常不参与授权（代次是发布期概念）。
    """

    bucket: str
    key_prefix: str = ""
    dataset: str = ""
    namespace: str = ""
    market: str = ""
    environment: str = ""

    def matches(self, resource: ObjectResource) -> bool:
        """判断资源是否落在本 scope 内。"""
        return (
            _wildcard_match(self.bucket, resource.bucket)
            and _wildcard_match(self.key_prefix, resource.key_prefix)
            and _wildcard_match(self.dataset, resource.dataset)
            and _wildcard_match(self.namespace, resource.namespace)
            and _wildcard_match(self.market, resource.market)
            and _wildcard_match(self.environment, resource.environment)
        )


def _wildcard_match(pattern: str, value: str) -> bool:
    """通配匹配：``*`` 前缀通配；空 pattern 视为不约束（匹配任意）。"""
    if not pattern:
        return True
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


# 可插拔的 canonical prefix resolver：layout 权威后续喂入。
# 签名：Callable[[str], tuple[str, str]]  # factor_id -> (bucket, key_prefix)
FactorWriteTargetResolver = Callable[[str], tuple[str, str]]


def _default_factor_target(factor_id: str) -> tuple[str, str]:
    """默认因子写目标：bucket="factor-lake"，prefix="factors/{factor_id}"。

    这是占位默认；canonical layout 权威（后续 wave）会替换为真实 resolver。
    """
    return ("factor-lake", f"factors/{factor_id}")


class ObjectStorageAuthorizationBoundary:
    """对象存储授权边界。

    用法：
        boundary = ObjectStorageAuthorizationBoundary(
            allowed_scopes=[ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*")],
            factor_target_resolver=my_resolver,  # 可插拔
        )
        boundary.authorize_factor_write("f1")   # 解析真实 bucket+prefix 后检查
        boundary.authorize_object(ObjectResource(...), ACTION_OBJECT_WRITE)
    """

    def __init__(
        self,
        allowed_scopes: Sequence[ObjectResourceScope],
        *,
        factor_target_resolver: FactorWriteTargetResolver | None = None,
        principal: Any = None,
    ) -> None:
        if not allowed_scopes:
            raise ValidationError("ObjectStorageAuthorizationBoundary 至少需要一个允许 scope")
        self.allowed_scopes = tuple(allowed_scopes)
        self.factor_target_resolver = factor_target_resolver or _default_factor_target
        self.principal = principal

    # ---- 因子写授权 --------------------------------------------------------

    def resolve_factor_write_target(self, factor_id: str) -> tuple[str, str]:
        """解析因子的实际写目标 (bucket, key_prefix)。"""
        bucket, key_prefix = self.factor_target_resolver(factor_id)
        if not bucket or not key_prefix:
            raise ValidationError(
                f"factor {factor_id} 无法解析写目标（bucket/prefix 为空）"
            )
        return bucket, key_prefix

    def authorize_factor_write(self, factor_id: str) -> ObjectResource:
        """授权因子写：解析真实 bucket+prefix 后对照允许 scope 检查。

        返回解析出的 ObjectResource（供调用方使用）。拒绝时抛 AccessDeniedError。
        """
        bucket, key_prefix = self.resolve_factor_write_target(factor_id)
        resource = ObjectResource(
            provider="COS",
            bucket=bucket,
            key_prefix=key_prefix,
            dataset=factor_id,
        )
        self.authorize_object(resource, ACTION_OBJECT_WRITE)
        return resource

    # ---- 通用对象授权 ------------------------------------------------------

    def authorize_object(self, resource: ObjectResource, action: str) -> None:
        """检查某对象资源 + 动作是否在允许 scope 内。

        Raises:
            AccessDeniedError: 资源不在任何允许 scope，或动作不在对象动作集合内
        """
        if action not in ALL_OBJECT_ACTIONS:
            raise AccessDeniedError(
                f"未知对象存储动作 {action!r}（允许：{sorted(ALL_OBJECT_ACTIONS)}）"
            )
        for scope in self.allowed_scopes:
            if scope.matches(resource):
                return
        raise AccessDeniedError(
            f"对象资源 {resource.to_dict()} 不在允许 scope 内，动作 {action} 被拒绝"
        )

    def require_any_object_write_action(self, resource: ObjectResource) -> None:
        """要求至少一个写类动作被授权。"""
        for action in ALL_OBJECT_WRITE_ACTIONS:
            try:
                self.authorize_object(resource, action)
                return
            except AccessDeniedError:
                continue
        raise AccessDeniedError(
            f"对象资源 {resource.to_dict()} 无任何写类动作授权"
        )


__all__ = [
    "ACTION_OBJECT_READ",
    "ACTION_OBJECT_LIST",
    "ACTION_OBJECT_WRITE",
    "ACTION_OBJECT_DELETE",
    "ACTION_MULTIPART_CREATE",
    "ACTION_MULTIPART_UPLOAD",
    "ACTION_MULTIPART_COMPLETE",
    "ACTION_METADATA_COMMIT",
    "ALL_OBJECT_ACTIONS",
    "ALL_OBJECT_WRITE_ACTIONS",
    "ObjectResource",
    "ObjectResourceScope",
    "ObjectStorageAuthorizationBoundary",
    "FactorWriteTargetResolver",
]
