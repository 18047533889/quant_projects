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

    Wave1-F action granularity: ``allowed_actions``（frozenset 或空）。空 = 该
    scope 的 path 覆盖即视为授权（legacy 语义）；非空 = 只对该 action 集合内
    的动作授权。``matches(resource, action)`` 必须 **同时** 满足 path 覆盖和
    action 覆盖 —— 只给 path 不给 action 视为不覆盖（fail-closed）。
    """

    bucket: str
    key_prefix: str = ""
    dataset: str = ""
    namespace: str = ""
    market: str = ""
    environment: str = ""
    allowed_actions: frozenset[str] = frozenset()

    def matches(self, resource: ObjectResource, action: str | None = None) -> bool:
        """判断资源 + 动作是否落在这个 scope 内。

        ``action`` 为空时只做 path 匹配（向后兼容），非空时必须同时通过
        由 ``allowed_actions`` 限定的动作检查。
        """
        if not (
            _wildcard_match(self.bucket, resource.bucket)
            and _wildcard_match(self.key_prefix, resource.key_prefix)
            and _wildcard_match(self.dataset, resource.dataset)
            and _wildcard_match(self.namespace, resource.namespace)
            and _wildcard_match(self.market, resource.market)
            and _wildcard_match(self.environment, resource.environment)
        ):
            return False
        if not self.allowed_actions:
            return True
        if action is None:
            # 显式限定了 actions 的 scope：未提供 action 视为未覆盖（fail-closed）。
            return False
        return action in self.allowed_actions


def _wildcard_match(pattern: str, value: str) -> bool:
    """通配匹配：``*`` 前缀通配；纯目录前缀（以 ``/`` 结尾）按目录子树匹配。

    语义：
      * 空 pattern 视为不约束（匹配任意）；
      * ``*`` 通配任意；
      * ``*`` 结尾按前缀展开（``factors/*`` 匹配 ``factors/f1``）
      * 以 ``/`` 结尾（``factors/``）匹配该目录子树（``factors/a/x``）；
      * 其余按精确相等。
    """
    if not pattern:
        return True
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    if pattern.endswith("/"):
        return value == pattern or value.startswith(pattern)
    return value == pattern


# 可插拔的 canonical prefix resolver：layout 权威后续喂入。
# 签名：Callable[[str], tuple[str, str]]  # factor_id -> (bucket, key_prefix)
FactorWriteTargetResolver = Callable[[str], tuple[str, str]]


def _default_factor_target(factor_id: str) -> tuple[str, str]:
    """默认因子写目标：bucket="factor-lake"，prefix="factors/{factor_id}"。

    这是占位默认；canonical layout 权威（后续 wave）会替换为真实 resolver。
    """
    return ("factor-lake", f"factors/{factor_id}")


class ProductionTargetResolver:
    """生产环境对象目标解析器 —— 无占位默认。

    Wave1-F：生产模式下解析器一旦缺少目标（bucket/prefix 为空的占位）必须失败，
    绝不回退到 ``factor-lake/factors/{factor_id}`` 这类占位默认 —— 否则一个未指定
    目标的生产写会静默落到错误的 canonical 路径，绕过 layout 权威。

    ``layout_map`` 是显式目标表（``{factor_id: (bucket, key_prefix)}`` 或一个
    可调用 resolver）；显式 table 命中失败即抛 ``ValidationError``。
    """

    def __init__(
        self,
        layout_map: Mapping[str, tuple[str, str]] | Callable[[str], tuple[str, str]] | None = None,
    ) -> None:
        self._layout_map = layout_map

    def __call__(self, factor_id: str) -> tuple[str, str]:
        """Make the resolver directly callable so it can be passed verbatim as
        ``factor_target_resolver`` (``FactorWriteTargetResolver`` protocol)."""
        return self.resolve(factor_id)

    def resolve(self, factor_id: str) -> tuple[str, str]:
        if callable(self._layout_map):
            bucket, key_prefix = self._layout_map(factor_id)
        elif self._layout_map:
            target = self._layout_map.get(factor_id)
            if target is None:
                raise ValidationError(
                    f"production target resolver: 未指定因子 {factor_id!r} 的写目标"
                    "（无占位默认）——请显式配置 canonical layout"
                )
            bucket, key_prefix = target
        else:
            raise ValidationError(
                f"production target resolver: 无 layout 权威可查，拒绝为 "
                f"{factor_id!r} 解析目标（无占位默认）"
            )
        if not bucket or not key_prefix:
            raise ValidationError(
                f"production target resolver: {factor_id!r} 目标不完整 "
                f"(bucket={bucket!r}, prefix={key_prefix!r})，拒绝占位默认"
            )
        return str(bucket), str(key_prefix)


class ObjectStorageAuthorizationBoundary:
    """对象存储授权边界。

    用法：
        boundary = ObjectStorageAuthorizationBoundary(
            allowed_scopes=[ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*")],
            factor_target_resolver=my_resolver,  # 可插拔
        )
        boundary.authorize_factor_write("f1")   # 解析真实 bucket+prefix 后检查
        boundary.authorize_object(ObjectResource(...), ACTION_OBJECT_WRITE)

    Wave1-F：``authorize_object`` 现在要求 **scope 的 path 覆盖 AND 该 scope 允许
    的动作** 同时满足；``allowed_scopes`` 里带 ``allowed_actions`` 的 scope 是
    按动作授出的 grant。生产环境下不传 ``factor_target_resolver`` 时（即
    ``environment="production"``）构造直接失败 —— 绝无占位默认。
    """

    def __init__(
        self,
        allowed_scopes: Sequence[ObjectResourceScope],
        *,
        factor_target_resolver: FactorWriteTargetResolver | None = None,
        principal: Any = None,
        environment: str = "research",
    ) -> None:
        if not allowed_scopes:
            raise ValidationError("ObjectStorageAuthorizationBoundary 至少需要一个允许 scope")
        self.allowed_scopes = tuple(allowed_scopes)
        self.environment = str(environment)
        self.principal = principal
        if str(environment) == "production":
            if factor_target_resolver is None:
                raise ValidationError(
                    "ObjectStorageAuthorizationBoundary: production 模式必须显式提供 "
                    "factor_target_resolver（无占位默认，拒绝 factor-lake/factors/{id} 回退）"
                )
            self.factor_target_resolver: FactorWriteTargetResolver = factor_target_resolver
        else:
            self.factor_target_resolver: FactorWriteTargetResolver = (
                factor_target_resolver or _default_factor_target
            )

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

        Wave1-F：必须 **同时** 满足 scope 的 path 覆盖和该 scope 允许的动作。
        ``require_any_object_write_action`` 语义保留：写类动作任一被单授予即通过。

        Raises:
            AccessDeniedError: 资源不在任何允许 scope，或动作不在对象动作集合内，
                或该 scope 未允许此动作。
        """
        if action not in ALL_OBJECT_ACTIONS:
            raise AccessDeniedError(
                f"未知对象存储动作 {action!r}（允许：{sorted(ALL_OBJECT_ACTIONS)}）"
            )
        for scope in self.allowed_scopes:
            if scope.matches(resource, action):
                return
        raise AccessDeniedError(
            f"对象资源 {resource.to_dict()} 不在允许 scope 内（或该 scope 未允许"
            f"动作 {action!r}），授权被拒绝"
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
    "ProductionTargetResolver",
    "FactorWriteTargetResolver",
]
