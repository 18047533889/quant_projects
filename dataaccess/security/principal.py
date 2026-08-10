# -*- coding: utf-8 -*-
"""DataPrincipal + AccessPolicy（R24 P0-S2 §4）。

## 模型

权限是交集，不是并集（§4「权限取交集」）：

    Principal
    ∩ AccessPolicy
    ∩ RegisteredDatasetBoundary
    ∩ CAM/IAM/STS
    ∩ LocalCacheOwnership

任何一层 deny 都必须 deny。``DataPrincipal`` 回答「当前是谁」，
``AccessPolicy`` 回答「这个身份能读哪些 dataset / factor namespace / action」。
物理路径边界（registry）与云端 IAM 是**另外两层**，不在这里重复实现。

## 动作（§4 最小集合）

    dataset:list / dataset:read / factor:list / factor:read /
    uri:read / metadata:read / factor:metadata_sensitive
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- 动作常量 ----
ACTION_DATASET_LIST = "dataset:list"
ACTION_DATASET_READ = "dataset:read"
ACTION_FACTOR_LIST = "factor:list"
ACTION_FACTOR_READ = "factor:read"
ACTION_URI_READ = "uri:read"
ACTION_METADATA_READ = "metadata:read"
ACTION_FACTOR_METADATA_SENSITIVE = "factor:metadata_sensitive"

ALL_ACTIONS = frozenset({
    ACTION_DATASET_LIST,
    ACTION_DATASET_READ,
    ACTION_FACTOR_LIST,
    ACTION_FACTOR_READ,
    ACTION_URI_READ,
    ACTION_METADATA_READ,
    ACTION_FACTOR_METADATA_SENSITIVE,
})

# 作用于具体数据集的动作（dataset:list 除外，它只校验 action 权限本身）。
DATASET_SCOPED_ACTIONS = ALL_ACTIONS - {ACTION_DATASET_LIST}


@dataclass(frozen=True)
class DataPrincipal:
    """当前 server/user 的逻辑身份。

    ``server_id`` 用于「不同服务器权限不同」的部署模型（§27）：代码完全相同，
    差异来自 principal + policy。None 表示非 server 上下文（本机 research）。
    """

    principal_id: str
    roles: tuple[str, ...] = ()
    server_id: str | None = None

    @property
    def is_server(self) -> bool:
        return bool(self.server_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "roles": list(self.roles),
            "server_id": self.server_id,
        }


# 未配置 principal 时的本机身份（research / dev）。
DEFAULT_LOCAL_PRINCIPAL = DataPrincipal(
    principal_id="local", roles=("local",), server_id=None
)


@dataclass(frozen=True)
class AccessPolicy:
    """逻辑授权层（§4）：允许的 dataset / factor namespace / action。

    ``dataset_rules``：允许的 dataset 集合；空集表示拒绝一切 dataset 访问。
    ``factor_namespaces``：允许的 factor namespace（如 ``internal.restricted``）。
    ``allow_uri_read``：是否允许 ``uri:read``（production 默认 False，§7 / §24）。
    """

    allowed_datasets: frozenset[str] = frozenset()
    allowed_factor_namespaces: frozenset[str] = frozenset()
    allowed_actions: frozenset[str] = frozenset()
    allow_uri_read: bool = False

    def allows_dataset(self, dataset: str) -> bool:
        # 空集 = 无限制（DEFAULT 策略放行一切已注册 dataset）；非空 = 显式 allowlist。
        if not self.allowed_datasets:
            return True
        return dataset in self.allowed_datasets

    def allows_namespace(self, namespace: str) -> bool:
        if not namespace:
            return True
        if not self.allowed_factor_namespaces:
            return True
        return namespace in self.allowed_factor_namespaces

    def allows_action(self, action: str) -> bool:
        if not self.allowed_actions:
            return True
        return action in self.allowed_actions

    def digest(self) -> str:
        """策略指纹（cache manifest scope 绑定 / lineage 记录用，§5.5 / §29）。"""
        import hashlib
        import json

        payload = {
            "allowed_datasets": sorted(self.allowed_datasets),
            "allowed_factor_namespaces": sorted(self.allowed_factor_namespaces),
            "allowed_actions": sorted(self.allowed_actions),
            "allow_uri_read": self.allow_uri_read,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_datasets": sorted(self.allowed_datasets),
            "allowed_factor_namespaces": sorted(self.allowed_factor_namespaces),
            "allowed_actions": sorted(self.allowed_actions),
            "allow_uri_read": self.allow_uri_read,
            "digest": self.digest(),
        }


# 未配置时的默认策略：允许一切已注册 dataset 与全部 action（保持旧行为）。
# ``allow_uri_read=False`` + DefaultAuthorizer 的 strict 判定共同保证：
#   - production/strict 下 uri:read 一律拒绝（§7 / §24）；
#   - research 下保留旧 dev 行为（仍受 PathAuthorizer 白名单约束）。
# production 部署必须显式提供更严格策略（HTTP / store 入口强制）。
DEFAULT_ACCESS_POLICY = AccessPolicy(
    allowed_actions=frozenset(ALL_ACTIONS),
    allow_uri_read=False,
)


__all__ = [
    "DataPrincipal",
    "AccessPolicy",
    "DEFAULT_LOCAL_PRINCIPAL",
    "DEFAULT_ACCESS_POLICY",
    "ALL_ACTIONS",
    "DATASET_SCOPED_ACTIONS",
    "ACTION_DATASET_LIST",
    "ACTION_DATASET_READ",
    "ACTION_FACTOR_LIST",
    "ACTION_FACTOR_READ",
    "ACTION_URI_READ",
    "ACTION_METADATA_READ",
    "ACTION_FACTOR_METADATA_SENSITIVE",
]
