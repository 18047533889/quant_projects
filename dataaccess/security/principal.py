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

from data_access.core.exceptions import ValidationError

# ---- 动作常量 ----
ACTION_DATASET_LIST = "dataset:list"
ACTION_DATASET_READ = "dataset:read"
ACTION_FACTOR_LIST = "factor:list"
ACTION_FACTOR_READ = "factor:read"
ACTION_URI_READ = "uri:read"
ACTION_METADATA_READ = "metadata:read"
ACTION_FACTOR_METADATA_SENSITIVE = "factor:metadata_sensitive"

# R27-F：写路径逻辑授权（write/delete/publish 与 read 进同一套 Principal/Policy，
# 不再只有 read 被保护）。
ACTION_DATASET_WRITE = "dataset:write"
ACTION_DATASET_DELETE = "dataset:delete"
ACTION_DATASET_PUBLISH = "dataset:publish"
ACTION_METADATA_WRITE = "metadata:write"
ACTION_FACTOR_WRITE = "factor:write"
ACTION_FACTOR_CATALOG_WRITE = "factor:catalog:write"

ALL_ACTIONS = frozenset({
    ACTION_DATASET_LIST,
    ACTION_DATASET_READ,
    ACTION_FACTOR_LIST,
    ACTION_FACTOR_READ,
    ACTION_URI_READ,
    ACTION_METADATA_READ,
    ACTION_FACTOR_METADATA_SENSITIVE,
    ACTION_DATASET_WRITE,
    ACTION_DATASET_DELETE,
    ACTION_DATASET_PUBLISH,
    ACTION_METADATA_WRITE,
    ACTION_FACTOR_WRITE,
    ACTION_FACTOR_CATALOG_WRITE,
})

# R26-P0-006：空集合语义三态。None / {"*"} 表示「未限制/通配」→ 放行一切；
# 空集合 frozenset() 表示「显式 deny all」；非空集合 = 精确 allowlist。
_UNRESTRICTED = ("*",)


def parse_strict_bool(value: object, *, context: str) -> bool:
    """R26-P0-008：安全配置的布尔值只接受真实 bool。

    ``bool("false") is True`` 是典型 fail-open；字符串 "false"/"true"、1/0、
    "0"/"1" 一律 reject（安全配置不允许字符串/数字伪装布尔）。
    """
    if isinstance(value, bool):
        return value
    raise ValidationError(
        f"{context} 必须是真正的布尔值 true/false，收到 {value!r}"
        "（R26-P0-008：拒绝字符串/数字伪装布尔——bool('false') 是 fail-open）"
    )

# 作用于具体数据集的动作（dataset:list 除外，它只校验 action 权限本身）。
DATASET_SCOPED_ACTIONS = ALL_ACTIONS - {ACTION_DATASET_LIST}


@dataclass(frozen=True)
class DataPrincipal:
    """当前 server/user 的逻辑身份。

    ``server_id`` 用于「不同服务器权限不同」的部署模型（§27）：代码完全相同，
    差异来自 principal + policy。None 表示非 server 上下文（本机 research）。

    R26-P0-009：``clearance``（public < basic < fundamental < premium < restricted）
    与 ``entitlements``（compartments，如 premium.wind / alt.news）用于因子
    classification 授权：``principal.clearance >= factor.classification AND
    factor.required_entitlements ⊆ principal.entitlements``。
    """

    principal_id: str
    roles: tuple[str, ...] = ()
    server_id: str | None = None
    clearance: str | None = None
    entitlements: tuple[str, ...] = ()

    @property
    def is_server(self) -> bool:
        return bool(self.server_id)

    def clearance_level(self) -> int:
        return FACTOR_CLEARANCE_LEVELS.get(
            str(self.clearance or "").strip().lower(), -1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "roles": list(self.roles),
            "server_id": self.server_id,
            "clearance": self.clearance,
            "entitlements": list(self.entitlements),
        }


# R26-P0-009：因子 classification 分级（低 → 高）。
FACTOR_CLASSIFICATION_LEVELS: dict[str, int] = {
    "public": 0,
    "basic": 10,
    "fundamental": 20,
    "premium": 30,
    "restricted": 40,
}
FACTOR_CLEARANCE_LEVELS = FACTOR_CLASSIFICATION_LEVELS


# 未配置 principal 时的本机身份（research / dev）。
DEFAULT_LOCAL_PRINCIPAL = DataPrincipal(
    principal_id="local", roles=("local",), server_id=None
)


@dataclass(frozen=True)
class AccessPolicy:
    """逻辑授权层（§4）：允许的 dataset / factor namespace / action。

    R26-P0-006：三个 ``allowed_*`` 集合字段采用**显式三态**，禁止用空集合
    同时表达「没配置」与「允许全部」：

        None / "*"   => unrestricted（放行一切）——仅显式构造时使用
        frozenset()  => deny all（空集合 = 明确不给任何权限）
        ["a","b"]    => 精确 allowlist

    ``allow_uri_read``：是否允许 ``uri:read``（production 默认 False，§7 / §24）。
    布尔严格解析见 ``parse_strict_bool``（P0-008）。
    """

    allowed_datasets: frozenset[str] | None = None
    allowed_factor_namespaces: frozenset[str] | None = None
    allowed_actions: frozenset[str] | None = None
    allow_uri_read: bool = False

    def __post_init__(self) -> None:
        for name in ("allowed_datasets", "allowed_factor_namespaces", "allowed_actions"):
            val = getattr(self, name)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                object.__setattr__(self, name, frozenset(val))
            elif not isinstance(val, frozenset):
                raise ValidationError(
                    f"AccessPolicy.{name} 必须是 set/frozenset/None，收到 {val!r}"
                )
        if not isinstance(self.allow_uri_read, bool):
            raise ValidationError(
                f"AccessPolicy.allow_uri_read 必须是布尔值，收到 {self.allow_uri_read!r}"
            )

    def allows_dataset(self, dataset: str) -> bool:
        return _allows(self.allowed_datasets, dataset)

    def allows_namespace(self, namespace: str) -> bool:
        if not namespace:
            return True
        return _allows(self.allowed_factor_namespaces, namespace)

    def allows_action(self, action: str) -> bool:
        return _allows(self.allowed_actions, action)

    def digest(self) -> str:
        """策略指纹（cache manifest scope 绑定 / lineage 记录用，§5.5 / §29）。"""
        import hashlib
        import json

        payload = {
            "allowed_datasets": _sorted_or_none(self.allowed_datasets),
            "allowed_factor_namespaces": _sorted_or_none(
                self.allowed_factor_namespaces
            ),
            "allowed_actions": _sorted_or_none(self.allowed_actions),
            "allow_uri_read": self.allow_uri_read,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_datasets": _sorted_or_none(self.allowed_datasets),
            "allowed_factor_namespaces": _sorted_or_none(
                self.allowed_factor_namespaces
            ),
            "allowed_actions": _sorted_or_none(self.allowed_actions),
            "allow_uri_read": self.allow_uri_read,
            "digest": self.digest(),
        }


def _allows(allowed: frozenset[str] | None, value: str) -> bool:
    """三态判定（R26-P0-006）：
    None / 含 "*" => unrestricted；空集 => deny all；否则精确匹配。
    """
    if allowed is None or "*" in allowed:
        return True
    if not allowed:
        return False
    return value in allowed


def _sorted_or_none(values: frozenset[str] | None) -> list[str] | None:
    return None if values is None else sorted(values)


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
    "ACTION_DATASET_WRITE",
    "ACTION_DATASET_DELETE",
    "ACTION_DATASET_PUBLISH",
    "ACTION_METADATA_WRITE",
    "ACTION_FACTOR_WRITE",
    "ACTION_FACTOR_CATALOG_WRITE",
    "FACTOR_CLASSIFICATION_LEVELS",
    "FACTOR_CLEARANCE_LEVELS",
    "parse_strict_bool",
]
