"""data_access.r30.policy —— PolicyManifest（R30-P1-017）。

把安全/授权策略折叠成可进入 ExecutionContext / audit / experiment snapshot 的
**不可变清单**：

    - policy_version：策略版本
    - principal_mappings：principal → 映射（角色/权限组/继承）
    - dataset_classifications：dataset → 数据分类（敏感级别）
    - factor_entitlements：factor → 授权（谁能用、何等级别）

``digest()`` 用 r30._shared.stable_digest 对全字段做确定性摘要；字段任一变化 →
digest 变化。``from_config`` 从扁平配置构造；``to_dict() / as_json() /
into_context_dict()`` 供下游审计/快照嵌入。

完全 additive：不修改 security/execution_context.py、security/policy.py。
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from data_access.r30._shared import stable_digest

__all__ = [
    "PolicyManifest",
]


class PolicyManifest:
    """策略清单：版本 + 三类映射 + 可选预置 digest。

    ``digest`` 参数为可选预置值（例如服务端已计算的策略指纹）；未提供时
    ``digest()`` 即时计算。
    """

    def __init__(
        self,
        policy_version: str,
        principal_mappings: Mapping[str, Any] | None = None,
        dataset_classifications: Mapping[str, Any] | None = None,
        factor_entitlements: Mapping[str, Any] | None = None,
        digest: str | None = None,
    ) -> None:
        self.policy_version = str(policy_version)
        self.principal_mappings = dict(principal_mappings or {})
        self.dataset_classifications = dict(dataset_classifications or {})
        self.factor_entitlements = dict(factor_entitlements or {})
        self._digest_override = digest

    # ---- 摘要 ----

    def digest(self) -> str:
        """稳定摘要：全字段折叠，字段变化即变化。"""
        if self._digest_override is not None:
            return self._digest_override
        return stable_digest(
            self.policy_version,
            self.principal_mappings,
            self.dataset_classifications,
            self.factor_entitlements,
        )

    # ---- 构造 ----

    @classmethod
    def from_config(
        cls,
        policy_version: str,
        principals: Any,
        classifications: Mapping[str, Any] | None = None,
        entitlements: Mapping[str, Any] | None = None,
    ) -> "PolicyManifest":
        """从扁平配置构造。

        ``principals`` 可以是：
            - mapping：{principal_id: {...}} → 直接用；
            - 序列：每项是 dict（含 principal_id/name/id）或对象（含
              principal_id 属性）→ 折叠成 {principal_id: 记录}。
        """
        principal_mappings: dict[str, Any] = {}
        if isinstance(principals, Mapping):
            for pid, rec in principals.items():
                principal_mappings[str(pid)] = (
                    dict(rec) if isinstance(rec, Mapping) else rec
                )
        else:
            for p in principals or []:
                if isinstance(p, Mapping):
                    pid = p.get("principal_id") or p.get("name") or p.get("id")
                    if pid is not None:
                        principal_mappings[str(pid)] = dict(p)
                else:
                    pid = getattr(p, "principal_id", None)
                    if pid is not None:
                        principal_mappings[str(pid)] = p
        return cls(
            policy_version=str(policy_version),
            principal_mappings=principal_mappings,
            dataset_classifications=dict(classifications or {}),
            factor_entitlements=dict(entitlements or {}),
        )

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "principal_mappings": dict(self.principal_mappings),
            "dataset_classifications": dict(self.dataset_classifications),
            "factor_entitlements": dict(self.factor_entitlements),
            "digest": self.digest(),
        }

    def as_json(self, **kw: Any) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            **kw,
        )

    def into_context_dict(self) -> dict[str, Any]:
        """嵌入 ExecutionContext / audit / experiment snapshot 的扁平字典。"""
        return {
            "policy_version": self.policy_version,
            "policy_digest": self.digest(),
            "principal_mappings": dict(self.principal_mappings),
            "dataset_classifications": dict(self.dataset_classifications),
            "factor_entitlements": dict(self.factor_entitlements),
        }

    def __repr__(self) -> str:  # pragma: no cover - 仅诊断
        return (
            f"PolicyManifest(policy_version={self.policy_version!r}, "
            f"digest={self.digest()!r})"
        )
