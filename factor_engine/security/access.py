# -*- coding: utf-8 -*-
"""派生因子数据权限继承（R24 P0-S4 §6 / §38 DoD security）。

## 泄露路径

    premium_alt_data → FactorEngine → alpha_secret_001 → factor_lake

如果低权限服务器不能读 ``premium_alt_data``、但能读 ``alpha_secret_001``，
restricted source 就被因子结果洗白了——这是实际的数据泄露。

## 修复

- Dataset contract 增加 ``access_tags`` / ``classification``；
- Factor materialize 时 ``derived_access_tags = union/max_sensitivity(source_access_tags)``，
  写入 FactorCatalog / full definition / lineage / factor lake metadata / factor
  matrix manifest；
- 读取（read_factors / factor_matrix / HTTP factor endpoints）统一校验；
- **禁止自动降密**：premium source → public factor 必须有显式审批
  ``declassification_approved / reviewer / policy_version / reason``。没有审批，
  派生数据权限不能低于输入。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

# 敏感性分级（高 → 低）。派生因子的敏感度 = max(source tags)。
_ACCESS_TAG_LEVELS: dict[str, int] = {
    "internal.restricted": 50,
    "alt.premium": 40,
    "fundamental": 30,
    "factor_engine.market.basic": 20,
    "public": 10,
}
_UNKNOWN_TAG_LEVEL = 60  # 未登记 tag 按最高敏感处理（fail-closed：宁可拒不可放）
# R40 #144: 空 tags 的敏感度 sentinel —— 空集不是「无敏感源」，而是
# UNKNOWN 分类（未声明 = 不可证明安全），必须高于任何已知级别。
_SENSITIVITY_UNKNOWN = 100


def access_tag_level(tag: str) -> int:
    """单 tag 的敏感度级；未登记 tag 按最高级（fail-closed）。"""
    return _ACCESS_TAG_LEVELS.get(str(tag).strip().lower(), _UNKNOWN_TAG_LEVEL)


def max_sensitivity(tags: Iterable[str]) -> int:
    """一组 tag 的最大敏感度级。

    R40 #144: 空集返回 ``_SENSITIVITY_UNKNOWN``（100）—— 未标记 source 不能当
    公开源处理（fail-closed：无分类证据 = 最高敏感，宁可拒不可放）。
    """
    vals = [access_tag_level(t) for t in (tags or ())]
    if not vals:
        return _SENSITIVITY_UNKNOWN
    return max(vals)


def derive_derived_access_tags(
    source_tags: Iterable[str],
    *,
    production: bool | None = None,
) -> frozenset[str]:
    """派生因子的 access tags = union(source tags)（不降密，除非显式审批）。

    不做「取最敏感单个」——保留全部源 tag，任何一层源权限收紧都会反映到因子。

    R40 #144: production 模式下任一 source tag 未登记在敏感词表（UNKNOWN
    CLASSIFICATION）即拒绝 —— 未分类的源标签派生出的访问标签是不可证明的权限
    声明，必须 fail-closed。
    """
    tags = frozenset(t for t in source_tags if t and str(t).strip())
    if production is None:
        # R40 #144 修复：``from runtime.production_policy`` 在 DA 共存上下文里会
        # 被 ``data_access/runtime`` 遮蔽（无 production_policy）→ 导入失败。
        # FE production 管线中 factor_engine 根在 sys.path 前部，取到 FE runtime；
        # 被遮蔽时保守取 research（DA 侧 caller 显式传 production= 决定严格度）。
        try:
            from factor_engine.runtime.production_policy import is_production_mode

            production = is_production_mode()
        except (ImportError, ModuleNotFoundError):
            production = False
    if production:
        for t in sorted(tags):
            if access_tag_level(t) >= _UNKNOWN_TAG_LEVEL:
                raise PermissionError(
                    f"UNKNOWN_CLASSIFICATION: access tag {t!r} is not in the "
                    "sensitivity vocabulary; production forbids deriving access "
                    "tags from an unclassified source tag (R40 #144 fail-closed)"
                )
    return tags


@dataclass(frozen=True)
class DeclassificationApproval:
    """premium source → public factor 的显式审批凭证（§6 禁止自动降密）。

    ``approved`` 为 True 且 ``reviewer`` / ``policy_version`` 非空才有效。
    """

    approved: bool = False
    reviewer: str | None = None
    policy_version: str | None = None
    reason: str | None = None
    # R40 #143: 审批凭证的完整审计字段 —— 时间戳 + 审批单号。approved=True 的
    # 审批缺这两个字段（以及 reviewer/policy_version）即不合法。
    timestamp: str | None = None
    approval_id: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DeclassificationApproval":
        if not raw:
            return cls()
        # R26-P0-008：approved 必须真实 bool；"false"/1/0 → 配置拒绝（fail-closed，
        # 禁止把 string "false" 解析成 True）。
        approved = raw.get("approved")
        if approved is not None and not isinstance(approved, bool):
            raise ValueError(
                "DeclassificationApproval.approved 必须是真正的布尔值，收到 "
                f"{approved!r}（R26-P0-008：拒绝字符串/数字伪装布尔）"
            )
        return cls(
            approved=bool(approved),
            reviewer=str(raw.get("reviewer") or "").strip() or None,
            policy_version=str(raw.get("policy_version") or "").strip() or None,
            reason=str(raw.get("reason") or "").strip() or None,
            timestamp=str(raw.get("timestamp") or "").strip() or None,
            approval_id=str(raw.get("approval_id") or "").strip() or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, *, strict: bool = True) -> None:
        """审批是否合法（strict 默认开启：缺 reviewer/policy_version/timestamp/
        approval_id 即拒绝 —— R40 #143 fail-closed，禁止匿名降密）。"""
        if not self.approved:
            return
        if strict and (
            not self.reviewer
            or not self.policy_version
            or not self.timestamp
            or not self.approval_id
        ):
            raise ValueError(
                "declassification approval 必须提供 reviewer、policy_version、"
                "timestamp 与 approval_id（fail-closed：禁止匿名/无凭证降密）"
            )


def require_declassification_approval(
    source_tags: Iterable[str],
    derived_tags: Iterable[str],
    *,
    approval: DeclassificationApproval | None = None,
    strict: bool = True,
) -> None:
    """校验派生因子是否**非法降密**。

    当 derived 的最大敏感度 < source 的最大敏感度（如 premium source → public
    factor）时，没有合法审批就抛 ``PermissionError``（fail-closed）。
    等于或高于源敏感度（union / max）是安全的，不需要审批。
    """
    src_level = max_sensitivity(source_tags)
    der_level = max_sensitivity(derived_tags)
    if der_level >= src_level:
        return
    approval = approval or DeclassificationApproval()
    approval.validate(strict=strict)
    if not approval.approved:
        raise PermissionError(
            f"派生因子降密被拒绝：source tags max_sensitivity={src_level}，"
            f"derived max_sensitivity={der_level}。没有 declassification approval，"
            "派生数据权限不能低于输入（R24 P0-S4 禁止自动降密）。"
        )


__all__ = [
    "access_tag_level",
    "max_sensitivity",
    "derive_derived_access_tags",
    "DeclassificationApproval",
    "require_declassification_approval",
]
