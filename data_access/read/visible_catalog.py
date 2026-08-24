"""R26-P0-025 —— VisibleFactorCatalog：按 principal 可见性过滤的因子目录。

问题（P0-025）：
    - ``/v1/factors`` 返回 ``catalog.root``（泄露服务器本地目录）；
    - ``count`` 是全 catalog 数量，不是当前 principal 可见数量；
    - factor list 先授权 ``factor_lake``，但没有先按每个 factor 的 derived
      classification/tag 过滤 catalog——premium factor 的名字/存在性本身是 metadata。

本模块构造 ``VisibleFactorCatalog(principal, policy, catalog)``：
    - ``records`` / ``visible_ids`` / ``__len__`` 全部基于 visible set；
    - 可见判定 = derived/source access tags **all-required** ⊆ allowed namespaces
      （与 store ``_authorize_factor_tags`` 同一逻辑，P0-009）；
    - classification/entitlements 额外收紧（有则必须满足）。
"""
from __future__ import annotations

from typing import Any, Mapping

from data_access.security.principal import (
    FACTOR_CLASSIFICATION_LEVELS,
    DataPrincipal,
)


def factor_visible_to(
    meta: Any,
    *,
    policy: Any,
    principal: DataPrincipal | None = None,
) -> bool:
    """单因子对 principal 是否可见（R26-P0-009/025 同一 all-required 逻辑）。"""
    allowed = getattr(policy, "allowed_factor_namespaces", None) if policy else None
    if allowed is not None and "*" in allowed:
        allowed = None
    if allowed is not None:
        tags = tuple(getattr(meta, "derived_access_tags", ()) or ()) or tuple(
            getattr(meta, "source_access_tags", ()) or ()
        )
        if tags and not all(t in allowed for t in tags):
            return False
    # classification ≤ principal clearance。
    if principal is not None and principal.clearance is not None:
        classification = str(getattr(meta, "classification", "") or "").strip().lower()
        if classification:
            if (
                FACTOR_CLASSIFICATION_LEVELS.get(classification, 100)
                > principal.clearance_level()
            ):
                return False
    # required entitlements ⊆ principal.entitlements。
    if principal is not None:
        required = tuple(getattr(meta, "required_entitlements", ()) or ())
        if required:
            ents = set(getattr(principal, "entitlements", ()) or ())
            if not all(e in ents for e in required):
                return False
    return True


class VisibleFactorCatalog:
    """按 principal 可见性过滤的只读因子目录（R26-P0-025）。

    不暴露 ``root``；``count`` / ``summary`` / ``metadata`` / ``read`` 全基于
    visible set。
    """

    def __init__(
        self,
        catalog: Any,
        *,
        policy: Any = None,
        principal: DataPrincipal | None = None,
    ) -> None:
        self._catalog = catalog
        self._policy = policy
        self._principal = principal

    def visible_ids(self) -> list[str]:
        return [
            fid
            for fid, meta in sorted((self._catalog.records or {}).items())
            if factor_visible_to(meta, policy=self._policy, principal=self._principal)
        ]

    def records(self) -> dict[str, Any]:
        return {
            fid: self._catalog.records[fid]
            for fid in self.visible_ids()
            if fid in self._catalog.records
        }

    def summary(self, *, redact_fn: Any = None) -> list[dict[str, Any]]:
        out = []
        for fid in self.visible_ids():
            meta = self._catalog.records.get(fid)
            if meta is None:
                continue
            d = meta.to_dict() if hasattr(meta, "to_dict") else {"factor_id": fid}
            out.append(redact_fn(d) if redact_fn else d)
        return out

    def __len__(self) -> int:
        return len(self.visible_ids())
