# -*- coding: utf-8 -*-
"""算子注册中心：canonical 名、别名、元数据与 runtime 实例的唯一索引。

命名约定（canonical）
--------------------
- **时序滚动** ``ts_*``：``ts_mean``、``ts_std``、``ts_pct``、``ts_decay_linear`` 等
- **截面** 无前缀或 ``cs_*``：``rank``、``zscore``、``cs_demean``、``cs_regression``
- **分组** ``group_*``：``group_neutralize``、``group_rank``、``group_mean``
- **扩展窗口** ``expanding_*``：``expanding_mean``、``expanding_zscore``
- **技术指标** 大写 TA-Lib 风格：``MACD``、``RSI``、``WMA``（DSL 可用 ``ts_rsi`` 等别名）
- **元素级** 小写 numpy 风格：``clip``、``log``、``where``、``abs``

旧名 / 方言名（``SMA``、``m_var``、``returns``、``cap`` 等）通过 ``_aliases.py`` 与
``_dedupe.py`` 映射到 canonical，**仍可计算**，不会进入重复 canonical 列表。

生命周期
--------
1. 各 ``cleaned_operators/*.py`` 在 import 时用 ``@register_operator`` 注册实现类；
2. ``_aliases.py`` 在 ``load_all()`` 末尾登记 DSL 别名 → canonical；
3. ``OperatorRegistry.get(name)`` 供 ``cleaned_bridge`` 与单测直接调用 ``calculate``。

数据结构
--------
- ``_operators[canonical][backend]``：如 ``pandas_numpy`` 上的算子实例；
- ``_aliases[alias]`` → canonical：DSL 名 ``ts_rsi`` 可解析到 ``RSI``；
- ``_catalog[canonical]``：description、param_names、status（``implemented`` / ``doc_only``）。

白名单与 runtime 一致性：仅 ``get(canon) is not None`` 的算子会进入 ``build_dsl_allowlist()``。
"""
from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple


def _merge_param_names(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Preserve the first canonical positional contract across backend adapters."""
    old = list(existing or [])
    cur = list(new or [])
    if not old:
        return cur
    if not cur:
        return old
    # Backend implementations frequently use local names (x/y, d/window,
    # value/method) for the same positional contract.  The canonical contract
    # remains the first registered declaration; backend-specific details are
    # retained in backend metadata and validated by execution tests.
    return old


class OperatorRegistry:
    """全局算子注册表（类级存储，无单例实例）。

    维护 canonical → backend → 算子实例、别名映射与 catalog 元数据。
    全部接口通过 ``classmethod`` 访问，线程安全由 import 时序保证。
    """

    class Lifecycle(StrEnum):
        BUILDING = "building"
        FINALIZED = "finalized"
        FROZEN = "frozen"

    _operators: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}
    _catalog: Dict[str, dict] = {}
    _lifecycle: Lifecycle = Lifecycle.BUILDING
    _version: int = 0

    @classmethod
    def lifecycle(cls) -> str:
        return cls._lifecycle.value

    @classmethod
    def version(cls) -> int:
        return cls._version

    @classmethod
    def _assert_writable(cls) -> None:
        if cls._lifecycle is cls.Lifecycle.FROZEN:
            raise RuntimeError("operator registry is frozen")

    @classmethod
    def finalize(cls) -> None:
        """Validate aliases and close the bootstrap registration phase."""
        cls._assert_writable()
        for alias, canonical in cls._aliases.items():
            if alias in cls._operators or alias in cls._catalog:
                raise ValueError(f"alias collides with canonical: {alias!r}")
            if canonical not in cls._operators and canonical not in cls._catalog:
                raise ValueError(f"dangling alias: {alias!r} -> {canonical!r}")
            if cls.resolve_canonical(canonical) != canonical:
                raise ValueError(f"alias chain is not flattened: {alias!r}")
        cls._lifecycle = cls.Lifecycle.FINALIZED
        cls._version += 1

    @classmethod
    def freeze(cls) -> None:
        """Freeze all registry mutation after import-time bootstrap."""
        if cls._lifecycle is cls.Lifecycle.BUILDING:
            cls.finalize()
        if cls._lifecycle is not cls.Lifecycle.FROZEN:
            cls._lifecycle = cls.Lifecycle.FROZEN
            cls._version += 1

    @classmethod
    def thaw_for_bootstrap(cls) -> None:
        """Explicit test/bootstrap escape hatch for a previously frozen registry."""
        if cls._lifecycle is cls.Lifecycle.FROZEN:
            cls._lifecycle = cls.Lifecycle.BUILDING
            cls._version += 1

    @classmethod
    def register(
        cls,
        operator: Any,
        *,
        canonical: str,
        backend: str = "pandas_numpy",
        aliases: Optional[List[str]] = None,
        source: str = "",
        status: str = "implemented",
        backend_explicit: bool = True,
    ) -> None:
        """注册一个已实现算子到 registry。

        参数:
            operator: 算子实例（须含 ``metadata``）。
            canonical: registry 主键；缺省时取 ``operator.metadata.name``。
            backend: 实现后端，如 ``pandas_numpy`` / ``polars`` / ``sql``。
            aliases: 可选 DSL 别名列表。
            source: 溯源标记，写入 catalog。
            status: 生命周期状态，默认 ``implemented``。
            backend_explicit: 是否显式声明 backend（Polars production 门禁用）。

        返回:
            None
        """
        canonical = canonical or operator.metadata.name
        cls._assert_writable()
        if canonical in cls._aliases:
            raise ValueError(f"canonical already declared as alias: {canonical!r}")
        cls._operators.setdefault(canonical, {})[backend] = operator
        existing = cls._catalog.get(canonical, {})
        if existing and status == "implemented":
            # Adding another backend is capability metadata, not a lifecycle
            # transition.  In particular, SQL marker registration must never
            # downgrade a reviewed production or research status.
            status = str(existing.get("status", "implemented") or "implemented")
        prev = existing
        backend_meta = dict(prev.get("backend_meta") or {})
        backend_meta[backend] = {"explicit": backend_explicit, "source": source}
        # A backend registration is additive.  Governance fields (surface,
        # PIT/scope, checkpoint contract, deprecation reason, …) are attached
        # by later audit layers and must survive when another backend (most
        # commonly the SQL marker) is registered.
        updated = dict(prev)
        # Canonical metadata is established by the first registration and is
        # never replaced by a later backend marker.  Backend-specific provenance
        # belongs exclusively under backend_meta.
        canonical_description = prev.get("description") or getattr(
            operator.metadata, "description", ""
        )
        canonical_params = _merge_param_names(
            prev.get("param_names"),
            getattr(operator.metadata, "param_names", []),
        )
        updated.update({
            "canonical": canonical,
            "backends": sorted(cls._operators[canonical].keys()),
            "status": status,
            "description": canonical_description,
            "param_names": canonical_params,
            "aliases": sorted(set((aliases or []) + prev.get("aliases", []))),
            "backend_meta": backend_meta,
        })
        # Keep legacy catalog readers from seeing a registration-order-dependent
        # source.  New consumers must use backend_meta[backend].source.
        updated.pop("selected_source", None)
        cls._catalog[canonical] = updated
        for alias in aliases or []:
            if alias != canonical:
                cls.register_alias(alias, canonical)

    @classmethod
    def resolve_canonical(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve aliases transitively and reject cycles or missing targets."""
        current = name
        seen: set[str] = set()
        for _ in range(max_depth + 1):
            if current in seen:
                raise ValueError(f"alias cycle detected at {current!r}")
            seen.add(current)
            target = cls._aliases.get(current)
            if target is None:
                if current in cls._catalog or current in cls._operators:
                    return current
                # Unknown names remain unchanged for optional lookup compatibility.
                return current
            current = target
        raise ValueError(f"alias resolution exceeded max_depth={max_depth}: {name!r}")

    @classmethod
    def register_alias(
        cls, alias: str, canonical: str, *, replace: bool = False,
        replacement_reason: str = "",
    ) -> None:
        """Register an alias with collision and canonical-name checks."""
        if alias == canonical:
            return
        if alias in cls._operators or alias in cls._catalog:
            # A legacy canonical wins over a late alias declaration; silently
            # replacing an active implementation would be worse than retaining
            # the explicit canonical entry.
            return
        existing = cls._aliases.get(alias)
        if existing is not None and existing != canonical and not replace:
            raise ValueError(f"alias already points to {existing!r}: {alias!r}")
        if replace and not replacement_reason.strip():
            raise ValueError("replacement_reason is required when replacing an alias")
        # Existing bootstrap aliases may be temporarily cyclic while dedupe
        # renames canonical keys. Validate only newly introduced edges once the
        # target has settled; repeated identical registrations are harmless.
        if cls._aliases.get(alias) == canonical:
            return
        if existing is not None and existing == canonical:
            return
        cls._aliases[alias] = canonical
        if canonical in cls._catalog:
            aliases = set(cls._catalog[canonical].get("aliases", []))
            aliases.add(alias)
            cls._catalog[canonical]["aliases"] = sorted(aliases)

    @classmethod
    def register_catalog_only(
        cls,
        canonical: str,
        *,
        aliases: Optional[List[str]] = None,
        status: str = "doc_only",
        business_category: str = "",
        description: str = "",
        source: str = "",
    ) -> None:
        """登记仅文档/catalog 占位条目（无 runtime 实现）。

        参数:
            canonical: 占位 canonical 名。
            aliases: 可选别名。
            status: 默认 ``doc_only``。
            business_category: 业务分类标签。
            description: 人类可读说明。
            source: 溯源标记。

        返回:
            None
        """
        cls._catalog[canonical] = {
            "canonical": canonical,
            "aliases": sorted(set(aliases or [])),
            "backends": [],
            "selected_source": source,
            "status": status,
            "description": description,
            "param_names": [],
            "business_category": business_category,
        }
        for alias in aliases or []:
            cls._aliases[alias] = canonical

    @classmethod
    def unregister(cls, canonical: str) -> None:
        """从 registry 移除 canonical 及其实现与 catalog 条目。

        参数:
            canonical: 待注销的 canonical 名。

        返回:
            None
        """
        cls._operators.pop(canonical, None)
        cls._catalog.pop(canonical, None)

    @classmethod
    def rename_canonical(cls, old: str, new: str) -> None:
        """将已注册 canonical 重命名为标准名，保留 runtime 与 backend。

        参数:
            old: 旧 canonical 名。
            new: 新 canonical 名；若已存在则把 ``old`` 的 backend **合并**进 ``new`` 再注销 ``old``。

        返回:
            None
        """
        if old == new or old not in cls._operators:
            return
        if new in cls._operators:
            # Merge backends (e.g. sql placeholder registered under new name before
            # pandas/polars were renamed onto it).
            for backend, op in list(cls._operators.get(old, {}).items()):
                if backend not in cls._operators[new]:
                    cls._operators[new][backend] = op
            old_cat = cls._catalog.pop(old, {})
            new_cat = cls._catalog.get(new, {})
            new_cat["backends"] = sorted(cls._operators[new].keys())
            new_cat["canonical"] = new
            # Prefer non-empty description / params from either side.
            if not new_cat.get("description") and old_cat.get("description"):
                new_cat["description"] = old_cat.get("description", "")
            if not new_cat.get("param_names") and old_cat.get("param_names"):
                new_cat["param_names"] = old_cat.get("param_names", [])
            aliases = set(new_cat.get("aliases") or []) | set(old_cat.get("aliases") or [])
            aliases.add(old)
            new_cat["aliases"] = sorted(a for a in aliases if a != new)
            cls._catalog[new] = new_cat
            cls._operators.pop(old, None)
            for alias, canon in list(cls._aliases.items()):
                if canon == old:
                    cls._aliases[alias] = new
            cls._aliases[old] = new
            return
        cls._operators[new] = cls._operators.pop(old)
        meta = cls._catalog.pop(old, {})
        meta["canonical"] = new
        aliases = set(meta.get("aliases") or [])
        aliases.add(old)
        meta["aliases"] = sorted(a for a in aliases if a != new)
        cls._catalog[new] = meta
        for op in cls._operators[new].values():
            if hasattr(op, "metadata"):
                op.metadata.name = new
        for alias, canon in list(cls._aliases.items()):
            if canon == old:
                cls._aliases[alias] = new
        cls._aliases[old] = new

    @classmethod
    def backends_for(cls, name: str) -> List[str]:
        """查询算子已注册的 backend 列表。

        参数:
            name: DSL 名或 canonical 名（先走别名解析）。

        返回:
            已注册 backend 名排序列表，如 ``["pandas_numpy", "polars"]``。
        """
        canonical = cls.resolve_canonical(name)
        return sorted(cls._operators.get(canonical, {}).keys())

    @classmethod
    def get_preferred(
        cls,
        name: str,
        *,
        prefer: str = "auto",
        mode: str = "production",
        allow_unverified_backend: bool = False,
    ) -> Tuple[Any | None, str]:
        """按策略选取最优可用 backend 及算子实例。

        参数:
            name: DSL 名或 canonical 名。
            prefer: 偏好后端，``auto`` / ``polars`` / ``pandas_numpy`` / ``sql``。

        返回:
            ``(算子实例或 None, 实际选用的 backend 名)`` 元组。
        """
        canonical = cls.resolve_canonical(name)
        backends = cls._operators.get(canonical, {})
        if prefer == "pandas_numpy":
            op = backends.get("pandas_numpy")
            if op is None:
                raise LookupError(f"{canonical!r} has no pandas_numpy backend")
            return op, "pandas_numpy"
        if prefer == "polars":
            from backend.operator_capability import get_best_backend

            try:
                return get_best_backend(
                    canonical, prefer="polars", mode=mode,
                    allow_unverified_backend=allow_unverified_backend,
                )
            except Exception:
                if mode == "production":
                    op = cls.get(canonical, "pandas_numpy")
                    if op is not None:
                        return op, "pandas_numpy"
                raise
        if prefer == "sql":
            from backend.sql_tiers import (
                is_sql_implemented,
                is_sql_parity_verified,
                effective_sql_production_safe,
            )

            permitted = (
                effective_sql_production_safe(canonical)
                if mode == "production"
                else is_sql_parity_verified(canonical)
                if mode == "validation"
                else is_sql_implemented(canonical)
            )
            if "sql" in backends and permitted:
                return backends["sql"], "sql"
            return cls.get_preferred(
                name, prefer="auto", mode=mode,
                allow_unverified_backend=allow_unverified_backend,
            )
        # auto：Hybrid 路由 — SQL 在 plan 层；此处 Polars safe vs Pandas fallback
        from backend.operator_capability import get_best_backend

        return get_best_backend(name, prefer="auto", mode=mode)

    @classmethod
    def get(cls, name: str, backend: str = "pandas_numpy") -> Any:
        """按名称与 backend 获取算子实例。

        参数:
            name: DSL 名或 canonical 名（先走别名解析）。
            backend: 目标后端，默认 ``pandas_numpy``。

        返回:
            算子实例；未注册时返回 ``None``。
        """
        canonical = cls.resolve_canonical(name)
        return cls._operators.get(canonical, {}).get(backend)

    @classmethod
    def list_canonical(cls) -> List[str]:
        """列出全部 canonical 名（含仅有 catalog、无 runtime 的条目）。

        返回:
            排序后的 canonical 名列表。
        """
        return sorted(set(cls._operators.keys()) | set(cls._catalog.keys()))

    @classmethod
    def catalog(cls) -> Dict[str, dict]:
        """导出完整 catalog 深拷贝，防止调用者修改 registry 内部状态。"""
        return copy.deepcopy(cls._catalog)
