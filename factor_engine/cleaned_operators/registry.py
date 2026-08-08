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
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

_BOOTSTRAP_TOKEN = object()


class RegistryInitializationError(RuntimeError):
    """Registry bootstrap was attempted from an impossible lifecycle state."""


def _impl_source_hash(operator: Any) -> str:
    """Deterministic source hash of a registered implementation for the
    overwrite audit log (P0-31).  Falls back to the class's module+qualname so
    the log is meaningful even when the operator exposes no callable source."""
    import hashlib
    import inspect as _inspect

    src: str | None = None
    fn = getattr(operator, "calculate", None)
    if fn is None and hasattr(operator, "_calculate_series"):
        fn = operator._calculate_series  # type: ignore
    if callable(fn):
        try:
            src = _inspect.getsource(fn)
        except (OSError, TypeError):  # pragma: no cover - interactive/no-source
            src = None
    if src is None:
        cls = operator.__class__
        src = f"{cls.__module__}.{cls.__qualname__}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


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

    class Lifecycle(str, Enum):
        BUILDING = "building"
        FINALIZED = "finalized"
        FROZEN = "frozen"

    _operators: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}
    _catalog: Dict[str, dict] = {}
    _lifecycle: Lifecycle = Lifecycle.BUILDING
    _version: int = 0
    # Intentional same-backend overwrites (replace=True): canonical/backend,
    # old/new source, old/new impl hash, reason (P0-31 audit trail).
    _overwrite_log: List[dict] = []
    # Hard-fail on undeclared same-backend duplicates (P0-31).  A bootstrap
    # probe may set this False to enumerate every collision site at once.
    _hard_fail_duplicates: bool = True
    # Sources that are declared *bootstrap override layers*.  FactorEngine's
    # loader deliberately layers audited implementations over base registrations
    # (faster native backends, compatibility semantics, parity repairs); each of
    # these modules is an explicit, durable declaration that its same-backend
    # re-registration is intentional.  Every such override is still written to
    # ``_overwrite_log`` with old/new source + implementation hashes, so nothing
    # is silent.  A collision from any source NOT in this set is a hard error.
    _DECLARED_OVERRIDE_SOURCES: frozenset[str] = frozenset({
        "ashare.ops",
        "bounded_structure_repairs",
        "candle_pattern_engine_repairs_v2",
        "composite_fastpath_native_polars",
        "composite_fastpath_primitives",
        "daily_panel",
        "factor_dsl_polars",
        "factor_dsl_polars_bridge",
        "factor_dsl_polars_native",
        "fundamental_transforms_repairs_v2",
        "gtja_compat",
        "layer_composite_fixes",
        "layer_governance_native_polars",
        "layer_governance_primitives",
        "operator_overhaul_audited",
        "operator_overhaul_compat",
        "operator_overhaul_native_polars",
        "polars_chip_tail",
        "polars_cs_misc",
        "polars_geometry_math",
        "polars_misc_utils",
        "polars_robust_stats",
        "polars_state_event",
        "production_repairs",
        "production_technical_extensions",
        "scalar_compare",
        "scalar_where",
        "semantic_hardening",
        "sequence_complexity",
        "stable_high_moments_v2",
        "structure_patterns_extra_repairs_v2",
        "technical_indicators_v2",
    })

    @classmethod
    def overwrite_log(cls) -> List[dict]:
        """Audit trail of intentional canonical+backend overrides."""
        return list(cls._overwrite_log)

    @classmethod
    def lifecycle(cls) -> str:
        return cls._lifecycle.value

    @classmethod
    def version(cls) -> int:
        return cls._version

    @classmethod
    def _assert_writable(cls) -> None:
        if cls._lifecycle is not cls.Lifecycle.BUILDING:
            raise RuntimeError(f"operator registry is not writable: {cls._lifecycle.value}")

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
            raise RuntimeError("operator registry must be finalized before freezing")
        if cls._lifecycle is not cls.Lifecycle.FROZEN:
            # R4-102: the catalog backend set must equal the runtime backend set
            # for every canonical.  A split (e.g. a catalog-only overlay that
            # cleared backends, or a runtime registration that never refreshed
            # the catalog) would let the manifest / surface layer advertise a
            # different backend capability than the runtime actually holds.
            # ``register_catalog_only(..., merge_existing=True)`` already blocks
            # the intentional split (P1-38); this freeze gate catches any
            # residual drift from renames / unregisters / overlay layers.
            for _canonical, _implementations in cls._operators.items():
                _runtime = sorted(_implementations.keys())
                _catalog_backends = sorted(
                    cls._catalog.get(_canonical, {}).get("backends") or []
                )
                if _runtime != _catalog_backends:
                    raise RuntimeError(
                        f"R4-102 catalog/runtime backend split for {_canonical!r}: "
                        f"catalog backends {_catalog_backends} != runtime {_runtime}. "
                        "A catalog overlay desynchronized the manifest from the "
                        "runtime; fix the registration, not the check."
                    )
            cls._lifecycle = cls.Lifecycle.FROZEN
            cls._version += 1

    @classmethod
    def thaw_for_bootstrap(cls, token: object) -> None:
        """Internal bootstrap escape hatch guarded by an unexported token."""
        if token is not _BOOTSTRAP_TOKEN:
            raise PermissionError("registry thaw requires the internal bootstrap token")
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
        replace: bool = False,
        replacement_reason: str = "",
        expected_old_source: str = "",
        semantic_version: str = "1.0",
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
            replace: 是否允许覆盖同一 canonical+backend 的既有实现。
            replacement_reason: 覆盖原因（replace=True 时必填）。
            expected_old_source: 期望被覆盖的旧 source；提供时与实测不一致会报错。

        返回:
            None

        覆盖治理（P0-31）：默认对同一 canonical+backend 的重复注册报错——静默
        覆盖会制造 load-order 相关的运行时语义漂移（DSL 认为参数=A、运行时实际
        参数=B）。只有显式 ``replace=True`` + ``replacement_reason`` 才允许覆盖，
        且每次覆盖都会写入 ``cls._overwrite_log`` 审计日志。
        """
        canonical = canonical or operator.metadata.name
        cls._assert_writable()
        if canonical in cls._aliases:
            raise ValueError(f"canonical already declared as alias: {canonical!r}")
        existing_ops = cls._operators.setdefault(canonical, {})
        if backend in existing_ops:
            old_source = str((cls._catalog.get(canonical, {}).get("backend_meta") or {}).get(backend, {}).get("source", "") or "")
            declared = source in cls._DECLARED_OVERRIDE_SOURCES
            if cls._hard_fail_duplicates and not declared:
                if not replace:
                    raise ValueError(
                        f"duplicate registration of canonical {canonical!r} backend "
                        f"{backend!r} (old source={old_source!r}, new source={source!r}). "
                        "Silent overwrite is forbidden; pass replace=True with a "
                        "replacement_reason, or add the source to "
                        "_DECLARED_OVERRIDE_SOURCES to declare it an override layer."
                    )
                # R4-101: a module-level blanket override is banned for undeclared
                # sources — every replacement must pin the exact old source it
                # expects to replace, so a future import-order shuffle cannot
                # silently swap implementations.
                if not expected_old_source:
                    raise ValueError(
                        f"replace of {canonical!r}/{backend} requires a non-empty "
                        "expected_old_source (review R4-101): pin the exact source "
                        "being replaced"
                    )
                if old_source != expected_old_source:
                    raise ValueError(
                        f"replace of {canonical!r}/{backend}: expected old source "
                        f"{expected_old_source!r} but registry holds {old_source!r}"
                    )
                if not replacement_reason:
                    raise ValueError(
                        f"replace of {canonical!r}/{backend} requires a non-empty "
                        "replacement_reason"
                    )
            # Record the audit trail for every same-backend overwrite — including
            # soft-probe mode (``_hard_fail_duplicates=False``) so a bootstrap
            # sweep can enumerate every collision site at once.
            cls._overwrite_log.append({
                "canonical": canonical,
                "backend": backend,
                "old_source": old_source,
                "new_source": source,
                "old_hash": _impl_source_hash(existing_ops[backend]),
                "new_hash": _impl_source_hash(operator),
                "reason": replacement_reason or f"declared override layer ({source})",
                "semantic_version": str(semantic_version or "1.0"),
            })
        existing_ops[backend] = operator
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
        # R4-95/98: surface the field-semantic metadata on the catalog dict so
        # catalog consumers see unit / window-semantics labels.  First
        # non-None wins: the pandas backend (registered first) typically carries
        # the annotations, while the polars/SQL markers must not wipe them.
        def _first_non_null(cur: Any, new: Any) -> Any:
            return cur if cur is not None else new

        _metadata = getattr(operator, "metadata", None)
        updated.update({
            "canonical": canonical,
            "backends": sorted(cls._operators[canonical].keys()),
            "status": status,
            "description": canonical_description,
            "param_names": canonical_params,
            "aliases": sorted(set((aliases or []) + prev.get("aliases", []))),
            "backend_meta": backend_meta,
            "window_semantics": _first_non_null(
                prev.get("window_semantics"),
                getattr(_metadata, "window_semantics", None),
            ),
            "input_units": _first_non_null(
                prev.get("input_units"),
                dict(getattr(_metadata, "input_units", None) or {}),
            ),
            "output_unit": _first_non_null(
                prev.get("output_unit"),
                getattr(_metadata, "output_unit", None),
            ),
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
    def resolve_canonical_strict(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve a registered name and fail immediately for unknown operators."""
        canonical = cls.resolve_canonical(name, max_depth=max_depth)
        if canonical not in cls._operators and canonical not in cls._catalog:
            raise KeyError(f"unknown operator canonical: {name!r}")
        return canonical

    @classmethod
    def resolve_canonical_optional(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve aliases while retaining optional lookup compatibility."""
        return cls.resolve_canonical(name, max_depth=max_depth)

    @classmethod
    def register_alias(
        cls, alias: str, canonical: str, *, replace: bool = False,
        replacement_reason: str = "",
    ) -> None:
        """Register an alias with collision and canonical-name checks."""
        cls._assert_writable()
        if alias == canonical:
            return
        if alias in cls._operators or alias in cls._catalog:
            raise ValueError(f"alias collides with canonical: {alias!r}")
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
        if canonical not in cls._operators and canonical not in cls._catalog:
            raise KeyError(f"alias target is not registered: {canonical!r}")
        probe = dict(cls._aliases)
        probe[alias] = canonical
        current = alias
        seen: set[str] = set()
        for _ in range(9):
            if current in seen:
                raise ValueError(f"alias cycle detected at {current!r}")
            seen.add(current)
            target = probe.get(current)
            if target is None:
                break
            current = target
        cls._aliases[alias] = canonical
        if canonical in cls._catalog:
            aliases = set(cls._catalog[canonical].get("aliases", []))
            aliases.add(alias)
            cls._catalog[canonical]["aliases"] = sorted(aliases)

    @classmethod
    def register_compat_alias(
        cls,
        alias: str,
        canonical: str,
        *,
        migration_reason: str,
        deprecated_since: str,
        removal_version: str,
    ) -> None:
        """Register an intentional migration alias with mandatory provenance."""
        if not all(str(x).strip() for x in (migration_reason, deprecated_since, removal_version)):
            raise ValueError("compat alias requires reason, deprecated_since and removal_version")
        cls.register_alias(
            alias,
            canonical,
            replace=alias in cls._aliases and cls._aliases.get(alias) != canonical,
            replacement_reason=migration_reason,
        )
        if alias != canonical:
            cls._catalog.setdefault(canonical, {}).setdefault("compat_aliases", {})[alias] = {
                "migration_reason": migration_reason,
                "deprecated_since": deprecated_since,
                "removal_version": removal_version,
            }

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
        merge_existing: bool = False,
    ) -> None:
        """登记仅文档/catalog 占位条目（无 runtime 实现）。

        参数:
            canonical: 占位 canonical 名。
            aliases: 可选别名。
            status: 默认 ``doc_only``。
            business_category: 业务分类标签。
            description: 人类可读说明。
            source: 溯源标记。
            merge_existing: 若 canonical 已有 runtime 而仍要覆盖 catalog，必须
                显式置 True（P1-38），否则拒绝——避免
                ``runtime 存在但 catalog.backends=[]/param_names=[]`` 的分裂状态。

        返回:
            None
        """
        cls._assert_writable()
        if canonical in cls._operators and not merge_existing:
            raise ValueError(
                f"canonical {canonical!r} already has runtime backends "
                f"{sorted(cls._operators[canonical])}; register_catalog_only would split "
                "the catalog (backends=[]) from the runtime. Pass merge_existing=True to "
                "declare it an intentional catalog-only overlay (P1-38)."
            )
        if canonical in cls._aliases:
            raise ValueError(f"canonical already declared as alias: {canonical!r}")
        previous = cls._catalog.get(canonical, {})
        stale_aliases = set(previous.get("aliases") or []) - set(aliases or [])
        for alias in stale_aliases:
            if cls._aliases.get(alias) == canonical:
                cls._aliases.pop(alias, None)
        # P1-38: with merge_existing=True, preserve the runtime-backed contract
        # (backends / param_names) instead of resetting it to [] — that split
        # state is what the guard above exists to prevent.
        cls._catalog[canonical] = {
            "canonical": canonical,
            "aliases": sorted(set(aliases or [])),
            "backends": list(
                previous.get("backends")
                or sorted(cls._operators[canonical].keys())
                if merge_existing
                else []
            ),
            "status": status,
            "description": description,
            "param_names": list(previous.get("param_names") or []) if merge_existing else [],
            "business_category": business_category,
        }
        for alias in aliases or []:
            cls.register_alias(alias, canonical)

    @classmethod
    def unregister(cls, canonical: str) -> None:
        """从 registry 移除 canonical 及其实现与 catalog 条目。

        参数:
            canonical: 待注销的 canonical 名。

        返回:
            None
        """
        cls._assert_writable()
        cls._operators.pop(canonical, None)
        cls._catalog.pop(canonical, None)
        for alias, target in list(cls._aliases.items()):
            if target == canonical:
                cls._aliases.pop(alias, None)

    @classmethod
    def rename_canonical(cls, old: str, new: str) -> None:
        """将已注册 canonical 重命名为标准名，保留 runtime 与 backend。

        参数:
            old: 旧 canonical 名。
            new: 新 canonical 名；若已存在则把 ``old`` 的 backend **合并**进 ``new`` 再注销 ``old``。

        返回:
            None
        """
        cls._assert_writable()
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
        fallback_policy: str = "error",
        data_source_kind: str = "memory",
        row_count_estimate: int | None = None,
    ) -> Tuple[Any | None, str]:
        """按策略选取最优可用 backend 及算子实例。

        参数:
            name: DSL 名或 canonical 名。
            prefer: 偏好后端，``auto`` / ``polars`` / ``pandas_numpy`` / ``sql``。

        返回:
            ``(算子实例或 None, 实际选用的 backend 名)`` 元组。
        """
        from backend.backend_router import BackendRouter

        selection = BackendRouter.select(
            cls.resolve_canonical(name),
            requested_backend=prefer,
            run_mode=mode,
            fallback_policy=fallback_policy,
            data_source_kind=data_source_kind,
            row_count_estimate=row_count_estimate,
            allow_unverified_backend=allow_unverified_backend,
        )
        return selection.operator, selection.backend

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

    @classmethod
    def snapshot(cls):
        """Return an immutable deep snapshot after the registry is frozen."""
        if cls._lifecycle is not cls.Lifecycle.FROZEN:
            raise RuntimeError("registry snapshot is available only after freeze")
        return MappingProxyType({
            "version": cls._version,
            "operators": MappingProxyType(copy.deepcopy(cls._operators)),
            "aliases": MappingProxyType(copy.deepcopy(cls._aliases)),
            "catalog": MappingProxyType(copy.deepcopy(cls._catalog)),
        })
