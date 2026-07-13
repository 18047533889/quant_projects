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

import os
from typing import Any, Dict, List, Optional, Tuple


def _merge_param_names(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """多 backend 注册时合并 param_names，保留更完整的参数契约。

    参数:
        existing: 已有 catalog 中的 param_names。
        new: 本次注册算子 metadata 中的 param_names。

    返回:
        合并后的参数名列表；同长度时优先含 ``benchmark_ret`` 的 CAPM 契约。
    """
    old = list(existing or [])
    cur = list(new or [])
    if not old:
        return cur
    if not cur:
        return old
    # 同长度时优先含 benchmark_ret/ret 的契约（CAPM 类算子）
    if len(cur) == len(old):
        if "benchmark_ret" in cur and "benchmark_ret" not in old:
            return cur
        if "benchmark_ret" in old:
            return old
    if len(cur) >= len(old):
        return cur
    return old


class OperatorRegistry:
    """全局算子注册表（类级存储，无单例实例）。

    维护 canonical → backend → 算子实例、别名映射与 catalog 元数据。
    全部接口通过 ``classmethod`` 访问，线程安全由 import 时序保证。
    """

    _operators: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}
    _catalog: Dict[str, dict] = {}

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
        cls._operators.setdefault(canonical, {})[backend] = operator
        existing = cls._catalog.get(canonical, {})
        if existing and status == "implemented":
            prev_status = str(existing.get("status", "implemented") or "implemented")
            if prev_status not in ("implemented", "production"):
                status = prev_status
        prev = existing
        backend_meta = dict(prev.get("backend_meta") or {})
        backend_meta[backend] = {"explicit": backend_explicit, "source": source}
        cls._catalog[canonical] = {
            "canonical": canonical,
            "backends": sorted(cls._operators[canonical].keys()),
            "selected_source": source,
            "status": status,
            "description": getattr(operator.metadata, "description", ""),
            "param_names": _merge_param_names(
                prev.get("param_names"),
                getattr(operator.metadata, "param_names", []),
            ),
            "aliases": sorted(set((aliases or []) + prev.get("aliases", []))),
            "backend_meta": backend_meta,
        }
        for alias in aliases or []:
            if alias != canonical:
                cls._aliases[alias] = canonical

    @classmethod
    def register_alias(cls, alias: str, canonical: str) -> None:
        """登记 DSL 别名 → canonical 映射。

        参数:
            alias: DSL 侧名称（如 ``ts_rsi``）。
            canonical: 目标 canonical 名（如 ``RSI``）。

        返回:
            None
        """
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
        canonical = cls._aliases.get(name, name)
        return sorted(cls._operators.get(canonical, {}).keys())

    @classmethod
    def get_preferred(
        cls,
        name: str,
        *,
        prefer: str = "auto",
    ) -> Tuple[Any | None, str]:
        """按策略选取最优可用 backend 及算子实例。

        参数:
            name: DSL 名或 canonical 名。
            prefer: 偏好后端，``auto`` / ``polars`` / ``pandas_numpy`` / ``sql``。

        返回:
            ``(算子实例或 None, 实际选用的 backend 名)`` 元组。
        """
        canonical = cls._aliases.get(name, name)
        backends = cls._operators.get(canonical, {})
        if prefer == "pandas_numpy":
            op = backends.get("pandas_numpy")
            return op, "pandas_numpy"
        if prefer == "polars":
            op = backends.get("polars")
            if op is not None:
                return op, "polars"
            return backends.get("pandas_numpy"), "pandas_numpy"
        if prefer == "sql":
            if "sql" in backends:
                return backends["sql"], "sql"
            return cls.get_preferred(name, prefer="auto")
        # auto：Hybrid 路由 — SQL 在 plan 层；此处 Polars safe vs Pandas fallback
        from backend.operator_capability import get_best_backend

        return get_best_backend(name, prefer="auto")

    @classmethod
    def get(cls, name: str, backend: str = "pandas_numpy") -> Any:
        """按名称与 backend 获取算子实例。

        参数:
            name: DSL 名或 canonical 名（先走别名解析）。
            backend: 目标后端，默认 ``pandas_numpy``。

        返回:
            算子实例；未注册时返回 ``None``。
        """
        canonical = cls._aliases.get(name, name)
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
        """导出完整 catalog 浅拷贝。

        返回:
            ``canonical → catalog 元数据`` 字典副本，供白名单导出与文档生成使用。
        """
        return dict(cls._catalog)
