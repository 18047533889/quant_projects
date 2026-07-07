# -*- coding: utf-8 -*-
"""算子注册中心：canonical 名、别名、元数据与 runtime 实例的唯一索引。

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
from typing import Any, Dict, List, Optional


class OperatorRegistry:
    """全局算子表；无单例类，全部classmethod访问。"""

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
    ) -> None:
        """注册一个已实现算子；``canonical`` 为内部主键，``aliases`` 为可选 DSL 别名。"""
        canonical = canonical or operator.metadata.name
        cls._operators.setdefault(canonical, {})[backend] = operator
        cls._catalog[canonical] = {
            "canonical": canonical,
            "backends": sorted(cls._operators[canonical].keys()),
            "selected_source": source,
            "status": status,
            "description": getattr(operator.metadata, "description", ""),
            "param_names": getattr(operator.metadata, "param_names", []),
            "aliases": sorted(set((aliases or []) + cls._catalog.get(canonical, {}).get("aliases", []))),
        }
        for alias in aliases or []:
            if alias != canonical:
                cls._aliases[alias] = canonical

    @classmethod
    def register_alias(cls, alias: str, canonical: str) -> None:
        """仅登记别名（实现须已通过 ``register`` 存在，或在 catalog 中占位）。"""
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
        """文档/catalog 占位，无 runtime；不会进入 DSL 白名单。"""
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
    def get(cls, name: str, backend: str = "pandas_numpy") -> Any:
        """按 DSL 名或 canonical 取算子实例；先走别名解析。"""
        canonical = cls._aliases.get(name, name)
        return cls._operators.get(canonical, {}).get(backend)

    @classmethod
    def list_canonical(cls) -> List[str]:
        """全部 canonical（含仅有 catalog、无实现的条目）。"""
        return sorted(set(cls._operators.keys()) | set(cls._catalog.keys()))

    @classmethod
    def catalog(cls) -> Dict[str, dict]:
        """导出完整 catalog 副本（脚本 ``export_dsl_allowlist`` 等使用）。"""
        return dict(cls._catalog)
