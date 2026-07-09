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

from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE


def _operator_backend_auto_aggressive() -> bool:
    return os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower() in {
        "auto_aggressive",
        "aggressive",
    }


def _merge_param_names(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """多 backend 注册时保留更完整的 param_names（避免 polars bridge 覆盖）。"""
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
        backend_explicit: bool = True,
    ) -> None:
        """注册一个已实现算子；``canonical`` 为内部主键，``aliases`` 为可选 DSL 别名。"""
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
    def unregister(cls, canonical: str) -> None:
        """移除重复 canonical（保留别名指向其他实现时使用）。"""
        cls._operators.pop(canonical, None)
        cls._catalog.pop(canonical, None)

    @classmethod
    def rename_canonical(cls, old: str, new: str) -> None:
        """将已注册 canonical 重命名为业界标准名（保留 runtime 与 backend）。"""
        if old == new or old not in cls._operators:
            return
        if new in cls._operators:
            cls.unregister(old)
            return
        cls._operators[new] = cls._operators.pop(old)
        meta = cls._catalog.pop(old, {})
        meta["canonical"] = new
        cls._catalog[new] = meta
        for op in cls._operators[new].values():
            if hasattr(op, "metadata"):
                op.metadata.name = new
        for alias, canon in list(cls._aliases.items()):
            if canon == old:
                cls._aliases[alias] = new

    @classmethod
    def backends_for(cls, name: str) -> List[str]:
        """返回 canonical 已注册的 backend 列表。"""
        canonical = cls._aliases.get(name, name)
        return sorted(cls._operators.get(canonical, {}).keys())

    @classmethod
    def get_preferred(
        cls,
        name: str,
        *,
        prefer: str = "auto",
    ) -> Tuple[Any | None, str]:
        """按策略选取最快可用 backend：``auto`` 优先 polars，否则 pandas。"""
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
        # auto：production 仅白名单走 polars；research 可 FACTOR_ENGINE_OPERATOR_BACKEND=auto_aggressive
        if "polars" in backends:
            if canonical in POLARS_PRODUCTION_SAFE or _operator_backend_auto_aggressive():
                return backends["polars"], "polars"
        return backends.get("pandas_numpy"), "pandas_numpy"

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
