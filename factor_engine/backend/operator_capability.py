# -*- coding: utf-8
"""统一 Backend 能力注册表：声明式覆盖 + 成本路由（不强制全量 Polars/SQL）。

路由原则（production）::

    能 SQL 下推的 → Hybrid/SQL 层优先（整树或子树）；
    SQL 不适合但 Polars parity 已验证 → Polars；
    语义不稳 / 未验证 / 不划算 → Pandas fallback（须可观测）。

本模块负责 **算子级能力声明** 与 **Polars/Pandas 热路径选型**；
SQL 下推在 ``backend/sql_pushdown`` + HybridBackend 层完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

BackendName = Literal[
    "pandas_numpy",
    "polars",
    "duckdb_sql",
    "clickhouse_sql",
]

CapabilityStatus = Literal[
    "unsupported",
    "implemented",
    "parity_verified",
    "production_safe",
]

# registry backend → capability backend
_REGISTRY_TO_CAPABILITY: dict[str, BackendName | None] = {
    "pandas_numpy": "pandas_numpy",
    "polars": "polars",
    "sql": "duckdb_sql",
}

_SQL_BACKENDS: tuple[BackendName, ...] = ("duckdb_sql", "clickhouse_sql")


@dataclass(frozen=True)
class BackendCapability:
    """单个 canonical × backend 的能力行。"""

    canonical: str
    backend: BackendName
    status: CapabilityStatus
    estimated_speedup: float = 1.0
    supports_nulls: bool = True
    supports_min_periods: bool = True
    supports_group: bool = False
    supports_window: bool = False
    notes: str = ""

    def to_csv_row(self) -> dict[str, str | float | bool]:
        """导出为 CSV 扁平行字典。

        返回:
            含 canonical、backend、status 及各能力标志字段的字典。
        """
        return {
            "canonical": self.canonical,
            "backend": self.backend,
            "status": self.status,
            "estimated_speedup": self.estimated_speedup,
            "supports_nulls": self.supports_nulls,
            "supports_min_periods": self.supports_min_periods,
            "supports_group": self.supports_group,
            "supports_window": self.supports_window,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class OperatorCapabilitySummary:
    """按 canonical 聚合的多 backend 视图（报表 / 路由）。"""

    canonical: str
    pandas_numpy: CapabilityStatus
    polars: CapabilityStatus
    duckdb_sql: CapabilityStatus
    clickhouse_sql: CapabilityStatus
    allow_in_production: bool
    parity_verified: bool
    polars_long_tier: str = "unsupported"
    notes: str = ""


def resolve_canonical(name: str) -> str:
    """将算子别名解析为 canonical 名称。

    参数:
        name: 算子名或别名。

    返回:
        解析后的 canonical 名称。
    """
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(name, name)


def polars_long_native(canonical: str) -> bool:
    """判断 canonical 是否在纯 Polars Expr native 白名单内。

    参数:
        canonical: 算子 canonical 名称或别名。

    返回:
        是否属于 ``POLARS_LONG_NATIVE`` 集合。
    """
    from backend.polars_long_policy import POLARS_LONG_NATIVE

    return resolve_canonical(canonical) in POLARS_LONG_NATIVE


def polars_long_tier(canonical: str) -> str:
    """查询 canonical 的 Polars long-table 能力 tier。

    参数:
        canonical: 算子 canonical 名称。

    返回:
        tier 标签，如 ``native``、``map_groups``、``unsupported`` 等。
    """
    from backend.polars_long_policy import infer_polars_long_tier

    return infer_polars_long_tier(canonical)


def polars_long_tier_status(canon: str) -> CapabilityStatus:
    """将 long-table tier 映射为 capability status（production 路由用）。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        对应 backend 能力状态标签。
    """
    tier = polars_long_tier(canon)
    if tier == "unsupported":
        return "unsupported"
    if tier == "native":
        from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

        return "production_safe" if canon in POLARS_PRODUCTION_SAFE else "parity_verified"
    if tier == "blocked_causal":
        return "unsupported"
    if tier in {"map_groups", "passthrough", "registry"}:
        return "implemented"
    return "unsupported"


def polars_expr_capable(canonical: str) -> bool:
    """判断 canonical 是否在手写 Polars expr / map_groups 编译白名单内。

    参数:
        canonical: 算子 canonical 名称或别名。

    返回:
        是否属于 ``POLARS_LONG_COMPATIBLE`` 集合。
    """
    from backend.polars_long_policy import POLARS_LONG_COMPATIBLE

    return resolve_canonical(canonical) in POLARS_LONG_COMPATIBLE


def polars_long_capable(canonical: str) -> bool:
    """判断 canonical 是否可走 polars_long（native + map_groups + registry bridge）。

    参数:
        canonical: 算子 canonical 名称或别名。

    返回:
        是否在 ``get_polars_long_capable()`` 返回的集合内。
    """
    from backend.polars_long_policy import get_polars_long_capable

    return resolve_canonical(canonical) in get_polars_long_capable()


def _sql_capable_canonicals() -> frozenset[str]:
    """返回 SQL 已实现 canonical 集合。"""
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    return SQL_IMPLEMENTED_CANONICALS


_EMITTER_OK_CACHE: dict[str, bool] = {}


def _sql_emitter_ok(canon: str) -> bool:
    """registry 声明可 SQL 且 emitter 能编译最小 plan（带缓存）。"""
    if canon in _EMITTER_OK_CACHE:
        return _EMITTER_OK_CACHE[canon]
    if canon in {"column", "literal"}:
        _EMITTER_OK_CACHE[canon] = True
        return True
    sql_set = _sql_capable_canonicals()
    if canon not in sql_set:
        _EMITTER_OK_CACHE[canon] = False
        return False
    ok = False
    try:
        from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
        from backend.sql_pushdown.plan_fixtures import minimal_plan

        plan = minimal_plan(canon)
        if plan_is_sql_capable(plan):
            compiled = compile_plan_to_sql(
                plan,
                dataset="_cap_check",
                time_column="ts",
                instrument_column="inst",
            )
            ok = compiled is not None and bool(compiled.query.strip())
    except Exception:
        ok = False
    _EMITTER_OK_CACHE[canon] = ok
    return ok


def _polars_status(canon: str) -> CapabilityStatus:
    """根据 registry 与 parity 白名单推断 Polars 能力状态。"""
    from cleaned_operators.operator_policy import (
        POLARS_PARITY_VERIFIED,
        POLARS_PRODUCTION_SAFE,
    )
    from cleaned_operators.registry import OperatorRegistry

    backends = OperatorRegistry.backends_for(canon)
    if "polars" not in backends:
        return "unsupported"
    if canon in POLARS_PRODUCTION_SAFE:
        return "production_safe"
    if canon in POLARS_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def _pandas_status(canon: str) -> CapabilityStatus:
    """根据 registry 推断 Pandas/Numpy 能力状态。"""
    from cleaned_operators.registry import OperatorRegistry

    if "pandas_numpy" in OperatorRegistry.backends_for(canon):
        return "implemented"
    return "unsupported"


def _sql_status(canon: str, *, dialect: BackendName) -> CapabilityStatus:
    """根据 SQL tier 与 emitter 能力推断指定方言的 SQL 状态。"""
    from backend.sql_tiers import (
        SQL_IMPLEMENTED_CANONICALS,
        SQL_PARITY_VERIFIED_CANONICALS,
    )

    if canon not in SQL_IMPLEMENTED_CANONICALS:
        return "unsupported"
    if not _sql_emitter_ok(canon):
        return "implemented"

    if dialect == "clickhouse_sql":
        from backend.sql_pushdown.clickhouse_capabilities import effective_clickhouse_production_safe

        if effective_clickhouse_production_safe(canon):
            return "production_safe"
    else:
        from backend.sql_tiers import effective_sql_production_safe

        if effective_sql_production_safe(canon):
            return "production_safe"

    if canon in SQL_PARITY_VERIFIED_CANONICALS:
        return "parity_verified"
    return "implemented"


def capability_for(canonical: str, backend: BackendName) -> BackendCapability:
    """导出单个 canonical × backend 能力行。

    参数:
        canonical: 算子 canonical 名称或别名。
        backend: 目标 backend 名称。

    返回:
        含状态、加速比与各能力标志的 ``BackendCapability``。
    """
    from cleaned_operators.operator_policy import infer_operator_policy
    from backend.operator_cost import default_backend_speedup

    canon = resolve_canonical(canonical)
    policy = infer_operator_policy(canon)
    scope = getattr(policy, "scope", "") or ""
    supports_group = canon.startswith("group_") or scope == "cs"
    supports_window = canon.startswith("ts_") or scope == "ts"

    if backend == "pandas_numpy":
        status = _pandas_status(canon)
    elif backend == "polars":
        status = _polars_status(canon)
    elif backend in _SQL_BACKENDS:
        status = _sql_status(canon, dialect=backend)
    else:
        status = "unsupported"

    notes = ""
    if backend == "clickhouse_sql" and status != "unsupported":
        # 部分算子 DuckDB 已通、CH 方言待验
        notes = "dialect=clickhouse; verify per deployment"

    return BackendCapability(
        canonical=canon,
        backend=backend,
        status=status,
        estimated_speedup=default_backend_speedup(canon, backend, status),
        supports_nulls=True,
        supports_min_periods=supports_window,
        supports_group=supports_group,
        supports_window=supports_window,
        notes=notes,
    )


def summarize_operator(canonical: str) -> OperatorCapabilitySummary:
    """按 canonical 聚合四 backend 能力状态。

    参数:
        canonical: 算子 canonical 名称或别名。

    返回:
        多 backend 聚合视图 ``OperatorCapabilitySummary``。
    """
    from cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED
    from cleaned_operators.operator_spec import build_operator_spec

    canon = resolve_canonical(canonical)
    spec = build_operator_spec(canon)
    return OperatorCapabilitySummary(
        canonical=canon,
        pandas_numpy=_pandas_status(canon),
        polars=_polars_status(canon),
        duckdb_sql=_sql_status(canon, dialect="duckdb_sql"),
        clickhouse_sql=_sql_status(canon, dialect="clickhouse_sql"),
        allow_in_production=bool(spec.allow_in_production),
        parity_verified=canon in POLARS_PARITY_VERIFIED,
        polars_long_tier=spec.polars_long_tier,
    )


def build_capability_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[OperatorCapabilitySummary]:
    """构建全量或子集 capability 矩阵（按 resolve_canonical 去重）。

    参数:
        canonicals: 可选 canonical 子集；为 ``None`` 时遍历 registry 全量。

    返回:
        ``OperatorCapabilitySummary`` 列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    if canonicals is None:
        seen: set[str] = set()
        names: list[str] = []
        for c in OperatorRegistry.list_canonical():
            if not OperatorRegistry.backends_for(c):
                continue
            rc = resolve_canonical(c)
            if rc in seen:
                continue
            seen.add(rc)
            names.append(rc)
    else:
        names = sorted({resolve_canonical(c) for c in canonicals})
    return [summarize_operator(c) for c in sorted(names)]


def export_flat_capabilities(
    canonicals: Sequence[str] | None = None,
) -> list[BackendCapability]:
    """展开为 canonical × backend 扁平行（CSV 导出）。

    参数:
        canonicals: 可选 canonical 子集。

    返回:
        ``BackendCapability`` 扁平行列表。
    """
    rows: list[BackendCapability] = []
    for summary in build_capability_matrix(canonicals):
        for backend in ("pandas_numpy", "polars", "duckdb_sql", "clickhouse_sql"):
            rows.append(capability_for(summary.canonical, backend))
    return rows


def supports_polars(canonical: str, *, mode: str = "production") -> bool:
    """判断 canonical 是否可走 Polars 热路径。

    参数:
        canonical: 算子 canonical 名称或别名。
        mode: ``production`` 仅 production_safe；``research`` 允许已实现/parity。

    返回:
        当前模式下是否支持 Polars 执行。
    """
    status = _polars_status(resolve_canonical(canonical))
    if status == "unsupported":
        return False
    if mode == "production":
        return status == "production_safe"
    return status in {"implemented", "parity_verified", "production_safe"}


def supports_sql(
    canonical: str,
    data_source_kind: str = "duckdb",
    *,
    mode: str = "production",
) -> bool:
    """判断 canonical 在指定数据源方言下是否可走 SQL 路径。

    参数:
        canonical: 算子 canonical 名称。
        data_source_kind: 数据源类型，``duckdb`` 或 ``clickhouse``/``ch``。
        mode: ``production`` 仅 parity/production_safe；``research`` 允许已实现。

    返回:
        是否支持 SQL 执行。
    """
    canon = resolve_canonical(canonical)
    dialect: BackendName = (
        "clickhouse_sql" if data_source_kind.lower() in {"clickhouse", "ch"} else "duckdb_sql"
    )
    status = _sql_status(canon, dialect=dialect)
    if status == "unsupported":
        return False
    if mode == "production":
        return status in {"parity_verified", "production_safe"}
    return status != "unsupported"


def supports_pandas(canonical: str) -> bool:
    """判断 canonical 是否有 Pandas/Numpy 实现。

    参数:
        canonical: 算子 canonical 名称。

    返回:
        registry 中是否注册了 ``pandas_numpy`` backend。
    """
    return _pandas_status(resolve_canonical(canonical)) != "unsupported"


def get_best_backend(
    name: str,
    *,
    mode: str = "production",
    data_source_kind: str = "memory",
    row_count_estimate: int | None = None,
    prefer: str = "auto",
) -> tuple[object | None, str]:
    """算子热路径 backend 选型：Polars（已验证） vs Pandas fallback。

    SQL 不在此函数选择——由 HybridBackend / SQL lowerer 在 plan 层处理。

    参数:
        name: 算子名称或别名。
        mode: 运行模式，影响 Polars 白名单严格程度。
        data_source_kind: 数据源类型（本函数内未直接用于 SQL 选型）。
        row_count_estimate: 可选行数估计，用于成本路由。
        prefer: 强制 backend（``pandas_numpy``/``polars``/``sql``/``auto``）。

    返回:
        ``(算子实例或 None, backend 名称)`` 元组。
    """
    import os

    from cleaned_operators.registry import OperatorRegistry
    from backend.operator_cost import estimate_backend_cost

    canonical = resolve_canonical(name)
    backends = OperatorRegistry.backends_for(canonical)

    if prefer == "pandas_numpy":
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        return op, "pandas_numpy"
    if prefer == "polars":
        op = OperatorRegistry.get(canonical, "polars")
        if op is not None:
            return op, "polars"
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        return op, "pandas_numpy"
    if prefer == "sql":
        op = OperatorRegistry.get(canonical, "sql")
        if op is not None:
            return op, "sql"
        prefer = "auto"

    use_cost = os.environ.get("FACTOR_ENGINE_COST_ROUTING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    aggressive = os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower() in {
        "auto_aggressive",
        "aggressive",
    }

    candidates: list[tuple[str, float]] = []

    if "polars" in backends:
        pol_status = _polars_status(canonical)
        pol_ok = pol_status == "production_safe" or (
            aggressive and pol_status != "unsupported"
        )
        if pol_ok:
            cost = estimate_backend_cost(
                canonical,
                "polars",
                row_count_estimate=row_count_estimate,
                requires_conversion=True,
            )
            candidates.append(("polars", cost))

    if "pandas_numpy" in backends:
        cost = estimate_backend_cost(
            canonical,
            "pandas_numpy",
            row_count_estimate=row_count_estimate,
            requires_conversion=False,
        )
        candidates.append(("pandas_numpy", cost))

    if not candidates:
        return None, "pandas_numpy"

    if use_cost and len(candidates) > 1:
        chosen = min(candidates, key=lambda x: x[1])[0]
    else:
        # 默认：与 registry.get_preferred 一致 — production 仅 safe 白名单
        from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

        if (
            "polars" in backends
            and (
                canonical in POLARS_PRODUCTION_SAFE
                or (aggressive and _polars_status(canonical) != "unsupported")
            )
        ):
            chosen = "polars"
        else:
            chosen = "pandas_numpy"

    op = OperatorRegistry.get(canonical, chosen)
    return op, chosen
