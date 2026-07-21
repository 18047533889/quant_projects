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


class UnsupportedOperatorBackendError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used."""


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
    execution_kind: str = "unsupported"
    estimated_speedup: float = 1.0
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False
    supports_scalar_broadcast: bool = False
    supports_min_periods: bool = False
    supports_group: bool = False
    supports_window: bool = False
    supports_lazy: bool = False
    supports_streaming: bool = False
    materializes_full_panel: bool = False
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
            "execution_kind": self.execution_kind,
            "estimated_speedup": self.estimated_speedup,
            "supports_nulls": self.supports_nulls,
            "supports_nan": self.supports_nan,
            "supports_inf": self.supports_inf,
            "supports_scalar_broadcast": self.supports_scalar_broadcast,
            "supports_min_periods": self.supports_min_periods,
            "supports_group": self.supports_group,
            "supports_window": self.supports_window,
            "supports_lazy": self.supports_lazy,
            "supports_streaming": self.supports_streaming,
            "materializes_full_panel": self.materializes_full_panel,
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

    return OperatorRegistry.resolve_canonical(name)


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


_EMITTER_OK_CACHE: dict[tuple[str, str, int], bool] = {}


def _sql_emitter_ok(canon: str, *, dialect: str = "duckdb_sql") -> bool:
    """registry 声明可 SQL 且 emitter 能编译最小 plan（按版本缓存）。"""
    from cleaned_operators.registry import OperatorRegistry

    cache_key = (canon, dialect, OperatorRegistry.version())
    if cache_key in _EMITTER_OK_CACHE:
        return _EMITTER_OK_CACHE[cache_key]
    if canon in {"column", "literal"}:
        _EMITTER_OK_CACHE[cache_key] = True
        return True
    sql_set = _sql_capable_canonicals()
    if canon not in sql_set:
        _EMITTER_OK_CACHE[cache_key] = False
        return False
    ok = False
    try:
        from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql, plan_is_sql_capable
        from backend.sql_pushdown.plan_fixtures import minimal_plan

        plan = minimal_plan(canon)
        if plan_is_sql_capable(plan):
            compiled = compile_plan_to_sql(
                plan,
                dataset="_cap_check",
                table="_cap_check",
                time_column="ts",
                instrument_column="inst",
                dialect=(
                    SqlDialect.CLICKHOUSE
                    if dialect == "clickhouse_sql"
                    else SqlDialect.DUCKDB
                ),
            )
            ok = compiled is not None and bool(compiled.query.strip())
    except Exception:
        ok = False
    _EMITTER_OK_CACHE[cache_key] = ok
    return ok


def _polars_status(canon: str) -> CapabilityStatus:
    """Derive Polars status only from tested backend evidence."""
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    from cleaned_operators.registry import OperatorRegistry

    backends = OperatorRegistry.backends_for(canon)
    if "polars" not in backends:
        return "unsupported"
    if (
        canon in POLARS_REFERENCE_PARITY_VERIFIED
        and canon in POLARS_EDGE_VERIFIED
        and canon in POLARS_NO_FALLBACK_VERIFIED
    ):
        return "production_safe"
    if canon in POLARS_REFERENCE_PARITY_VERIFIED:
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
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if canon not in SQL_IMPLEMENTED_CANONICALS:
        return "unsupported"
    if not _sql_emitter_ok(canon, dialect=dialect):
        return "implemented"

    if dialect == "clickhouse_sql":
        from backend.sql_tiers import CLICKHOUSE_SQL_PARITY_VERIFIED
        from backend.sql_pushdown.clickhouse_capabilities import effective_clickhouse_production_safe

        if effective_clickhouse_production_safe(canon):
            return "production_safe"
        if canon in CLICKHOUSE_SQL_PARITY_VERIFIED:
            return "parity_verified"
        return "implemented"
    else:
        from backend.sql_tiers import DUCKDB_SQL_PARITY_VERIFIED, effective_sql_production_safe

        if effective_sql_production_safe(canon):
            return "production_safe"
        if canon in DUCKDB_SQL_PARITY_VERIFIED:
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
    from cleaned_operators.registry import OperatorRegistry
    from backend.operator_cost import default_backend_speedup

    canon = resolve_canonical(canonical)
    if canon == "if_else":
        canon = "where"
    policy = infer_operator_policy(canon)
    scope = getattr(policy, "scope", "") or ""
    supports_group = scope == "group"
    supports_window = scope == "ts"
    backend_key = "sql" if backend in _SQL_BACKENDS else backend
    backend_meta = dict(
        ((OperatorRegistry._catalog.get(canon, {}).get("backend_meta") or {}).get(backend_key) or {})
    )

    if backend == "pandas_numpy":
        status = _pandas_status(canon)
    elif backend == "polars":
        status = _polars_status(canon)
    elif backend in _SQL_BACKENDS:
        status = _sql_status(canon, dialect=backend)
    else:
        status = "unsupported"

    required_metadata = {
        "execution_kind", "supports_lazy", "supports_streaming",
        "materializes_full_panel", "supports_nulls", "supports_nan",
        "supports_inf", "supports_scalar_broadcast", "supports_group",
        "supports_window", "supports_min_periods",
    }
    if status == "production_safe":
        missing = sorted(required_metadata.difference(backend_meta))
        if missing:
            raise RuntimeError(
                f"{canon}/{backend}: production-safe capability metadata missing {missing}"
            )

    notes = ""
    if backend == "clickhouse_sql" and status != "unsupported":
        # 部分算子 DuckDB 已通、CH 方言待验
        notes = "dialect=clickhouse; verify per deployment"

    return BackendCapability(
        canonical=canon,
        backend=backend,
        status=status,
        execution_kind=str(backend_meta.get("execution_kind", "unsupported")),
        estimated_speedup=default_backend_speedup(canon, backend, status),
        supports_nulls=bool(backend_meta.get("supports_nulls", False)),
        supports_nan=bool(backend_meta.get("supports_nan", False)),
        supports_inf=bool(backend_meta.get("supports_inf", False)),
        supports_scalar_broadcast=bool(backend_meta.get("supports_scalar_broadcast", False)),
        supports_min_periods=bool(backend_meta.get("supports_min_periods", False)),
        supports_group=bool(backend_meta.get("supports_group", supports_group)),
        supports_window=bool(backend_meta.get("supports_window", supports_window)),
        supports_lazy=bool(backend_meta.get("supports_lazy", False)),
        supports_streaming=bool(backend_meta.get("supports_streaming", False)),
        materializes_full_panel=bool(backend_meta.get("materializes_full_panel", False)),
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
    if canon == "if_else":
        canon = "where"
    spec = build_operator_spec(canon)
    return OperatorCapabilitySummary(
        canonical=canon,
        pandas_numpy=_pandas_status(canon),
        polars=_polars_status(canon),
        duckdb_sql=_sql_status(canon, dialect="duckdb_sql"),
        clickhouse_sql=_sql_status(canon, dialect="clickhouse_sql"),
        allow_in_production=bool(spec.allow_in_production) if spec is not None else False,
        parity_verified=canon in POLARS_PARITY_VERIFIED,
        polars_long_tier=spec.polars_long_tier if spec is not None else "unsupported",
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
        mode: ``production`` 仅 production-safe；``research`` 允许已实现。

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
        return status == "production_safe"
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
    allow_unverified_backend: bool = False,
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
        if op is None:
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} has no pandas_numpy backend"
            )
        return op, "pandas_numpy"
    if prefer == "polars":
        op = OperatorRegistry.get(canonical, "polars")
        permitted = _polars_status(canonical) == "production_safe" or (
            mode != "production" and allow_unverified_backend
        )
        if op is not None and permitted:
            return op, "polars"
        if mode == "production":
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} polars backend is not production-safe"
            )
        if not allow_unverified_backend:
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} polars backend is not verified"
            )
        raise UnsupportedOperatorBackendError(f"{canonical!r} has no usable polars backend")
    if prefer == "sql":
        op = OperatorRegistry.get(canonical, "sql")
        dialect: BackendName = (
            "clickhouse_sql" if data_source_kind.lower() in {"clickhouse", "ch"} else "duckdb_sql"
        )
        permitted = _sql_status(canonical, dialect=dialect) == "production_safe" or (
            mode != "production" and allow_unverified_backend
        )
        if op is not None and permitted:
            return op, "sql"
        if mode == "production" or not allow_unverified_backend:
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} SQL backend is not usable for {dialect}"
            )
        raise UnsupportedOperatorBackendError(f"{canonical!r} has no usable SQL backend")

    use_cost = os.environ.get("FACTOR_ENGINE_COST_ROUTING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    aggressive_requested = os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower() in {
        "auto_aggressive",
        "aggressive",
    }
    aggressive = aggressive_requested and mode == "research" and allow_unverified_backend

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
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is not None:
            return op, "pandas_numpy"
        raise UnsupportedOperatorBackendError(
            f"no usable backend for {canonical!r}"
        )

    if use_cost and len(candidates) > 1:
        chosen = min(candidates, key=lambda x: x[1])[0]
    else:
        # Default production route consumes the same evidence-backed status as
        # reports and explicit backend selection.
        if (
            "polars" in backends
            and (
                _polars_status(canonical) == "production_safe"
                or (aggressive and _polars_status(canonical) != "unsupported")
            )
        ):
            chosen = "polars"
        else:
            chosen = "pandas_numpy"

    op = OperatorRegistry.get(canonical, chosen)
    return op, chosen
