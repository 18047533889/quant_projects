# -*- coding: utf-8 -*-
"""Single authority for backend capability registry and evidence-constrained cost router.

This module is the unified source of truth for all backend capability queries.
Production admission is two-dimensional: the canonical operator must be a
reviewed production target, and the selected physical backend must carry valid
execution evidence.  Registry lifecycle labels, implementation presence and
Pandas-first tier membership are never sufficient by themselves.

Prior duplication note: backend/capability_registry.py was a facade wrapper
that claimed to be the "unified authority" but only delegated to this module.
It has been deprecated in favor of this single authority.

FE-P0-003: All enums now imported from backend.contracts for ABI consistency.
FE-P0-004: PhysicalImplementationSpec imported from unified authority.
"""
from __future__ import annotations

import hashlib
import dataclasses
import enum
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

# Import unified enums from contracts (FE-P0-003)
from backend.contracts import (
    Accelerator,
    BackendFamily,
    BackendKind,
    CapabilityLevel,
    ExecutionKind,
    PhysicalImplementationID,
    PhysicalImplementationSpec,
)

BackendName = Literal[
    "pandas_numpy",
    "polars",
    "duckdb_sql",
    "clickhouse_sql",
    "q_kdb",
]
CapabilityStatus = Literal[
    "unsupported",
    "implemented",
    "parity_verified",
    "production_safe",
]

# Version: incremented when capability semantics change (merged from capability_registry)
CAPABILITY_REGISTRY_VERSION = "v2.1.0"  # Bumped for FE-P0-003/004/005 unification


class UnsupportedOperatorBackendError(RuntimeError):
    """Raised when the requested/automatic backend has no eligible implementation."""


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend is unavailable at runtime."""


class CapabilityInfrastructureError(RuntimeError):
    """Raised when capability identity computation hits an infrastructure error.

    #213：production capability identity 不允许 repr fallback——typed serializer
    error（payload 不可序列化/不稳定 hash）是 infrastructure failure，不是可以
    悄悄降级成 repr 的普通错误。调用方必须把该错误当作 capability 计算失败处理，
    而不是拿到一个基于 repr 的伪 identity。
    """


_REGISTRY_TO_CAPABILITY: dict[str, BackendName | None] = {
    "pandas_numpy": "pandas_numpy",
    "polars": "polars",
    "sql": "duckdb_sql",
    "q_kdb": "q_kdb",
}
_SQL_BACKENDS: tuple[BackendName, ...] = ("duckdb_sql", "clickhouse_sql")


@dataclass(frozen=True)
class BackendCapability:
    """Unified backend capability record (FE-BE-P0-002: merged with BackendCapabilityRecord).

    Single authority for backend capability metadata. Uses typed enums for all
    classification fields to ensure ABI consistency across the platform.

    FE-BE-P0-001: Single authority - all capability queries return this type.
    FE-BE-P0-002: Unified record - no duplicate capability dataclasses.
    """
    canonical: str
    backend: BackendKind
    level: CapabilityLevel
    execution_kind: ExecutionKind

    # Semantic support flags
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False
    supports_scalar_broadcast: bool = False
    supports_min_periods: bool = False
    supports_group: bool = False
    supports_window: bool = False

    # Execution mode flags
    supports_lazy: bool = False
    supports_streaming: bool = False
    materializes_full_panel: bool = False

    # Cost estimation
    estimated_speedup: float = 1.0

    # Version tracking
    registry_version: str = CAPABILITY_REGISTRY_VERSION

    # Notes
    notes: str = ""

    def to_csv_row(self) -> dict[str, str | float | bool]:
        return {
            "canonical": self.canonical,
            "backend": self.backend.value if isinstance(self.backend, BackendKind) else str(self.backend),
            "level": self.level.value if isinstance(self.level, CapabilityLevel) else str(self.level),
            "execution_kind": self.execution_kind.value if isinstance(self.execution_kind, ExecutionKind) else str(self.execution_kind),
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

    def is_production_eligible(self) -> bool:
        """Check if this capability is production-safe."""
        return self.level == CapabilityLevel.PRODUCTION_SAFE

    def is_native_execution(self) -> bool:
        """Check if execution is truly native (not delegate).

        FE-BE-P0-002: Updated to use unified ExecutionKind enum members.
        R21-IS-NATIVE-EXECUTION-SYNC: All native ExecutionKind members are
        covered. New backends (Numba CPU kernel, ClickHouse SQL, Q/kdb) must
        be listed here so that downstream routing can rely on a single
        capability query instead of ad-hoc enum checks.
        """
        return self.execution_kind in {
            ExecutionKind.NATIVE_EXPR,
            ExecutionKind.NATIVE_GROUP,
            ExecutionKind.NATIVE_STREAMING,
            ExecutionKind.POLARS_NATIVE_EXPR,
            ExecutionKind.POLARS_NUMPY_KERNEL,
            ExecutionKind.DUCKDB_NATIVE_SQL,
            ExecutionKind.NUMBA_CPU_KERNEL,
            ExecutionKind.CLICKHOUSE_NATIVE_SQL,
            ExecutionKind.Q_NATIVE,
        }

    # Legacy compatibility properties
    @property
    def status(self) -> CapabilityStatus:
        """Legacy property mapping level to status string."""
        return self.level.value  # type: ignore


@dataclass(frozen=True)
class OperatorCapabilitySummary:
    canonical: str
    pandas_numpy: CapabilityStatus
    polars: CapabilityStatus
    duckdb_sql: CapabilityStatus
    clickhouse_sql: CapabilityStatus
    q_kdb: CapabilityStatus
    allow_in_production: bool
    parity_verified: bool
    polars_long_tier: str = "unsupported"
    notes: str = ""


@dataclass(frozen=True)
class CapabilityDecision:
    """调用级（canonical + bound params × dialect）SQL capability 判定。"""

    canonical: str
    dialect: str
    supported: bool
    production_safe: bool
    reason: str
    estimated_cost: float = 1.0


@dataclass(frozen=True)
class PhysicalInventoryAdmission:
    """Fail-closed admission truth for one planner-selectable implementation."""

    production_surface: bool
    policy_allows_production: bool
    evidence_production_safe: bool
    spec_complete: bool
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalInventoryRecord:
    """Immutable canonical×physical-path inventory row."""

    canonical: str
    registry_slot: str
    backend: str
    implementation_id: PhysicalImplementationID | None
    spec: PhysicalImplementationSpec | None
    admission: PhysicalInventoryAdmission


def _declared_physical_spec(
    operator: Any,
    backend: str,
) -> PhysicalImplementationSpec | None:
    """Read an explicit backend-specific declaration; never infer from code."""
    if operator is None:
        return None
    for attribute in ("_physical_specs", "physical_specs"):
        specs = getattr(operator, attribute, None)
        if isinstance(specs, dict):
            spec = specs.get(backend)
            return spec if isinstance(spec, PhysicalImplementationSpec) else None
    factory = getattr(operator, "physical_spec_for_backend", None)
    if callable(factory):
        try:
            spec = factory(backend)
        except Exception:
            return None
        return spec if isinstance(spec, PhysicalImplementationSpec) else None
    spec = getattr(operator, "_physical_spec", None)
    if isinstance(spec, PhysicalImplementationSpec):
        return spec
    factory = getattr(operator, "physical_spec", None)
    if callable(factory):
        try:
            spec = factory()
        except Exception:
            return None
        if isinstance(spec, PhysicalImplementationSpec):
            return spec
    return None


def _default_production_surface(canonical: str, catalog: dict[str, Any]) -> bool:
    """Query DirectUse rather than treating registry presence as production."""
    try:
        from mining.direct_use import PublicMiningDisposition, public_mining_disposition, resolve_direct_use

        contract = resolve_direct_use(canonical, catalog)
        return public_mining_disposition(contract.status) == PublicMiningDisposition.DIRECT_VISIBLE
    except Exception:
        return False


def _default_policy_admission(canonical: str) -> bool:
    """Query the canonical spec/policy authority, failing closed on any error."""
    try:
        from cleaned_operators.operator_spec import build_operator_spec

        spec = build_operator_spec(canonical)
        return bool(spec is not None and spec.allow_in_production)
    except Exception:
        return False


def enumerate_physical_inventory(
    *,
    registry: Any | None = None,
    production_surface: Any | None = None,
    policy_admission: Any | None = None,
    evidence_admission: Any | None = None,
) -> tuple[PhysicalInventoryRecord, ...]:
    """Enumerate every selectable registry slot with fail-closed admission truth.

    Injectable authorities keep the oracle focused in tests. Production defaults
    query OperatorRegistry, DirectUse/mining role, OperatorSpec/policy, and backend
    evidence. Missing, incomplete, or identity-mismatched declarations remain
    visible inventory rows but are never admitted.
    """
    if registry is None:
        from cleaned_operators.registry import OperatorRegistry

        registry = OperatorRegistry
    surface_query = production_surface or _default_production_surface
    policy_query = policy_admission or _default_policy_admission
    evidence_query = evidence_admission

    rows: list[PhysicalInventoryRecord] = []
    for canonical in sorted(set(registry.list_canonical())):
        catalog = dict(getattr(registry, "_catalog", {}).get(canonical, {}) or {})
        try:
            on_surface = bool(surface_query(canonical, catalog))
        except Exception:
            on_surface = False
        try:
            policy_ok = bool(policy_query(canonical))
        except Exception:
            policy_ok = False
        for slot in sorted(set(registry.backends_for(canonical))):
            physical_backends = (
                ("duckdb_sql", "clickhouse_sql") if slot == "sql" else (slot,)
            )
            registry_get = registry.get
            try:
                parameters = inspect.signature(registry_get).parameters.values()
                supports_mode = any(
                    parameter.name == "mode"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_mode = True
            try:
                if supports_mode:
                    operator = registry_get(canonical, slot, mode="any")
                else:
                    # Compatibility only when the callable signature proves that
                    # the injected/legacy registry has no mode parameter.
                    operator = registry_get(canonical, slot)
            except Exception:
                operator = None
            for backend in physical_backends:
                spec = _declared_physical_spec(operator, backend)
                reasons: list[str] = []
                if operator is None:
                    reasons.append("selectable registry slot has no implementation")
                if not on_surface:
                    reasons.append("not on DirectUse production surface")
                if not policy_ok:
                    reasons.append("operator spec/policy denies production")
                if spec is None:
                    spec_complete = False
                    implementation_id = None
                    reasons.append("missing explicit PhysicalImplementationSpec")
                else:
                    errors = list(spec.validation_errors())
                    if spec.canonical != canonical:
                        errors.append("canonical identity mismatch")
                    if spec.backend != backend:
                        errors.append("backend identity mismatch")
                    if not spec.is_production_eligible():
                        errors.append("spec is not production eligible")
                    spec_complete = not errors
                    implementation_id = spec.physical_implementation_id if spec_complete else None
                    reasons.extend(errors)
                # Production evidence must name the exact immutable physical ID.
                # Legacy canonical/backend evidence remains descriptive elsewhere
                # but cannot admit an inventory row.
                if evidence_query is None or implementation_id is None:
                    evidence_ok = False
                else:
                    try:
                        evidence_ok = bool(
                            evidence_query(canonical, backend, implementation_id)
                        )
                    except Exception:
                        evidence_ok = False
                if not evidence_ok:
                    reasons.append("physical ID lacks matching production evidence")
                admitted = bool(
                    operator is not None
                    and on_surface
                    and policy_ok
                    and evidence_ok
                    and spec_complete
                    and implementation_id is not None
                )
                rows.append(
                    PhysicalInventoryRecord(
                        canonical=canonical,
                        registry_slot=slot,
                        backend=backend,
                        implementation_id=implementation_id,
                        spec=spec,
                        admission=PhysicalInventoryAdmission(
                            production_surface=on_surface,
                            policy_allows_production=policy_ok,
                            evidence_production_safe=evidence_ok,
                            spec_complete=spec_complete,
                            admitted=admitted,
                            reasons=tuple(dict.fromkeys(reasons)),
                        ),
                    )
                )
    return tuple(rows)


# FE-BE-P0-002: BackendCapabilityRecord is now an alias for BackendCapability
BackendCapabilityRecord = BackendCapability


@dataclass(frozen=True)
class CapabilityQueryResult:
    """Result of a capability query with reasoning (merged from capability_registry).

    FE-BE-P0-002: Uses unified BackendCapability (BackendCapabilityRecord is an alias).
    """
    supported: bool
    production_safe: bool
    record: BackendCapability | None
    reason: str
    registry_version: str = CAPABILITY_REGISTRY_VERSION


def resolve_canonical(name: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.resolve_canonical(name)


def polars_long_native(canonical: str) -> bool:
    from backend.polars_long_policy import POLARS_LONG_NATIVE

    return resolve_canonical(canonical) in POLARS_LONG_NATIVE


def polars_long_tier(canonical: str) -> str:
    from backend.polars_long_policy import infer_polars_long_tier

    return infer_polars_long_tier(canonical)


def polars_long_tier_status(canon: str) -> CapabilityStatus:
    from backend.polars_long_production import polars_long_production_tier

    tier = polars_long_production_tier(resolve_canonical(canon))
    if tier == "production_safe":
        return "production_safe"
    if tier == "parity_verified":
        return "parity_verified"
    if tier in {
        "implemented",
        "stateful",
        "python_rolling",
        "map_groups",
        "registry",
        "passthrough",
        "nonstandard_alg",
    }:
        return "implemented"
    return "unsupported"


def polars_expr_capable(canonical: str) -> bool:
    from backend.polars_long_policy import POLARS_LONG_COMPATIBLE

    return resolve_canonical(canonical) in POLARS_LONG_COMPATIBLE


def polars_long_capable(canonical: str) -> bool:
    from backend.polars_long_policy import get_polars_long_capable

    return resolve_canonical(canonical) in get_polars_long_capable()


def _sql_capable_canonicals() -> frozenset[str]:
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    return SQL_IMPLEMENTED_CANONICALS


#: #364：per-canonical×dialect 覆盖——即使 backend_meta 声明为 streaming，这些
#: canonical 的 SQL 实现实际是 self-join / rolling OLS / 全局结构，并非真 streaming。
_SQL_STREAMING_LIMITATIONS: frozenset[str] = frozenset(
    {
        # EWM 家族：EWMA self join 依赖前序状态，不是真 streaming
        "ts_ema",
        "ewm_std",
        "ewm_var",
        "ewm_cov",
        "ewm_corr",
        "ts_ewm_std",
        "ts_ewm_var",
        "ts_ewm_cov",
        "ts_ewm_corr",
        "RSI_WILDER",
        "ATR_WILDER",
        "ADX",
        "DX",
        "DMI_plus",
        "DMI_minus",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "DEMA",
        "TEMA",
        # rolling regression：窗口内 OLS 需要每窗口状态，不是 streaming
        "ts_beta",
        "rolling_beta",
        "return_volume_beta",
        "return_turnover_beta",
        "ts_regression_slope",
        "ts_regression_tstat",
        "ts_time_slope",
        # KNN / graph / matrix_profile：全局结构，整宽物化
        "knn",
        "graph",
        "matrix_profile",
        "ts_matrix_profile",
    }
)


_EMITTER_SOURCE_HASH: str | None = None


def _emitter_source_hash() -> str:
    """``backend.sql_pushdown.emitter`` 模块源码的 sha256 前缀（#390）。"""
    global _EMITTER_SOURCE_HASH
    if _EMITTER_SOURCE_HASH is not None:
        return _EMITTER_SOURCE_HASH
    import hashlib
    import inspect

    try:
        from backend.sql_pushdown import emitter as _emitter_mod

        src = inspect.getsourcefile(_emitter_mod)
        if src:
            with open(src, "rb") as fh:
                _EMITTER_SOURCE_HASH = hashlib.sha256(fh.read()).hexdigest()[:16]
                return _EMITTER_SOURCE_HASH
    except Exception:
        pass
    _EMITTER_SOURCE_HASH = "unknown"
    return _EMITTER_SOURCE_HASH


def _safe_payload_hash(payload: Any) -> str:
    """对任意 payload 计算稳定 hash（#213）。

    production capability identity 不允许 repr fallback：typed serializer error
    → :class:`CapabilityInfrastructureError`，绝不返回基于 ``repr()`` 的伪 identity。
    """
    try:
        from backend.evidence_provenance import compute_payload_hash

        return compute_payload_hash(payload)
    except CapabilityInfrastructureError:
        raise
    except Exception as exc:
        raise CapabilityInfrastructureError(
            f"capability payload hash failed for {type(payload).__name__!r}: {exc}"
        ) from exc


def _json_contract_value(value: Any) -> Any:
    """R21-DEF5：typed serialization of a registry contract value to plain JSON.

    The registry catalog entry is a *contract view* whose ``param_specs`` dict
    carries live ``cleaned_operators.base.ParamSpec`` dataclass objects (and
    ``dtype`` holds a live ``type``, ``param_role`` a live ``ParamRole`` enum).
    Hashing the contract must serialize those declared fields explicitly — a
    raw ``json.dumps`` of live objects raises TypeError, and a ``repr()``
    fallback would manufacture pseudo-identity (#213 forbids that).

    Everything else in a catalog entry is already JSON-native (str/int/float/
    bool/None/list/tuple/dict with str keys — verified across the catalog);
    those pass through unchanged. Any OTHER live object type still raises
    ``CapabilityInfrastructureError`` (fail-closed, no silent coercion).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_contract_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_contract_value(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        from cleaned_operators.base import MISSING, ParamSpec

        if not isinstance(value, ParamSpec):
            raise CapabilityInfrastructureError(
                f"registry contract carries unsupported dataclass "
                f"{type(value).__module__}.{type(value).__name__!r}"
            )
        out: dict[str, Any] = {}
        for field in dataclasses.fields(ParamSpec):
            raw = getattr(value, field.name)
            if field.name == "dtype":
                # live ``type`` object (int/float/str/bool) -> its __name__
                out[field.name] = None if raw is None else raw.__name__
            elif field.name == "default":
                # MISSING sentinel = "no default declared"; None is a real default
                out[field.name] = "__MISSING__" if raw is MISSING else _json_contract_value(raw)
            else:
                out[field.name] = _json_contract_value(raw)
        return out
    raise CapabilityInfrastructureError(
        f"registry contract carries unsupported object type "
        f"{type(value).__module__}.{type(value).__name__!r}"
    )


def _sql_contract(canon: str) -> dict:
    """读取 OperatorRegistry 中 canonical 的 logical contract（#363）。

    SQL 实现不拥有 parameter signature，capability 一律读 registry contract；
    SQL marker 元数据里的 ``param_names=[]`` 不做签名审计。

    R21-DEF5：返回 plain-JSON 视图——live ``ParamSpec``/``dtype``/``ParamRole``
    对象按声明的字段 typed-serialize（见 :func:`_json_contract_value`），使
    hash/manifest 消费方拿到的是 deterministic、可直接 ``json.dumps`` 的
    contract，而不是 registry 内部对象图。未知对象类型仍然 fail-closed。
    """
    from cleaned_operators.registry import OperatorRegistry

    return _json_contract_value(
        dict(OperatorRegistry._catalog.get(canon, {}) or {})
    )


def _sql_bound_param_names(canon: str) -> frozenset[str]:
    """SQL canonical 的合法 bound-param 名集合：contract param_names/param_specs + 生产签名。"""
    contract = _sql_contract(canon)
    names: set[str] = set(contract.get("param_names") or ())
    specs = contract.get("param_specs") or {}
    if isinstance(specs, dict):
        names.update(specs.keys())
    try:
        from backend.production_signature import PRODUCTION_SIGNATURES

        sig = PRODUCTION_SIGNATURES.get(canon)
        if sig is not None:
            names.update(p.name for p in sig.params)
    except Exception:
        pass
    return frozenset(names)


def _validate_bound_params(canon: str, bound_params: dict) -> None:
    """校验 bound params 都存在于 registry logical contract（#363）。

    非法 param 名直接 raise ``UnsupportedOperatorBackendError``（调用级契约违约）。
    """
    allowed = _sql_bound_param_names(canon)
    unknown = sorted(set(bound_params) - allowed)
    if unknown:
        raise UnsupportedOperatorBackendError(
            f"{canon}: SQL bound params {unknown} not in registry contract/signature "
            f"param_names={sorted(allowed)}"
        )


def _build_bound_plan(canon: str, bound_params: dict | None):
    """用 ``minimal_plan(canon)`` 打底、把 bound_params 填入 attrs 构造 bound-call plan。"""
    from planner.logical_plan import PlanNode
    from backend.sql_pushdown.plan_fixtures import minimal_plan

    plan = minimal_plan(canon)
    if bound_params:
        attrs = dict(plan.attrs or {})
        attrs.update(bound_params)
        plan = PlanNode(op=plan.op, inputs=plan.inputs, attrs=attrs, node_id=plan.node_id)
    return plan


_EMITTER_OK_CACHE: dict[tuple[Any, ...], bool] = {}


def _sql_emitter_ok(
    canon: str,
    *,
    dialect: str = "duckdb_sql",
    bound_params: dict | None = None,
) -> bool:
    """emitter 能否编译该 canonical（可选：带 bound params 的调用组合，#390/#391）。

    缓存 key 由 ``(canon, dialect, registry_version, emitter_hash, contract_hash,
    params_key)`` 组成，参数摘要独立缓存——同一 canonical 的不同参数组合结果分开。
    """
    from cleaned_operators.registry import OperatorRegistry

    registry_version = OperatorRegistry.version()
    emitter_hash = _emitter_source_hash()
    contract_hash = _safe_payload_hash(_sql_contract(canon))
    params_key = _safe_payload_hash(bound_params) if bound_params else ""
    cache_key = (
        canon,
        dialect,
        registry_version,
        emitter_hash,
        contract_hash,
        params_key,
    )
    if cache_key in _EMITTER_OK_CACHE:
        return _EMITTER_OK_CACHE[cache_key]
    if canon in {"column", "literal"}:
        _EMITTER_OK_CACHE[cache_key] = True
        return True
    if canon not in _sql_capable_canonicals():
        _EMITTER_OK_CACHE[cache_key] = False
        return False

    ok = False
    try:
        from backend.sql_pushdown.emitter import (
            SqlDialect,
            compile_plan_to_sql,
            plan_is_sql_capable,
        )

        plan = _build_bound_plan(canon, bound_params)
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


def _polars_status(canon: str, *, production_mode: bool = True) -> CapabilityStatus:
    """Return Polars status from evidence, rejecting delegates from production_safe.

    Polars delegates (polars_udf_pandas_delegate) round-trip through to_pandas(),
    materialize the full panel, and execute the pandas reference kernel. They are
    NOT native Polars implementations and must never reach production_safe, even
    if they appear in evidence sets.

    Evidence certifies correctness but does not distinguish native vs delegate
    execution. This function consults polars_backend_kind to enforce that only
    genuine native implementations can be production_safe.

    FE-P0-005: Defaults to production_mode=True (fail closed). Operators without
    explicit PhysicalImplementationSpec are classified as unsupported in production.
    """
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    from cleaned_operators.registry import OperatorRegistry

    if "polars" not in OperatorRegistry.backends_for(canon):
        return "unsupported"

    # Delegates must never reach production_safe (they round-trip through pandas)
    # FE-P0-005: Pass production_mode flag to enforce explicit spec requirement
    from backend.polars_backend_kind import canonical_polars_is_delegate, canonical_polars_kind, PolarsImplementationKind

    polars_kind = canonical_polars_kind(canon, production_mode=production_mode)
    if production_mode and polars_kind == PolarsImplementationKind.UNSUPPORTED:
        # No explicit spec in production mode
        return "unsupported"

    is_delegate = canonical_polars_is_delegate(canon, production_mode=production_mode)

    if (
        canon in POLARS_REFERENCE_PARITY_VERIFIED
        and canon in POLARS_EDGE_VERIFIED
        and canon in POLARS_NO_FALLBACK_VERIFIED
    ):
        # Delegates cannot be production_safe even with full evidence
        if is_delegate:
            return "parity_verified"
        return "production_safe"

    if canon in POLARS_REFERENCE_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def _pandas_status(canon: str) -> CapabilityStatus:
    """Return Pandas status strictly from valid immutable evidence.

    ``catalog.status == 'production'`` means reviewed semantic target only.  It
    must never grant physical execution admission.  Daily primitives are bound
    to primitive evidence; non-Daily factor operators are bound to the
    factor-operator evidence artifact through the certification overlay.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "pandas_numpy" not in OperatorRegistry.backends_for(canon):
        return "unsupported"

    catalog = OperatorRegistry._catalog.get(canon, {})
    meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
    if bool(meta.get("production_certified")):
        source = str(meta.get("certification_source") or "")
        if source in {"primitive_verified.json", "factor_operator_verified.json"}:
            return "production_safe"

    try:
        from backend.evidence_provenance import evidence_artifact_valid
        from backend.primitive_evidence import PRIMITIVE_BACKEND_EXECUTION_CERTIFIED

        if evidence_artifact_valid() and canon in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED:
            return "production_safe"
    except Exception:
        pass

    # Deliberately no tier/status fallback here.  An implementation without
    # current evidence remains implemented and is unavailable in production.
    return "implemented"


def _sql_status(canon: str, *, dialect: BackendName) -> CapabilityStatus:
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if canon not in SQL_IMPLEMENTED_CANONICALS:
        return "unsupported"
    # #363：SQL canonical 的 parameter signature 一律读 registry logical contract；
    # SQL marker 的 ``param_names=[]`` 不做签名审计。调用级（bound params）校验
    # 在 check_call_capability 完成，这里保持无参基态。
    contract = _sql_contract(canon)
    _ = contract
    if not _sql_emitter_ok(canon, dialect=dialect):
        return "implemented"
    if dialect == "clickhouse_sql":
        from backend.sql_pushdown.clickhouse_capabilities import (
            effective_clickhouse_production_safe,
        )
        from backend.sql_tiers import CLICKHOUSE_SQL_PARITY_VERIFIED

        if effective_clickhouse_production_safe(canon):
            return "production_safe"
        if canon in CLICKHOUSE_SQL_PARITY_VERIFIED:
            return "parity_verified"
        return "implemented"

    from backend.sql_tiers import (
        DUCKDB_SQL_PARITY_VERIFIED,
        effective_sql_production_safe,
    )

    if effective_sql_production_safe(canon):
        return "production_safe"
    if canon in DUCKDB_SQL_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def _q_status(canon: str) -> CapabilityStatus:
    """Q/KDB status derived from the q physical-implementation evidence authority.

    Mirrors ``backend.q_backend.q_capability.QBackendCapability`` admission
    exactly: production_safe only for registry production-ready (fully
    certified) operators; ``implemented`` for declared operators with a
    lowering (research-ready). Everything else — including any failure to
    reach the q authority — fails closed to ``unsupported``.
    """
    try:
        from backend.q_backend.q_physical_implementation_registry import (
            get_q_physical_implementation_registry,
        )

        registry = get_q_physical_implementation_registry()
        if canon in registry.get_production_ready():
            return "production_safe"
        if registry.has_lowering(canon) and canon in registry.declared_targets():
            return "implemented"
    except Exception:
        return "unsupported"
    return "unsupported"


def backend_status(
    canonical: str,
    backend: BackendName,
    *,
    data_source_kind: str = "duckdb",
    production_mode: bool = True,  # FE-P0-005: Default to fail-closed
) -> CapabilityStatus:
    """Get backend capability status.

    FE-P0-005: Defaults to production_mode=True (fail closed). Requires explicit
    PhysicalImplementationSpec for production_safe classification. Set
    production_mode=False explicitly for research/diagnostic heuristic fallback.
    """
    canon = resolve_canonical(canonical)
    if backend == "pandas_numpy":
        return _pandas_status(canon)
    if backend == "polars":
        return _polars_status(canon, production_mode=production_mode)
    if backend in _SQL_BACKENDS:
        return _sql_status(canon, dialect=backend)
    if backend == "q_kdb":
        # R21-BACKENDNAME-Q-ALIGN: route through the q evidence authority
        # (fail-closed). Previously hardcoded "unsupported".
        return _q_status(canon)
    return "unsupported"


def check_call_capability(
    canonical: str,
    bound_params: dict | None = None,
    *,
    dialect: str = "duckdb_sql",
) -> CapabilityDecision:
    """调用级 SQL capability 判定（#362）。

    对 ``canonical + bound_params × dialect`` 组合，先取算子级 base status
    （``_sql_status``），再用「带参数的 bound-call plan」编译验证该参数组合能否
    由 emitter 下推：

    - 能编译且 base status == production_safe → ``production_safe=True``；
    - 能编译但 base status 较低 → ``supported=True`` / ``production_safe=False``；
    - 编译失败（该参数组合不受支持）→ ``supported=False``。

    参数:
        canonical: 算子名或别名。
        bound_params: 调用参数（如 ``{'window': 20, 'ddof': 1}``）。参数名必须
            存在于 registry logical contract / 生产签名，否则 raise
            ``UnsupportedOperatorBackendError``（#363）。
        dialect: SQL 方言，``duckdb_sql`` 或 ``clickhouse_sql``。
    """
    from backend.operator_cost import default_backend_speedup

    canon = resolve_canonical(canonical)
    if canon == "if_else":
        canon = "where"
    if bound_params is not None:
        if not isinstance(bound_params, dict):
            bound_params = dict(bound_params)
        _validate_bound_params(canon, bound_params)

    if canon not in _sql_capable_canonicals():
        return CapabilityDecision(
            canonical=canon,
            dialect=dialect,
            supported=False,
            production_safe=False,
            reason="not in SQL_IMPLEMENTED_CANONICALS",
            estimated_cost=1.0,
        )

    base_status = _sql_status(canon, dialect=dialect)  # type: ignore[arg-type]
    compile_ok = _sql_emitter_ok(canon, dialect=dialect, bound_params=bound_params or None)
    cost = default_backend_speedup(canon, dialect, base_status)
    if not compile_ok:
        return CapabilityDecision(
            canonical=canon,
            dialect=dialect,
            supported=False,
            production_safe=False,
            reason=f"bound params {bound_params or {}} fail SQL compile",
            estimated_cost=cost,
        )
    if base_status == "production_safe":
        return CapabilityDecision(
            canonical=canon,
            dialect=dialect,
            supported=True,
            production_safe=True,
            reason=f"SQL compiles with {bound_params or {}}; production_safe",
            estimated_cost=cost,
        )
    return CapabilityDecision(
        canonical=canon,
        dialect=dialect,
        supported=True,
        production_safe=False,
        reason=f"SQL compiles with {bound_params or {}} but status={base_status}",
        estimated_cost=cost,
    )


def production_eligible_backends(
    canonical: str,
    *,
    data_source_kind: str = "duckdb",
) -> tuple[str, ...]:
    """Return only independently production-certified physical backends."""
    canon = resolve_canonical(canonical)
    eligible: list[str] = []
    if _pandas_status(canon) == "production_safe":
        eligible.append("pandas_numpy")
    if _polars_status(canon) == "production_safe":
        eligible.append("polars")
    dialect: BackendName = (
        "clickhouse_sql"
        if data_source_kind.lower() in {"clickhouse", "ch"}
        else "duckdb_sql"
    )
    if _sql_status(canon, dialect=dialect) == "production_safe":
        eligible.append("sql")
    return tuple(eligible)


def capability_for(canonical: str, backend: BackendName) -> BackendCapability:
    """Get capability record for canonical×backend.

    FE-BE-P0-002: Returns unified BackendCapability with enum types.
    """
    from backend.operator_cost import default_backend_speedup
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    canon = resolve_canonical(canonical)
    if canon == "if_else":
        canon = "where"
    policy = infer_operator_policy(canon)
    scope = getattr(policy, "scope", "") or ""
    backend_key = "sql" if backend in _SQL_BACKENDS else backend
    backend_meta = dict(
        (
            (OperatorRegistry._catalog.get(canon, {}).get("backend_meta") or {}).get(
                backend_key
            )
            or {}
        )
    )
    if backend == "pandas_numpy":
        status = _pandas_status(canon)
    elif backend == "polars":
        status = _polars_status(canon)
    elif backend == "q_kdb":
        status = _q_status(canon)
    else:
        status = _sql_status(canon, dialect=backend)

    required_metadata = {
        "execution_kind",
        "supports_lazy",
        "supports_streaming",
        "materializes_full_panel",
        "supports_nulls",
        "supports_nan",
        "supports_inf",
        "supports_scalar_broadcast",
        "supports_group",
        "supports_window",
        "supports_min_periods",
    }
    if status == "production_safe" and backend != "pandas_numpy":
        missing = sorted(required_metadata.difference(backend_meta))
        if missing:
            raise RuntimeError(
                f"{canon}/{backend}: production-safe capability metadata missing {missing}"
            )

    notes = ""
    if backend == "q_kdb" and status != "unsupported":
        notes = "q/kdb physical backend; evidence authority: q_physical_implementation_registry"
    if backend == "clickhouse_sql" and status != "unsupported":
        notes = "dialect=clickhouse; verify per deployment"
    supports_streaming = bool(backend_meta.get("supports_streaming", False))
    materializes_full_panel = bool(
        backend_meta.get("materializes_full_panel", backend == "pandas_numpy")
    )
    if backend in _SQL_BACKENDS and canon in _SQL_STREAMING_LIMITATIONS:
        # #364：per-canonical×dialect 覆盖——EWM self join / rolling regression /
        # KNN/graph/matrix_profile 并非真 streaming，即使 backend_meta 声明过强也强制降级。
        supports_streaming = False
        materializes_full_panel = True

    # Map backend string to BackendKind enum
    backend_kind_map = {
        "pandas_numpy": BackendKind.PANDAS_NUMPY,
        "polars": BackendKind.POLARS,
        "duckdb_sql": BackendKind.DUCKDB_SQL,
        "clickhouse_sql": BackendKind.CLICKHOUSE_SQL,
        "q_kdb": BackendKind.Q_KDB,
    }
    backend_kind = backend_kind_map.get(backend, BackendKind.PANDAS_NUMPY)

    # Map status string to CapabilityLevel enum
    level_map = {
        "unsupported": CapabilityLevel.UNSUPPORTED,
        "implemented": CapabilityLevel.IMPLEMENTED,
        "parity_verified": CapabilityLevel.PARITY_VERIFIED,
        "production_safe": CapabilityLevel.PRODUCTION_SAFE,
    }
    level = level_map.get(status, CapabilityLevel.UNSUPPORTED)

    # Map execution_kind string to ExecutionKind enum
    execution_kind_str = str(
        backend_meta.get(
            "execution_kind",
            "pandas_numpy_reference" if backend == "pandas_numpy" else "unsupported",
        )
    )
    execution_kind_map = {
        "unsupported": ExecutionKind.UNSUPPORTED,
        "native_expr": ExecutionKind.NATIVE_EXPR,
        "native_streaming": ExecutionKind.NATIVE_STREAMING,
        "python_udf": ExecutionKind.DELEGATE_PYTHON,
        "pandas_fallback": ExecutionKind.DELEGATE_PANDAS,
        "pandas_materialization_fallback": ExecutionKind.DELEGATE_PANDAS,
        "pandas_numpy_reference": ExecutionKind.REFERENCE,
        "polars_native_expr": ExecutionKind.POLARS_NATIVE_EXPR,
        "polars_numpy_kernel": ExecutionKind.POLARS_NUMPY_KERNEL,
        "polars_pandas_delegate": ExecutionKind.POLARS_PANDAS_DELEGATE,
        "duckdb_native_sql": ExecutionKind.DUCKDB_NATIVE_SQL,
        "clickhouse_native_sql": ExecutionKind.CLICKHOUSE_NATIVE_SQL,
        "numba_cpu_kernel": ExecutionKind.NUMBA_CPU_KERNEL,
        "q_native": ExecutionKind.Q_NATIVE,
    }
    exec_kind = execution_kind_map.get(execution_kind_str, ExecutionKind.UNSUPPORTED)

    return BackendCapability(
        canonical=canon,
        backend=backend_kind,
        level=level,
        execution_kind=exec_kind,
        estimated_speedup=default_backend_speedup(canon, backend, status),
        supports_nulls=bool(backend_meta.get("supports_nulls", backend == "pandas_numpy")),
        supports_nan=bool(backend_meta.get("supports_nan", backend == "pandas_numpy")),
        supports_inf=bool(backend_meta.get("supports_inf", backend == "pandas_numpy")),
        supports_scalar_broadcast=bool(
            backend_meta.get("supports_scalar_broadcast", backend == "pandas_numpy")
        ),
        supports_min_periods=bool(
            backend_meta.get("supports_min_periods", backend == "pandas_numpy")
        ),
        supports_group=bool(
            backend_meta.get("supports_group", scope in {"cs", "group"})
        ),
        supports_window=bool(backend_meta.get("supports_window", scope == "ts")),
        supports_lazy=bool(backend_meta.get("supports_lazy", False)),
        supports_streaming=supports_streaming,
        materializes_full_panel=materializes_full_panel,
        notes=notes,
    )


def summarize_operator(canonical: str) -> OperatorCapabilitySummary:
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
        q_kdb=_q_status(canon),
        allow_in_production=bool(spec.allow_in_production) if spec is not None else False,
        parity_verified=canon in POLARS_PARITY_VERIFIED,
        polars_long_tier=spec.polars_long_tier if spec is not None else "unsupported",
    )


def build_capability_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[OperatorCapabilitySummary]:
    from cleaned_operators.registry import OperatorRegistry

    if canonicals is None:
        names = sorted(
            {
                resolve_canonical(c)
                for c in OperatorRegistry.list_canonical()
                if OperatorRegistry.backends_for(c)
            }
        )
    else:
        names = sorted({resolve_canonical(c) for c in canonicals})
    return [summarize_operator(c) for c in names]


def export_flat_capabilities(
    canonicals: Sequence[str] | None = None,
) -> list[BackendCapability]:
    rows: list[BackendCapability] = []
    for summary in build_capability_matrix(canonicals):
        for backend in (
            "pandas_numpy",
            "polars",
            "duckdb_sql",
            "clickhouse_sql",
            "q_kdb",
        ):
            rows.append(capability_for(summary.canonical, backend))
    return rows


def supports_polars(canonical: str, *, mode: str = "production") -> bool:
    status = _polars_status(resolve_canonical(canonical))
    return status == "production_safe" if mode == "production" else status != "unsupported"


def supports_sql(
    canonical: str,
    data_source_kind: str = "duckdb",
    *,
    mode: str = "production",
) -> bool:
    canon = resolve_canonical(canonical)
    dialect: BackendName = (
        "clickhouse_sql"
        if data_source_kind.lower() in {"clickhouse", "ch"}
        else "duckdb_sql"
    )
    status = _sql_status(canon, dialect=dialect)
    return status == "production_safe" if mode == "production" else status != "unsupported"


def supports_pandas(canonical: str, *, mode: str = "production") -> bool:
    status = _pandas_status(resolve_canonical(canonical))
    return status == "production_safe" if mode == "production" else status != "unsupported"


def _backend_cost(
    canonical: str,
    backend: str,
    *,
    row_count_estimate: int | None,
    requires_conversion: bool,
) -> float:
    from backend.operator_cost import estimate_backend_cost

    return estimate_backend_cost(
        canonical,
        backend,
        row_count_estimate=row_count_estimate,
        requires_conversion=requires_conversion,
    )


def get_best_backend(
    name: str,
    *,
    mode: str = "production",
    data_source_kind: str = "memory",
    row_count_estimate: int | None = None,
    prefer: str = "auto",
    allow_unverified_backend: bool = False,
    registry: Any | None = None,
) -> tuple[object | None, str]:
    """Select the cheapest eligible operator backend.

    Production never silently falls back to an implementation that lacks
    current evidence.  SQL is normally selected at the plan/subtree layer, but
    explicit ``prefer='sql'`` remains supported.  ``registry`` is an
    injectable OperatorRegistry for tests (defaults to the global one).

    R21-ROUTING-AUTHORITY: this function is a capability/cost CANDIDATE
    PROVIDER only.  It may propose a backend but never finalizes the production
    route.  The Global Physical Planner
    (``runtime.multibackend.batch_global_optimizer.PhysicalBatchGlobalOptimizer``)
    is the sole production routing authority.  When a caller supplies an
    explicit ``prefer`` (a whole-plan route already chosen by the planner), the
    candidate is returned as-is and the caller is responsible for honoring it.
    """
    import os

    from cleaned_operators.registry import OperatorRegistry

    if registry is None:
        registry = OperatorRegistry

    def resolve(name_: str) -> str:
        resolver = getattr(registry, "resolve_canonical", None)
        return resolver(name_) if callable(resolver) else resolve_canonical(name_)

    canonical = resolve(name)
    mode = str(mode or "research").lower()
    backends = registry.backends_for(canonical)
    prod = mode == "production"

    def permitted(registry_backend: str) -> bool:
        if registry_backend == "pandas_numpy":
            status = _pandas_status(canonical)
        elif registry_backend == "polars":
            status = _polars_status(canonical)
        elif registry_backend == "sql":
            dialect: BackendName = (
                "clickhouse_sql"
                if data_source_kind.lower() in {"clickhouse", "ch"}
                else "duckdb_sql"
            )
            status = _sql_status(canonical, dialect=dialect)
        elif registry_backend == "q_kdb":
            # R21-BACKENDNAME-Q-ALIGN: q_kdb routes through the q evidence
            # authority. Fail-closed: unavailable authority => unsupported.
            status = _q_status(canonical)
        else:
            return False
        if prod:
            return status == "production_safe"
        return status != "unsupported" and (
            allow_unverified_backend
            or status in {"parity_verified", "production_safe"}
        )

    requested = str(prefer or "auto").lower()
    if requested in {"pandas_numpy", "polars", "sql", "q_kdb"}:
        # R21-BACKENDNAME-Q-ALIGN: explicit prefer="q_kdb" selects the Q/KDB
        # physical backend. The q evidence authority gates production eligibility;
        # research admission is allowed when the request is explicit and the
        # operator object is registered in the injected registry.
        if requested == "q_kdb":
            if prod and not permitted("q_kdb"):
                raise UnsupportedOperatorBackendError(
                    f"{canonical!r} backend='q_kdb' is not eligible in mode={mode!r}"
                )
            op = registry.get(canonical, "q_kdb")
            if op is None:
                raise UnsupportedOperatorBackendError(
                    f"{canonical!r} backend='q_kdb' has no registered q operator"
                )
            return op, "q_kdb"
        op = registry.get(canonical, requested)
        if op is None or not permitted(requested):
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} backend={requested!r} is not eligible in mode={mode!r}"
            )
        return op, requested
    if requested != "auto":
        raise UnsupportedOperatorBackendError(
            f"unknown backend preference {prefer!r}"
        )

    aggressive_requested = (
        os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower()
        in {"auto_aggressive", "aggressive"}
    )
    aggressive = aggressive_requested and not prod and allow_unverified_backend
    candidates: list[tuple[str, float]] = []

    if "pandas_numpy" in backends and (permitted("pandas_numpy") or aggressive):
        candidates.append(
            (
                "pandas_numpy",
                _backend_cost(
                    canonical,
                    "pandas_numpy",
                    row_count_estimate=row_count_estimate,
                    requires_conversion=False,
                ),
            )
        )
    if "polars" in backends and (permitted("polars") or aggressive):
        candidates.append(
            (
                "polars",
                _backend_cost(
                    canonical,
                    "polars",
                    row_count_estimate=row_count_estimate,
                    requires_conversion=True,
                ),
            )
        )

    if not candidates:
        raise UnsupportedOperatorBackendError(
            f"no {'production-certified ' if prod else ''}operator backend for {canonical!r}"
        )

    env_cost = os.environ.get("FACTOR_ENGINE_COST_ROUTING", "").strip().lower()
    use_cost = prod or env_cost in {"1", "true", "yes", "on"}
    if use_cost and len(candidates) > 1:
        chosen = min(candidates, key=lambda item: (item[1], item[0]))[0]
    else:
        # R11 P1-06: research auto must NOT prefer a polars backend that is only
        # a python bridge (pl -> pandas -> pandas kernel -> pl).  That path adds
        # a materialisation + round-trip and is slower, not faster, than the
        # pandas reference.  Prefer polars ONLY when it is a real fast path
        # (native expressions / per-column numpy UDF without a full-frame pandas
        # round-trip); a ``pandas_materialization_fallback`` stays behind the
        # certified pandas reference unless benchmark evidence says otherwise.
        if any(candidate == "polars" for candidate, _ in candidates):
            pl_op = registry.get(canonical, "polars")
            from backend.polars_backend_kind import polars_backend_kind

            kind = polars_backend_kind(pl_op)
            if kind.value != "polars_udf_pandas_delegate":
                chosen = "polars"
            else:
                chosen = "pandas_numpy"
        else:
            chosen = candidates[0][0]

    op = registry.get(canonical, chosen)
    if op is None:
        raise UnsupportedOperatorBackendError(
            f"selected backend {chosen!r} disappeared for {canonical!r}"
        )
    return op, chosen


class _OperatorBackendRegistryStub:
    """Minimal injectable stub for ``get_best_backend`` tests.

    This is intentionally private; production callers still use
    ``cleaned_operators.registry.OperatorRegistry`` by default.
    """

    def __init__(
        self,
        *,
        backends_for: Mapping[str, Sequence[str]] | None = None,
        get_impl: Callable[[str, str], object | None] | None = None,
        resolve_canonical: Callable[[str], str] | None = None,
    ) -> None:
        self._backends = backends_for or {}
        self._get = get_impl or (lambda canonical, backend: None)
        self._resolve = resolve_canonical

    def backends_for(self, canonical: str) -> Sequence[str]:
        return self._backends.get(canonical, ())

    def get(self, canonical: str, backend: str) -> object | None:
        return self._get(canonical, backend)

    def resolve_canonical(self, canonical: str) -> str:
        return self._resolve(canonical) if self._resolve is not None else canonical


class BackendCapabilityRegistry:
    """Unified capability authority (merged from capability_registry).

    This registry provides the single source of truth for all backend capability
    queries. Design principles:
    - Single version number for all capability state
    - No inline capability checks in router/cost/emitter
    - Explicit version tracking for physical plan cache invalidation
    - Clear separation: registry owns capability, cost model owns estimates
    """

    _version_hash: str | None = None

    @classmethod
    def version(cls) -> str:
        """Return current capability registry version."""
        return CAPABILITY_REGISTRY_VERSION

    @classmethod
    def version_hash(cls) -> str:
        """Return hash of capability state for cache keys."""
        if cls._version_hash is not None:
            return cls._version_hash

        from backend.polars_long_policy import POLARS_LONG_NATIVE
        from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

        components = [
            CAPABILITY_REGISTRY_VERSION,
            str(sorted(POLARS_LONG_NATIVE)),
            str(sorted(SQL_IMPLEMENTED_CANONICALS)),
        ]

        try:
            from backend.primitive_evidence import evidence_generation
            components.append(str(evidence_generation()))
        except Exception:
            pass

        combined = "|".join(components)
        cls._version_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return cls._version_hash

    @classmethod
    def query(
        cls,
        canonical: str,
        backend: BackendKind | str,
        *,
        mode: Literal["production", "research"] = "production",
        data_source_kind: str = "duckdb",
        bound_params: dict[str, Any] | None = None,
    ) -> CapabilityQueryResult:
        """Query capability for canonical×backend×params.

        Args:
            canonical: Operator canonical name
            backend: Physical backend kind
            mode: Execution mode (production requires production_safe)
            data_source_kind: Data source dialect for SQL backends
            bound_params: Bound call parameters (for SQL validation)

        Returns:
            Complete capability query result with reasoning
        """
        canon = resolve_canonical(canonical)

        # Normalize backend kind
        if isinstance(backend, str):
            backend_str = backend.lower()
            if backend_str in {"polars_long", "polars"}:
                backend_kind = BackendKind.POLARS
            elif backend_str == "pandas_numpy":
                backend_kind = BackendKind.PANDAS_NUMPY
            elif backend_str in {"duckdb_sql", "sql"}:
                backend_kind = BackendKind.DUCKDB_SQL
            elif backend_str == "clickhouse_sql":
                backend_kind = BackendKind.CLICKHOUSE_SQL
            elif backend_str == "q_kdb":
                backend_kind = BackendKind.Q_KDB
            else:
                return CapabilityQueryResult(
                    supported=False,
                    production_safe=False,
                    record=None,
                    reason=f"Unknown backend: {backend}",
                )
        else:
            backend_kind = backend

        backend_kind_map = {
            BackendKind.PANDAS_NUMPY: "pandas_numpy",
            BackendKind.POLARS: "polars",
            BackendKind.DUCKDB_SQL: "duckdb_sql",
            BackendKind.CLICKHOUSE_SQL: "clickhouse_sql",
            BackendKind.Q_KDB: "q_kdb",
        }
        if backend_kind not in backend_kind_map:
            return CapabilityQueryResult(
                supported=False,
                production_safe=False,
                record=None,
                reason=f"Backend {backend_kind!r} not implemented",
            )

        # Q_KDB: bypass SQL bound-param path; use q evidence authority.
        if backend_kind is BackendKind.Q_KDB:
            q_status = _q_status(canon)
            record = cls._build_record(canon, backend_kind, data_source_kind)
            return CapabilityQueryResult(
                supported=q_status != "unsupported",
                production_safe=q_status == "production_safe",
                record=record,
                reason=f"Status: {q_status}",
            )

        # SQL bound-param validation for DuckDB/ClickHouse only.
        if backend_kind in {
            BackendKind.DUCKDB_SQL,
            BackendKind.CLICKHOUSE_SQL,
        } and bound_params:
            dialect = "duckdb_sql" if backend_kind == BackendKind.DUCKDB_SQL else "clickhouse_sql"
            decision = check_call_capability(canon, bound_params, dialect=dialect)

            if not decision.supported:
                return CapabilityQueryResult(
                    supported=False,
                    production_safe=False,
                    record=None,
                    reason=decision.reason,
                )

            record = cls._build_record(canon, backend_kind, data_source_kind)
            return CapabilityQueryResult(
                supported=True,
                production_safe=decision.production_safe,
                record=record,
                reason=decision.reason,
            )

        backend_name: BackendName = backend_kind_map[backend_kind]  # type: ignore
        status = backend_status(canon, backend_name, data_source_kind=data_source_kind)

        supported = status != "unsupported"
        production_safe = status == "production_safe"

        if mode == "production" and not production_safe:
            record = cls._build_record(canon, backend_kind, data_source_kind) if supported else None
            return CapabilityQueryResult(
                supported=False,
                production_safe=False,
                record=record,
                reason=f"Status {status} insufficient for production",
            )

        record = cls._build_record(canon, backend_kind, data_source_kind)
        return CapabilityQueryResult(
            supported=supported,
            production_safe=production_safe,
            record=record,
            reason=f"Status: {status}",
        )

    @classmethod
    def _build_record(
        cls,
        canonical: str,
        backend: BackendKind,
        data_source_kind: str,
    ) -> BackendCapability:
        """Build complete capability record.

        FE-BE-P0-002: Returns unified BackendCapability.
        """
        backend_name_map = {
            BackendKind.PANDAS_NUMPY: "pandas_numpy",
            BackendKind.POLARS: "polars",
            BackendKind.DUCKDB_SQL: "duckdb_sql",
            BackendKind.CLICKHOUSE_SQL: "clickhouse_sql",
            BackendKind.Q_KDB: "q_kdb",
        }

        backend_name: BackendName = backend_name_map[backend]  # type: ignore
        cap = capability_for(canonical, backend_name)

        # capability_for already returns BackendCapability with proper enum types
        return cap

    @classmethod
    def supports_backend(
        cls,
        canonical: str,
        backend: BackendKind | str,
        *,
        mode: Literal["production", "research"] = "production",
        data_source_kind: str = "duckdb",
    ) -> bool:
        """Check if canonical supports backend at given mode."""
        result = cls.query(
            canonical,
            backend,
            mode=mode,
            data_source_kind=data_source_kind,
        )
        return result.supported and (
            result.production_safe if mode == "production" else True
        )

    @classmethod
    def list_backends(
        cls,
        canonical: str,
        *,
        mode: Literal["production", "research"] = "production",
    ) -> list[BackendKind]:
        """List all backends supporting canonical at given mode."""
        backends = []
        for backend in [
            BackendKind.PANDAS_NUMPY,
            BackendKind.POLARS,
            BackendKind.DUCKDB_SQL,
            BackendKind.Q_KDB,
        ]:
            if cls.supports_backend(canonical, backend, mode=mode):
                backends.append(backend)
        return backends
