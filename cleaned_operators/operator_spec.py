# -*- coding: utf-8
"""算子生产契约：metadata + policy + lifecycle 统一视图。

聚合 registry 元数据与 ``OperatorPolicy``，提供 production 准入判定、
manifest 导出及 CI 契约校验函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cleaned_operators.operator_policy import OperatorPolicy, infer_operator_policy

OperatorStatus = Literal["production", "research", "experimental", "deprecated", "stub", "doc_only"]
NumericalStability = Literal["high", "medium", "low"]

# stateful / deferred：允许 production 但不计入 strict dual-backend core
PRODUCTION_ALLOWED_DEFERRED_CANONICALS: frozenset[str] = frozenset()

# legacy PIT core（不含 deferred stateful）
_LEGACY_PRODUCTION_CORE: frozenset[str] = frozenset()

# DSL alias；canonical 为 ``where``
_FASTPATH_ALIAS_ONLY: frozenset[str] = frozenset({"if_else"})


def _build_production_authoring_canonicals() -> frozenset[str]:
    """Return the reviewed daily authoring target, not a capability claim."""
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    return frozenset(DAILY_CANONICALS)


PRODUCTION_AUTHORING_CANONICALS: frozenset[str] = _build_production_authoring_canonicals()
PRODUCTION_INTERNAL_CANONICALS: frozenset[str] = frozenset()


def _build_production_dual_backend_core() -> frozenset[str]:
    """Derive certified three-engine capability from current evidence."""
    from backend.evidence_provenance import evidence_artifact_valid
    from backend.primitive_evidence import PRIMITIVE_BACKEND_EXECUTION_CERTIFIED

    if not evidence_artifact_valid():
        return frozenset()
    return frozenset(
        canonical
        for canonical in PRODUCTION_AUTHORING_CANONICALS
        if canonical in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED
    )


PRODUCTION_DUAL_BACKEND_CORE_CANONICALS: frozenset[str] = _build_production_dual_backend_core()


def _build_production_core_canonicals() -> frozenset[str]:
    """Build the policy target independently from current backend evidence."""
    return PRODUCTION_AUTHORING_CANONICALS | PRODUCTION_INTERNAL_CANONICALS


# SQL 下推 / DSL 常用 production 核心 canonical（PIT 门禁 / 契约脚本对齐此清单）
PRODUCTION_CORE_CANONICALS: frozenset[str] = _build_production_core_canonicals()


def production_allowed_composite_canonicals() -> frozenset[str]:
    """证据齐全的 composite（非 micro_* / PRODUCTION_DENIED）。"""
    from backend.composite_evidence import composite_production_safe
    from planner.composite_lowering import list_composite_lowerings

    return frozenset(c for c in list_composite_lowerings() if composite_production_safe(c))

# 向后兼容别名
PRODUCTION_PIT_REQUIRED: frozenset[str] = PRODUCTION_CORE_CANONICALS

# R30 §2: tombstoned names (rand_*/shuffle/sample/Lead/next/bfill/causal_bfill/
# fillna_interpolate/interpolate/dropna/norm*) are physically removed operators,
# NOT PIT-exempt special categories.  They are excluded via
# ``tombstones.is_tombstoned`` (single authority) so ``get()`` /
# ``resolve_canonical()`` raise ``RemovedOperatorError``.  Only genuinely
# non-temporal elementwise scaffolds remain exempt.
_PIT_EXEMPT: frozenset[str] = frozenset({"column", "literal", "col"})

# ``PERMANENTLY_FORBIDDEN_CANONICALS`` retains a small set of NEVER-operator
# utility names that are not tombstoned because they never had an operator
# identity; everything that once was a random/future/noncausal-fill operator is
# handled by ``tombstones``.
PERMANENTLY_FORBIDDEN_CANONICALS: frozenset[str] = frozenset(
    {
        "constant",
    }
)

ProductionPolicy = Literal["allowed", "pending", "denied", "permanently_forbidden"]

# production DSL 禁止（可 research / experimental，不可 production 投递）
PRODUCTION_DENIED_CANONICALS: frozenset[str] = frozenset(
    {
        # R30 §2: rand_*/shuffle/sample/Lead/next/bfill/causal_bfill/
        # fillna_interpolate/interpolate/norm* are physically removed via
        # ``tombstones`` (single authority) — they no longer belong in any
        # operator-governance set.  ``constant`` remains a real internal
        # canonical and ``dropna`` a real research-status operator that simply
        # are not production factor terminals.
        "constant",
        "dropna",
        # Raw matrix / signal-processing primitives — MOVE_INTERNAL capability
        # (R22-083), never a public factor terminal.
        "fft",
        "ifft",
        "wavelet",
        "convolve",
        "correlate",
        "mat_inverse",
        "eig",
        "svd",
        "pca",
        # Legacy row-shift / non-PIT fundamental names — kept denied so the
        # strict PIT-aware fiscal operators are the only production forms
        # (R22-116..117: the deny list holds only genuinely-forbidden items;
        # everything else lives in DirectUse admission).
        "quarter",
        "ttm",
        "yoy",
        "avg2",
        "downside_beta",
        "operating_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "ts_poly2_coeff",
        "ts_poly2_resid",
        # NOTE R22-064..070: tail_beta / residual_momentum_capm / coskewness_to_market /
        # idio_vol / idio_skew / rank_corr / digital_count / rolling_beta_to_market
        # are legitimately DIRECT (benchmark-required alpha, dependency alpha, state
        # statistic) and were removed from this deny list — see the hardening
        # ``_remove_promoted_legacy_denials`` path.  ``intraday_vwap_deviation`` is
        # a contextual DIRECT_ALPHA (A-share minute source present; US blocked) and
        # is denied at the source-context gate, not here.
        # R23-P0-PIT11: 7 financial expectation/surprise operators that compute
        # expectations/surprises only knowable AFTER an earnings event — a leak
        # vector in production terminal placement (the label is knowable before
        # the signal).  Certified contextual but P0 blocker for terminal use.
        "fin_actual_expectation_divergence",
        "fin_beat_streak",
        "fin_miss_streak",
        "fin_surprise",
        "fin_surprise_event_percentile",
        "fin_surprise_event_zscore",
        "fin_surprise_zscore",
        # R23-P0-PIT18: 14 financial revision/restatement operators whose
        # point-in-time revision event provenance (whether the revision was
        # knowable at signal time) is UNPROVEN — cannot be certified for
        # production terminal placement without evidence.  Certified contextual
        # but P0 blocker for terminal production use.
        "fin_days_since_expectation_revision",
        "fin_days_since_update",
        "fin_expectation_revision",
        "fin_expectation_revision_count",
        "fin_expectation_revision_magnitude",
        "fin_expectation_revision_pct",
        "fin_expectation_revision_speed",
        "fin_restated_flag",
        "fin_revision_count",
        "fin_revision_delta",
        "fin_revision_direction",
        "fin_revision_magnitude",
        "fin_revision_pct",
        "fin_staleness",
    }
)


def is_production_denied(canon: str) -> bool:
    """判断 canonical 是否禁止用于 production 投递。

    NEW-012: a ``micro_*`` PREFIX is no longer a production denial.  A name
    cannot represent safety — eligibility is decided per canonical by its
    source contract, grain contract, surface classification, six-gate evidence
    and ``production_certified`` (see :func:`_compute_allow_in_production`,
    which requires the canonical to classify as ``daily``/``extended`` and carry
    full evidence before it can be production-eligible).  The legacy blanket
    prefix deny silently blocked any genuinely production-usable minute-derived
    daily factor that happened to be named ``micro_*``; it is removed.

    参数:
        canon: canonical 算子名。

    返回:
        禁止返回 ``True``（仅显式列出的 denied canonical）。
    """
    if canon in PRODUCTION_DENIED_CANONICALS:
        return True
    return False


def infer_production_policy(canon: str) -> ProductionPolicy:
    """推断算子 production policy（与 execution_kind 独立）。"""
    from cleaned_operators.registry import OperatorRegistry
    from planner.composite_lowering import has_composite_lowering

    resolved = OperatorRegistry._aliases.get(canon, canon)
    if resolved in PERMANENTLY_FORBIDDEN_CANONICALS:
        return "permanently_forbidden"
    backends_map = OperatorRegistry._operators.get(resolved)
    if not backends_map:
        return "denied"
    chosen = "pandas_numpy" if "pandas_numpy" in backends_map else next(iter(backends_map))
    op = backends_map.get(chosen)
    if op is None:
        return "denied"
    catalog = OperatorRegistry._catalog.get(resolved, {})
    status = _infer_status(catalog)
    policy = infer_operator_policy(op, canonical=resolved)
    allow = _compute_allow_in_production(
        resolved,
        status=status,
        pit_safe=policy.pit_safe,
        shape_preserving=policy.shape_preserving,
    )
    if allow:
        return "allowed"
    if has_composite_lowering(resolved):
        from backend.composite_evidence import composite_evidence_complete, composite_production_safe

        if composite_production_safe(resolved):
            return "allowed"
        if composite_evidence_complete(resolved):
            return "pending"
        return "pending"
    if is_production_denied(resolved):
        return "denied"
    return "denied"


def is_production_permanently_forbidden(canon: str) -> bool:
    """永久禁止 production（含 PIT 泄漏类算子）。"""
    from cleaned_operators.registry import OperatorRegistry

    resolved = OperatorRegistry._aliases.get(canon, canon)
    return resolved in PERMANENTLY_FORBIDDEN_CANONICALS


@dataclass(frozen=True)
class OutputShapeContract:
    """R34 P0-037：显式的 grain 变换契约。

    shape-changing（非 shape_preserving）的 production 算子必须声明此契约——
    不允许 silent axis drop / silent reindex。合法变换如 ``minute -> daily``、
    ``event table -> entity-date panel``、``snapshot -> daily``。
    """

    input_grain: str
    output_grain: str
    preserves_index: bool = False
    preserves_columns: bool = False
    aggregation_keys: tuple[str, ...] = ()

    def is_valid(self) -> bool:
        return bool(self.input_grain) and bool(self.output_grain)


@dataclass(frozen=True)
class OperatorSpec:
    """算子生产契约（metadata + policy + lifecycle 聚合视图）。

    字段涵盖生命周期、PIT 安全、backend 支持、数值稳定性等生产门禁属性。
    """

    canonical: str
    status: OperatorStatus
    allow_in_production: bool
    deterministic: bool
    pit_safe: bool
    backends: tuple[str, ...]
    policy: OperatorPolicy
    param_names: tuple[str, ...] = ()
    # round-7: the PANEL-input names (vs scalar parameters).  The manifest must
    # NOT list ``window`` / ``lag`` as ``input_fields`` — they are scalar knobs,
    # not data fields.  Derived from ``OperatorMetadata.panel_params`` /
    # ``input_fields``, else inferred from the kernel signature (params without a
    # default are panel inputs).
    panel_params: tuple[str, ...] = ()
    supports_panel: bool = True
    supports_polars: bool = False
    polars_long_tier: str = "unsupported"
    shape_preserving: bool = True
    index_preserving: bool = True
    columns_preserving: bool = True
    # R34 P0-037：shape-changing 算子的显式 grain 契约（None = 无声明）。
    output_shape: OutputShapeContract | None = None
    numerical_stability: NumericalStability = "medium"
    description: str = ""
    execution_kind: str = "primitive"
    production_policy: ProductionPolicy = "denied"
    lowering_available: bool = False
    dual_backend_target: bool = False
    # R40 #127/#128: manifest 的真实频率 / 输出类型 —— 从 operator metadata 的
    # grain/input_grain 与 return_type 推断，不再是硬编码 "any"/"series"。
    frequency: str = "any"
    output_type: str = "series"

    def to_dict(self) -> dict[str, Any]:
        """将 ``OperatorSpec`` 序列化为普通字典。

        返回:
            含 canonical、status、policy 等字段的字典。
        """
        return {
            "canonical": self.canonical,
            "status": self.status,
            "allow_in_production": self.allow_in_production,
            "deterministic": self.deterministic,
            "pit_safe": self.pit_safe,
            "backends": list(self.backends),
            "policy": self.policy.to_dict(),
            "param_names": list(self.param_names),
            "panel_params": list(self.panel_params),
            "supports_panel": self.supports_panel,
            "supports_polars": self.supports_polars,
            "polars_long_tier": self.polars_long_tier,
            "shape_preserving": self.shape_preserving,
            "index_preserving": self.index_preserving,
            "columns_preserving": self.columns_preserving,
            "numerical_stability": self.numerical_stability,
            "description": self.description,
            "execution_kind": self.execution_kind,
            "production_policy": self.production_policy,
            "lowering_available": self.lowering_available,
            "dual_backend_target": self.dual_backend_target,
            "frequency": self.frequency,
            "output_type": self.output_type,
        }


def _infer_status(catalog_entry: dict[str, Any] | None) -> OperatorStatus:
    """从 catalog 条目推断算子生命周期状态。

    参数:
        catalog_entry: registry catalog 中的元数据字典，可为 ``None``。

    返回:
        ``OperatorStatus`` 枚举值；未显式声明时 fail-closed 为 ``research``。
    """
    raw = str((catalog_entry or {}).get("status", "implemented") or "implemented").lower()
    if raw.endswith("stub"):
        return "stub"
    if raw in ("doc_only", "documentation"):
        return "doc_only"
    if raw in ("deprecated",):
        return "deprecated"
    if raw in ("experimental",):
        return "experimental"
    if raw in ("production",):
        return "production"
    if raw in ("research",):
        return "research"
    # fail-closed：catalog ``implemented`` 等未显式声明的一律 research
    return "research"


def _compute_allow_in_production(
    resolved: str,
    *,
    status: OperatorStatus,
    pit_safe: bool,
    shape_preserving: bool = True,
) -> bool:
    """计算算子是否允许用于 production。

    单一准入规则（review §2.5）：算子必须在 daily/extended surface、六证
    ``production_certified=True``、至少一个 evidence 认证后端、且非
    compatibility/diagnostic/benchmark-only。静态 core 与 migrated 不再区分。

    参数:
        resolved: 解析后的 canonical 名。
        status: 算子生命周期状态。
        pit_safe: 是否 PIT 安全。
        shape_preserving: 是否保持 panel 形状。

    返回:
        允许 production 返回 ``True``。
    """
    from cleaned_operators.registry import OperatorRegistry

    if is_production_denied(resolved):
        return False
    # R30 §5 (P0-004): production admission is PIT-safe-gated.  A factor that
    # consumes any future observation can never be production-admitted even if
    # downstream evidence overlays exist.  This is a hard gate, not a
    # downstream-evidence check.
    if not pit_safe:
        return False
    # R30 §5 (P0-005): lifecycle is fail-closed.  ``research`` is an explicit
    # rejection (research tools are never production-admitted); experimental /
    # deprecated / stub / doc_only are likewise rejected.  Only a reviewed
    # ``production`` lifecycle passes this gate (the remaining surface / six-gate
    # checks below still apply).
    if status in ("research", "experimental", "deprecated", "stub", "doc_only"):
        return False
    if not shape_preserving:
        # NEW-013: a legit grain transform (minute -> daily, snapshot -> daily,
        # event table -> entity-date panel) is naturally shape-changing and must
        # NOT be rejected by the old blanket ``shape_preserving=True`` rule.
        # The operator is admissible when it DECLARES the frequency change via
        # its metadata grain contract (``input_grain``/``output_grain``) so the
        # shape change is a provable, intended transform — not a silent reshape.
        catalog = OperatorRegistry._catalog.get(resolved, {})
        declared_grain = bool(
            catalog.get("input_grain") and catalog.get("output_grain")
            and catalog.get("input_grain") != catalog.get("output_grain")
        )
        if not declared_grain:
            return False
        # fall through: a declared grain transform passes the shape gate; the
        # remaining surface/evidence gates below still apply.
    catalog = OperatorRegistry._catalog.get(resolved, {})
    if any(
        catalog.get(flag)
        for flag in (
            "compatibility_only", "diagnostic_only", "benchmark_only",
            "hidden_from_default_mining",
        )
    ):
        return False
    from cleaned_operators.operator_surface import classify_canonical

    if classify_canonical(resolved) not in {"daily", "extended"}:
        return False
    # 六证（implementation/semantic/temporal/source_pit/edge/backend）是唯一
    # 生产认证权威；``should_fail_closed`` 或 status 本身不得单独授信。
    if catalog.get("production_certified") is not True:
        return False
    from backend.operator_capability import production_eligible_backends

    if not production_eligible_backends(resolved):
        return False
    return True


def _has_pit_declaration(canon: str, meta: Any) -> bool:
    """检查算子是否有显式 PIT 策略声明。

    参数:
        canon: canonical 名。
        meta: 算子 ``metadata`` 对象。

    返回:
        有显式 policy 或 ``pit_safe``/``causal`` tag 返回 ``True``。
    """
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    if canon in _EXPLICIT_POLICIES:
        return True
    tags = [str(t).lower() for t in (getattr(meta, "tags", None) or [])]
    return "pit_safe" in tags or "causal" in tags


def _infer_shape_contract(resolved: str, policy: OperatorPolicy) -> tuple[bool, bool, bool]:
    """从 policy 推断 shape/index/columns 保持契约。

    参数:
        resolved: canonical 名。
        policy: 算子执行策略对象。

    返回:
        ``(shape_preserving, index_preserving, columns_preserving)`` 三元组。
    """
    sp = getattr(policy, "shape_preserving", True)
    ip = getattr(policy, "index_preserving", True)
    cp = getattr(policy, "columns_preserving", True)
    return bool(sp), bool(ip), bool(cp)


def _infer_panel_params(op: Any, meta: Any, catalog: dict[str, Any]) -> tuple[str, ...]:
    """Panel-input names for an operator (round-7, audit item 10).

    Resolution order: ``OperatorMetadata.panel_params`` (authoritative when
    declared) -> ``input_fields`` -> kernel-signature inference (positional
    parameters WITHOUT a default are panel inputs; ``window`` / ``lag`` /
    ``alpha`` have defaults, so they are scalar parameters).  A panel must never
    be mislabelled a scalar field and vice versa.
    """
    declared = tuple(getattr(meta, "panel_params", None) or ())
    if declared:
        return declared
    names = tuple(getattr(meta, "param_names", None) or ())
    if not names:
        return ()
    ifields = tuple(getattr(meta, "input_fields", None) or ())
    if ifields:
        matched = tuple(n for n in names if n in ifields)
        return matched or ifields
    import inspect

    fn = getattr(op, "_calculate_series", None) or getattr(op, "calculate", None)
    if fn is None:
        return ()
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ()
    # R30 §9: a positional parameter with NO default is not necessarily a panel
    # input — scalar thresholds / bounds (``lower`` / ``upper`` / ``threshold``)
    # are required scalars that legitimately have no default.  Treating them as
    # panels misroutes the kernel (e.g. ``ts_threshold_cycle_period(x, lower,
    # upper, window)`` would feed ``lower`` as a second price panel).  These
    # names are semantically scalar and are never panel inputs.
    _SCALAR_THRESHOLD_NAMES = frozenset({
        "lower", "upper", "threshold", "low", "high", "min_value", "max_value",
        "alpha", "beta", "gamma", "theta", "sigma", "mu", "rho", "phi",
        "eps", "epsilon", "tol", "tolerance", "k", "q", "p", "n", "seed",
    })
    panels: list[str] = []
    for n in names:
        if n in _SCALAR_THRESHOLD_NAMES:
            continue  # required scalar threshold / bound, never a panel
        param = sig.parameters.get(n)
        if param is None:
            continue
        if param.default is inspect.Parameter.empty:
            panels.append(n)
    return tuple(panels)


def build_operator_spec(canon: str, *, backend: str | None = None) -> OperatorSpec | None:
    """从 Registry 构建单个算子的生产契约视图。

    参数:
        canon: DSL 名或 canonical 名。
        backend: 指定 backend；默认优先 ``pandas_numpy``。

    返回:
        ``OperatorSpec`` 实例；无 runtime 时返回 ``None``。
    """
    from cleaned_operators.registry import OperatorRegistry

    resolved = OperatorRegistry._aliases.get(canon, canon)
    backends_map = OperatorRegistry._operators.get(resolved)
    if not backends_map:
        return None
    chosen_backend = backend or (
        "pandas_numpy" if "pandas_numpy" in backends_map else next(iter(backends_map))
    )
    op = backends_map.get(chosen_backend)
    if op is None:
        return None

    meta = getattr(op, "metadata", None)
    catalog = OperatorRegistry._catalog.get(resolved, {})
    status = _infer_status(catalog)
    policy = infer_operator_policy(op, canonical=resolved)
    all_backends = tuple(sorted(backends_map.keys()))
    tags = [str(t).lower() for t in (getattr(meta, "tags", None) or [])]

    allow_in_production = _compute_allow_in_production(
        resolved,
        status=status,
        pit_safe=policy.pit_safe,
        shape_preserving=policy.shape_preserving,
    )
    shape_preserving, index_preserving, columns_preserving = _infer_shape_contract(resolved, policy)

    deterministic = "shuffle" not in canon and "rand_" not in canon and "random" not in tags
    stability: NumericalStability = "high"
    if policy.scope in ("hypothesis",):
        stability = "low"
    elif policy.scope in ("cs",) and "regression" in canon:
        stability = "medium"

    from backend.polars_long_policy import infer_polars_long_tier

    polars_long_tier = infer_polars_long_tier(resolved)

    from planner.composite_lowering import has_composite_lowering, infer_execution_kind

    execution_kind = infer_execution_kind(resolved)
    production_policy = infer_production_policy(resolved)
    lowering_available = has_composite_lowering(resolved)
    panel_params = _infer_panel_params(op, meta, catalog)
    # R40 #129: supports_panel 由真实 panel_params 推导（不再对全部算子硬编码
    # True）。声明了 panel_params / input_fields / kernel 无默认参数的都是 panel
    # 输入算子；纯标量算子（如 ``add`` 单标量模式）不再误报 panel 能力。
    supports_panel = bool(panel_params)
    # R40 #130: dual_backend_target 从 certified backend evidence 导出 —— 不再
    # 按 execution_kind 猜。production 允许且 ≥2 个 production-certified backend
    # 的算子才是 dual-backend 目标。
    try:
        from backend.operator_capability import production_eligible_backends

        _eligible = production_eligible_backends(resolved)
        dual_backend_target = bool(allow_in_production and len(_eligible) >= 2)
    except Exception:  # noqa: BLE001 - 证据不可用时保守 False
        dual_backend_target = False
    # R40 #127: frequency 从 grain 契约推断（output_grain 优先，其次 input_grain）。
    frequency = (
        str(catalog.get("output_grain") or "").strip()
        or str(getattr(meta, "output_grain", None) or "").strip()
        or str(catalog.get("input_grain") or "").strip()
        or str(getattr(meta, "input_grain", None) or "").strip()
        or "any"
    )
    # R40 #128: output_type 从 return_type 契约推断（metadata 优先，catalog 兜底）。
    output_type = (
        str(getattr(meta, "return_type", None) or "").strip()
        or str(catalog.get("return_type") or "").strip()
        or "series"
    )

    return OperatorSpec(
        canonical=resolved,
        status=status,
        allow_in_production=allow_in_production,
        deterministic=deterministic,
        pit_safe=policy.pit_safe,
        backends=all_backends,
        policy=policy,
        param_names=tuple(getattr(meta, "param_names", None) or ()),
        panel_params=panel_params,
        supports_panel=supports_panel,
        supports_polars="polars" in all_backends,
        polars_long_tier=polars_long_tier,
        shape_preserving=shape_preserving,
        index_preserving=index_preserving,
        columns_preserving=columns_preserving,
        numerical_stability=stability,
        description=str(getattr(meta, "description", "") or catalog.get("description", "")),
        execution_kind=execution_kind,
        production_policy=production_policy,
        lowering_available=lowering_available,
        dual_backend_target=dual_backend_target,
        frequency=frequency,
        output_type=output_type,
    )


def iter_operator_specs() -> list[OperatorSpec]:
    """遍历全部已注册算子并构建 ``OperatorSpec`` 列表。

    返回:
        按 canonical 排序的 ``OperatorSpec`` 列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    specs: list[OperatorSpec] = []
    for canon in sorted(OperatorRegistry._operators):
        spec = build_operator_spec(canon)
        if spec is not None:
            specs.append(spec)
    return specs


def spec_to_manifest_entry(spec: OperatorSpec) -> dict[str, Any]:
    """将 ``OperatorSpec`` 转为审查清单风格的 manifest 条目。

    参数:
        spec: 算子生产契约对象。

    返回:
        供 YAML/JSON 导出的 manifest 字典。
    """
    pol = spec.policy
    scope_map = {
        "ts": "time_series",
        "cs": "cross_sectional",
        "elementwise": "elementwise",
        "aggregate": "aggregate",
        "hypothesis": "hypothesis",
        "unknown": "unknown",
    }
    panel_inputs = list(spec.panel_params)
    scalar_params = [p for p in spec.param_names if p not in panel_inputs]
    return {
        "name": spec.canonical,
        "scope": scope_map.get(pol.scope, pol.scope),
        # R40 #127: 真实 frequency（从 grain 契约推断，不再硬编码 "any"）。
        "frequency": spec.frequency,
        # round-7: input_fields is now ONLY the panel data inputs.  Scalar knobs
        # (window / lag / alpha …) live under scalar_parameters — the machine
        # contract must not advertise ``window`` as a data field (audit item 10).
        "input_fields": panel_inputs,
        "scalar_parameters": scalar_params,
        "param_names": list(spec.param_names),
        # R40 #128: 真实 output_type（从 return_type / OutputShapeContract 推断，
        # 不再硬编码 "series"；ts_last_if 等 scalar/frame 输出算子如实标注）。
        "output_type": spec.output_type,
        "pit_safe": spec.pit_safe,
        "domain_policy": pol.domain_policy,
        "overflow_policy": pol.overflow_policy,
        "null_policy": pol.null_policy,
        "includes_current": pol.includes_current_bar,
        "lookback": pol.lookback_window,
        "min_periods": pol.min_periods,
        "nan_policy": pol.nan_policy,
        "status": spec.status,
        "allow_in_production": spec.allow_in_production,
        "backends": list(spec.backends),
        "description": spec.description,
        "execution_kind": spec.execution_kind,
        "production_policy": spec.production_policy,
        "lowering_available": spec.lowering_available,
        "dual_backend_target": spec.dual_backend_target,
    }


def export_operator_manifest(
    *,
    production_only: bool = False,
    core_only: bool = False,
) -> list[dict[str, Any]]:
    """导出全部或过滤后的算子 manifest 列表。

    参数:
        production_only: 仅导出 ``allow_in_production=True`` 的算子。
        core_only: 仅导出 ``PRODUCTION_CORE`` 集合内的算子。

    返回:
        manifest 条目字典列表。
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in iter_operator_specs():
        if spec.canonical in seen:
            continue
        if production_only and not spec.allow_in_production:
            continue
        if core_only and spec.canonical not in PRODUCTION_CORE_CANONICALS:
            continue
        seen.add(spec.canonical)
        entries.append(spec_to_manifest_entry(spec))
    return entries


def check_production_pit_declarations() -> list[str]:
    """校验 production 核心算子须显式 PIT 声明。

    返回:
        违规描述字符串列表；无问题时为空列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon in sorted(PRODUCTION_PIT_REQUIRED):
        if canon in _PIT_EXEMPT:
            continue
        op = OperatorRegistry.get(canon)
        if op is None:
            errors.append(f"PRODUCTION_PIT_REQUIRED {canon!r} 无 runtime 实现")
            continue
        meta = getattr(op, "metadata", None)
        if not _has_pit_declaration(canon, meta):
            errors.append(f"PRODUCTION_PIT_REQUIRED {canon!r} 缺少显式 PIT 声明（policy 或 pit_safe tag）")
    return errors


def check_production_shape_contracts() -> list[str]:
    """校验 production 算子的 shape/index/columns 契约。

    R34 P0-037：shape-changing 算子若显式声明 ``output_shape``（input_grain /
    output_grain 均非空）则合法（如 minute->daily、event->panel、snapshot->daily）；
    无声明则仍 fail-closed。禁止 silent axis drop / silent reindex。

    返回:
        违规描述字符串列表。
    """
    errors: list[str] = []
    for spec in iter_operator_specs():
        if not spec.allow_in_production:
            continue
        if not spec.shape_preserving:
            shape = spec.output_shape
            if shape is None or not shape.is_valid():
                errors.append(
                    f"production 算子 {spec.canonical!r} shape_preserving=False 且 "
                    "无显式 OutputShapeContract（input_grain/output_grain）"
                )
        if not spec.index_preserving:
            errors.append(f"production 算子 {spec.canonical!r} index_preserving=False")
        if not spec.columns_preserving:
            errors.append(f"production 算子 {spec.canonical!r} columns_preserving=False")
    return errors


def check_polars_production_backend_explicit() -> list[str]:
    """校验 POLARS_PRODUCTION_SAFE 算子的 polars backend 须显式注册。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon in sorted(POLARS_PRODUCTION_SAFE):
        resolved = OperatorRegistry._aliases.get(canon, canon)
        entry = OperatorRegistry._catalog.get(resolved, {})
        meta = (entry.get("backend_meta") or {}).get("polars")
        if meta is None:
            continue
        if not meta.get("explicit", False):
            errors.append(
                f"POLARS_PRODUCTION_SAFE {canon!r} 的 polars backend 须显式 backend='polars'"
            )
    return errors


def production_allowed_canonicals() -> frozenset[str]:
    """获取允许用于 production 的 canonical 集合。

    返回:
        ``allow_in_production=True`` 的 canonical ``frozenset``。
    """
    return frozenset(spec.canonical for spec in iter_operator_specs() if spec.allow_in_production)


def check_microstructure_param_contracts() -> list[str]:
    """校验微观结构算子须声明非空 ``param_names``。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon, entry in OperatorRegistry._catalog.items():
        if not str(canon).startswith("micro_"):
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        params = entry.get("param_names") or []
        if not params:
            errors.append(f"微观算子 {canon!r} param_names 为空")
    return errors


_FUNDAMENTAL_PERIOD_OPS: frozenset[str] = frozenset(
    {
        "quarter",
        "ttm",
        "yoy",
        "avg2",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
    }
)

_INTRADAY_PARAM_OPS: frozenset[str] = frozenset({"intraday_vwap_deviation"})


def check_intraday_param_contracts() -> list[str]:
    """校验日内算子须声明非空 ``param_names``。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon in _INTRADAY_PARAM_OPS:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None or OperatorRegistry.get(canon) is None:
            continue
        params = entry.get("param_names") or []
        if not params:
            errors.append(f"日内算子 {canon!r} param_names 为空")
    return errors

_CAPM_DUAL_INPUT_OPS: frozenset[str] = frozenset(
    {
        "idio_vol",
        "idio_skew",
        "downside_beta",
        "tail_beta",
        "residual_momentum_capm",
        "coskewness_to_market",
    }
)


def check_fundamental_param_contracts() -> list[str]:
    """校验基本面 period 算子须声明 ``x`` + ``fiscal_quarter`` 参数契约。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon in _FUNDAMENTAL_PERIOD_OPS:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None or OperatorRegistry.get(canon) is None:
            continue
        params = list(entry.get("param_names") or [])
        if "x" not in params:
            errors.append(f"基本面算子 {canon!r} param_names 缺少 'x'")
        if "fiscal_quarter" not in params:
            errors.append(f"基本面算子 {canon!r} param_names 缺少 'fiscal_quarter'")
    return errors


def check_capm_param_contracts() -> list[str]:
    """校验 CAPM 类算子须声明 ``ret`` + ``benchmark_ret`` + ``window``。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    required = ("ret", "benchmark_ret", "window")
    optional_tail = {"tail_beta": ("q",)}
    errors: list[str] = []
    for canon in _CAPM_DUAL_INPUT_OPS:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None or OperatorRegistry.get(canon) is None:
            continue
        params = list(entry.get("param_names") or [])
        for key in required:
            if key not in params:
                errors.append(f"CAPM 算子 {canon!r} param_names 缺少 {key!r}")
        for key in optional_tail.get(canon, ()):
            if key not in params:
                errors.append(f"CAPM 算子 {canon!r} param_names 缺少可选参数 {key!r}")
    return errors


def check_production_plan_ops(plan: Any) -> list[str]:
    """遍历逻辑计划/IR，检查算子是否 production 允许。

    参数:
        plan: 逻辑计划根节点（含 ``op`` 与 ``inputs`` 属性）。

    返回:
        违规描述字符串列表。
    """
    from cleaned_operators.registry import OperatorRegistry

    allowed = production_allowed_canonicals()
    errors: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op and op not in {"column", "literal", "plan_ref"}:
            canon = OperatorRegistry._aliases.get(op, op)
            if canon not in allowed:
                spec = build_operator_spec(canon)
                if spec is None:
                    errors.append(f"算子 {op!r} 无 runtime 实现")
                else:
                    errors.append(f"算子 {op!r} 不允许用于 production（status={spec.status}）")
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return errors


_QOQ_FAMILY = {"fin_qoq", "fin_pct_change", "fin_log_change", "fin_diff"}
# Report periods per fiscal year for quarterly statements.  A cumulative-ytd
# comparison is only meaningful between the SAME fiscal-quarter position across
# years (同比), i.e. an offset that is a whole number of years.
_CUMULATIVE_PERIODS_PER_YEAR = 4


def check_financial_grain_contract(
    formula: str,
    market_context=None,
    data_snapshot=None,
    periods_per_year: int | None = None,
) -> list[str]:
    """Reject single-period growth operators on cumulative (``flow_ytd``) fields.

    ``fin_qoq`` always compares one report period back; a year-to-date cumulative
    field (grain ``flow_ytd`` / flow_semantics ``cumulative_ytd_flow``) is a
    running sum, so QoQ on it computes a spurious current-YTD vs previous-YTD
    change.  ``fin_pct_change`` / ``fin_log_change`` / ``fin_diff`` are also
    single-period by default and inherit the same trap; a whole-year offset
    (``periods`` a multiple of ``periods_per_year``) is valid on cumulative
    because the two points are the same fiscal-quarter position in their
    respective YTD curves (review §5.1).

    R34 P0-021/022: ``market_context``（默认 A 股，显式传入 US 亦受支持）来自
    execution context；``periods_per_year`` 不再硬编码为 4，可从 fiscal 契约/
    market context 解析（A 股季度财报 = 4）。US 年/半年报场景传入实际值。

    P0-33: the check uses the field's typed ``flow_semantics``
    (``cumulative_ytd_flow``) instead of AST-name guessing.  When the field has
    no typed flow semantics it falls back to the legacy grain-name rule and logs.

    The correct sequence for a quarterly change of a cumulative flow is
    ``fin_qoq(fin_quarter_from_cumulative(x, period_id), period_id)``.
    """
    import ast
    import logging

    from fields.resolver import resolve_market_field

    if market_context is None:
        from market.context import ASHARE_CONTEXT

        market_context = ASHARE_CONTEXT
    if periods_per_year is None:
        # 从 market context 的 fiscal 声明解析；A 股季度财报 = 4。
        periods_per_year = int(
            getattr(market_context, "periods_per_year", None) or 4
        )

    logger = logging.getLogger(__name__)
    errors: list[str] = []
    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError:
        return errors

    def _periods_of(call: ast.Call) -> int:
        for keyword in call.keywords:
            if keyword.arg == "periods" and isinstance(keyword.value, ast.Constant):
                return int(keyword.value.value)
        # Positional signature is ``(x, period_id, periods=1)``: periods is the
        # third positional argument (``call.args[2]``).
        if len(call.args) >= 3 and isinstance(call.args[2], ast.Constant):
            return int(call.args[2].value)
        return 1

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in _QOQ_FAMILY:
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        field_name = node.args[0].id
        periods = _periods_of(node)
        resolved = resolve_market_field(field_name, market_context, strict=False)
        spec = resolved.spec if resolved is not None else None
        if spec is None:
            continue
        flow = getattr(spec, "flow_semantics", None)
        if flow == "cumulative_ytd_flow":
            # Cumulative YTD fields: only same fiscal-quarter-position YTD
            # comparisons across years (同比) are meaningful.  A cross-quarter
            # offset subtracts a running total against a different position of
            # the fiscal year and is an accounting error.
            if periods % periods_per_year != 0:
                errors.append(
                    f"{node.func.id}({field_name}) 作用于累计字段 "
                    f"(flow_semantics=cumulative_ytd_flow): 跨季累计值相减无意义，"
                    f"仅同比（同一财年季度位置、periods 为 "
                    f"{periods_per_year} 的整数倍）有效；"
                    f"单期变化前需先用 fin_quarter_from_cumulative 去累计"
                )
        else:
            # Untyped fallback: the field carries no typed flow_semantics, so
            # keep the legacy grain-name rule for fiscal-YTD cumulative fields.
            if "ytd" in tuple(spec.grain or ()):
                if periods == 1:
                    errors.append(
                        f"{node.func.id}({field_name}) 作用于累计字段 "
                        f"(grain=flow_ytd): 单期变化前需先用 "
                        f"fin_quarter_from_cumulative 去累计"
                    )
                logger.info(
                    "check_financial_grain_contract fell back to grain-based rule "
                    "for untyped field %r",
                    field_name,
                )
    return errors


def check_production_formula_ops(formula: str) -> list[str]:
    """解析公式中出现的算子名，检查是否均 production 允许。

    参数:
        formula: DSL 公式字符串。

    返回:
        违规或语法错误描述列表。
    """
    import ast

    from cleaned_operators.registry import OperatorRegistry

    allowed = production_allowed_canonicals()
    errors: list[str] = []
    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError as exc:
        return [f"公式语法错误: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "col":
                continue
            canon = OperatorRegistry._aliases.get(name, name)
            if canon not in allowed:
                spec = build_operator_spec(canon)
                if spec is None:
                    errors.append(f"算子 {name!r} 无 runtime 实现")
                else:
                    errors.append(f"算子 {name!r} 不允许用于 production（status={spec.status}）")
    return errors
