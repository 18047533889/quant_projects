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
PRODUCTION_ALLOWED_DEFERRED_CANONICALS: frozenset[str] = frozenset(
    {
        "ts_ema",
        "ts_decay_linear",
        "RSI_WILDER",
        "ATR_WILDER",
    }
)

# legacy PIT core（不含 deferred stateful）
_LEGACY_PRODUCTION_CORE: frozenset[str] = frozenset(
    {
        "ts_rank",
        "ts_sharpe",
        "ts_autocorr",
        "cs_resid",
        "cs_regression",
        "ffill",
    }
)

# DSL alias；canonical 为 ``where``
_FASTPATH_ALIAS_ONLY: frozenset[str] = frozenset({"if_else"})


def _build_production_dual_backend_core() -> frozenset[str]:
    """构建 strict dual-backend production core（不含 deferred stateful）。"""
    from backend.production_fastpath_tiers import (
        P0_PRODUCTION_FASTPATH_CANONICALS,
        P1_POLARS_CORE_PRODUCTION_SAFE,
    )

    return (
        P0_PRODUCTION_FASTPATH_CANONICALS
        | P1_POLARS_CORE_PRODUCTION_SAFE
        | _LEGACY_PRODUCTION_CORE
    ) - _FASTPATH_ALIAS_ONLY


PRODUCTION_DUAL_BACKEND_CORE_CANONICALS: frozenset[str] = _build_production_dual_backend_core()


def _build_production_core_canonicals() -> frozenset[str]:
    """构建 PRODUCTION_CORE canonical 集合（dual-backend core + deferred allowed）。"""
    return PRODUCTION_DUAL_BACKEND_CORE_CANONICALS | PRODUCTION_ALLOWED_DEFERRED_CANONICALS


# SQL 下推 / DSL 常用 production 核心 canonical（PIT 门禁 / 契约脚本对齐此清单）
PRODUCTION_CORE_CANONICALS: frozenset[str] = _build_production_core_canonicals()

# 向后兼容别名
PRODUCTION_PIT_REQUIRED: frozenset[str] = PRODUCTION_CORE_CANONICALS

_PIT_EXEMPT: frozenset[str] = frozenset(
    {"column", "literal", "col", "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate", "shuffle"}
)

PERMANENTLY_FORBIDDEN_CANONICALS: frozenset[str] = frozenset(
    {
        "bfill",
        "causal_bfill",
        "fillna_interpolate",
        "shuffle",
        "Lead",
        "next",
        "dropna",
        "constant",
    }
)

ProductionPolicy = Literal["allowed", "pending", "denied", "permanently_forbidden"]

# production DSL 禁止（可 research / experimental，不可 production 投递）
PRODUCTION_DENIED_CANONICALS: frozenset[str] = frozenset(
    {
        "dropna",
        "bfill",
        "causal_bfill",
        "fillna_interpolate",
        "shuffle",
        "constant",
        "fft",
        "ifft",
        "wavelet",
        "convolve",
        "correlate",
        "mat_inverse",
        "eig",
        "svd",
        "pca",
        "rolling_beta_to_market",
        "downside_beta",
        "tail_beta",
        "residual_momentum_capm",
        "coskewness_to_market",
        "idio_vol",
        "idio_skew",
        "quarter",
        "ttm",
        "yoy",
        "avg2",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        "operating_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "intraday_vwap_deviation",
        "rank_corr",
        "ts_poly2_coeff",
        "ts_poly2_resid",
        "digital_count",
    }
)


def is_production_denied(canon: str) -> bool:
    """判断 canonical 是否禁止用于 production 投递。

    参数:
        canon: canonical 算子名。

    返回:
        禁止返回 ``True``（含 ``micro_*`` 前缀算子）。
    """
    if canon in PRODUCTION_DENIED_CANONICALS:
        return True
    if str(canon).startswith("micro_"):
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
    supports_panel: bool = True
    supports_polars: bool = False
    polars_long_tier: str = "unsupported"
    shape_preserving: bool = True
    index_preserving: bool = True
    columns_preserving: bool = True
    numerical_stability: NumericalStability = "medium"
    description: str = ""
    execution_kind: str = "primitive"
    production_policy: ProductionPolicy = "denied"
    lowering_available: bool = False
    dual_backend_target: bool = False

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

    参数:
        resolved: 解析后的 canonical 名。
        status: 算子生命周期状态。
        pit_safe: 是否 PIT 安全。
        shape_preserving: 是否保持 panel 形状。

    返回:
        允许 production 返回 ``True``。
    """
    if is_production_denied(resolved):
        return False
    if status in ("experimental", "deprecated", "stub", "doc_only"):
        return False
    if not shape_preserving:
        return False
    if resolved in PRODUCTION_CORE_CANONICALS:
        return True
    if status == "production":
        return pit_safe
    return False


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

    return OperatorSpec(
        canonical=resolved,
        status=status,
        allow_in_production=allow_in_production,
        deterministic=deterministic,
        pit_safe=policy.pit_safe,
        backends=all_backends,
        policy=policy,
        param_names=tuple(getattr(meta, "param_names", None) or ()),
        supports_panel=True,
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
        dual_backend_target=execution_kind in {"composite", "primitive", "stateful"},
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
    return {
        "name": spec.canonical,
        "scope": scope_map.get(pol.scope, pol.scope),
        "frequency": "any",
        "input_fields": list(spec.param_names),
        "output_type": "series",
        "pit_safe": spec.pit_safe,
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
    """校验 production 算子须保持 panel shape/index/columns。

    返回:
        违规描述字符串列表。
    """
    errors: list[str] = []
    for spec in iter_operator_specs():
        if not spec.allow_in_production:
            continue
        if not spec.shape_preserving:
            errors.append(f"production 算子 {spec.canonical!r} shape_preserving=False")
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
