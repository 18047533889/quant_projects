# -*- coding: utf-8 -*-
"""Point-in-Time safety audit for operator IR and logical SourceRef dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factor_engine.ir.nodes import IRNode


class PitSafetyError(RuntimeError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("PIT 安全审计失败: " + ", ".join(sorted(set(violations))))


@dataclass
class PitAuditReport:
    violations: list[str] = field(default_factory=list)
    checked_ops: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "checked_ops": list(self.checked_ops),
        }


_FORWARD_FILL_OPS = frozenset({
    "ffill", "fillna_ffill", "forward_fill", "fill_forward",
    "bfill", "fillna_backfill", "backfill",
})
# R13 P1-12: ``fillna`` only counts as forward fill when its ``method`` argument
# is an ffill-class spelling.  The ``method`` literal can appear either in
# ``node.attrs`` (keyword) or as a literal input at index 1 (positional).
_FFILL_METHOD_ALIASES = frozenset({"ffill", "pad", "forward_fill"})


def _fillna_uses_forward_fill(node: IRNode) -> bool:
    """Whether a ``fillna`` IR node is an ffill-class forward fill.

    ``fillna(x, "ffill")`` is canonical-rewritten onto the gated ``ffill``
    implementation at execution time; the audit must treat the two spellings
    identically so ``forbid_forward_fill=True`` cannot be bypassed by the
    ``fillna`` spelling.
    """
    method = (node.attrs or {}).get("method")
    if method is None and len(node.inputs) > 1:
        child = node.inputs[1]
        if child.op == "literal":
            method = (child.attrs or {}).get("value")
    return isinstance(method, str) and method.strip().lower() in _FFILL_METHOD_ALIASES


_POSITIVE_LAG_PARAMS = {
    "ts_delay": ("n", 1),
    "ts_delta": ("n", 1),
    "ts_pct": ("d", 1),
}
_FINANCIAL_TABLES = frozenset({"StockIncome", "StockCashFlow", "StockBalance", "StockIndicator"})
_EXACT_DAILY_TABLES = frozenset({
    "DailyBar", "StockDailyBar", "StockDailyBarAdj", "StockValuationDaily",
    "SizeDaily", "StockCapitalDaily", "IndexConstituent", "BenchmarkIndexDailyBar",
    "IndexDailyBar", "EtfDailyBar", "ETFDailyBar", "Calendar",
})
_ASOF_DAILY_TABLES = frozenset({
    "IndustryDaily", "StockIndustry", "StockStatus", "StockList", "ETFList", "IndexList",
})
_RELATION_TABLES = frozenset({"StockTopTenShareholder", "StockTopTenFloatShareholder"})
_EFFECTIVE_TABLES = frozenset({"StockDividend"})
_MINUTE_TABLES = frozenset({"StockMinuteBar", "MinuteBar", "StockMinuteBarAdj", "MinuteBarAdj"})
_PROFILE_FEATURES = frozenset({
    "profile_zscore",
    "profile_deviation",
    "abnormal_volume_profile",
    "abnormal_return_profile",
    "abnormal_vol_profile",
})
_TIMESTAMP_CONVENTIONS = frozenset({"bar_end", "bar_start"})

# ---------------------------------------------------------------------------
# R13 P1-13: the DataAccess contract registry (``storage.sources.logical_tables``
# ``LogicalTableContract``) is the single source of truth for table temporal /
# PIT classification.  The hardcoded table sets above are retained ONLY as a
# research fallback when the contract layer cannot be imported.
# ---------------------------------------------------------------------------
# join_policy -> transforms the contract allows.  ``None`` is the identity read.
_JOIN_POLICY_ALLOWED_TRANSFORMS: dict[str, frozenset[str | None]] = {
    "anchor": frozenset({None}),
    "exact": frozenset({None}),
    "exact_date": frozenset({None}),
    "asof_backward": frozenset({None, "asof_backward"}),
    "financial_pit": frozenset({None, "financial_asof", "financial_lag"}),
    "relation_pit": frozenset({None, "relation_asof", "relation_aggregate"}),
    "effective_only": frozenset({None}),
    "minute_session": frozenset({
        "intraday_feature", "minute_at", "minute_range", "minute_bar",
        "minute_resample",
    }),
    "special": frozenset({None}),
}
_LEGACY_REQUIRED_PARAM_ALIAS = {
    "IndexSymbol": "index",
    "IndustrySource": "industry_source",
}
_CONTRACT_LAYER_UNAVAILABLE = object()


def _literal_number(node: IRNode, *, attr: str, input_index: int) -> float | None:
    raw = (node.attrs or {}).get(attr)
    if raw is None and input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            raw = (child.attrs or {}).get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _audit_derived_field(
    field,
    *,
    forbid_forward_fill,
    fail_on_missing,
    violations,
    checked,
    source_guard,
):
    key = f"DerivedField:{field}"
    if key in source_guard:
        violations.append(f"{key}(cycle)")
        return
    source_guard.add(key)
    try:
        from factor_engine.runtime.derived_field_registry import load_derived_field_definition
        from factor_engine.api.dsl_parser import parse_factor
        from factor_engine.ir.analyzer import Analyzer

        definition = load_derived_field_definition(field)
        factor = parse_factor(
            definition.expression,
            name=f"pit::{field}@{definition.version}",
            surface="compat",
            dialect="lqtp",
            dialect_version="2026-07-19",
        )
        report = audit_ir(
            Analyzer().lower(factor.expr).ir,
            forbid_forward_fill=forbid_forward_fill,
            fail_on_missing=fail_on_missing,
            _source_guard=source_guard,
        )
        checked.extend(f"{key}->{op}" for op in report.checked_ops)
        violations.extend(f"{key}->{item}" for item in report.violations)
    except Exception as exc:
        violations.append(f"{key}(unverifiable:{type(exc).__name__})")
    finally:
        source_guard.discard(key)


def _valid_hhmm(text: str) -> bool:
    try:
        hour, minute = (int(value) for value in text.split(":"))
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except Exception:
        return False


def _audit_intraday_feature(
    label: str,
    transform_params: dict[str, Any],
    violations: list[str],
) -> None:
    feature = str(transform_params.get("feature") or "").strip()
    if not feature:
        violations.append(f"{label}(missing_feature)")

    try:
        bar_minutes = int(transform_params.get("bar_minutes", 5))
    except (TypeError, ValueError):
        bar_minutes = 0
    if bar_minutes <= 0:
        violations.append(f"{label}(bar_minutes<=0)")

    try:
        minimum_bars = int(transform_params.get("min_bars", 2))
    except (TypeError, ValueError):
        minimum_bars = 0
    if minimum_bars < 2:
        violations.append(f"{label}(min_bars<2)")

    try:
        coverage = float(transform_params.get("min_coverage", 0.8))
    except (TypeError, ValueError):
        coverage = 0.0
    if not 0 < coverage <= 1:
        violations.append(f"{label}(invalid_min_coverage)")

    convention = str(
        transform_params.get("timestamp_convention", "bar_end")
    ).lower()
    if convention not in _TIMESTAMP_CONVENTIONS:
        violations.append(f"{label}(invalid_timestamp_convention)")

    cutoff = str(transform_params.get("cutoff_time") or "session_close")
    if cutoff != "session_close" and not _valid_hhmm(cutoff):
        violations.append(f"{label}(invalid_cutoff_time)")
    for key in ("session_open", "session_close", "split_time"):
        if (
            key in transform_params
            and transform_params[key] is not None
            and not _valid_hhmm(str(transform_params[key]))
        ):
            violations.append(f"{label}(invalid_{key})")

    # history_days=0 is valid for same-session features. Only profile features
    # require a positive multi-day baseline.
    if "history_days" in transform_params:
        try:
            history_days = int(transform_params["history_days"])
        except (TypeError, ValueError):
            history_days = -1
        if history_days < 0:
            violations.append(f"{label}(history_days<0)")
    else:
        history_days = 0

    for key in ("minutes", "session_minutes", "lag"):
        if key not in transform_params:
            continue
        try:
            value = int(transform_params[key])
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            violations.append(f"{label}({key}<=0)")

    if feature in _PROFILE_FEATURES and history_days < 2:
        violations.append(f"{label}(profile_history_days<2)")

    if "q" in transform_params:
        try:
            quantile = float(transform_params["q"])
        except (TypeError, ValueError):
            quantile = 0.0
        if not 0 < quantile < 1:
            violations.append(f"{label}(q_not_in_0_1)")


def _resolve_table_contract(table: str) -> LogicalTableContract | None | object:
    """Resolve a SourceRef table's contract from the DataAccess contract registry.

    Returns:
        * a ``LogicalTableContract`` — the registry is authoritative;
        * ``None`` — the contract layer loaded but the table is not registered
          (an unknown availability contract);
        * ``_CONTRACT_LAYER_UNAVAILABLE`` — the contract module could not be
          imported; callers fall back to the hardcoded classification.
    """
    try:
        from factor_engine.storage.sources.logical_tables import logical_table_contract
    except Exception:
        return _CONTRACT_LAYER_UNAVAILABLE
    try:
        return logical_table_contract(table)
    except Exception:
        return None


def _audit_source_ref_by_contract(
    specification,
    contract: "LogicalTableContract",
    *,
    label: str,
    params: dict[str, Any],
    transform_params: dict[str, Any],
    forbid_forward_fill: bool,
    fail_on_missing: bool,
    violations: list[str],
    checked: list[str],
    source_guard: set[str],
) -> bool:
    """Validate a SourceRef against its ``LogicalTableContract`` (R13 P1-13).

    The contract is the single source of truth: transform legality, required
    resolution parameters, relation cardinality and temporal policy all come from
    the contract's ``join_policy`` / ``required_parameter`` / ``cardinality``
    instead of a hand-written table classification.
    """
    join_policy = contract.join_policy

    if join_policy == "special":
        if specification.table == "Intermediate":
            try:
                from factor_engine.runtime.intermediate_registry import intermediate_dependency_lineage

                intermediate_dependency_lineage(
                    str(params.get("name", "")), int(params.get("version", 0))
                )
            except Exception as exc:
                violations.append(f"{label}(unversioned:{type(exc).__name__})")
        elif specification.table == "DerivedField":
            _audit_derived_field(
                specification.field,
                forbid_forward_fill=forbid_forward_fill,
                fail_on_missing=fail_on_missing,
                violations=violations,
                checked=checked,
                source_guard=source_guard,
            )
        else:
            violations.append(f"{label}(unknown_availability_contract)")
        return True

    # Required resolution parameter declared by the contract (IndexSymbol /
    # IndustrySource).  Accept both the canonical and the legacy spelling; a
    # conflict between the two is rejected (mirrors the executor).
    required = contract.required_parameter
    if required:
        legacy = _LEGACY_REQUIRED_PARAM_ALIAS.get(required)
        canonical_value = str(params.get(required, "")).strip()
        legacy_value = str(params.get(legacy, "")).strip() if legacy else ""
        if not canonical_value and not legacy_value:
            violations.append(f"{label}(missing_{required})")
        if canonical_value and legacy_value and canonical_value != legacy_value:
            violations.append(f"{label}(conflicting_{required})")

    if join_policy == "minute_session":
        if specification.transform == "intraday_feature":
            _audit_intraday_feature(label, transform_params, violations)
        elif specification.transform not in {
            "minute_at", "minute_range", "minute_bar", "minute_resample"
        }:
            violations.append(f"{label}(minute_transform_required)")
        else:
            if (
                specification.transform == "minute_at"
                and not str(transform_params.get("hhmm", "")).strip()
            ):
                violations.append(f"{label}(missing_hhmm)")
            if specification.transform == "minute_range":
                start = str(transform_params.get("start", ""))
                end = str(transform_params.get("end", ""))
                if not start or not end or start >= end:
                    violations.append(f"{label}(invalid_minute_range)")
            if specification.transform in {"minute_bar", "minute_resample"}:
                try:
                    period = int(transform_params.get("period", 1))
                    index = int(transform_params.get("index", 0))
                except (TypeError, ValueError):
                    period, index = 0, -1
                if period <= 0 or index < 0:
                    violations.append(f"{label}(invalid_minute_period_or_index)")
        return True

    if join_policy == "effective_only":
        # An effective-only table is not strict PIT regardless of transform.
        violations.append(f"{label}(effective_only_not_strict_pit)")
        return True

    allowed = _JOIN_POLICY_ALLOWED_TRANSFORMS.get(join_policy, frozenset({None}))
    if specification.transform not in allowed:
        violations.append(
            f"{label}(unsupported_transform={specification.transform})"
        )

    if join_policy == "financial_pit" and specification.transform == "financial_lag":
        try:
            quarters = int(transform_params.get("quarters", 1))
        except (TypeError, ValueError):
            quarters = 0
        if quarters <= 0:
            violations.append(f"{label}(financial_lag_quarters<=0)")

    if join_policy == "relation_pit":
        if (
            contract.cardinality == "one_to_many"
            and specification.transform not in {"relation_asof", "relation_aggregate"}
        ):
            violations.append(f"{label}(relation_requires_scalar_selector)")

    return True


def _audit_source_ref_hardcoded(
    specification,
    *,
    label: str,
    params: dict[str, Any],
    transform_params: dict[str, Any],
    forbid_forward_fill: bool,
    fail_on_missing: bool,
    violations: list[str],
    checked: list[str],
    source_guard: set[str],
) -> bool:
    """Research fallback classification used only when the contract layer is
    unavailable (import failure) or the table is not registered.  Production /
    ``fail_on_missing`` never reaches here for unknown tables — the contract path
    fails them closed (R13 P1-13)."""
    if specification.table in _EXACT_DAILY_TABLES:
        if specification.table in {"BenchmarkIndexDailyBar", "IndexDailyBar", "IndexConstituent"}:
            canonical_index = str(params.get("IndexSymbol", "")).strip()
            legacy_index = str(params.get("index", "")).strip()
            if not canonical_index and not legacy_index:
                violations.append(f"{label}(missing_IndexSymbol)")
            if canonical_index and legacy_index and canonical_index != legacy_index:
                violations.append(f"{label}(conflicting_IndexSymbol)")
        if specification.transform is not None:
            violations.append(
                f"{label}(unexpected_transform={specification.transform})"
            )
        return True

    if specification.table in _ASOF_DAILY_TABLES:
        if specification.transform not in {None, "asof_backward"}:
            violations.append(
                f"{label}(unsupported_transform={specification.transform})"
            )
        return True

    if specification.table in _FINANCIAL_TABLES:
        if specification.transform not in {None, "financial_asof", "financial_lag"}:
            violations.append(
                f"{label}(unsupported_transform={specification.transform})"
            )
        if specification.transform == "financial_lag":
            try:
                quarters = int(transform_params.get("quarters", 1))
            except (TypeError, ValueError):
                quarters = 0
            if quarters <= 0:
                violations.append(f"{label}(financial_lag_quarters<=0)")
        return True

    if specification.table in _RELATION_TABLES:
        if specification.transform not in {None, "relation_asof", "relation_aggregate"}:
            violations.append(f"{label}(unsupported_relation_transform={specification.transform})")
        return True

    if specification.table in _EFFECTIVE_TABLES:
        violations.append(f"{label}(effective_only_not_strict_pit)")
        return True

    if specification.table in _MINUTE_TABLES:
        if specification.transform == "intraday_feature":
            _audit_intraday_feature(label, transform_params, violations)
            return True
        if specification.transform not in {
            "minute_at", "minute_range", "minute_bar", "minute_resample"
        }:
            violations.append(f"{label}(minute_transform_required)")
            return True
        if (
            specification.transform == "minute_at"
            and not str(transform_params.get("hhmm", "")).strip()
        ):
            violations.append(f"{label}(missing_hhmm)")
        if specification.transform == "minute_range":
            start = str(transform_params.get("start", ""))
            end = str(transform_params.get("end", ""))
            if not start or not end or start >= end:
                violations.append(f"{label}(invalid_minute_range)")
        if specification.transform in {"minute_bar", "minute_resample"}:
            try:
                period = int(transform_params.get("period", 1))
                index = int(transform_params.get("index", 0))
            except (TypeError, ValueError):
                period, index = 0, -1
            if period <= 0 or index < 0:
                violations.append(f"{label}(invalid_minute_period_or_index)")
        return True

    if specification.table == "Intermediate":
        try:
            from factor_engine.runtime.intermediate_registry import intermediate_dependency_lineage

            intermediate_dependency_lineage(
                str(params.get("name", "")), int(params.get("version", 0))
            )
        except Exception as exc:
            violations.append(f"{label}(unversioned:{type(exc).__name__})")
        return True

    if specification.table == "DerivedField":
        _audit_derived_field(
            specification.field,
            forbid_forward_fill=forbid_forward_fill,
            fail_on_missing=fail_on_missing,
            violations=violations,
            checked=checked,
            source_guard=source_guard,
        )
        return True

    if specification.table == "TurnoverBaseDaily":
        if specification.transform is not None:
            violations.append(
                f"{label}(unexpected_transform={specification.transform})"
            )
        return True

    violations.append(f"{label}(unknown_availability_contract)")
    return True


def _audit_source_ref(
    name,
    *,
    forbid_forward_fill,
    fail_on_missing,
    violations,
    checked,
    source_guard,
) -> bool:
    from factor_engine.api.source_ref import decode_source_ref

    specification = decode_source_ref(name)
    if specification is None:
        return False
    label = f"SourceRef[{specification.table}.{specification.field}]"
    checked.append(label)
    params = specification.params_dict()
    transform_params = specification.transform_params_dict()

    contract = _resolve_table_contract(specification.table)
    if contract is not _CONTRACT_LAYER_UNAVAILABLE:
        if contract is None:
            # The contract layer loaded but the table is not registered.  In
            # production this is fail-closed (R13 P1-13): an unknown table must
            # not silently pass through a hardcoded classification.
            if fail_on_missing:
                violations.append(f"{label}(unknown_availability_contract)")
                return True
        else:
            return _audit_source_ref_by_contract(
                specification,
                contract,
                label=label,
                params=params,
                transform_params=transform_params,
                forbid_forward_fill=forbid_forward_fill,
                fail_on_missing=fail_on_missing,
                violations=violations,
                checked=checked,
                source_guard=source_guard,
            )

    # Research fallback: contract layer unavailable (import failure) or the
    # table is not in the registry.  Production never reaches here for unknown
    # tables — they fail closed above.
    return _audit_source_ref_hardcoded(
        specification,
        label=label,
        params=params,
        transform_params=transform_params,
        forbid_forward_fill=forbid_forward_fill,
        fail_on_missing=fail_on_missing,
        violations=violations,
        checked=checked,
        source_guard=source_guard,
    )


def audit_ir(
    ir: IRNode,
    *,
    forbid_forward_fill: bool = False,
    fail_on_missing: bool = True,
    _source_guard: set[str] | None = None,
) -> PitAuditReport:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    violations: list[str] = []
    checked: list[str] = []
    source_guard = _source_guard if _source_guard is not None else set()

    def walk(node: IRNode) -> None:
        if node.op == "column":
            _audit_source_ref(
                str((node.attrs or {}).get("name") or ""),
                forbid_forward_fill=forbid_forward_fill,
                fail_on_missing=fail_on_missing,
                violations=violations,
                checked=checked,
                source_guard=source_guard,
            )
            return
        if node.op in {"literal", "plan_ref"}:
            return
        checked.append(node.op)
        canonical = OperatorRegistry.resolve_canonical_optional(node.op)
        # Semantic IR-level checks first: they inspect the IR shape (literal
        # params, operator semantics) and must fire regardless of whether the
        # runtime is registered.  Production unregisters ``ffill`` / ``fillna``
        # via layer_governance, so the forward-fill / future-data / lag-param
        # violations are semantic ones and must not be masked by a later
        # ``missing_runtime`` early-return (R13 P1-12).
        rule = _POSITIVE_LAG_PARAMS.get(canonical)
        if rule is not None:
            parameter, index = rule
            value = _literal_number(node, attr=parameter, input_index=index)
            if value is not None and value < 1:
                violations.append(f"{canonical}({parameter}={value:g})")
        if canonical in {
            "bfill", "backfill", "fillna_backfill", "fillna_bfill",
            "lead", "Lead", "next",
        }:
            violations.append(f"{canonical}(future_data)")
        if forbid_forward_fill:
            if canonical in _FORWARD_FILL_OPS:
                violations.append(f"{canonical}(forward_fill)")
            elif canonical == "fillna" and _fillna_uses_forward_fill(node):
                # R13 P1-12: fillna(method="ffill"/"pad"/"forward_fill") is a
                # forward-fill and must be blocked by the same gate as ``ffill``.
                violations.append("fillna(forward_fill)")

        implementation = OperatorRegistry.get(canonical)
        if implementation is None:
            if fail_on_missing:
                violations.append(f"{canonical}(missing_runtime)")
            return
        policy = infer_operator_policy(implementation, canonical=canonical)
        if not policy.pit_safe or policy.lag < 0:
            violations.append(canonical)
        for child in node.inputs:
            walk(child)

    walk(ir)
    return PitAuditReport(violations=violations, checked_ops=checked)


def assert_pit_safe(
    ir: IRNode,
    *,
    enforce: bool = True,
    forbid_forward_fill: bool = False,
    fail_on_missing: bool = True,
) -> PitAuditReport:
    report = audit_ir(
        ir,
        forbid_forward_fill=forbid_forward_fill,
        fail_on_missing=fail_on_missing,
    )
    if enforce and not report.passed:
        raise PitSafetyError(report.violations)
    return report
