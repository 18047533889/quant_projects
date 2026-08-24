# -*- coding: utf-8 -*-
"""R30 §24 (P1-025): explicit ParamRole for every production scalar.

``effective_param_role`` has a fallback that maps an undeclared-role scalar to
ECONOMIC when ``searchable`` — that silent default is the exact search-space
pollution R30 targets.  This module backfills an explicit role for production
scalars from a *reviewed, name-based* rule table and marks each backfilled spec
``role_source="rule"`` so the audit can tell rule-based roles from authored
ones.  A scalar that matches no rule stays unroled and is reported by
``audit_r30`` (never silently ECONOMIC for production).
"""
from __future__ import annotations

from factor_engine.cleaned_operators.base import ParamRole

# name-suffix / name-substring -> role.  Rule application order matters: the
# first matching rule wins.  Roles that are governance (never searched) take
# priority so a ``*_window``-looking governance knob is not mis-rolled.
_RULE_ROLES: tuple[tuple[str, ParamRole], ...] = (
    # Governance / data-policy knobs — never a search dimension.
    ("missing_policy", ParamRole.MISSING_POLICY),
    ("impute", ParamRole.MISSING_POLICY),
    ("min_finite", ParamRole.SUPPORT_POLICY),
    ("min_group_size", ParamRole.SUPPORT_POLICY),
    ("min_cross", ParamRole.SUPPORT_POLICY),
    ("min_periods", ParamRole.SUPPORT_POLICY),
    ("min_line", ParamRole.SUPPORT_POLICY),
    ("coverage_threshold", ParamRole.SUPPORT_POLICY),
    ("side", ParamRole.POLICY),
    ("add_intercept", ParamRole.POLICY),
    ("center", ParamRole.POLICY),
    ("seed", ParamRole.POLICY),
    ("shock_quantile", ParamRole.STATE_THRESHOLD),
    ("quantile", ParamRole.STATE_THRESHOLD),
    ("event_effective_lag", ParamRole.HORIZON),
    ("label_horizon", ParamRole.HORIZON),
    ("horizon", ParamRole.HORIZON),
    # Estimator resolution knobs — small reviewed grid only.
    ("n_components", ParamRole.MODEL_ORDER),
    ("n_experts", ParamRole.MODEL_ORDER),
    ("n_regimes", ParamRole.MODEL_ORDER),
    ("order", ParamRole.MODEL_ORDER),
    ("dim", ParamRole.MODEL_ORDER),
    ("component", ParamRole.MODEL_ORDER),
    ("delay", ParamRole.ESTIMATOR_RESOLUTION),
    ("lag", ParamRole.HORIZON),
    ("penetration", ParamRole.STATE_THRESHOLD),
    ("eps_fraction", ParamRole.NUMERICAL),
    ("epsilon", ParamRole.NUMERICAL),
    ("bins", ParamRole.ESTIMATOR_RESOLUTION),
    ("theiler", ParamRole.ESTIMATOR_RESOLUTION),
    ("tau", ParamRole.ESTIMATOR_RESOLUTION),
    ("min_embeddings", ParamRole.SUPPORT_POLICY),
    ("min_patterns", ParamRole.SUPPORT_POLICY),
    ("min_segment", ParamRole.SUPPORT_POLICY),
    ("min_contiguous_fraction", ParamRole.SUPPORT_POLICY),
    ("min_effective_n", ParamRole.SUPPORT_POLICY),
    ("k_min", ParamRole.SUPPORT_POLICY),
    ("l1_ratio", ParamRole.ESTIMATOR_RESOLUTION),
    ("normalized", ParamRole.POLICY),
    ("bias_correction", ParamRole.POLICY),
    ("include_current", ParamRole.POLICY),
    ("residual_fraction", ParamRole.STATE_THRESHOLD),
    ("q", ParamRole.STATE_THRESHOLD),
    ("alpha", ParamRole.STATE_THRESHOLD),
    # Windows / horizons — the dominant searched dimension.
    ("_window", ParamRole.HORIZON),
    ("window", ParamRole.HORIZON),
    ("_lag", ParamRole.HORIZON),
    ("history_days", ParamRole.HORIZON),
    ("max_lookback", ParamRole.HORIZON),
    ("level", ParamRole.MODEL_ORDER),
    ("k", ParamRole.STATE_THRESHOLD),
    ("cutoff", ParamRole.STATE_THRESHOLD),
)

_APPLIED = False


def backfill_scalar_roles() -> int:
    """Backfill explicit roles on all registered operator param_specs.

    Returns the number of scalars backfilled by rule.
    """
    global _APPLIED
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    backfilled = 0
    for backends_map in OperatorRegistry._operators.values():
        # R30 §24: prefer the pandas_numpy reference backend's metadata (the
        # canonical contract); a polars wrapper may carry a different, thinner
        # param_specs dict and would mask the canonical window spec.
        op = (
            backends_map.get("pandas_numpy")
            or backends_map.get("polars")
            or next(iter(backends_map.values()))
        )
        meta = getattr(op, "metadata", None)
        if meta is None:
            continue
        specs = getattr(meta, "param_specs", None) or {}
        for name, spec in specs.items():
            if getattr(spec, "param_role", None) is not None:
                continue  # authored role is authoritative
            for needle, role in _RULE_ROLES:
                if needle in name:
                    # ParamSpec is a frozen dataclass; use object.__setattr__ so
                    # the backfill is explicit and auditable (role_source marks
                    # it as rule-derived rather than authored).
                    object.__setattr__(spec, "param_role", role)
                    object.__setattr__(spec, "role_source", "rule")
                    backfilled += 1
                    break
    _APPLIED = True
    return backfilled


def rule_role_for(param_name: str) -> ParamRole | None:
    for needle, role in _RULE_ROLES:
        if needle in param_name:
            return role
    return None
