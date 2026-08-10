# -*- coding: utf-8 -*-
"""R30 master audit: aggregate the hard gates across the phases implemented so
far.  Reports each gate TRUE/FALSE and the exit code reflects R30_HARD_BLOCKERS_ZERO.
"""
from __future__ import annotations

import sys
import json

sys.path.insert(0, ".")
from cleaned_operators import load_all
load_all()

from cleaned_operators.registry import OperatorRegistry as R
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.tombstones import (
    ALL_TOMBSTONED_NAMES, RANDOM_TOMBSTONED_NAMES,
    FUTURE_TOMBSTONED_NAMES, NONCAUSAL_FILL_TOMBSTONED_NAMES,
)
from cleaned_operators.base import missing_role_defaults_to_searchable
from cleaned_operators.operator_spec import _infer_panel_params
from runtime.execution_contract import (
    _STATEFUL_CANONICALS, _DECLARED_STATEFUL, execution_contract,
)

gates: dict[str, bool] = {}

# --- Phase 2: random/future/noncausal physically removed -------------------
gates["R30_ACTIVE_RANDOM_FACTOR_PRIMITIVES_ZERO"] = all(
    n not in R._operators and n not in R._catalog and n not in R._aliases
    for n in RANDOM_TOMBSTONED_NAMES
)
gates["R30_ACTIVE_FUTURE_REFERENCE_PRIMITIVES_ZERO"] = all(
    n not in R._operators and n not in R._catalog and n not in R._aliases
    for n in FUTURE_TOMBSTONED_NAMES
)
gates["R30_ACTIVE_NONCAUSAL_FILL_PRIMITIVES_ZERO"] = all(
    n not in R._operators and n not in R._catalog and n not in R._aliases
    for n in NONCAUSAL_FILL_TOMBSTONED_NAMES
)

# --- Phase 3: default production loader research zero -----------------------
from cleaned_operators import RESEARCH_LOAD_MODULES, INTERNAL_KERNEL_MODULES, PRODUCTION_LOAD_MODULES, _LOAD_MODULES
gates["R30_LOADER_SPLIT_DECLARED"] = bool(
    RESEARCH_LOAD_MODULES and INTERNAL_KERNEL_MODULES and PRODUCTION_LOAD_MODULES
    and set(RESEARCH_LOAD_MODULES) <= set(_LOAD_MODULES)
)

# --- Phase 4: raw registry production bypass zero ----------------------------
# Tombstoned names RAISE RemovedOperatorError (never return executable); they
# also have no runtime dict entry.  Non-production-surface operators return None
# under mode="production".
gates["R30_RAW_REGISTRY_PRODUCTION_BYPASS_ZERO"] = all(
    n not in R._operators and n not in R._catalog and n not in R._aliases
    for n in ALL_TOMBSTONED_NAMES
) and all(
    R.get(c, mode="production") is None
    for c in R.list_canonical()
    if classify_canonical(c) in ("research", "unsafe", "internal", "legacy")
    and R._operators.get(c)
)

# --- Phase 5: production admission requires pit_safe ------------------------
from cleaned_operators.operator_spec import _compute_allow_in_production, _infer_status
from cleaned_operators.operator_policy import infer_operator_policy
gates["R30_PRODUCTION_ADMISSION_REQUIRES_PIT_SAFE"] = (
    _compute_allow_in_production("x", status="production", pit_safe=False) is False
)
gates["R30_RESEARCH_STATUS_PRODUCTION_ZERO"] = (
    _compute_allow_in_production("x", status="research", pit_safe=True) is False
)

# --- Phase 6: param kind/role explicit ---------------------------------------
viol_roles = 0
for c in R.list_canonical():
    if classify_canonical(c) not in ("daily", "extended"):
        continue
    op = R.get(c, mode="any")
    if op is None:
        continue
    meta = getattr(op, "metadata", None)
    if meta is None:
        continue
    declared = tuple(getattr(meta, "panel_params", None) or ())
    panels = set(declared) or set(_infer_panel_params(op, meta, R._catalog.get(c, {})))
    specs = getattr(meta, "param_specs", None) or {}
    names = tuple(getattr(meta, "param_names", None) or ())
    for p in names:
        if p in panels:
            continue
        spec = specs.get(p)
        if spec is None:
            continue
        if missing_role_defaults_to_searchable(spec):
            viol_roles += 1
gates["R30_ALL_PRODUCTION_SCALAR_PARAM_ROLES_EXPLICIT"] = viol_roles == 0

# --- Phase 9: concrete bugs closed ------------------------------------------
from cleaned_operators.dmd import _log_finite_horizon_sum
import numpy as np
gates["R30_DMD_GEOMETRIC_LOGSUM_OVERFLOW_ZERO"] = all(
    np.isfinite(_log_finite_horizon_sum(lr, 50)) for lr in (100.0, 1000.0, 10000.0)
)

from cleaned_operators.stateful.survival import _survival_kernel
ages = _survival_kernel(np.array([0.0, 1.0, np.nan, 1.0, 0.0]), 60, 1, 1, 1, 1.0,
                        inactive_policy="zero", gap_policy="lower_bound")[0]
gates["R30_SURVIVAL_GAP_STATE_BUG_CLOSED"] = ages[3] == 1.0

from runtime.execution_contract import minimum_effective_samples
gates["R30_REGRESSION_SUPPORT_FLOOR_PARAMETER_AWARE"] = all(
    minimum_effective_samples("ts_regression", {"n_regressors": n}) == n + 1
    for n in (1, 2, 5, 10)
)

# --- Phase 13: legacy stateful seed fallback zero ----------------------------
fallback_prod = [
    c for c in _STATEFUL_CANONICALS
    if classify_canonical(c) in ("daily", "extended")
    and getattr(execution_contract(c), "legacy_seed_fallback", False)
]
gates["R30_PRODUCTION_LEGACY_STATEFUL_SEED_FALLBACK_ZERO"] = not fallback_prod

# --- misc structural ----------------------------------------------------------
# R34 P0-002：把 "Phase 14 fills per-canonical" 的硬编码 True 换成真检查——
# 每个 current production canonical 都必须在 reviewed migration manifest 里有
# review_id（版本绑定）。manifest 里 semantic_hash 为空的情况由 R34 P0-013
# 单独暴露，这里只做 review_id 的真实存在性检查。
from cleaned_operators.operator_surface import DAILY_CANONICALS, REVIEWED_MIGRATION_MANIFEST  # noqa: E402

# R34 P0-002：review 机制的语义范围是 daily 迁移表面（manifest 只登记 daily）。
# 逐 daily canonical 检查 review_id 真实存在；extended/research 走 semantic
# certification，不在此 gate 范围。
_reviewed = {
    c
    for c, meta in REVIEWED_MIGRATION_MANIFEST.items()
    if str(meta.get("review_id") or "").strip()
}
_daily_unreviewed = [c for c in sorted(DAILY_CANONICALS) if c not in _reviewed]
gates["R30_EVERY_CURRENT_CANONICAL_REVIEWED"] = not _daily_unreviewed

gates = {k: bool(v) for k, v in gates.items()}
blockers = [k for k, v in gates.items() if not v]
print("=== R30 HARD GATES ===")
for k in sorted(gates):
    print(f"  [{'OK' if gates[k] else 'FAIL'}] {k}")
print(f"\nR30_HARD_BLOCKERS_ZERO = {not blockers}")
json.dump(gates, open("docs/evidence/r30/R30_HARD_GATES.json", "w"), indent=2, sort_keys=True)
sys.exit(1 if blockers else 0)
