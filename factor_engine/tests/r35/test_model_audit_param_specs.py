# -*- coding: utf-8 -*-
"""R35 Phase 4 (search-space hygiene): production-lane model-like canonicals
must declare explicit ``ParamSpec`` dicts.

The model-operators audit (M-115 / M-162 / M-170 / M-240) found that without
declared specs every scalar param defaults to FULL search — window/order/
min_periods get freely searched and estimator-resolution knobs pollute the
default AlphaProbe/GP grammar.  This gate asserts:

(a) every production-lane model-like canonical declares a non-empty
    ``param_specs`` dict;
(b) every ``param_specs`` key is a member of the operator's declared
    ``param_names`` (the central ``keys(param_specs) ⊆ param_names`` invariant);
(c) window / horizon params are declared ``HORIZON`` + searchable (they ARE the
    alpha mechanism), while estimator-resolution params (``order`` / ``bins`` /
    ``lag`` / ``m`` / ``subsequence_length`` / ``coefficient_index`` /
    ``n_components`` / ``rank`` / ``dim`` / ``delay`` / ``tau`` / …) are never
    full-resolution searches.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.base import ParamRole, param_search_grade  # noqa: E402
from cleaned_operators.model_lane import assign_model_lane  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402

#: The four production lanes (DIAGNOSTIC_RESEARCH / DELETE_TOMBSTONE are not
#: default-search surfaces).
PRODUCTION_LANES = {
    "FAST_NATIVE_ALPHA",
    "EXPENSIVE_CERTIFIED_ALPHA",
    "STATE_CONDITION_EVENT",
    "MODEL_FEATURE_SCORE",
}

#: window / horizon params that are the alpha mechanism (must be HORIZON +
#: searchable + full-resolution).
WINDOW_HORIZON_NAMES = {
    "window", "history_window", "horizon", "periods", "history",
    "scale_window", "path_window", "max_lookback", "history_days",
}


def _load() -> None:
    load_all()


def _production_model_like_canonicals() -> list[str]:
    return [
        c for c in sorted(OperatorRegistry.list_canonical())
        if assign_model_lane(c) in PRODUCTION_LANES
    ]


def _canonical_metadata(name: str):
    """Prefer the pandas_numpy reference backend (the canonical contract),
    mirroring ``param_role_contract.backfill_scalar_roles``."""
    backends = OperatorRegistry._operators.get(name, {})
    op = (
        backends.get("pandas_numpy")
        or backends.get("polars")
        or next(iter(backends.values()))
    )
    return getattr(op, "metadata", None)


def test_every_production_model_like_canonical_declares_param_specs():
    _load()
    missing = []
    for c in _production_model_like_canonicals():
        md = _canonical_metadata(c)
        specs = getattr(md, "param_specs", None) if md else None
        if not specs:
            missing.append(c)
    assert missing == [], (
        f"production-lane model-like canonicals with NO param_specs: {missing}"
    )


def test_param_spec_keys_are_subset_of_param_names():
    _load()
    bad = []
    for c in _production_model_like_canonicals():
        md = _canonical_metadata(c)
        specs = getattr(md, "param_specs", None) or {}
        params = set(getattr(md, "param_names", []) or [])
        for k in specs:
            if k not in params:
                bad.append((c, k))
    assert bad == [], f"param_specs keys not declared in param_names: {bad}"


def test_window_horizon_params_are_horizon_searchable():
    _load()
    bad = []
    for c in _production_model_like_canonicals():
        md = _canonical_metadata(c)
        specs = getattr(md, "param_specs", None) or {}
        for name, spec in specs.items():
            if name not in WINDOW_HORIZON_NAMES:
                continue
            if spec.param_role != ParamRole.HORIZON:
                bad.append((c, name, "role", spec.param_role))
            if not spec.searchable:
                bad.append((c, name, "searchable", spec.searchable))
            if param_search_grade(spec) != "full":
                bad.append((c, name, "grade", param_search_grade(spec)))
    assert bad == [], f"window/horizon params not HORIZON+searchable: {bad}"


def test_estimator_resolution_params_never_full_search():
    _load()
    bad = []
    for c in _production_model_like_canonicals():
        md = _canonical_metadata(c)
        specs = getattr(md, "param_specs", None) or {}
        for name, spec in specs.items():
            if spec.param_role in (ParamRole.ESTIMATOR_RESOLUTION, ParamRole.MODEL_ORDER):
                if param_search_grade(spec) == "full":
                    bad.append((c, name, "grade", param_search_grade(spec)))
    assert bad == [], f"estimator-resolution params pollute full search: {bad}"


def test_min_periods_and_support_policy_are_non_searchable():
    """Statistical-support floors and boolean governance must never be full
    search dimensions (M-115/M-162)."""
    _load()
    bad = []
    for c in _production_model_like_canonicals():
        md = _canonical_metadata(c)
        specs = getattr(md, "param_specs", None) or {}
        for name, spec in specs.items():
            if name in ("min_periods", "min_count", "min_anchors", "add_intercept", "retval"):
                if spec.searchable:
                    bad.append((c, name, "searchable", spec.searchable))
                if param_search_grade(spec) == "full":
                    bad.append((c, name, "grade", param_search_grade(spec)))
    assert bad == [], f"support-policy/governance params leaked into full search: {bad}"
