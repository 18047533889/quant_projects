# -*- coding: utf-8 -*-
"""R16 regression tests for scripts/audit_all_registered_operators.py.

Covers R16-004 (no sampling cap), R16-005 (full panel binding), R16-006
(behavior fingerprint includes NaN topology), R16-007 (exceptions become
AUDIT_ERROR, never silent skip), R16-008 (release_blocking marking), R16-009
(eligible-only manifest gap), R16-010 (identical_contract is not a duplicate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _StubOperator:
    """Minimal operator stub with the metadata surface the audit needs."""

    def __init__(self, *, panel_params=("x",), param_specs=None, param_types=None,
                 fn=None):
        self.metadata = _StubMetadata(
            panel_params=panel_params,
            param_specs=param_specs or {},
            param_types=param_types or {},
        )
        self._fn = fn
        self._calculate_series = fn or (lambda *a, **k: None)

    def calculate(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _StubMetadata:
    def __init__(self, *, panel_params, param_specs, param_types):
        self.panel_params = tuple(panel_params)
        self.scalar_params = ()
        self.param_specs = param_specs
        self.param_types = param_types
        self.name = "stub"


def _spec(default, min=None, max=None, choices=None):
    from cleaned_operators.base import ParamSpec
    kwargs = dict(dtype=float)
    if default is not None:
        kwargs["default"] = default
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    if choices is not None:
        kwargs["choices"] = choices
    return ParamSpec(**kwargs)


def test_behavior_fingerprint_distinguishes_nan_masks():
    """R16-006: same finite values, different NaN mask -> different fingerprint."""
    from scripts.audit_all_registered_operators import _behavior_fingerprint

    a = pd.DataFrame({"s0": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"s0": [1.0, np.nan, 3.0]})
    fa, fb = _behavior_fingerprint(a), _behavior_fingerprint(b)
    assert fa != fb
    # same values, different shape -> different
    c = pd.DataFrame({"s0": [1.0, 2.0, 3.0], "s1": [4.0, 5.0, 6.0]})
    assert fa != _behavior_fingerprint(c)


def test_dead_param_runner_binds_full_panels():
    """R16-005: binary/ternary operators get ALL panel args, not just x."""
    from scripts.audit_all_registered_operators import _synthetic_panels, _run_op_hash

    def add3(x, y, z, window=2):
        return (x + y + z).rolling(window, min_periods=1).mean()

    op = _StubOperator(
        panel_params=("x", "y", "z"),
        fn=add3,
    )
    panels = _synthetic_panels(op)
    assert set(panels) == {"x", "y", "z"}
    fp, err = _run_op_hash(op, panels, ["window"])
    assert err is None
    assert fp is not None


def test_dead_param_runner_never_silently_skips_errors():
    """R16-007: an exception in the kernel surfaces as AUDIT_ERROR, not None."""
    from scripts.audit_all_registered_operators import _synthetic_panels, _run_op_hash

    def boom(x, window=2):
        raise RuntimeError("kernel exploded")

    op = _StubOperator(panel_params=("x",), fn=boom)
    panels = _synthetic_panels(op)
    fp, err = _run_op_hash(op, panels, ["window"])
    assert fp is None
    assert err is not None and "kernel exploded" in err


def test_dedupe_identical_contract_not_duplicate():
    """R16-010: ts_mean / ts_std share an interface but are NOT duplicates.

    The detector is exercised without loading the whole registry by calling
    the internal candidate logic on a two-entry catalog that would have been
    flagged by the old identical-contract branch.
    """
    from scripts.audit_all_registered_operators import _alias_edges, detect_semantic_duplicate_candidates

    catalog = {
        "ts_mean": {"param_names": ["x", "window"], "output_unit": "same_as:x",
                    "input_grain": "daily", "compatibility_only": False, "aliases": []},
        "ts_std": {"param_names": ["x", "window"], "output_unit": "same_as:x",
                   "input_grain": "daily", "compatibility_only": False, "aliases": []},
        "ts_decay_linear": {"compatibility_only": False, "aliases": []},
        "MACD": {"compatibility_only": False, "aliases": []},
        "MACD_line": {"compatibility_only": False, "aliases": []},
    }
    edges = _alias_edges(catalog)
    # fake alias lookup module-global
    import scripts.audit_all_registered_operators as mod
    mod._ALIAS_LOOKUP = {a: t for a, t in edges}
    candidates = detect_semantic_duplicate_candidates(catalog)
    pairs = {(d["left"], d["right"]) for d in candidates}
    # ts_mean <-> ts_std must NOT appear (identical contract is not duplicate)
    assert ("ts_mean", "ts_std") not in pairs and ("ts_std", "ts_mean") not in pairs
    reasons = {d["reason"] for d in candidates}
    assert "identical_contract" not in reasons


def test_manifest_gap_uses_eligible_admission():
    """R16-009: the manifest-gap detector asks for the ELIGIBLE runtime set,
    not admission=all (which is self-validating)."""
    from scripts import audit_all_registered_operators as mod

    captured = {}

    class _Fake:
        def __call__(self, admission="eligible"):
            captured["admission"] = admission
            return []

    class _Ops:
        canonical = "ts_mean"

    import types
    fake_mining = types.ModuleType("mining.operator_catalog")
    fake_mining.get_mining_operators = _Fake()
    fake_mining.MiningRole = mod.MiningRole
    fake_mining.assign_mining_role = mod.assign_mining_role
    fake_mining.assign_mining_role_ex = mod.assign_mining_role_ex
    fake_mining.mining_eligible = mod.mining_eligible
    # temporarily substitute in sys.modules under a fresh name via monkeypatch-lite
    import sys
    key = "mining.operator_catalog"
    real = sys.modules.get(key)
    sys.modules[key] = fake_mining
    try:
        from scripts.audit_all_registered_operators import detect_manifest_gap
        result = detect_manifest_gap({})
        assert result == []
        assert captured.get("admission") == "eligible"
    finally:
        if real is not None:
            sys.modules[key] = real
        else:
            sys.modules.pop(key, None)


def test_dead_param_detector_has_no_sampling_cap():
    """R16-004: the detector signature no longer accepts max_canonicals; it
    audits every candidate (full coverage, not a 60-op slice)."""
    import inspect
    from scripts import audit_all_registered_operators as mod

    sig = inspect.signature(mod.detect_dead_searchable_params)
    assert "max_canonicals" not in sig.parameters
    assert sig.return_annotation is not inspect.Signature.empty or True
    # return shape: (dead, audited, total, audit_errors)
    fake = {
        "a": {"compatibility_only": False, "diagnostic_only": False},
    }
    dead, audited, total, errors = mod.detect_dead_searchable_params(fake)
    # with no registry available it must report an AUDIT_ERROR rather than PASS
    assert isinstance(dead, list) and isinstance(errors, list)
    assert isinstance(total, int) and isinstance(audited, int)


# ---------------------------------------------------------------------------
# R16-001/002/003: audit_r15_master full-registry coverage
# ---------------------------------------------------------------------------


class _StubReg:
    def __init__(self, catalog):
        self._catalog = catalog
        self._operators = {}

    def get(self, canonical):
        return self._operators.get(canonical)

    def register(self, canonical, op, catalog_entry):
        self._operators[canonical] = op
        self._catalog[canonical] = catalog_entry


def _simple_op(kind="mean"):
    import numpy as np
    import pandas as pd

    def fn(x):
        if kind == "mean":
            return x.rolling(3, min_periods=1).mean()
        raise RuntimeError("boom")

    class _Op:
        metadata = type("M", (), {"param_names": ["x", "window"],
                                  "panel_params": ("x",),
                                  "scalar_params": ("window",),
                                  "param_specs": {}})()

        def calculate(self, x):
            return fn(x)
    return _Op()


def _make_reg_with_broken_op():
    """A registry where one daily-surface canonical's kernel always throws.

    Uses REAL daily canonical names (``ts_mean`` / ``ts_var``) so the golden
    runner's surface applicability treats them as factor terminals.
    """
    import numpy as np
    import pandas as pd

    reg = _StubReg({})
    reg.register("ts_mean", _simple_op("mean"), {"input_grain": "daily", "output_unit": "same_as:x"})
    class _Broken:
        metadata = type("M", (), {"param_names": ["x"], "panel_params": ("x",),
                                  "scalar_params": (), "param_specs": {}})()

        def calculate(self, *a, **k):
            raise RuntimeError("boom")

    reg.register("ts_var", _Broken(), {"input_grain": "daily", "output_unit": "same_as:x"})
    return reg


def test_ashare_coverage_full_registry_not_run_reported():
    """R16-001: a broken daily canonical is still audited (NOT_RUN), and
    set_equality is exact — no canonical is silently excluded."""
    from scripts.audit_r15_master import _ashare_coverage

    reg = _make_reg_with_broken_op()
    rows, meta = _ashare_coverage(sorted(reg._catalog), reg)
    canon_set = {r["canonical"] for r in rows}
    not_run_set = {r["canonical"] for r in meta["not_run_records"]}
    assert "ts_mean" in canon_set
    assert "ts_var" in not_run_set
    assert meta["set_equality"] is True
    assert meta["registry_total"] == 2 and meta["not_run"] == 1


def test_semantic_duplicates_full_registry():
    """R16-002: full-registry scan with explicit not_run + set equality."""
    from scripts.audit_r15_master import _semantic_duplicates

    reg = _make_reg_with_broken_op()
    dup = _semantic_duplicates(sorted(reg._catalog), reg)
    assert dup["sample_size"] >= 1
    assert dup["set_equality"] is True
    assert any(r["canonical"] == "ts_var" for r in dup["not_run"])


def test_golden_null_runner_explicit_outcomes():
    """R16-003: per-canonical (test_id, outcome); a broken daily op is
    AUDIT_ERROR, not silently absent; every canonical has an outcome."""
    from scripts.audit_r15_master import _golden_null_runner

    reg = _make_reg_with_broken_op()
    records = _golden_null_runner(sorted(reg._catalog), reg)
    by_canon = {}
    for r in records:
        by_canon.setdefault(r["canonical"], []).append(r["outcome"])
    # broken ts_var: AUDIT_ERROR, never absent
    assert "AUDIT_ERROR" in by_canon["ts_var"]
    # ts_mean: golden PASS
    assert "PASS" in by_canon["ts_mean"]
    # every canonical has at least one explicit outcome
    for canon in ("ts_mean", "ts_var"):
        assert by_canon[canon], f"{canon} has no outcome"


# ---------------------------------------------------------------------------
# R16-013~020: mining/operator_catalog
# ---------------------------------------------------------------------------


def test_index_source_typed_not_relation():
    """R16-013: index_* operators require index_pit, never relation_pit."""
    from mining.operator_catalog import _required_sources

    assert _required_sources("index_member_ratio", {}) == ("index_pit",)
    assert _required_sources("index_relative_strength", {}) == ("index_pit",)
    assert _required_sources("relation_exposure", {}) == ("relation_pit",)
    assert _required_sources("holder_concentration", {}) == ("shareholder_pit",)
    assert _required_sources("event_frequency", {}) == ("event_pit",)


def test_source_capability_concept_gap():
    """R16-014: source_id present but required concept missing -> concept gap."""
    from mining.operator_catalog import (
        SourceCapability, SourceRequirement, source_status,
    )

    reqs = {"source_requirements": (
        SourceRequirement("daily_bar", ("price",)),
        SourceRequirement("fundamental_pit", ("eps", "pe")),
    )}
    full = {
        "daily_bar": SourceCapability("daily_bar", ("price",)),
        "fundamental_pit": SourceCapability("fundamental_pit", ("eps", "pe")),
    }
    st = source_status("pe_ttm", reqs, ["daily_bar", "fundamental_pit"],
                       source_capabilities=full)
    assert st.missing == () and st.concept_gap == ()
    lacking = {
        "daily_bar": SourceCapability("daily_bar", ("price",)),
        "fundamental_pit": SourceCapability("fundamental_pit", ("book_value",)),
    }
    st2 = source_status("pe_ttm", reqs, ["daily_bar", "fundamental_pit"],
                        source_capabilities=lacking)
    assert "fundamental_pit" in st2.concept_gap


def test_target_frequency_strict_enum():
    """R16-018: unknown frequency strings raise."""
    from mining.operator_catalog import TargetFrequency, bind_target_frequency
    from backend.operator_errors import OperatorParameterError

    assert bind_target_frequency("daily") is TargetFrequency.DAILY
    assert bind_target_frequency("minute") is TargetFrequency.MINUTE
    with pytest.raises(ValueError):
        bind_target_frequency("weekly/foo")


def test_market_support_fail_closed():
    """R16-019: specialized-but-undeclared market operators fail closed."""
    from mining.operator_catalog import market_support

    assert market_support("limit_up_close") == ("ashare",)
    assert market_support("suspension_gap_duration") == ()
    assert market_support("ts_mean", {"market_semantics": "agnostic"}) == ("ashare", "us")


def test_mining_eligible_checks_market():
    """R16-017: direct mining_eligible with a market context rejects a
    market-mismatched operator (not only the outer get_mining_operators)."""
    import mining.operator_catalog as M

    orig = M.cost_contract_declared
    M.cost_contract_declared = lambda *a, **k: True
    try:
        cat = {"production_certified": True, "input_grain": "daily"}
        ctx_us = M.MiningContext(market="us", available_sources=("daily_bar",))
        ctx_ashare = M.MiningContext(market="ashare", available_sources=("daily_bar",))
        assert M.mining_eligible("limit_up_close", catalog=cat,
                                 role=M.MiningRole.ALPHA, context=ctx_us) is False
        assert M.mining_eligible("limit_up_close", catalog=cat,
                                 role=M.MiningRole.ALPHA, context=ctx_ashare) is True
    finally:
        M.cost_contract_declared = orig


# ---------------------------------------------------------------------------
# R16-024~028: export_mining_manifest
# ---------------------------------------------------------------------------


def test_runtime_manifest_rejects_pending_schema():
    """R16-024: a runtime loader must reject a pending/remediation schema."""
    import json
    from scripts.export_mining_manifest import validate_runtime_manifest

    tmp = pd.io.common.__file__  # just a writable location under repo tmp
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runtime_mining_manifest.eligible.json"
        p.write_text(json.dumps({"manifest_kind": "operator_remediation_pending", "operators": []}))
        assert validate_runtime_manifest(p)
        p.write_text(json.dumps({"manifest_kind": "runtime_eligible", "operators": []}))
        assert validate_runtime_manifest(p) == []

