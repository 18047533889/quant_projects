# -*- coding: utf-8 -*-
"""R19 regression tests for scripts/audit_all_registered_operators.py.

Covers:
  R19-009  main()/release_blocking is the single gate — no second
           ``_HARD_INVARIANTS`` list.
  R19-010  duplicate pairs are tagged by reason and split into four disjoint
           buckets; only CONFIRMED_EXACT_DUPLICATES blocks release.
  R19-011  dead-param probe uses REAL declared choices, never an invented
           ``lo + 1``; a parameter with no alternative is not-probeable.
  R19-012  the MISSING sentinel is not treated as a declared default (None is);
           the sentinel object is never fed into a kernel.
  R19-013  dead-param coverage is reported at PARAMETER level and gated
           (probe_error_params == 0 and untested_params == 0).
  R19-014  behavior fingerprint canonicalizes NaN payload bits.
  R19-015  manifest gap is split into intrinsic + per-context invariants.

The audit's real ``load_all()`` is expensive and, at HEAD, blocked by an
unrelated concurrent-session registry issue; these tests therefore drive the
internal detectors with minimal catalog / operator fixtures and monkeypatched
``assign_mining_role`` / ``OperatorRegistry.get``.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import scripts.audit_all_registered_operators as mod


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _StubOp:
    """Minimal operator with the metadata surface the audit reads."""

    def __init__(self, *, panel_params=("x",), param_specs=None, param_types=None,
                 fn=None):
        self.metadata = types.SimpleNamespace(
            panel_params=tuple(panel_params),
            scalar_params=(),
            param_specs=param_specs or {},
            param_types=param_types or {},
            name="stub",
        )
        self._calculate_series = fn or (lambda *a, **k: None)
        self._fn = fn or (lambda *a, **k: None)

    def calculate(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _spec(default=..., *, choices=None, min=None, max=None, searchable=True,
          param_role=None):
    """A ParamSpec-shaped stand-in (the audit only reads these attributes)."""
    kwargs = dict(min=min, max=max, choices=choices,
                  searchable=searchable, param_role=param_role)
    if default is not ...:
        kwargs["default"] = default
    return types.SimpleNamespace(**kwargs)


def _basic_panel_op(fn):
    return _StubOp(panel_params=("x",), fn=fn)


# ---------------------------------------------------------------------------
# R19-009: release gate is the single release_blocking set
# ---------------------------------------------------------------------------


def test_no_second_hard_invariants_list():
    """R19-009: main() gates on release_blocking — there is no _HARD_INVARIANTS."""
    assert not hasattr(mod, "_HARD_INVARIANTS")
    assert "CONFIRMED_EXACT_DUPLICATES" in mod._RELEASE_BLOCKING_INVARIANTS
    assert "DEAD_SEARCHABLE_PARAMS" in mod._RELEASE_BLOCKING_INVARIANTS
    # the old merged duplicate invariant is gone from the gate
    assert "SEMANTIC_DUPLICATE_CANONICALS" not in mod._RELEASE_BLOCKING_INVARIANTS
    # R19-015: intrinsic gates, per-context does not
    assert "CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST" in mod._RELEASE_BLOCKING_INVARIANTS
    assert "CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST" not in mod._RELEASE_BLOCKING_INVARIANTS


def test_main_fails_on_nonempty_release_blocking(monkeypatch, tmp_path, capsys):
    """R19-009: strict mode fails when ANY release-blocking invariant is non-empty."""
    fake_result = {
        "release_blocking": ["UNCLASSIFIED_FACTOR_CANONICALS"],
        "invariants": {"UNCLASSIFIED_FACTOR_CANONICALS": ["foo"], "RELATED_FAMILY_PAIRS": ["a <-> b"]},
        "detector_coverage": {"DEAD_SEARCHABLE_PARAMS": {}},
        "counts": {},
    }
    monkeypatch.setattr(mod, "audit_all_registered_operators", lambda: fake_result)
    monkeypatch.setattr(mod, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(tmp_path / "r.md"),
                                      "--json", str(tmp_path / "r.json")])
    assert mod.main() == 1
    assert "STRICT FAIL" in capsys.readouterr().out


def test_main_non_blocking_invariant_does_not_fail(monkeypatch, tmp_path):
    """R19-009/010: advisory duplicate buckets must NOT gate a release."""
    fake_result = {
        "release_blocking": [],
        "invariants": {
            "SEMANTIC_DUPLICATE_CANDIDATES": ["a <-> b (candidate)"],
            "RELATED_FAMILY_PAIRS": ["c <-> d (name_family)"],
        },
        "detector_coverage": {"DEAD_SEARCHABLE_PARAMS": {
            "audit_errors": [], "probe_error_params": 0, "untested_params": 0,
        }},
        "counts": {},
    }
    monkeypatch.setattr(mod, "audit_all_registered_operators", lambda: fake_result)
    monkeypatch.setattr(mod, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(tmp_path / "r.md"),
                                      "--json", str(tmp_path / "r.json")])
    assert mod.main() == 0


# ---------------------------------------------------------------------------
# R19-010: duplicate classification into four buckets
# ---------------------------------------------------------------------------


def test_duplicate_pairs_bucket_by_reason():
    pairs = [
        {"left": "a", "right": "b", "reason": "alias"},
        {"left": "c", "right": "d", "reason": "name_family"},
        {"left": "e", "right": "f", "reason": "monotonic"},
        {"left": "g", "right": "h", "reason": "candidate"},
    ]
    buckets = mod._bucket_duplicate_pairs(pairs)
    assert buckets["CONFIRMED_EXACT_DUPLICATES"] == ["a <-> b (alias)"]
    assert buckets["RELATED_FAMILY_PAIRS"] == ["c <-> d (name_family)"]
    assert buckets["MONOTONIC_EQUIVALENT_PAIRS"] == ["e <-> f (monotonic)"]
    assert buckets["SEMANTIC_DUPLICATE_CANDIDATES"] == ["g <-> h (candidate)"]
    # all four buckets exist and are disjoint
    assert set(buckets) == set(mod._DUPLICATE_REASON_BUCKET.values())
    assert sum(len(v) for v in buckets.values()) == 4


def test_duplicate_detection_tags_reasons_and_buckets():
    catalog = {
        "rank": {"compatibility_only": False, "aliases": []},
        "ts_rank": {"compatibility_only": False, "aliases": []},
        "exp": {"compatibility_only": False, "aliases": []},
        "log": {"compatibility_only": False, "aliases": []},
        "sigmoid": {"compatibility_only": False, "aliases": []},
        "ts_log_return": {"compatibility_only": False, "aliases": []},
    }
    mod._ALIAS_LOOKUP = {"ts_rank": "rank"}
    candidates = mod.detect_semantic_duplicate_candidates(catalog)
    reasons = {(d["left"], d["right"]): d["reason"] for d in candidates}
    # a direct alias edge is a PROVEN exact duplicate
    assert reasons.get(("rank", "ts_rank")) == "alias"
    # two strict-monotonic re-encodings of the same family -> mining dedup
    assert reasons.get(("exp", "log")) == "monotonic"
    assert reasons.get(("exp", "sigmoid")) == "monotonic"
    # transform base + composite longer name -> advisory candidate, NOT monotonic
    assert reasons.get(("log", "ts_log_return")) == "candidate"

    buckets = mod._bucket_duplicate_pairs(candidates)
    confirmed = "".join(buckets["CONFIRMED_EXACT_DUPLICATES"])
    assert "rank <-> ts_rank" in confirmed
    assert len(buckets["MONOTONIC_EQUIVALENT_PAIRS"]) == 3  # exp/log, exp/sigmoid, log/sigmoid
    assert len(buckets["SEMANTIC_DUPLICATE_CANDIDATES"]) == 1
    assert buckets["RELATED_FAMILY_PAIRS"] == []


def test_duplicate_reason_bucket_mapping_is_total():
    for reason in ("alias", "name_family", "monotonic", "candidate"):
        assert reason in mod._DUPLICATE_REASON_BUCKET


# ---------------------------------------------------------------------------
# R19-011: dead-param probe uses REAL choices
# ---------------------------------------------------------------------------


def test_param_alternatives_uses_real_choices():
    spec = _spec(default=2, choices=(1, 2, 3, 4))
    alts, reason = mod._param_alternatives(spec, 2)
    assert reason is None
    assert alts == [1, 3, 4]  # every real choice distinct from the base

    spec_single = _spec(default=1, choices=(1,))
    alts2, reason2 = mod._param_alternatives(spec_single, 1)
    assert alts2 == []
    assert reason2 is not None  # not dead, not an error — just not probeable

    spec_no_choices = _spec(default=3, min=1, max=10)
    alts3, reason3 = mod._param_alternatives(spec_no_choices, 3)
    assert reason3 is None and alts3 and alts3[0] != 3


def test_run_op_hash_override_passes_real_choice():
    captured = {}

    def fn(x, method=1):
        captured["method"] = method
        return x

    op = _basic_panel_op(fn)
    op.metadata.param_specs = {"method": _spec(default=1, choices=(1, 2, 3))}
    op.metadata.param_types = {"method": int}
    panels = mod._synthetic_panels(op)
    fp, err = mod._run_op_hash(op, panels, ["method"], override={"method": 3})
    assert err is None
    assert captured["method"] == 3  # a real choice, never an invented lo+1


# ---------------------------------------------------------------------------
# R19-012: MISSING is not a declared default
# ---------------------------------------------------------------------------


def test_base_param_value_never_uses_missing_sentinel_as_default(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mod, "_MISSING_SENTINEL", sentinel)
    op = _basic_panel_op(lambda x, **k: x)
    op.metadata.param_types = {"w": int, "p": object}
    missing_spec = _spec(default=sentinel)  # spec.default IS the MISSING sentinel
    assert mod._base_param_value(op, "w", missing_spec) == 10  # type fallback
    # None is a legitimate DECLARED default and must be used as-is
    none_spec = _spec(default=None)
    assert mod._base_param_value(op, "p", none_spec) is None


def test_run_op_hash_never_feeds_missing_into_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mod, "_MISSING_SENTINEL", sentinel)
    captured = {}

    def fn(x, w=1):
        captured["w"] = w
        return x

    op = _basic_panel_op(fn)
    op.metadata.param_specs = {"w": _spec(default=sentinel)}
    op.metadata.param_types = {"w": int}
    panels = mod._synthetic_panels(op)
    fp, err = mod._run_op_hash(op, panels, ["w"])
    assert err is None
    assert captured["w"] == 10
    assert captured["w"] is not sentinel


def test_missing_sentinel_is_imported_when_available():
    # In a healthy environment the module resolves the REAL sentinel; when the
    # shared import chain is broken the defensive fallback is None.  Either way
    # the module must expose the attribute and never crash at import.
    assert hasattr(mod, "_MISSING_SENTINEL")


# ---------------------------------------------------------------------------
# R19-013: parameter-level dead-param coverage
# ---------------------------------------------------------------------------


def test_dead_param_parameter_level_coverage(monkeypatch):
    from cleaned_operators.registry import OperatorRegistry

    calls = []

    def fn(x, alive=1.0, dead=3, choice=1, locked=9):
        calls.append({"alive": alive, "dead": dead, "choice": choice, "locked": locked})
        if choice == 3:
            return x + alive + 100.0
        return x + alive

    op = _StubOp(
        panel_params=("x",),
        param_specs={
            "alive": _spec(default=1.0),
            "dead": _spec(default=3),
            "choice": _spec(default=1, choices=(1, 2, 3)),
            "locked": _spec(default=9, choices=(9,)),
        },
        param_types={"alive": float, "dead": int, "choice": int, "locked": int},
        fn=fn,
    )
    monkeypatch.setattr(mod, "assign_mining_role",
                        lambda canon, entry: mod.MiningRole.ALPHA)
    monkeypatch.setattr(OperatorRegistry, "get",
                        lambda cls, name, backend="pandas_numpy": op)

    catalog = {"foo": {"production_certified": True}}
    detail = mod.detect_dead_searchable_params_detail(catalog)

    assert detail["total_operators"] == 1
    assert detail["audited_operators"] == 1
    assert detail["total_searchable_params"] == 4
    assert detail["successfully_probed_params"] == 3  # alive, dead, choice
    assert detail["dead_params"] == 1                  # dead (injectivity)
    assert detail["probe_error_params"] == 0
    assert detail["untested_params"] == 1              # locked: single choice
    assert [d["param"] for d in detail["dead"]] == ["dead"]
    assert [n["param"] for n in detail["not_probeable"]] == ["locked"]
    assert detail["not_probeable"][0]["reason"] == "no choice distinct from base"

    # R19-011: the choice probe must have exercised the REAL choices [2, 3]
    choice_values = [c["choice"] for c in calls if c["choice"] != 1]
    assert 2 in choice_values and 3 in choice_values
    assert not any(c["choice"] == 4 for c in calls)  # no invented lo+1


def test_dead_param_backward_compat_four_tuple(monkeypatch):
    from cleaned_operators.registry import OperatorRegistry

    op = _basic_panel_op(lambda x, **k: x)
    monkeypatch.setattr(mod, "assign_mining_role",
                        lambda canon, entry: mod.MiningRole.ALPHA)
    monkeypatch.setattr(OperatorRegistry, "get",
                        lambda cls, name, backend="pandas_numpy": op)
    dead, audited, total, errors = mod.detect_dead_searchable_params(
        {"foo": {"production_certified": True}}
    )
    assert isinstance(dead, list) and isinstance(errors, list)
    assert isinstance(total, int) and isinstance(audited, int)


def test_main_fails_on_untested_params(monkeypatch, tmp_path, capsys):
    """R19-013: release requires untested_params == 0."""
    fake_result = {
        "release_blocking": [],
        "invariants": {},
        "detector_coverage": {"DEAD_SEARCHABLE_PARAMS": {
            "audit_errors": [],
            "probe_error_params": 0,
            "untested_params": 2,
            "not_probeable": [{"canonical": "a", "param": "locked", "reason": "no choice distinct from base"}],
        }},
        "counts": {},
    }
    monkeypatch.setattr(mod, "audit_all_registered_operators", lambda: fake_result)
    monkeypatch.setattr(mod, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(tmp_path / "r.md"),
                                      "--json", str(tmp_path / "r.json")])
    assert mod.main() == 1
    assert "untested" in capsys.readouterr().out


def test_main_fails_on_probe_error_params(monkeypatch, tmp_path, capsys):
    """R19-013: release requires probe_error_params == 0."""
    fake_result = {
        "release_blocking": [],
        "invariants": {},
        "detector_coverage": {"DEAD_SEARCHABLE_PARAMS": {
            "audit_errors": [{"canonical": "a", "param": "w", "error": "boom"}],
            "probe_error_params": 1,
            "untested_params": 0,
        }},
        "counts": {},
    }
    monkeypatch.setattr(mod, "audit_all_registered_operators", lambda: fake_result)
    monkeypatch.setattr(mod, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(tmp_path / "r.md"),
                                      "--json", str(tmp_path / "r.json")])
    assert mod.main() == 1


# ---------------------------------------------------------------------------
# Report writer handles the new invariant shapes
# ---------------------------------------------------------------------------


def test_write_report_handles_dict_invariants(tmp_path):
    """R19-015: the per-context manifest invariant is a dict; write_report must
    not crash and must render context keys."""
    result = {
        "counts": {"registered public total": 0, "static certified usable total": 0,
                   "mining usable total": 0, "internal total": 0, "source transform total": 0,
                   "deleted total": 0, "unclassified total": 0, "unresolved role total": 0,
                   "unused total": 0},
        "buckets": {},
        "verdicts": [],
        "invariants": {
            "CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST": {
                "market=ashare,sources=none": ["cert_missing"],
            },
            "CONFIRMED_EXACT_DUPLICATES": ["rank <-> ts_rank (alias)"],
        },
        "detector_coverage": {
            "CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST": {"ran": True, "coverage": "x"},
            "SEMANTIC_DUPLICATE_DETECTOR": {"ran": True, "coverage": "y"},
            "DEAD_SEARCHABLE_PARAMS": {"ran": True, "coverage": "z"},
        },
    }
    path = tmp_path / "report.md"
    mod.write_report(result, path)
    text = path.read_text(encoding="utf-8")
    assert "market=ashare,sources=none" in text
    assert "rank <-> ts_rank" in text


# ---------------------------------------------------------------------------
# R19-014: behavior fingerprint canonicalizes NaN payloads
# ---------------------------------------------------------------------------


def test_behavior_fingerprint_stable_to_nan_payload():
    a = pd.DataFrame({"s0": [1.0, np.nan, 3.0]})
    b = pd.DataFrame({"s0": [1.0, -np.nan, 3.0]})
    # prove the two inputs differ at the bit level (signed vs unsigned NaN)
    assert np.asarray(a["s0"]).tobytes() != np.asarray(b["s0"]).tobytes()
    # ... but are semantically identical and must hash the same
    assert mod._behavior_fingerprint(a) == mod._behavior_fingerprint(b)


def test_behavior_fingerprint_still_distinguishes_nan_topology():
    a = pd.DataFrame({"s0": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"s0": [1.0, np.nan, 3.0]})
    assert mod._behavior_fingerprint(a) != mod._behavior_fingerprint(b)


def test_behavior_fingerprint_still_distinguishes_inf_masks():
    a = pd.DataFrame({"s0": [np.inf, 2.0, 3.0]})
    b = pd.DataFrame({"s0": [-np.inf, 2.0, 3.0]})
    assert mod._behavior_fingerprint(a) != mod._behavior_fingerprint(b)


# ---------------------------------------------------------------------------
# R19-015: manifest gap intrinsic / contextual split
# ---------------------------------------------------------------------------


class _FakeMining:
    """Fake ``mining.operator_catalog`` module for manifest-gap tests."""

    MiningRole = mod.MiningRole
    assign_mining_role = staticmethod(mod.assign_mining_role)
    assign_mining_role_ex = staticmethod(mod.assign_mining_role_ex)
    mining_eligible = staticmethod(mod.mining_eligible)
    _KNOWN_SOURCES = {"daily_bar", "fundamental_pit"}

    @staticmethod
    def market_support(canonical, entry=None):
        return ("ashare", "us")

    def __init__(self, present):
        self._present = set(present)
        self.calls = []

    def get_mining_operators(self, *, admission="eligible", available_sources=None,
                             market=None, **kwargs):
        self.calls.append({
            "admission": admission,
            "available_sources": available_sources,
            "market": market,
        })
        return [types.SimpleNamespace(canonical=c)
                for c in sorted(self._present)]


@pytest.fixture()
def fake_mining_module(monkeypatch):
    """Swap ``sys.modules['mining.operator_catalog']`` for the test duration."""
    fake = _FakeMining(present={"a", "b"})
    monkeypatch.setitem(sys.modules, "mining.operator_catalog", fake)
    yield fake


def _certified_catalog():
    return {
        "a": {"production_certified": True},
        "b": {"production_certified": True},
        "cert_missing": {"production_certified": True},
    }


def test_manifest_gap_intrinsic_uses_full_source_universe(fake_mining_module, monkeypatch):
    monkeypatch.setattr(mod, "assign_mining_role",
                        lambda canon, entry: mod.MiningRole.ALPHA)
    intrinsic = mod.detect_manifest_gap_intrinsic(_certified_catalog())
    assert intrinsic == ["cert_missing"]
    # intrinsic query passes the FULL known source universe, never fail-closed None
    assert fake_mining_module.calls[0]["available_sources"] == ("daily_bar", "fundamental_pit")
    assert fake_mining_module.calls[0]["market"] is None
    assert fake_mining_module.calls[0]["admission"] == "eligible"


def test_manifest_gap_contextual_is_per_context(fake_mining_module, monkeypatch):
    monkeypatch.setattr(mod, "assign_mining_role",
                        lambda canon, entry: mod.MiningRole.ALPHA)
    contextual = mod.detect_manifest_gap_contextual(_certified_catalog())
    # cert_missing is checked in each market it supports, with its own sources
    keys = set(contextual)
    assert "market=ashare,sources=none" in keys
    assert "market=us,sources=none" in keys
    for members in contextual.values():
        assert "cert_missing" in members
        assert "a" not in members and "b" not in members
    # every contextual query carries a market + an explicit source set
    for call in fake_mining_module.calls:
        assert call["market"] is not None
        assert call["available_sources"] is not None


def test_manifest_gap_shim_backward_compat(fake_mining_module, monkeypatch):
    """R19-015: the old context-free detect_manifest_gap stays as a shim."""
    monkeypatch.setattr(mod, "assign_mining_role",
                        lambda canon, entry: mod.MiningRole.ALPHA)
    # shim calls get_mining_operators(admission=...) with no source/market context
    result = mod.detect_manifest_gap(_certified_catalog())
    assert result == ["cert_missing"]
    assert fake_mining_module.calls[0]["admission"] == "eligible"
    assert fake_mining_module.calls[0]["available_sources"] is None
