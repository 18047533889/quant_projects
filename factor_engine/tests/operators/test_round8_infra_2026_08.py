# -*- coding: utf-8 -*-
"""AI audit round 8 — P10+P11 infrastructure acceptance tests (2026-08).

Covers the audit items assigned to this package:

* #386 weight normalisation contract (``positive_homogeneous_degree_0``);
* #387 mixed-type / non-finite sequence parameters are invalid (no partial drop);
* #389 constant-folding finite gate (1e308 * 1e308 must NOT fold into an inf
  literal);
* #392 ``looks_like_source_ref`` / ``decode_source_ref_strict`` helpers;
* #382/#383 certify-script JUnit per-case recording and skip != pass;
* #384 finite-coverage gate helper;
* #385 ``certified_parameter_domain`` helper;
* P11 ClockSemantics global metadata design.
"""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

FE_ROOT = Path(__file__).resolve().parents[2]


def _load_script(rel: str):
    path = FE_ROOT / rel
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(path.stem, path)
    )
    assert module is not None
    module.__loader__.exec_module(module)  # type: ignore[attr-defined]
    return module


# ---------------------------------------------------------------------------
# #386 / #387 — ParameterCanonicalizer
# ---------------------------------------------------------------------------


def test_phd0_group_normalization_contract():
    from parameter_canonicalizer import ParamNormalizer, ParameterCanonicalizer

    pc = ParameterCanonicalizer(
        "weighted_mean",
        [
            ParamNormalizer("w1", normalize_with=("w2",), positive_homogeneous_degree_0=True),
            ParamNormalizer("w2", normalize_with=("w1",), positive_homogeneous_degree_0=True),
        ],
    )
    # f(c*w) == f(w) for weighted mean -> (1, 2) and (2, 4) are the same factor.
    assert pc.canonical_key({"w1": 1.0, "w2": 2.0}) == pc.canonical_key({"w1": 2.0, "w2": 4.0})
    assert pc.canonical_key({"w1": 1.0, "w2": 2.0}) == pc.canonical_key({"w1": 3.0, "w2": 6.0})


def test_non_phd0_group_is_not_normalized():
    from parameter_canonicalizer import ParamNormalizer, ParameterCanonicalizer

    pc = ParameterCanonicalizer(
        "weighted_sum",
        [
            ParamNormalizer("w1", normalize_with=("w2",), positive_homogeneous_degree_0=False),
            ParamNormalizer("w2", normalize_with=("w1",), positive_homogeneous_degree_0=False),
        ],
    )
    # weighted sum is NOT scale-invariant: f(c*w) == c*f(w), so (1,2) != (2,4).
    assert pc.canonical_key({"w1": 1.0, "w2": 2.0}) != pc.canonical_key({"w1": 2.0, "w2": 4.0})
    # Unchanged values must enter the key verbatim.
    key = pc.canonical_key({"w1": 1.0, "w2": 2.0})
    assert ("w1", "1.0") in key and ("w2", "2.0") in key


def test_mixed_type_weight_sequence_raises():
    from parameter_canonicalizer import ParameterCanonicalizer

    pc = ParameterCanonicalizer("op", [])
    with pytest.raises(ValueError, match="non-numeric element"):
        pc.canonical_key({"weights": [1.0, 2.0, "x"]})
    with pytest.raises(ValueError, match="non-numeric element"):
        pc.canonical_key({"weights": [1.0, 2.0, object()]})


def test_non_finite_sequence_raises():
    from parameter_canonicalizer import ParameterCanonicalizer

    pc = ParameterCanonicalizer("op", [])
    with pytest.raises(ValueError, match="non-numeric element"):
        pc.canonical_key({"weights": [1.0, float("nan"), 3.0]})
    with pytest.raises(ValueError, match="non-numeric element"):
        pc.canonical_key({"weights": [1.0, float("inf"), 3.0]})


def test_valid_numeric_sequence_is_accepted():
    from parameter_canonicalizer import ParameterCanonicalizer

    pc = ParameterCanonicalizer("op", [])
    key = pc.canonical_key({"weights": [1.0, 2.0, 3.0], "window": 5})
    assert ("weights", "[1.0, 2.0, 3.0]") in key
    # Mixed int/float/bool is numeric per the #387 contract.
    pc.canonical_key({"weights": [1, 2.0, True]})


# ---------------------------------------------------------------------------
# #389 — constant-folding finite gate
# ---------------------------------------------------------------------------


def test_fold_literals_multiply_inf_not_folded():
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        "multiply",
        [
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
        ],
        attrs={},
    )
    out = Optimizer()._fold_literals(plan)
    # 1e308 * 1e308 == inf -> must NOT fold into an inf literal.
    assert out.op == "multiply"
    assert out.inputs[0].op == "literal" and out.inputs[1].op == "literal"


def test_fold_literals_add_inf_not_folded():
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        "add",
        [
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
        ],
        attrs={},
    )
    out = Optimizer()._fold_literals(plan)
    assert out.op == "add"


def test_fold_nary_sum_inf_not_folded():
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        "nary_add",
        [
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
        ],
        attrs={},
    )
    out = Optimizer()._fold_literals(plan)
    assert out.op == "nary_add"


def test_fold_nary_mul_inf_not_folded():
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        "nary_mul",
        [
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
            PlanNode("literal", attrs={"value": 1e308}, inputs=[]),
        ],
        attrs={},
    )
    out = Optimizer()._fold_literals(plan)
    assert out.op == "nary_mul"


def test_fold_literals_finite_still_folds():
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    plan = PlanNode(
        "multiply",
        [
            PlanNode("literal", attrs={"value": 2.0}, inputs=[]),
            PlanNode("literal", attrs={"value": 3.0}, inputs=[]),
        ],
        attrs={},
    )
    out = Optimizer()._fold_literals(plan)
    assert out.op == "literal"
    assert out.attrs["value"] == 6.0


# ---------------------------------------------------------------------------
# #392 — SourceRef strict decode helpers
# ---------------------------------------------------------------------------


def test_source_ref_helpers_roundtrip():
    from api.source_ref import (
        decode_source_ref,
        decode_source_ref_strict,
        encode_source_ref,
        looks_like_source_ref,
        make_source_ref,
    )

    spec = make_source_ref("StockMinuteBar", "Close", params={"n": 5})
    name = encode_source_ref(spec)
    assert looks_like_source_ref(name)
    decoded = decode_source_ref_strict(name)
    assert decoded.table == "StockMinuteBar"
    assert decoded.field == "Close"
    assert decoded.params_dict() == {"n": 5}
    # legacy decoder keeps returning the spec for a well-formed ref.
    assert decode_source_ref(name) is not None


def test_looks_like_source_ref_rejects_non_string():
    from api.source_ref import looks_like_source_ref

    assert not looks_like_source_ref(None)
    assert not looks_like_source_ref(123)
    assert not looks_like_source_ref("plain_column")


def test_decode_source_ref_strict_raises_on_bad_base64():
    from api.source_ref import decode_source_ref_strict, looks_like_source_ref

    bad = "__fe_source_ref_v1__not-base64!!!"
    assert looks_like_source_ref(bad)
    with pytest.raises(ValueError):
        decode_source_ref_strict(bad)


# ---------------------------------------------------------------------------
# Review-8 #468/#469/#470 — SourceRef scalar / integer / duplicate-param gates
# ---------------------------------------------------------------------------

def test_source_ref_scalar_rejects_nan_inf():
    """#468: NaN / Inf must never enter an encoded SourceRef identity."""
    from api.source_ref import make_source_ref

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            make_source_ref("DailyBar", "Close", params={"threshold": bad})
    with pytest.raises(ValueError, match="finite"):
        make_source_ref("DailyBar", "Close",
                        transform="financial_lag", transform_params={"quarters": float("nan")})


def test_source_ref_strict_int_rejects_ambiguous_literals():
    """#469: True / 1.9 / '2' / NaN / Inf must not silently coerce to 1/2."""
    from api.source_ref import _strict_int, intermediate_col

    for bad in (True, 1.9, "2", float("nan"), float("inf"), None):
        with pytest.raises(ValueError):
            _strict_int(bad, name="test")
    # integral floats are the same integer -> accepted
    assert _strict_int(2.0, name="test") == 2
    assert _strict_int(3, name="test") == 3

    with pytest.raises(ValueError):
        intermediate_col("alpha_x", True)
    with pytest.raises(ValueError):
        intermediate_col("alpha_x", 1.9)
    with pytest.raises(ValueError):
        intermediate_col("alpha_x", "2")


def test_source_col_rejects_duplicate_parameters():
    """#470: a parameter given both positionally and as a keyword must raise."""
    from api.source_ref import source_col

    with pytest.raises(ValueError, match="both positionally and as a keyword"):
        source_col("Intermediate", "value", "name", "alpha_x", name="alpha_y")
    with pytest.raises(ValueError, match="more than once"):
        source_col("Intermediate", "value", "name", "alpha_x", "name", "alpha_y")


def test_source_ref_transform_rejects_unconsumed_parameters():
    """#470: known transforms must reject parameters they never consume."""
    from api.columns import col
    from api.source_ref import (
        encode_source_ref,
        make_source_ref,
        transform_source_col,
    )

    with pytest.raises(ValueError, match="does not consume"):
        make_source_ref(
            "DailyBar", "Close",
            transform="financial_lag", transform_params={"quarters": 1, "bogus": 2},
        )
    # transform_source_col path goes through with_transform — same gate.
    base = make_source_ref("StockMinuteBar", "Close")
    ref = col(encode_source_ref(base))
    with pytest.raises(ValueError, match="does not consume"):
        transform_source_col(ref, "minute_bar", period=5, index=0, bogus=1)


def test_decode_source_ref_strict_raises_on_missing_field():
    from api.source_ref import decode_source_ref_strict

    raw = base64.urlsafe_b64encode(json.dumps({"table": "T"}).encode()).decode().rstrip("=")
    with pytest.raises(ValueError, match="missing required field"):
        decode_source_ref_strict("__fe_source_ref_v1__" + raw)


def test_decode_source_ref_strict_raises_on_non_object_payload():
    from api.source_ref import decode_source_ref_strict

    raw = base64.urlsafe_b64encode(json.dumps([1, 2, 3]).encode()).decode().rstrip("=")
    with pytest.raises(ValueError, match="JSON object"):
        decode_source_ref_strict("__fe_source_ref_v1__" + raw)


# ---------------------------------------------------------------------------
# P11 — ClockSemantics global metadata design
# ---------------------------------------------------------------------------


def test_clock_for_family_rules():
    from planner.clock_semantics import ClockSemantics, clock_for

    assert clock_for("ts_mean") == ClockSemantics.TRADING_BAR
    assert clock_for("ts_std") == ClockSemantics.TRADING_BAR
    assert clock_for("event_response_effective_events") == ClockSemantics.EVENT
    assert clock_for("fin_revision_count") == ClockSemantics.FISCAL_PERIOD
    assert clock_for("fiscal_yoy") == ClockSemantics.FISCAL_PERIOD
    assert clock_for("intraday_slot_*") == ClockSemantics.SESSION_SLOT
    assert clock_for("intraday_morning_return") == ClockSemantics.SESSION_SLOT
    assert clock_for("rank") == ClockSemantics.OBSERVATION


def test_clock_explicit_table_wins_over_prefix():
    from planner.clock_semantics import ClockSemantics, clock_for, declare_clock

    # declared override wins over the ts_ -> TRADING_BAR prefix rule.
    declare_clock("ts_my_event_op", ClockSemantics.EVENT)
    assert clock_for("ts_my_event_op") == ClockSemantics.EVENT


def test_clock_registry_view():
    from planner.clock_semantics import clock_for, clock_registry

    reg = clock_registry()
    assert reg["ts_mean"] == "trading_bar"
    assert reg["event_response_effective_events"] == "event"
    assert reg["fin_revision_count"] == "fiscal_period"
    assert reg["intraday_slot"] == "session_slot"
    # every registry entry resolves back to itself.
    for name, value in reg.items():
        assert clock_for(name).value == value


# ---------------------------------------------------------------------------
# #382/#383/#384/#385 — certify-script helpers (imported without running them)
# ---------------------------------------------------------------------------


def test_primitive_certifier_finite_coverage_gate(tmp_path):
    mod = _load_script("scripts/certify_primitive_evidence.py")
    # Non-materialised in-memory fixture -> vacuously OK.
    ok, ratio = mod._finite_coverage_ok(str(tmp_path / "missing.npy"))
    assert ok and ratio == 1.0
    # NaN-dominated fixture -> downgraded.
    np.save(tmp_path / "allnan.npy", np.full((4, 4), np.nan))
    ok, ratio = mod._finite_coverage_ok(str(tmp_path / "allnan.npy"))
    assert not ok and ratio == 0.0
    # A healthy fixture passes.
    np.save(tmp_path / "good.npy", np.arange(16.0).reshape(4, 4))
    ok, ratio = mod._finite_coverage_ok(str(tmp_path / "good.npy"))
    assert ok and ratio == 1.0


def test_primitive_certifier_parameter_domain_default():
    mod = _load_script("scripts/certify_primitive_evidence.py")
    assert mod._certified_parameter_domain("ts_mean") == {"bounds": ["default"]}


def test_primitive_certifier_junit_parse_skip_fails(tmp_path):
    mod = _load_script("scripts/certify_primitive_evidence.py")
    junit = tmp_path / "stage.xml"
    junit.write_text(
        "<testsuites><testsuite name='pytest'>"
        "<testcase classname='pkg.test' name='test_x[ts_mean-1]'/>"
        "<testcase classname='pkg.test' name='test_y'><skipped message='skip'/></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    ok, skipped, records = mod._parse_junit_records(
        junit, ["pkg/test.py"], frozenset({"ts_mean"}), "polars_reference_parity"
    )
    # skip != pass: any skip fails the file.
    assert not ok
    assert skipped == ["pkg.test"]
    # the executed+passed case is still recorded per-test-case.
    assert len(records) == 1
    assert records[0]["canonical"] == "ts_mean"
    assert records[0]["backend"] == "polars"
    assert records[0]["status"] == "passed"
    assert records[0]["case_id"] == "pkg.test::test_x[ts_mean-1]"


def test_primitive_certifier_junit_parse_all_pass_ok(tmp_path):
    mod = _load_script("scripts/certify_primitive_evidence.py")
    junit = tmp_path / "stage.xml"
    junit.write_text(
        "<testsuites><testsuite name='pytest'>"
        "<testcase classname='pkg.test' name='test_x[ts_mean-1]'/>"
        "<testcase classname='pkg.test' name='test_x[ts_std-5]'/>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    ok, skipped, records = mod._parse_junit_records(
        junit, ["pkg/test.py"], frozenset({"ts_mean", "ts_std"}), "duckdb_reference_parity"
    )
    assert ok
    assert skipped == []
    assert {r["canonical"] for r in records} == {"ts_mean", "ts_std"}
    assert all(r["backend"] == "duckdb" for r in records)
