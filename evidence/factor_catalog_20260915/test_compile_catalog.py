from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("compile_catalog.py")
SPEC = importlib.util.spec_from_file_location("compile_catalog", SCRIPT)
assert SPEC and SPEC.loader
compile_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compile_catalog)


@pytest.fixture(scope="module")
def runtime():
    return compile_catalog.build_runtime()


def test_real_compile_succeeds_without_reading_and_is_only_preflight(runtime):
    from factor_engine.storage.sources.datasource import DataSource

    parser, engine = runtime
    assert isinstance(engine.data_source, DataSource)
    row = compile_catalog.compile_record(
        {"source_row": 1, "id": "catalog_1", "formula": "ts_mean(close, 5)"},
        parser=parser,
        engine=engine,
    )
    assert row["status"] == "COMPILED"
    assert "execut" not in row["status"].lower()
    assert row["current_formula"]


def test_compile_failure_preserves_exact_exception(runtime):
    parser, engine = runtime
    formula = "ts_mean(close, 0)"
    row = compile_catalog.compile_record(
        {"source_row": 2, "id": "catalog_2", "formula": formula},
        parser=parser,
        engine=engine,
    )
    assert row["status"] == "COMPILE_FAILED"
    assert row["error_type"]
    assert row["error"]
    assert not row["error"].endswith("...")


def test_drawdown_activity_contract_checks_level_input_not_activity(runtime):
    parser, engine = runtime
    valid = compile_catalog.compile_record(
        {
            "source_row": 5,
            "id": "catalog_5",
            "formula": "ts_max_drawdown_activity_cost(vwap, abs(ret), window=60)",
        },
        parser=parser,
        engine=engine,
    )
    invalid = compile_catalog.compile_record(
        {
            "source_row": 6,
            "id": "catalog_6",
            "formula": "ts_max_drawdown_activity_cost(ret, turnover_ratio, window=60)",
        },
        parser=parser,
        engine=engine,
    )
    assert valid["status"] == "COMPILED"
    assert invalid["status"] == "COMPILE_FAILED"
    assert "drawdown requires a level / wealth-index input" in invalid["error"]


@pytest.mark.parametrize(
    "formula",
    [
        "trade_when(gt(close, open), close, fallback=neg(close))",
        "ts_cross_quantilogram(target=ret, source=turnover_ratio, window=60)",
    ],
)
def test_declared_panel_or_mixed_keyword_expression_compiles(runtime, formula):
    parser, engine = runtime
    row = compile_catalog.compile_record(
        {"source_row": 7, "id": "catalog_7", "formula": formula},
        parser=parser,
        engine=engine,
    )
    assert row["status"] == "COMPILED", row


def test_panel_keyword_becomes_ordered_input_while_scalars_stay_attrs(runtime):
    parser, engine = runtime
    plan, _ = engine.compile(
        __import__("factor_engine.api.factor", fromlist=["Factor"]).Factor(
            name="keyword_binding",
            expr=parser.parse(
                "ts_cross_quantilogram(target=ret, source=turnover_ratio, window=60)"
            ),
        )
    )
    assert [child.op for child in plan.inputs] == ["column", "column"]
    assert plan.inputs[0].attrs["name"] == "ret"
    assert plan.inputs[1].attrs["name"] != "ret"
    assert plan.attrs["window"] == 60


def test_expression_in_declared_scalar_keyword_still_fails_closed(runtime):
    parser, engine = runtime
    row = compile_catalog.compile_record(
        {"source_row": 8, "id": "catalog_8", "formula": "ts_mean(close, window=open)"},
        parser=parser,
        engine=engine,
    )
    assert row["status"] == "COMPILE_FAILED"
    assert row["error"] == "cleaned op 'ts_mean' kwargs must be literals, got 'window'"


def test_analyzer_rejects_positional_plus_keyword_panel_without_parser(runtime):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    _, engine = runtime
    expr = CleanedCall(
        "ts_cross_quantilogram",
        (ColumnRef("ret"),),
        (("target", ColumnRef("turnover_ratio")), ("source", ColumnRef("volume"))),
    )
    with pytest.raises(TypeError, match="multiple values.*target"):
        engine.analyzer.lower(expr)


def test_analyzer_rejects_two_aliases_for_same_panel_slot(runtime, monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    _, engine = runtime
    metadata = OperatorRegistry.get("ts_cross_quantilogram").metadata
    aliases = metadata.param_aliases
    monkeypatch.setattr(metadata, "param_aliases", {**aliases, "lhs": "target"})
    expr = CleanedCall(
        "ts_cross_quantilogram",
        (),
        (
            ("target", ColumnRef("ret")),
            ("lhs", ColumnRef("close")),
            ("source", ColumnRef("turnover_ratio")),
        ),
    )
    with pytest.raises(TypeError, match="multiple keyword values.*target"):
        engine.analyzer.lower(expr)


def test_scalar_gap_does_not_change_panel_input_order(runtime, monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_engine.expr.literal import Literal

    _, engine = runtime
    metadata = OperatorRegistry.get("ts_cross_quantilogram").metadata
    monkeypatch.setattr(metadata, "param_names", ["target", "window", "source"])
    monkeypatch.setattr(metadata, "panel_params", ("target", "source"))
    monkeypatch.setattr(metadata, "scalar_params", ("window",))
    expr = CleanedCall(
        "ts_cross_quantilogram",
        (),
        (
            ("source", ColumnRef("turnover_ratio")),
            ("window", Literal(60)),
            ("target", ColumnRef("ret")),
        ),
    )
    analysis = engine.analyzer.lower(expr)
    assert len(analysis.ir.inputs) == 2
    assert analysis.ir.inputs[0].attrs["name"] == "ret"
    assert analysis.ir.inputs[1].attrs["name"] != "ret"
    assert analysis.ir.attrs["window"] == 60


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"source_row": 3, "id": "", "formula": "close"}, "NON_FACTOR_OR_MISSING_ID"),
        ({"source_row": 4, "id": "catalog_4", "formula": ""}, "MISSING_FORMULA"),
    ],
)
def test_every_input_row_has_compile_status_and_exact_error(runtime, record, message):
    parser, engine = runtime
    row = compile_catalog.compile_record(record, parser=parser, engine=engine)
    assert row["status"] == "COMPILE_FAILED"
    assert row["error"] == message


def test_stream_writes_one_row_at_a_time_and_bounded_summary(runtime):
    parser, engine = runtime
    records = [
        {"source_row": 1, "id": "a", "formula": "close"},
        {"source_row": 2, "id": "b", "formula": "not_a_real_operator(close)"},
    ]
    dst = io.StringIO()
    summary = compile_catalog.audit_stream(
        iter(records), dst, parser=parser, engine=engine, progress_every=0
    )
    rows = [json.loads(line) for line in dst.getvalue().splitlines()]
    assert [row["status"] for row in rows] == ["COMPILED", "COMPILE_FAILED"]
    assert summary["processed"] == 2
    assert summary["scope"] == "compile_preflight_only_not_execution_certification"
