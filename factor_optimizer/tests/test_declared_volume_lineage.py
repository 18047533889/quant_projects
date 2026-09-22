import pytest
from factor_optimizer.research_manifest import _declared_lineage
from factor_optimizer.research_baseline import compile_baseline
from factor_preprocess.contracts.treatment_lineage import build_signature_from_lineage

VOLUME = "fillna_const(divide(ts_sum(gt(col('Volume'), ts_ema(col('Volume'), 20)), 20), 20), 0.5)"


def test_real_volume_recipe_keeps_fill_lineage_and_enables_baseline():
    lineage = _declared_lineage(VOLUME)
    assert lineage is not None
    assert [s.name for s in lineage] == ["fillna_const"]
    assert dict(lineage.steps[0].parameters) == {"value": 0.5}
    assert not build_signature_from_lineage(lineage).is_unknown_or_incomplete
    assert compile_baseline(lineage, training_context_ref="TRAIN").operations == ("winsor", "cs_rank")


@pytest.mark.parametrize("expression", [
    "ts_ema(col('Volume'), 20)",
    "ts_ema(col('Volume'), span=20)",
])
def test_outer_ema_is_recorded_as_existing_smoothing(expression):
    lineage = _declared_lineage(expression)
    assert lineage is not None
    signature = build_signature_from_lineage(lineage)
    assert signature.temporal_smoothing
    assert signature.temporal_smoothing_params["span"] == 20
    assert not signature.is_unknown_or_incomplete


def test_rank_fill_smoothing_order_is_not_erased():
    lineage = _declared_lineage("rank(ts_ema(fillna_const(col('Volume'), 0.5), 20))")
    assert lineage is not None
    assert [s.name for s in lineage] == ["fillna_const", "ts_ema", "rank"]
    signature = build_signature_from_lineage(lineage)
    assert signature.cs_rank and signature.temporal_smoothing
    assert compile_baseline(lineage, training_context_ref="TRAIN").operations == ("winsor",)


@pytest.mark.parametrize("expression", [
    "fillna_const(custom(col('Volume')), 0.5)",
    "fillna_const(rank(col('Volume')), float('nan'))",
    "ts_sum(col('Volume'), True)",
    "ts_ema(col('Volume'), span=0)",
    "ts_ema(col('Volume'), span=20, adjust=True)",
    "ts_ema(col('Volume'), 20, span=30)",
    "fillna_const(col('Volume'), True)",
    "gt(rank(col('Volume')), col('close'))",
])
def test_unsupported_semantics_still_fail_closed(expression):
    assert _declared_lineage(expression) is None
