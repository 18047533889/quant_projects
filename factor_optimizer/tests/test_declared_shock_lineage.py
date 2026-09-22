import pytest
from factor_optimizer.research_manifest import _declared_lineage
from factor_optimizer.research_baseline import compile_baseline

SHOCK = "multiply(ts_sum(ts_delay(where(lt(ts_pct(col('AdjClose'), 1), -0.02), 1, 0), 1), 60), flex_max(subtract(ts_std(maximum(neg(ts_pct(col('AdjClose'), 1)), 0), 20), ts_std(maximum(ts_pct(col('AdjClose'), 1), 0), 20)), 0))"


@pytest.mark.parametrize('ranked', [False, True])
def test_shock_features_enable_baseline_without_duplicate_outer_rank(ranked):
    lineage = _declared_lineage('rank(' + SHOCK + ')' if ranked else SHOCK)
    assert lineage is not None
    assert [s.name for s in lineage] == (['rank'] if ranked else [])
    plan = compile_baseline(lineage, training_context_ref='TRAIN')
    assert plan.operations == (('winsor',) if ranked else ('winsor', 'cs_rank'))
    assert 'lineage_unknown' not in plan.omissions


@pytest.mark.parametrize('expression', [
    "ts_delay(col('x'), -1)", "ts_delay(col('x'), True)",
    "ts_pct(col('x'), 0)", "ts_std(col('x'), 1)",
    "ts_std(col('x'), 20, 0)", "ts_std(col('x'), 20, ddof=0)",
    "flex_max(col('x'), 20)", "flex_max(col('x'), True)",
    "where(lt(col('x'), 0), rank(col('x')), 0)",
    "where(lt(col('x'), 0), col('x'), custom(col('x')))",
    "where(lt(col('x'), 0), col('x'))", "maximum(col('x'), 0, 1)",
])
def test_unrecognized_or_ambiguous_forms_remain_unresolved(expression):
    assert _declared_lineage(expression) is None
