"""Reject structurally impossible fits rather than silently returning all NaN."""
import numpy as np
import pandas as pd
import pytest

def test_one_lag_is_rejected_at_planning_and_runtime():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.cleaned_operators.sequence_complexity import TsAutocorrDecayHalfLife
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    source="ts_autocorr_decay_half_life(ret,60,1,False,20)"
    with pytest.raises(ValueError, match="max_lag"):
        engine.compile(Factor(name="invalid_fit",expr=parser.parse(source),source_expr=source,surface="compat_research"))
    with pytest.raises(ValueError, match="max_lag"):
        TsAutocorrDecayHalfLife()._calculate_series(pd.DataFrame({"A":np.arange(100.)}),60,1,False,20)

def test_corrected_recipe_has_finite_half_life_on_persistent_series():
    from factor_engine.cleaned_operators.sequence_complexity import TsAutocorrDecayHalfLife
    from factor_engine.tools.catalog_r19_parameter_recipes import migrate_formula
    formula,notes=migrate_formula("ts_autocorr_decay_half_life(ret, 60, 1, 20, 0.05)")
    assert formula=="ts_autocorr_decay_half_life(ret, 60, 10, False, 20)"
    assert "impossible one-lag" in notes[0]
    rng=np.random.default_rng(19)
    x=np.zeros(500)
    for i in range(1,len(x)):x[i]=0.9*x[i-1]+rng.normal()
    result=TsAutocorrDecayHalfLife()._calculate_series(pd.DataFrame({"A":x}),60,10,False,20)
    finite=result.to_numpy()[np.isfinite(result.to_numpy())]
    assert len(finite)>100 and (finite>0).all()
