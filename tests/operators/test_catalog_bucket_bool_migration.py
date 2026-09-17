import numpy as np
import pandas as pd
import pytest
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize("source, expected", [
    ("rank(cs_bucket(ret, 7, 1))", "rank(cs_bucket(ret, 7, True))"),
    ("cs_bucket(ret, buckets=7, ascending=0)", "cs_bucket(ret, buckets=7, ascending=False)"),
])
def test_bucket_bool_literal_migration(source, expected):
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == expected
    assert len(result.changes) == 1
    assert "cs_bucket.ascending" in result.changes[0]
    assert migrate_catalog_recipe_formula(expected).changes == ()


@pytest.mark.parametrize("source", [
    "cs_bucket(ret,7,2)", "cs_bucket(ret,7,flag)", "cs_bucket(ret,7,1,extra)",
    "cs_bucket(ret,7,1,ascending=False)", "cs_bucket(ret,ascending=1,**opts)",
    "ns.cs_bucket(ret,7,1)", "cs_bucket(ret,7,1.0)",
])
def test_ambiguous_bucket_calls_untouched(source):
    assert migrate_catalog_recipe_formula(source).formula == source


@pytest.mark.parametrize("flag", (0, 1))
def test_legacy_bucket_bool_values_identical(flag):
    from factor_engine.cleaned_operators.common.daily_panel import cs_bucket
    x = pd.DataFrame([[1., 2., 2., np.nan, -3., 4.]])
    import ast
    converted = migrate_catalog_recipe_formula(f"cs_bucket(ret,7,{flag})")
    literal = ast.literal_eval(ast.parse(converted.formula, mode="eval").body.args[2])
    assert type(literal) is bool
    pd.testing.assert_frame_equal(cs_bucket(x,7,literal), cs_bucket(x,7,bool(flag)))
    with pytest.raises(TypeError, match="bool"):
        cs_bucket(x,7,flag)
