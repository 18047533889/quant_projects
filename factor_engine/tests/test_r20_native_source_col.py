import pytest


def _parser():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all
    load_all()
    return DSLParser(surface="compat_research", dialect="native")


def _only_ref(expr):
    from factor_engine.api.source_ref import decode_source_ref
    from factor_engine.expr.column import ColumnRef
    refs = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ColumnRef):
            refs.append(node)
        stack.extend(node.children())
        for _, value in getattr(node, "kwargs", ()):
            if hasattr(value, "children"):
                stack.append(value)
    assert len(refs) == 1
    return decode_source_ref(refs[0].name)


def test_native_source_col_parameter_identity_roundtrips():
    spec = _only_ref(_parser().parse(
        'source_col("IndexConstituent","Weight","IndexSymbol","000300.SH")'
    ))
    assert spec is not None
    assert (spec.table, spec.field) == ("IndexConstituent", "Weight")
    assert spec.params_dict() == {"IndexSymbol": "000300.SH"}


def test_native_source_col_two_indices_have_distinct_identity():
    parser = _parser()
    a = _only_ref(parser.parse(
        'source_col("BenchmarkIndexDailyBar","Close","index","000300.SH")'
    ))
    b = _only_ref(parser.parse(
        'source_col("BenchmarkIndexDailyBar","Close","index","000905.SH")'
    ))
    assert a is not None and b is not None and a != b
    assert a.params_dict()["index"] == "000300.SH"
    assert b.params_dict()["index"] == "000905.SH"


@pytest.mark.parametrize("formula", [
    'source_col("BenchmarkIndexDailyBar","Close")',
    'source_col("IndexConstituent","Weight")',
])
def test_native_source_col_rejects_missing_identity(formula):
    with pytest.raises(Exception, match="requires exact parameters"):
        _parser().parse(formula)


def test_ranked_source_preserves_numeric_identity():
    parser=_parser()
    refs=[_only_ref(parser.parse(f'source_col("StockTopTenShareholder","ShareRatio","ShareholderRank",{rank})')) for rank in (1,2)]
    assert refs[0]!=refs[1]
    assert refs[0].params_dict()=={"ShareholderRank":1}


@pytest.mark.parametrize("value",['"1"',"True","0","11"])
def test_ranked_source_rejects_invalid_selector(value):
    with pytest.raises(Exception,match="rank must be integer"):
        _parser().parse(f'source_col("StockTopTenShareholder","ShareRatio","ShareholderRank",{value})')


def test_daily_holder_churn_macro_has_exact_rank_identity_and_bounded_expansion():
    from factor_engine.api.source_ref import decode_source_ref
    from factor_engine.expr.column import ColumnRef
    expr=_parser().parse("holder_top10_daily_id_churn('StockTopTenShareholder')")
    found={}
    stack=[expr]
    while stack:
        node=stack.pop()
        if isinstance(node,ColumnRef):
            spec=decode_source_ref(node.name)
            found[(spec.field,spec.params_dict()["ShareholderRank"])]=spec
        stack.extend(node.children())
    assert set(found)=={(field_name,rank) for field_name in ("ShareRatio","ShareholderId") for rank in range(1,11)}
    with pytest.raises(Exception,match="official top-ten"):
        _parser().parse("holder_top10_daily_id_churn('StockDailyBarAdj')")
