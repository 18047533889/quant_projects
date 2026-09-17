import pytest
from factor_engine.tools.catalog_r19_fiscal_recipes import migrate_formula

@pytest.mark.parametrize("source",[
 "fin_goodwill_intensity(goodwill,total_assets)",
 "fiscal_sign_consistency(signal=operating_revenue,period_id=ReportPeriodEndDate)",
 "fin_net_debt_issuance(ShorttermLoan,LongtermLoan,BondsPayable,total_assets,FiscalPeriodId)",
 "fin_lease_intensity(usufruct_assets,lease_liability,total_assets)",
 "fin_other_earnings_dependence(other_earnings,total_profit)",
 "fin_impairment_intensity(asset_impairment_loss,credit_impairment_loss,operating_revenue)",
])
def test_unique_source_period_compiles(source):
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from evidence.factor_catalog_20260915.smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    formula,changes=migrate_formula(source)
    assert changes
    expr=parser.parse(formula)
    assert not bind_fields(expr)[1]
    engine.compile(Factor(name="fiscal",expr=expr,source_expr=formula,surface="compat_research"))

@pytest.mark.parametrize("source",[
 "fin_goodwill_intensity(unknown,total_assets)",
 "fin_goodwill_intensity(operating_revenue,total_assets)",
 "fin_goodwill_intensity(goodwill,total_assets,period_id=custom_period)",
 "x(",
])
def test_unknown_or_mixed_table_not_invented(source):
    assert migrate_formula(source)==(source,[])
