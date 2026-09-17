from factor_engine.tools.catalog_r20_return_units import migrate_formula

def test_remove_only_explicit_adjusted_bps_double_conversion():
    f="ts_std(safe_div_null(field('ret', table='StockDailyBarAdj'), 10000), 20)"
    new,notes=migrate_formula(f)
    assert new=="ts_std(field('ret', table='StockDailyBarAdj'), 20)"
    assert len(notes)==1 and "UNIT_CORRECTION" in notes[0]
    assert migrate_formula(new)==(new,[])

def test_keep_unknown_units_and_deliberate_scaling():
    for f in [
        "safe_div_null(IndexReturn,10000)", "safe_div_null(ret,10000)",
        "safe_div_null(field('ret',table='StockDailyBar'),10000)",
        "safe_div_null(field('close',table='StockDailyBarAdj'),10000)",
        "safe_div_null(field('ret',table='StockDailyBarAdj'),100)",
    ]:
        assert migrate_formula(f)==(f,[])
