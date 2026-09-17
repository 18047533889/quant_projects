import pandas as pd
import numpy as np
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.tools.catalog_r17_limit_recipes import migrate_catalog_r17_limit_formula


UP="ashare_limit_up_touch(field('open', table='StockDailyBar'), field('close', table='StockDailyBar'), field('high_limit', table='StockDailyBarAdj'), field('low_limit', table='StockDailyBarAdj'))"
DOWN=UP.replace("ashare_limit_up_touch","ashare_limit_down_touch")
FAILED=UP.replace("ashare_limit_up_touch","ashare_limit_failed").replace("table='StockDailyBarAdj'), field('low_limit'","table='StockDailyBar'), field('low_limit'",1)


def test_strict_reviewed_shapes_rewrite_to_raw_directional_contracts():
    up=migrate_catalog_r17_limit_formula(UP,logic="涨停触板事件",enabled=True)
    down=migrate_catalog_r17_limit_formula(DOWN,logic="跌停触板事件",enabled=True)
    failed=migrate_catalog_r17_limit_formula(FAILED,logic="Historical failed_limit response conditioned on turnover",enabled=True)
    assert up.formula=="ashare_limit_up_touch(field('high', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"
    assert down.formula=="ashare_limit_down_touch(field('low', table='StockDailyBar'), field('low_limit', table='StockDailyBar'), 0.005)"
    assert failed.formula=="ashare_limit_failed(field('high', table='StockDailyBar'), field('close', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"
    assert all("default tick_tolerance=0.005" in item for item in (up.changes+down.changes+failed.changes))


def test_reviewed_catalog_logic_labels_are_supported():
    assert migrate_catalog_r17_limit_formula(UP,logic="Historical limit_up response",enabled=True).changes
    assert migrate_catalog_r17_limit_formula(DOWN,logic="Historical limit_down response",enabled=True).changes
    assert migrate_catalog_r17_limit_formula(FAILED,logic="Historical failed_limit response",enabled=True).changes


def test_ambiguous_shape_or_logic_is_unchanged():
    assert migrate_catalog_r17_limit_formula(UP,logic="unrelated",enabled=True).formula==UP
    wrong=UP.replace("field('open'","field('high'")
    assert migrate_catalog_r17_limit_formula(wrong,logic="涨停触板",enabled=True).formula==wrong
    assert migrate_catalog_r17_limit_formula(UP,logic="涨停触板",enabled=False).formula==UP


@pytest.mark.parametrize("formula,logic", [
    (UP, "跌停触板"),
    (UP, "limit_down touch"),
    (DOWN, "涨停touch"),
    (DOWN, "limit_up touch"),
    (UP, "limit_up and limit_down touch"),
    (DOWN, "涨停与跌停双向触板"),
])
def test_directional_conflict_is_rejected(formula, logic):
    migrated=migrate_catalog_r17_limit_formula(formula,logic=logic,enabled=True)
    assert migrated.formula==formula
    assert migrated.changes==()


def test_raw_official_limit_boundary_oracles():
    load_all()
    idx=pd.Index(["below","touch","above"])
    upper=pd.Series([10.0,10.0,10.0],index=idx); lower=pd.Series([9.0,9.0,9.0],index=idx)
    up=OperatorRegistry.get("ashare_limit_up_touch").calculate(pd.Series([9.994,9.995,10.0],index=idx),upper,0.005)
    down=OperatorRegistry.get("ashare_limit_down_touch").calculate(pd.Series([9.006,9.005,9.0],index=idx),lower,0.005)
    failed=OperatorRegistry.get("ashare_limit_failed").calculate(pd.Series([9.994,9.995,10.0],index=idx),pd.Series([9.8,9.994,9.996],index=idx),upper,0.005)
    assert up.tolist()==[0.0,1.0,1.0]
    assert down.tolist()==[0.0,1.0,1.0]
    assert failed.tolist()==[0.0,1.0,0.0]


def test_raw_limit_boundary_and_null_parity_pandas_polars():
    pl=pytest.importorskip("polars")
    load_all()
    idx=pd.RangeIndex(4); columns=["A"]
    frames={
        "high":pd.DataFrame([9.994,9.995,10.0,np.nan],index=idx,columns=columns),
        "low":pd.DataFrame([9.006,9.005,9.0,np.nan],index=idx,columns=columns),
        "close":pd.DataFrame([9.8,9.994,9.996,np.nan],index=idx,columns=columns),
        "upper":pd.DataFrame([10.0,10.0,10.0,10.0],index=idx,columns=columns),
        "lower":pd.DataFrame([9.0,9.0,9.0,9.0],index=idx,columns=columns),
    }
    for name,args in (
        ("ashare_limit_up_touch",("high","upper")),
        ("ashare_limit_down_touch",("low","lower")),
        ("ashare_limit_failed",("high","close","upper")),
    ):
        pandas_op=OperatorRegistry.get(name,backend="pandas_numpy")
        polars_op=OperatorRegistry.get(name,backend="polars")
        pandas_out=pandas_op.calculate(*(frames[key] for key in args),tick_tolerance=0.005)
        polars_out=polars_op.calculate(*(
            pl.DataFrame({"A":frames[key]["A"].tolist()}) for key in args
        ),tick_tolerance=0.005)
        np.testing.assert_allclose(
            pandas_out["A"].to_numpy(),polars_out["A"].to_numpy(),equal_nan=True
        )
