import numpy as np
import pytest
import factor_engine.api.dsl_parser as dsl

def test_compact_statistics_use_every_rank_and_match_numpy(monkeypatch):
    amounts=np.arange(1.,11.)*100
    weights=np.arange(1.,11.)/100
    pledges=np.arange(10.)*5
    calls=[]
    def source(table,field,key,rank):
        calls.append((table,field,key,rank))
        values={"ShareNumber":amounts,"ShareRatio":weights,"SharePledge":pledges}
        return values[field][rank-1]
    ops={"add":lambda x,y:x+y,"multiply":lambda x,y:x*y,"subtract":lambda x,y:x-y,
         "safe_div_null":lambda x,y:x/y if y else np.nan,"square":np.square,"sqrt":np.sqrt}
    monkeypatch.setattr(dsl,"_native_source_col",source)
    monkeypatch.setattr(dsl,"build_dsl_allowlist",lambda **kw:ops)
    actual=dsl._native_holder_top10_stat("StockTopTenShareholder","weighted_std")
    expected=np.sqrt(np.average((amounts-np.average(amounts,weights=weights))**2,weights=weights))
    np.testing.assert_allclose(actual,expected)
    assert {c[3] for c in calls}==set(range(1,11))
    assert len(calls)==20
    calls.clear()
    actual=dsl._native_holder_top10_stat("StockTopTenShareholder","pledge_ratio")
    np.testing.assert_allclose(actual,pledges.sum()/amounts.sum())
    assert len(calls)==20
    with pytest.raises(ValueError,match="official top-ten"):
        dsl._native_holder_top10_stat("StockDailyBarAdj","pledge_ratio")

@pytest.mark.parametrize("statistic,kernel,lag",[
    ("disclosure_count","holder_disclosure_count",None),
    ("two_day_rank_migration","holder_share_weighted_rank_migration",2),
    ("daily_id_churn","holder_id_matched_churn",1),
])
def test_id_macros_preserve_all_rank_id_pairs_and_exact_session_lag(monkeypatch,statistic,kernel,lag):
    calls=[]
    def source(table,field,key,rank):
        calls.append((table,field,key,rank))
        return (field,rank)
    ops={kernel:lambda *args:args,"delay":lambda value,n:("delay",value,n)}
    monkeypatch.setattr(dsl,"_native_source_col",source)
    monkeypatch.setattr(dsl,"build_dsl_allowlist",lambda **kw:ops)
    table="StockTopTenFloatShareholder"
    actual=(dsl._native_holder_top10_daily_id_churn(table) if statistic=="daily_id_churn"
            else dsl._native_holder_top10_id_stat(table,statistic))
    current=tuple((field,rank) for field in ("ShareRatio","ShareholderId") for rank in range(1,11))
    expected=current if lag is None else current+tuple(("delay",value,lag) for value in current)
    assert actual==expected
    assert len(calls)==20
    assert all(call[0]==table and call[2]=="ShareholderRank" for call in calls)
