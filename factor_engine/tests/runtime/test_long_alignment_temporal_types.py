from datetime import date, datetime
import polars as pl
import pytest
from factor_engine.backend.long_alignment import AlignmentError, anchor_left_join_binary, anchor_left_join_triple


def frame(ts, dtype):
    return pl.DataFrame({"ts":pl.Series(ts,dtype=dtype),"inst":["A"]*len(ts),"_v":[1.]*len(ts)}).lazy()


def test_date_midnight_timestamp_join_without_reopening_data():
    a=frame([date(2024,1,2)],pl.Date)
    b=frame([datetime(2024,1,2)],pl.Datetime("ns"))
    result=anchor_left_join_binary(a,b).collect()
    assert result.height==1
    assert result.schema["ts"]==pl.Datetime("ns")
    assert result["_y"].to_list()==[1.]
    assert anchor_left_join_triple(a,b,frame([datetime(2024,1,2)],pl.Datetime("us"))).collect().height==1


def test_intraday_and_timezone_are_not_silently_collapsed():
    a=frame([date(2024,1,2)],pl.Date)
    with pytest.raises(AlignmentError):
        anchor_left_join_binary(a,frame([datetime(2024,1,2,9,30)],pl.Datetime("ns")))
    with pytest.raises(AlignmentError):
        anchor_left_join_binary(a,frame([datetime(2024,1,2)],pl.Datetime("ns","UTC")))


def test_timestamp_unit_conversion_retains_nanoseconds():
    a=frame([0],pl.Datetime("us"))
    b=frame([1],pl.Datetime("ns"))
    with pytest.raises(AlignmentError):
        anchor_left_join_binary(a,b)
