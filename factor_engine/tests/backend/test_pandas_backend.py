import pytest

pd = pytest.importorskip("pandas")

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def test_pandas_backend_end_to_end_single_factor():
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ["A", "B"],
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 20.0, 11.0, 21.0, 12.0, 18.0], index=idx)

    source = InMemorySeriesSource(data={"close": close})
    factor = Factor(name="mom_2_rank", expr=rank(ts_mean(col("close"), 2)))

    out = FactorEngine(backend=PandasBackend(), data_source=source).run(factor)
    result = out["result"]

    assert isinstance(result, pd.Series)
    assert result.index.names == ["timestamp", "instrument"]
    assert result.loc[(pd.Timestamp("2024-01-03"), "A")] == 0.0
    assert result.loc[(pd.Timestamp("2024-01-03"), "B")] == 1.0
