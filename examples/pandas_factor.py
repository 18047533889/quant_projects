import pandas as pd

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource


class InMemorySeriesSource(DataSource):
    def __init__(self, data: dict[str, pd.Series]) -> None:
        self.data = data

    def load_column(self, name: str):
        return self.data[name]


idx = pd.MultiIndex.from_product(
    [
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ["A", "B"],
    ],
    names=["timestamp", "instrument"],
)
close = pd.Series([10.0, 20.0, 11.0, 21.0, 12.0, 18.0], index=idx)

factor = Factor(name="mom_2_rank", expr=rank(ts_mean(col("close"), 2)))
engine = FactorEngine(backend=PandasBackend(), data_source=InMemorySeriesSource({"close": close}))

out = engine.run(factor)
print(out["result"])
