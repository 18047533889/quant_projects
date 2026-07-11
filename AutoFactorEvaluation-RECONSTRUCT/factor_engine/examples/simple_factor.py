from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.debug_backend import DebugBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource


class DemoDataSource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError("Use DebugBackend for this demo.")


expr = rank(ts_mean(col("close"), 20) - ts_mean(col("close"), 5))
factor = Factor(name="mom_20_5_rank", expr=expr, freq="1d", universe="equities")

engine = FactorEngine(backend=DebugBackend(), data_source=DemoDataSource())
out = engine.run(factor)
print(out["result"])
