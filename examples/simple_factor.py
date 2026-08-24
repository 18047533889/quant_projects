from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean
from factor_engine.backend.debug_backend import DebugBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.datasource import DataSource


class DemoDataSource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError("Use DebugBackend for this demo.")


expr = rank(ts_mean(col("close"), 20) - ts_mean(col("close"), 5))
factor = Factor(name="mom_20_5_rank", expr=expr, freq="1d", universe="equities")

engine = FactorEngine(backend=DebugBackend(), data_source=DemoDataSource())
out = engine.run(factor)
print(out["result"])
