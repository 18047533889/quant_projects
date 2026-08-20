from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.debug_backend import DebugBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource


class DemoDataSource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError("Use DebugBackend for this demo.")


f1 = Factor(name="mom_20_5", expr=ts_mean(col("close"), 20) - ts_mean(col("close"), 5))
f2 = Factor(name="mom_20_5_rank", expr=rank(ts_mean(col("close"), 20) - ts_mean(col("close"), 5)))

engine = FactorEngine(backend=DebugBackend(), data_source=DemoDataSource())
dag = engine.compile_many([f1, f2])
print([x.factor_name for x in dag.roots])
