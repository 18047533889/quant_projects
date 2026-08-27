"""q/K Backend vs Pandas/Polars 性能基准测试。

对比 q/K 与 pandas/polars 在不同场景下的性能：
- 滚动窗口聚合
- 截面操作
- 大规模数据处理
"""

import time
import pytest
import pandas as pd
import numpy as np

from factor_engine.backend.q_backend.q_backend import QBackend
from factor_engine.backend.q_backend.q_process_manager import is_q_available
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.backend.context import ExecutionContext
from factor_engine.planner.logical_plan import PlanNode


# Benchmark 配置
BENCHMARK_SIZES = {
    "small": (252, 100),      # 252 days × 100 instruments
    "medium": (1000, 500),    # 1000 days × 500 instruments
}


def generate_benchmark_data(n_periods: int, n_instruments: int) -> pd.DataFrame:
    """生成基准测试数据。"""
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    instruments = [f"STOCK_{i:04d}" for i in range(n_instruments)]

    data = []
    for inst in instruments:
        values = np.random.randn(n_periods).cumsum() + 100
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "instrument": inst,
                "close": values[i],
                "volume": np.random.randint(1000, 100000),
            })

    return pd.DataFrame(data)


def benchmark_operation(backend, plan: PlanNode, data: pd.DataFrame, n_runs: int = 3) -> dict:
    """执行单次基准测试。"""
    ctx = ExecutionContext()
    ctx.base_data = data

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        try:
            result = backend.execute(plan, ctx)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            return {"success": False, "error": str(e), "mean_time": None}

    return {
        "success": True,
        "mean_time": np.mean(times),
        "min_time": np.min(times),
    }


class TestQBackendBenchmark:
    """q/K Backend 基准测试套件。"""

    @pytest.mark.benchmark
    @pytest.mark.skipif(
        not is_q_available(),
        reason="Q runtime unavailable (pykx/q binary/Q_LICENSED missing) - NOT_RUN, not a failure",
    )
    def test_rolling_mean_benchmark(self):
        """滚动均值性能测试。预期：q 在向量化操作上有 5-20x 加速。"""
        n_periods, n_instruments = BENCHMARK_SIZES["small"]
        data = generate_benchmark_data(n_periods, n_instruments)

        plan = PlanNode(
            op="ts_mean",
            inputs=(),
            attrs={"window": 20},
            node_id="ts_mean_20",
        )

        pandas_backend = PandasBackend()
        pandas_result = benchmark_operation(pandas_backend, plan, data)

        q_backend = QBackend(fallback_to_pandas=True)
        q_result = benchmark_operation(q_backend, plan, data)

        if pandas_result["success"] and q_result["success"]:
            speedup = pandas_result["mean_time"] / q_result["mean_time"]
            print(f"\nRolling mean benchmark:")
            print(f"  Pandas: {pandas_result['mean_time']*1000:.2f}ms")
            print(f"  q/K:    {q_result['mean_time']*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])
