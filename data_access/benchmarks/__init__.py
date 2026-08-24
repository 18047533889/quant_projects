"""R30-P0-001 —— Benchmark Suite：固定可复现基准（B01-B09）。

全部基于 synthetic fixtures（data_access/benchmarks/fixtures.py），
确定性 seed，同一 tmp 重建 → 同一数据。不依赖生产数据 / COS。

    run_benchmarks.py             编排（默认 B01/B02/B03/B04, small scale）
    fixtures.py                   build_fixture_store / tiny_fixture
    report.py                     BenchmarkReport / benchmark_env
    benchmark_local_daily.py      B01 10y daily panel
    benchmark_local_minute.py     B02 分钟级读 + 分钟→日聚合
    benchmark_join.py             B03 daily + fundamental PIT join + universe
    benchmark_factor_batch.py     B04/B05/B06 因子批量读
    benchmark_session_reuse.py    同 DataReadSession 复用
    benchmark_incremental.py      B09 单日增量更新
    benchmark_cos.py              B07/B08 COS cold/warm（无真实 COS → SKIP）
"""
from __future__ import annotations

from .report import BenchmarkReport, benchmark_env
from .fixtures import build_fixture_store, tiny_fixture

__all__ = [
    "BenchmarkReport",
    "benchmark_env",
    "build_fixture_store",
    "tiny_fixture",
]
