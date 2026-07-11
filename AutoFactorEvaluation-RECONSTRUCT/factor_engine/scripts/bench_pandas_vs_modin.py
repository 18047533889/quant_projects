#!/usr/bin/env python3
"""对比纯 pandas 与 Modin（``FACTOR_ENGINE_USE_MODIN=1``）在同一因子上的耗时。

用法::

    cd share/projects/factor_engine
    python scripts/bench_pandas_vs_modin.py

需已 ``pip install 'factor-engine[pandas,modin]'``；未安装 modin 时脚本会跳过 Modin 段。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.factory import build_backend
from backend.pandas_compat import reset_pandas_module_cache_for_tests, resolve_pandas_module
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _panel(n_inst: int = 30, n_days: int = 60) -> dict[str, pd.Series]:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=n_days, freq="D"), [f"S{i:03d}" for i in range(n_inst)]],
        names=["timestamp", "instrument"],
    )
    rng = np.random.default_rng(0)
    return {"close": pd.Series(rng.standard_normal(len(idx)), index=idx)}


def _bench(label: str, n_runs: int = 5) -> float:
    data = _panel()
    src = InMemorySeriesSource(data=data)
    fac = Factor(name="b", expr=rank(ts_mean(col("close"), 20)))
    eng = FactorEngine(backend=build_backend(label), data_source=src)
    # 预热
    eng.run(fac)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        eng.run(fac)
    return (time.perf_counter() - t0) / n_runs


def main() -> None:
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        t_pd = _bench("pandas")
        print(f"pandas backend (resolve_pandas_module={resolve_pandas_module().__name__}): {t_pd*1000:.2f} ms/run")

        reset_pandas_module_cache_for_tests()
        os.environ["FACTOR_ENGINE_USE_MODIN"] = "1"
        try:
            import modin.pandas  # noqa: F401
        except ImportError:
            print("modin 未安装，跳过 Modin 对比。安装: pip install 'factor-engine[modin]'", file=sys.stderr)
            return
        t_md = _bench("pandas_modin")
        impl = resolve_pandas_module().__name__
        print(f"pandas_modin (resolved={impl}): {t_md*1000:.2f} ms/run")
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
        os.environ.pop("FACTOR_ENGINE_USE_MODIN", None)
        reset_pandas_module_cache_for_tests()


if __name__ == "__main__":
    main()
