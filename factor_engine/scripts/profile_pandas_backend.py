#!/usr/bin/env python3
"""对 ``PandasBackend`` 代表性因子跑 ``cProfile``， stdout 输出热点（可自行重定向到文件）。

用法::

    cd share/projects/factor_engine
    python scripts/profile_pandas_backend.py
    python -m cProfile -s cumtime scripts/profile_pandas_backend.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _panel(n_inst: int = 20, n_days: int = 50) -> dict[str, pd.Series]:
    """构造用于 profiling 的多标的日频面板数据。"""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=n_days, freq="D"), [f"S{i:03d}" for i in range(n_inst)]],
        names=["timestamp", "instrument"],
    )
    rng = np.random.default_rng(0)
    return {
        "close": pd.Series(rng.standard_normal(len(idx)), index=idx),
        "vol": pd.Series(rng.random(len(idx)) * 1e6, index=idx),
    }


def main() -> None:
    """对 PandasBackend 代表性因子运行 cProfile 并输出热点统计。"""
    data = _panel()
    src = InMemorySeriesSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    fac = Factor(name="prof", expr=rank(ts_mean(col("close"), 20)))
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        eng.run(fac)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(40)
    print(buf.getvalue())
    print(
        "\n说明：关注 ``pandas_backend`` 内 ``groupby``/``apply``/``rolling`` 及 "
        "``_ts_roll_via_bottleneck``；Bottleneck/Numba 可显著改变排序。",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
