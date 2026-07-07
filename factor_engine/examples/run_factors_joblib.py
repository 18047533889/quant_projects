#!/usr/bin/env python3
"""多因子并行求值示例：在「因子」粒度用 Joblib 并行，不在单个 rolling 窗口内并行。

依赖：``pip install "factor-engine[pandas,parallel]"``（或单独安装 joblib）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "需要 joblib：pip install 'factor-engine[parallel]' 或 pip install joblib"
    ) from exc

from api.dsl_parser import parse_factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _panel():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {
        "x": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx),
        "y": pd.Series([10.0, 11.0, 12.0, 13.0], index=idx),
    }


def _run_one(data: dict, name: str, expr_text: str) -> tuple[str, pd.Series]:
    src = InMemorySeriesSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    fac = parse_factor(expr_text, name=name)
    out = eng.run(fac)["result"]
    return name, out


def main() -> None:
    data = _panel()
    jobs = [
        ("f_rank_x", 'rank(col("x"))'),
        ("f_ts_mean", 'ts_mean(col("x"), 2)'),
        ("f_spread", 'col("y") - col("x")'),
    ]
    results = Parallel(n_jobs=2, prefer="threads")(
        delayed(_run_one)(data, n, t) for n, t in jobs
    )
    for name, s in results:
        print(name)
        print(s)


if __name__ == "__main__":
    main()
