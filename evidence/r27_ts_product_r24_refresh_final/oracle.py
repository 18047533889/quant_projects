"""Independent post-contract oracle for the explicit R24 ts_product recipe."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def main():
    load_all()
    rng = np.random.default_rng(2720)
    values = 1.0 + rng.normal(0.0, 0.02, size=(96, 2))
    values[[8, 37], 0] = np.nan
    frame = pd.DataFrame(values, columns=["A", "B"])
    expected = np.full_like(values, np.nan)
    for row in range(96):
        start = max(0, row - 11)
        for column in range(2):
            sample = values[start : row + 1, column]
            finite = sample[~np.isnan(sample)]
            if len(finite) >= 3:
                expected[row, column] = np.prod(finite)
    result = {"canonical": "ts_product", "params": {"window": 12, "min_periods": 3, "skipna": True}, "backends": {}}
    arrays = {}
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("ts_product", backend, mode="research")
        source = pl.from_pandas(frame) if backend == "polars" else frame
        actual = op.calculate(source, window=12, min_periods=3, skipna=True)
        arrays[backend] = actual.to_numpy()
        np.testing.assert_allclose(arrays[backend], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        result["backends"][backend] = {"oracle_equal": True, "finite": int(np.isfinite(arrays[backend]).sum())}
    np.testing.assert_allclose(arrays["pandas_numpy"], arrays["polars"], rtol=1e-12, atol=1e-12, equal_nan=True)
    result["pandas_polars_equal"] = True
    output = Path(__file__).with_name("oracle_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
