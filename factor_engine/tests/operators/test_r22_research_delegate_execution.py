"""Execution contracts for six final-routed research Polars delegates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.polars_backend_kind import PolarsImplementationKind
from factor_engine.backend.contracts import ExecutionKind


@dataclass(frozen=True)
class Case:
    name: str
    params: dict[str, Any]
    invalid: dict[str, Any]
    rows: int = 80


CASES = (
    Case("ts_adaptive_noise_kalman", {"q_init": .02, "r_init": .3, "window": 8, "adapt_rate": .1}, {"window": "bad"}),
    Case("ts_causal_local_linear_smoother", {"window": 12, "min_periods": 6}, {"window": 1, "min_periods": 1}),
    Case("ts_causal_savgol_endpoint", {"window": 9, "polyorder": 2}, {"window": 3, "polyorder": 3}),
    Case("ts_bds_statistic", {"window": 40, "embedding_dim": 2, "distance_multiplier": 1.5}, {"embedding_dim": 1}),
    Case("ts_betti_1_max_persistence", {"window": 24, "tau": 1, "embedding_dim": 3}, {"window": 6, "tau": 2, "embedding_dim": 3}),
    Case("ts_bicoherence_top_decile_excess", {"window": 64, "n_segments": 4, "n_surrogates": 2}, {"window": 16, "n_segments": 3, "n_surrogates": 2}, rows=72),
)


def _frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(2206)
    t = np.arange(rows, dtype=float)
    values = (0.015 * t + np.sin(2*np.pi*t/13) + .45*np.sin(2*np.pi*t/7+.2)
              + .08*rng.standard_normal(rows))
    return pd.DataFrame({"A": values})


def _ops(name: str):
    load_all()
    return (OperatorRegistry.get(name, "pandas_numpy", mode="any"),
            OperatorRegistry.get(name, "polars", mode="any"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_final_delegate_matches_pandas_and_is_future_stable(case: Case) -> None:
    pl = pytest.importorskip("polars")
    frame = _frame(case.rows)
    pandas_op, polars_op = _ops(case.name)
    expected = pandas_op.calculate(frame, **case.params).to_numpy()
    actual = polars_op.calculate(pl.from_pandas(frame), **case.params).to_numpy()
    assert np.isfinite(expected).sum() > 0
    assert np.isfinite(actual).sum() > 0
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    if case.name in {
        "ts_causal_local_linear_smoother",
        "ts_causal_savgol_endpoint",
        "ts_betti_1_max_persistence",
    }:
        physical = getattr(polars_op, "physical_spec", None)
        spec = physical() if callable(physical) else polars_op._physical_spec
        assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE

    cut = case.rows - 9
    perturbed = frame.copy()
    perturbed.iloc[cut:, 0] += np.linspace(100., 900., case.rows-cut)
    changed = polars_op.calculate(pl.from_pandas(perturbed), **case.params).to_numpy()
    np.testing.assert_allclose(changed[:cut], actual[:cut], equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_nan_inf_fixture_matches_pandas(case: Case) -> None:
    pl = pytest.importorskip("polars")
    frame = _frame(case.rows)
    frame.iloc[7, 0] = np.nan
    frame.iloc[19, 0] = np.inf
    frame.iloc[31, 0] = -np.inf
    pandas_op, polars_op = _ops(case.name)
    expected = pandas_op.calculate(frame, **case.params).to_numpy()
    actual = polars_op.calculate(pl.from_pandas(frame), **case.params).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_invalid_parameters_raise_the_same_error(case: Case) -> None:
    pl = pytest.importorskip("polars")
    frame = _frame(case.rows)
    pandas_op, polars_op = _ops(case.name)
    bad = dict(case.params)
    bad.update(case.invalid)
    errors = []
    for op, data in ((pandas_op, frame), (polars_op, pl.from_pandas(frame))):
        with pytest.raises((TypeError, ValueError)) as caught:
            op.calculate(data, **bad)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[1] == errors[0]


def test_load_all_and_bridge_bootstrap_choose_identical_exact_delegates() -> None:
    script = r'''
import hashlib, json, sys
import numpy as np, pandas as pd, polars as pl
if sys.argv[1] == "load_all":
    from factor_engine.cleaned_operators import load_all
    load_all()
else:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    ensure_cleaned_loaded()
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.polars_backend_kind import polars_backend_kind
t=np.arange(40,dtype=float)
x=pd.DataFrame({"A":.02*t+np.sin(2*np.pi*t/9)})
cases={
 "ts_adaptive_noise_kalman":{"q_init":.02,"r_init":.3,"window":8,"adapt_rate":.1},
 "ts_causal_local_linear_smoother":{"window":12,"min_periods":6},
 "ts_causal_savgol_endpoint":{"window":9,"polyorder":2},
 "ts_betti_1_max_persistence":{"window":24,"tau":1,"embedding_dim":3},
}
out={}
for name,params in cases.items():
    op=OperatorRegistry.get(name,"polars",mode="any")
    values=op.calculate(pl.from_pandas(x),**params).to_numpy()
    payload=np.nan_to_num(values,nan=0.,posinf=1e300,neginf=-1e300).tobytes()+np.isnan(values).tobytes()
    physical=getattr(op,"physical_spec",None)
    spec=physical() if callable(physical) else op._physical_spec
    out[name]={"params":list(op.metadata.param_names),"hash":hashlib.sha256(payload).hexdigest(),
               "finite":int(np.isfinite(values).sum()),
               "kind":polars_backend_kind(op,source="r22-bootstrap",production_mode=True).value,
               "execution_kind":spec.execution_kind.value,
               "binding_hashes":[spec.implementation_source_hash,spec.parameter_domain_hash,
                                  spec.semantic_contract_hash,spec.implementation_closure_hash]}
print("R22_BOOTSTRAP="+json.dumps(out,sort_keys=True))
'''
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", POLARS_MAX_THREADS="2")
    results = []
    for entrypoint in ("load_all", "ensure_cleaned_loaded"):
        completed = subprocess.run(
            [sys.executable, "-c", script, entrypoint], check=True, text=True,
            capture_output=True, timeout=90, env=env,
        )
        line = next(line for line in completed.stdout.splitlines() if line.startswith("R22_BOOTSTRAP="))
        results.append(json.loads(line.removeprefix("R22_BOOTSTRAP=")))
    assert results[1] == results[0]
    assert all(item["finite"] > 0 for item in results[0].values())
    adaptive = results[0]["ts_adaptive_noise_kalman"]
    assert adaptive["kind"] == PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE.value
    assert adaptive["execution_kind"] == ExecutionKind.POLARS_PANDAS_DELEGATE.value
    assert all(len(digest) == 64 for digest in adaptive["binding_hashes"])
