"""factor_matrix 宽表物化编排。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from api.factor import Factor

if TYPE_CHECKING:
    from runtime.engine import FactorEngine


def execute_materialize_matrix(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    *,
    factor_ids: Sequence[str] | None = None,
    universe: str,
    frequency: str = "1d",
    matrix_root: str | Path | None = None,
    partition_columns: list[str] | None = None,
    value_dtype: str = "float32",
    **run_kwargs: Any,
) -> dict[str, Any]:
    """``FactorEngine.materialize_matrix`` 实现体。"""
    ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
    if len(ids) != len(factors):
        raise ValueError("factor_ids 长度必须与 factors 一致")
    run_out = engine.run_many(factors, **run_kwargs)
    results = {
        fid: run_out["results"][factor.name]
        for factor, fid in zip(factors, ids)
    }
    from storage.factor_matrix_materializer import FactorMatrixMaterializer

    materializer = FactorMatrixMaterializer(matrix_root=matrix_root)
    summary = materializer.materialize(
        results,
        universe=universe,
        frequency=frequency,
        partition_columns=partition_columns,
        value_dtype=value_dtype,
    )
    summary["run_many"] = {
        "factor_names": [f.name for f in factors],
        "shared_nodes": len(run_out.get("dag").shared_nodes)
        if run_out.get("dag")
        else 0,
    }
    if "rolling_cache" in run_out:
        summary["rolling_cache"] = run_out["rolling_cache"]
    return summary
