"""Single-call, bounded-wave execution for streams of independent factor definitions."""
from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from operator import index
from threading import Lock
from time import monotonic
from typing import Any

from factor_engine.api.factor import Factor
from factor_engine.planner.dag import DuplicateFactorNameError


def execute_run_many_stream(
    engine: Any,
    factors: Iterable[Factor],
    *,
    sink: Any,
    wave_size: int | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Consume one input wave only after its predecessor has completely finished.

    All execution, resource admission, production checks and sink failure behavior
    delegate to run_many_parallel. Names are kept globally to reject duplicates;
    factor definitions, DAGs, paths and outputs are not retained across waves.
    """
    from factor_engine.runtime.batch_service import _validate_result_policy
    from factor_engine.planner.physical_lowerer import (
        _get_adaptive_chunk_size, _get_adaptive_dag_width_limit,
    )

    _validate_result_policy("sink", sink)
    width_limit = _get_adaptive_dag_width_limit()
    if wave_size is None:
        wave_size = min(_get_adaptive_chunk_size(), width_limit)
    if isinstance(wave_size, bool):
        raise ValueError("wave_size must be a positive integer")
    try:
        wave_size = index(wave_size)
    except TypeError as exc:
        raise ValueError("wave_size must be a positive integer") from exc
    if not 1 <= wave_size <= width_limit:
        raise ValueError(f"wave_size must be between 1 and host DAG width limit {width_limit}")

    pending = iter(factors)
    names: set[str] = set()
    requested = completed = waves = 0
    expected: set[str] = set()
    write_lock = Lock()
    started = monotonic()

    def write(name, result):
        nonlocal completed
        with write_lock:
            if name not in expected:
                raise RuntimeError(f"streaming wave delivered unexpected or duplicate factor {name!r}")
            accepted = sink(name, result)
            if accepted is not False:
                expected.remove(name)
                completed += 1
            return accepted

    while True:
        wave = list(islice(pending, wave_size))
        if not wave:
            break
        for factor in wave:
            if factor.name in names:
                raise DuplicateFactorNameError(
                    f"run_many_stream requires globally unique factor names: {factor.name!r}"
                )
            names.add(factor.name)
        requested += len(wave)
        expected = {factor.name for factor in wave}
        before = completed
        output = engine.run_many_parallel(
            wave, result_policy="sink", sink=write, **run_kwargs,
        )
        if output.get("results"):
            raise RuntimeError("streaming wave unexpectedly retained factor results")
        if expected or completed - before != len(wave):
            raise RuntimeError("streaming wave did not deliver every factor to the sink")
        waves += 1
        del output, wave, factor

    return {
        "results": {},
        "result_policy": "sink",
        "executor": "streaming_waves",
        "requested_factors": requested,
        "completed_factors": completed,
        "completed_waves": waves,
        "wave_size": wave_size,
        "cse_scope": "per_wave",
        "seconds": monotonic() - started,
    }
