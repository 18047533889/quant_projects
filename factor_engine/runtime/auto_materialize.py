# -*- coding: utf-8 -*-
"""Auto batch materialization: all factors in, values land, zero knobs.

``materialize_auto`` is the single entry point users need.  Every resource
decision is derived from live system state -- the user passes factors and a
write target, nothing else:

- **memory ceiling**   -- derived from ``/proc/meminfo`` MemAvailable: the
  number of factor panels allowed in flight at once is
  ``mem_available x memory_fraction / per-panel bytes`` (clamped).  It
  re-shrinks automatically because the snapshot is taken per wave.
- **worker/wave admission** -- delegated to the certified resource broker
  (CPU-token + memory-lease admission inside ``run_many``; the adaptive
  scheduler derives the read-wave budget as
  ``min(calibrated optimum, SafeEnvelope x wave_fraction, job lease)``).
- **factor queue**     -- the DAG wave planner queues the full factor list
  (lookback-clustered waves, CSE, auto-sharding).  No manual ordering.
- **output**           -- each wave's results stream straight into
  ``execute_materialize_batch`` (shared-work hoist, one generation
  transaction per wave) and are dropped from memory the moment the write
  receipt is validated.  Memory goes to storage; nothing accumulates.

``write_target="staging"`` (default) lands values through the data-access
store (dataset ``factor_lake_staging`` -> COS).  ``local``/``both``/
``clickhouse`` are supported by the same receipt machinery.

NOTE on the streaming sink: ``engine.materialize_many_fast`` (R39 streaming
writer) is the eventual replacement, but its write-receipt round-trip
currently fails against the WIP in engine.py/adaptive_batch_scheduler.py
(WriteReceipt in_doubt).  This module uses the committed stable path
(``run_many`` + ``execute_materialize_batch``) until that lands.
"""
from __future__ import annotations

from typing import Any, Sequence


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _panel_bytes(engine: Any) -> int:
    """Estimated bytes of one full factor panel (dates x instruments x 8)."""
    ds = getattr(engine, "data_source", None)
    try:
        spec = ds.execution_spec() if ds is not None else {}
        n_dates = int(spec.get("n_dates", 0) or 0)
        n_inst = int(spec.get("n_instruments", 0) or 0)
    except Exception:
        n_dates = n_inst = 0
    if n_dates <= 0 or n_inst <= 0:
        n_dates, n_inst = 250, 50  # conservative default daily panel
    return n_dates * n_inst * 8


def auto_chunk_size(engine: Any, *, memory_fraction: float = 0.25,
                    floor: int = 16, ceiling: int = 256) -> int:
    """Factors allowed in flight, derived from live MemAvailable."""
    avail = _mem_available_bytes()
    per = max(_panel_bytes(engine) * 8, 1)  # x8: intermediates/CSE copies
    n = int(avail * memory_fraction / per)
    return max(floor, min(ceiling, n))


def snapshot_auto_resources(engine: Any, chunk: int) -> dict[str, Any]:
    return {
        "mode": "auto",
        "mem_available_bytes": _mem_available_bytes(),
        "factors_in_flight": chunk,
        "wave_budget": "resource-broker derived per plan (SafeEnvelope fraction)",
        "sink_queue_bytes": "per-wave batch write (no resident queue)",
        "admission": "resource-broker CPU tokens + memory leases",
        "queue": "DAG wave planner (lookback-clustered, CSE, auto-shard)",
    }


def _canon(v, depth=0):
    if depth > 6 or v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _canon(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x, depth + 1) for x in v]
    return str(v)


def materialize_auto(
    engine: Any,
    factors: Sequence[Any],
    *,
    market: str | None = "ashare",
    factor_ids: Sequence[str] | None = None,
    write_target: str = "staging",
    staging_dataset: str | None = None,
    lake_root: str | None = None,
    materialize_kwargs: dict[str, Any] | None = None,
    memory_fraction: float = 0.25,
    chunk_size: int | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ALL given factors as DAG batches and land values directly.

    Zero resource knobs: the memory ceiling drives how many factors are in
    flight, the resource broker admits workers, the DAG wave planner queues
    the factors, and results stream from memory into the write target
    (default ``staging`` = data-access dataset -> COS) wave by wave.
    """
    from factor_engine.runtime.materialize_batch import (
        execute_materialize_batch,
    )

    factors = list(factors)
    mk = dict(materialize_kwargs or {})
    if market is not None:
        mk.setdefault("market", market)
    mk.setdefault("write_target", write_target)
    if staging_dataset is not None:
        mk.setdefault("staging_dataset", staging_dataset)
    if lake_root is not None:
        mk.setdefault("lake_root", lake_root)

    # Sanitize the lineage context: the receipt canonicalizer only accepts
    # primitives; opaque objects in execution_spec (FactorDependencyEdge)
    # would poison the write receipt.
    ds = getattr(engine, "data_source", None)
    if ds is not None and "data_source_config" not in mk:
        try:
            cfg = ds.execution_spec()
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            mk["data_source_config"] = _canon(cfg)

    ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
    if len(ids) != len(factors):
        raise ValueError("factor_ids length must match factors")
    if len(set(ids)) != len(ids):
        raise ValueError("factor_ids must be unique")

    chunk = chunk_size or auto_chunk_size(engine, memory_fraction=memory_fraction)
    rk = dict(run_kwargs or {})

    all_ids: list[str] = []
    materializations: dict[str, Any] = {}
    writer_errors: list[str] = []
    per_wave: list[dict[str, Any]] = []

    for k in range(0, len(factors), chunk):
        wave_factors = factors[k:k + chunk]
        wave_ids = ids[k:k + chunk]
        out = engine.run_many(wave_factors, result_policy="return", **rk)
        results = out.get("results") if isinstance(out, dict) else out
        results = results if isinstance(results, dict) else {}
        analyses = out.get("analyses") if isinstance(out, dict) else None
        analyses = analyses if isinstance(analyses, dict) else {}

        from factor_engine.runtime.materialize_batch import MaterializeItem

        items = []
        for f, fid in zip(wave_factors, wave_ids):
            v = results.get(f.name)
            if v is None:
                writer_errors.append(f"{fid}: no result")
                continue
            items.append(MaterializeItem(
                factor=f,
                output={"analysis": analyses.get(f.name), "result": v},
                factor_id=fid,
            ))
        if not items:
            continue
        batch_out = execute_materialize_batch(engine, items, shared_options=mk)
        receipt = batch_out.get("write_receipt") or {}
        ritems = receipt.get("items") or {}
        for fid, it in ritems.items():
            state = str((it or {}).get("state", ""))
            if state.upper().rstrip(".") not in ("WRITTEN", "OK", "COMMITTED"):
                writer_errors.append(f"{fid}: receipt state {state or 'missing'}")
        materializations.update(batch_out.get("materializations") or {})
        all_ids.extend(i.factor_id or i.factor.name for i in items)
        per_wave.append({
            "wave": k // chunk,
            "factors": len(items),
        })
        # results are dropped when `results` goes out of scope next iteration

    return {
        "factor_ids": all_ids,
        "materializations": materializations,
        "writer_errors": writer_errors,
        "waves": per_wave,
        "write_target": write_target,
        "auto_resource": snapshot_auto_resources(engine, chunk),
    }
