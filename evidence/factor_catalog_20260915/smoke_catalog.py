"""Bounded real-data smoke execution for the imported factor catalog.

This runner is evidence-only: it reads a small, explicit A-share window through
DataAccessSource, executes each selected formula once with FactorEngine, and
writes only compressed per-factor summaries (never panels or production data).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gzip
import faulthandler
import gc
import hashlib
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_SYMBOLS = (
    "000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ",
    "000651.SZ", "000858.SZ", "600000.SH", "600519.SH",
)
SUPPORTED_DATASET = "ashare_stock_daily_adj"
MAX_ERROR_CHARS = 500
MAX_SMOKE_SYMBOLS = 128
WATCHDOG_SCOPE_ENV = (
    "FACTOR_CATALOG_EXTERNAL_WATCHDOG",
    "FACTOR_CATALOG_WATCHDOG_PARENT_PID",
    "FACTOR_CATALOG_WATCHDOG_TOKEN",
)
SUPPORTED_BACKENDS = ("pandas", "polars", "auto")


class FactorTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def factor_deadline(seconds: int):
    """Install a cooperative POSIX deadline around Python-visible execution.

    Native kernels may defer SIGALRM until they return to Python.  The external
    watchdog remains the hard wall-clock/RSS authority for the process.
    """
    if seconds <= 0:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def _expired(_signum, _frame):
        raise FactorTimeout(f"factor exceeded {seconds}s deadline")

    signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def validate_external_watchdog_scope() -> None:
    """Require a live, direct watchdog parent and a per-launch scope token."""
    enabled, parent_pid, token = (os.environ.get(name, "") for name in WATCHDOG_SCOPE_ENV)
    try:
        expected_parent = int(parent_pid)
    except ValueError:
        expected_parent = -1
    if (enabled != "1" or expected_parent != os.getppid()
            or re.fullmatch(r"[0-9a-f]{32,}", token) is None):
        raise RuntimeError(
            "external watchdog mode requires a valid direct-parent watchdog scope; "
            "refusing unbounded execution"
        )


@contextlib.contextmanager
def execution_deadline(seconds: int, mode: str):
    if mode == "external-watchdog":
        validate_external_watchdog_scope()
        yield
    elif mode == "legacy-cooperative":
        with factor_deadline(seconds):
            yield
    else:
        raise ValueError(f"unsupported deadline mode: {mode}")


def build_execution_backend(name: str):
    """Build the requested engine backend through the public factory."""
    from factor_engine.backend.factory import build_backend
    return build_backend(name)


def execution_scope_fields(start: str, end: str, symbols) -> dict[str, Any]:
    return {
        "execution_window": [start, end],
        "execution_symbol_count": len(symbols),
    }


def preflight_factors(engine, factors, rows, *, timeout_seconds,
                      deadline_mode="external-watchdog"):
    """Isolate compile failures before DAG batching; success is NOT execution.

    Uses the same compiler as execution. No source reads are necessary for this
    phase; a read-rejecting source is exercised in the regression suite.
    """
    ready = []
    for factor in factors:
        row = rows[factor.name]
        started = time.monotonic()
        try:
            with execution_deadline(timeout_seconds, deadline_mode):
                engine.compile(factor)
        except Exception as exc:
            row.update(
                status="COMPILE_TIMEOUT" if isinstance(exc, FactorTimeout) else "COMPILE_FAILED",
                compile_status="FAILED",
                error_type=type(exc).__name__,
                error=str(exc)[:MAX_ERROR_CHARS],
            )
        else:
            row["compile_status"] = "COMPILED"
            ready.append(factor)
        finally:
            row["compile_seconds"] = round(time.monotonic() - started, 6)
    return ready


def current_rss_mib() -> float:
    """Read current resident bytes from procfs; return NaN if unavailable."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return float("nan")


def collect_field_refs(expr: Any) -> list[Any]:
    """Collect unique AST ColumnRef leaves, preserving table identity."""
    from factor_engine.expr.column import ColumnRef

    leaves: dict[tuple[str, str | None], Any] = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ColumnRef):
            leaves[(node.name, getattr(node, "table", None))] = node
        stack.extend(node.children())
        for _, value in getattr(node, "kwargs", ()):
            if hasattr(value, "children"):
                stack.append(value)
    return [leaves[key] for key in sorted(leaves, key=lambda item: str(item))]


def bind_fields(
    expr: Any,
    resolver: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Resolve AST leaves while preserving cross-table SourceRef identities."""
    if resolver is None:
        from factor_engine.fields.resolver import resolve_market_field
        resolver = resolve_market_field
    bindings: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for leaf in collect_field_refs(expr):
        name = leaf.name
        try:
            resolved = resolver(leaf, "ashare", strict=True)
            spec = resolved.spec
            dataset = getattr(spec, "dataset", None)
            binding = {
                "input": name,
                "table": getattr(spec, "table", None),
                "dataset": dataset,
                "column": getattr(spec, "source_name", None),
                "unit": str(getattr(spec, "unit", None)),
                "source_unit": str(getattr(spec, "source_unit", None)),
                "scale": getattr(spec, "scale_to_canonical", None),
            }
            if not dataset:
                failures.append({
                    "input": name, "reason": "MISSING_DATASET_IDENTITY", "detail": str(binding),
                })
            elif not binding["column"]:
                failures.append({
                    "input": name, "reason": "MISSING_PHYSICAL_COLUMN", "detail": str(binding),
                })
            else:
                bindings.append(binding)
        except Exception as exc:
            failures.append({
                "input": name,
                "reason": "FIELD_RESOLUTION_FAILED",
                "detail": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            })
    return bindings, failures


def summarize_result(value: Any) -> dict[str, Any]:
    """Return bounded result evidence without persisting the factor panel."""
    import numpy as np
    import pandas as pd

    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    if isinstance(value, pd.DataFrame):
        series = value.stack(future_stack=True)
    elif isinstance(value, pd.Series):
        series = value
    else:
        series = pd.Series(value)
    numeric = pd.to_numeric(series, errors="coerce")
    finite_count = int(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return {
        "value_count": int(series.size),
        "finite_count": finite_count,
        "result_hash": hashlib.sha256(hashed).hexdigest(),
    }


def execute_default_batch(engine: Any, factors: Any, sink: Any):
    """Exercise public defaults; never override CSE, workers or DAG options."""
    return engine.run_many(factors, result_policy="sink", sink=sink)


def mark_batch_aborted(rows: dict, exc: Exception) -> None:
    """Unsunk roots are not individually diagnosed by a batch exception."""
    for row in rows.values():
        if "status" not in row:
            row.update(
                status="BATCH_ABORTED", retry_after_batch_abort=False,
                error_type=type(exc).__name__, error=str(exc)[:MAX_ERROR_CHARS],
            )


def retry_factor_summary(engine: Any, factor: Any, *, requested_backend: str) -> dict[str, Any]:
    """Keep auto retries on the planner-admitted batch API; persist no panels."""
    if requested_backend != "auto":
        return summarize_result(engine.run(factor))
    summaries: dict[str, dict[str, Any]] = {}

    def sink(name: str, value: Any) -> bool:
        if name != factor.name or name in summaries:
            raise RuntimeError("unexpected or duplicate terminal sink in singleton retry")
        summaries[name] = summarize_result(value)
        return True

    outcome = engine.run_many(
        [factor], enable_cse=True, result_policy="sink", sink=sink,
    )
    if factor.name not in summaries:
        raise RuntimeError("run_many retry returned without invoking the terminal sink")
    summary = summaries[factor.name]
    if isinstance(outcome, dict):
        route = (outcome.get("backend_paths") or {}).get(factor.name)
        if route is not None:
            summary["backend_path"] = route
    return summary


def _load_records(path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as src:
        for index, line in enumerate(src):
            if index < offset:
                continue
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.jsonl.gz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeout-seconds", type=int, default=30)
    ap.add_argument("--deadline-mode", choices=("external-watchdog", "legacy-cooperative"),
                    default="external-watchdog",
                    help="external watchdog is native-safe; legacy cooperative is diagnostic only")
    ap.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="pandas")
    ap.add_argument("--max-rss-mib", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batch-only", action="store_true",
                    help="report batch aborts without singleton retries")
    ap.add_argument("--prepare-chunk", type=int, default=100)
    ap.add_argument("--rss-rotate-mib", type=int, default=900)
    ap.add_argument("--daily-only", action="store_true")
    ap.add_argument("--semantic-redesign-intraday-limit", action="store_true")
    args = ap.parse_args()
    if args.limit <= 0 or args.offset < 0:
        ap.error("--limit must be positive; full-catalog execution requires explicit sharding")
    if (args.timeout_seconds <= 0 or args.max_rss_mib <= 0 or args.batch_size <= 0 or args.prepare_chunk <= 0
            or args.rss_rotate_mib <= 0 or args.rss_rotate_mib >= args.max_rss_mib):
        ap.error("timeout and RSS limit must be positive")
    if args.deadline_mode == "external-watchdog":
        validate_external_watchdog_scope()

    base = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = base / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base / output_path
    if output_path.suffix != ".gz":
        ap.error("--output must end in .gz")

    # Scope the real mirror to this process; never mutate registry files/global config.
    os.environ["ASHARE_PARQUET_ROOT"] = "/home/sunhaiwei/cos_data"
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    os.environ["DATA_ACCESS_RUN_MODE"] = "interactive_research"

    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
    from factor_engine.tools.catalog_recipe_migration import (
        migrate_catalog_recipe_formula, redesign_catalog_intraday_limit_formula,
    )
    from factor_engine.tools.catalog_sketch_migration import migrate_catalog_sketch_formula

    load_all()
    parser = DSLParser(surface="compat_research", dialect="native")
    effective_limit = min(args.limit, args.prepare_chunk)
    scan_limit = effective_limit if not args.daily_only else min(114_132, effective_limit * 100)
    records = _load_records(input_path, scan_limit, args.offset)
    prepared: list[dict[str, Any]] = []
    source_fields: dict[str, str] = {}
    source_records_consumed = 0
    for record in records:
        source_records_consumed += 1
        item = {"record": record}
        try:
            sketch_migration = migrate_catalog_sketch_formula(
                str(record.get("formula") or ""), enabled=True, parser=parser
            )
            redesign_migration = redesign_catalog_intraday_limit_formula(
                sketch_migration.formula,
                enabled=args.semantic_redesign_intraday_limit,
            )
            migration = migrate_adjusted_price_fields(
                redesign_migration.formula, market="ashare"
            )
            recipe_migration = migrate_catalog_recipe_formula(migration.formula)
            expr = parser.parse(recipe_migration.formula)
            bindings, failures = bind_fields(expr)
            item.update(
                expr=expr,
                migrated_formula=recipe_migration.formula,
                migration_changes=(
                    ([f"catalog_sketch:{sketch_migration.status}"]
                    if sketch_migration.converted
                     else [])
                    + list(redesign_migration.changes)
                    + list(migration.changes)
                    + list(recipe_migration.changes)
                ),
                bindings=bindings,
                binding_failures=failures,
            )
            if args.daily_only and (
                failures or any(b["dataset"] != SUPPORTED_DATASET for b in bindings)
            ):
                continue
            if not failures:
                for binding in bindings:
                    if binding["dataset"] == SUPPORTED_DATASET:
                        source_fields[binding["input"]] = binding["column"]
        except Exception as exc:
            item["prepare_error"] = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        prepared.append(item)
        if args.daily_only and len(prepared) >= effective_limit:
            break

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols or len(symbols) > MAX_SMOKE_SYMBOLS or len(set(symbols)) != len(symbols):
        ap.error(f"--symbols must contain between 1 and {MAX_SMOKE_SYMBOLS} unique instruments")
    source = DataAccessSource(
        dataset=SUPPORTED_DATASET,
        fields=source_fields,
        start_date=args.start,
        end_date=args.end,
        instrument_filter=symbols,
        run_mode="interactive_research",
        production=False,
        read_auto=False,
    )
    # FE and DataAccess intentionally use their own validated mode vocabularies:
    # FE="research"; DA request propagation="interactive_research".
    engine = FactorEngine(build_execution_backend(args.backend), source, run_mode="research")

    counts: collections.Counter[str] = collections.Counter()
    emitted = 0
    stopped_reason: str | None = None
    started = time.monotonic()
    def base_row(item: dict[str, Any]) -> dict[str, Any]:
        record = item["record"]
        row = {
            key: record.get(key)
            for key in ("source_row", "id", "formula", "fields", "tables", "domain", "pit", "logic", "batch")
        }
        row["bindings"] = item.get("bindings", [])
        row["executed_formula"] = item.get("migrated_formula", row["formula"])
        row["migration_changes"] = item.get("migration_changes", [])
        row.update(execution_scope_fields(args.start, args.end, symbols))
        return row

    # Publish only a closed gzip member. A watchdog-killed shard may leave this
    # bounded private temp file, but can never corrupt a previously durable shard.
    temp_output_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with gzip.open(temp_output_path, "wt", encoding="utf-8") as dst:
        ready: list[dict[str, Any]] = []
        for item in prepared:
            if "prepare_error" not in item and not item["binding_failures"]:
                ready.append(item)
                continue
            record = item["record"]
            row = base_row(item)
            if "prepare_error" in item:
                row.update(status="PREPARE_FAILED", error_type="PreparationError", error=item["prepare_error"])
            else:
                row.update(status="SOURCE_UNAVAILABLE", unresolved=item["binding_failures"])
            row["elapsed_seconds"] = 0.0
            row["rss_mib"] = round(current_rss_mib(), 3)
            counts[row["status"]] += 1
            emitted += 1
            dst.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            dst.flush()

        for offset in range(0, len(ready), args.batch_size):
            chunk = ready[offset:offset + args.batch_size]
            rss_before = current_rss_mib()
            # DuckDB/Arrow native arenas may not return all prior financial buffers
            # to the OS. Rotate the process before the next batch can cross the
            # hard 2 GiB watchdog; resume with --offset=next_offset.
            if rss_before >= args.rss_rotate_mib:
                stopped_reason = f"RSS_ROTATION_AT_{rss_before:.3f}_MIB"
                break
            # source_row is the total catalog identity, including non-factor rows
            # whose id is intentionally blank.
            factor_names = [f"source_row_{item['record']['source_row']}" for item in chunk]
            rows = {name: base_row(item) for name, item in zip(factor_names, chunk, strict=True)}
            batch_started = time.monotonic()
            factors = []
            for name, item in zip(factor_names, chunk, strict=True):
                try:
                    factors.append(Factor(
                        name=name,
                        expr=item["expr"],
                        source_expr=item["migrated_formula"],
                        surface="compat_research",
                    ))
                except Exception as exc:
                    rows[name].update(
                        status="CONTRACT_FAILED",
                        error_type=type(exc).__name__,
                        error=str(exc)[:MAX_ERROR_CHARS],
                        terminal_elapsed_seconds=round(time.monotonic() - batch_started, 6),
                    )

            factors = preflight_factors(
                engine, factors, rows, timeout_seconds=args.timeout_seconds,
                deadline_mode=args.deadline_mode,
            )

            def sink(name: str, result: Any) -> bool:
                summary = summarize_result(result)
                status = "EXECUTED" if summary["finite_count"] else "EXECUTED_ALL_NONFINITE"
                rows[name].update(
                    status=status,
                    terminal_elapsed_seconds=round(time.monotonic() - batch_started, 6),
                    **summary)
                print(json.dumps({"event": "factor_terminal",
                                  "source_row": rows[name]["source_row"],
                                  "status": status,
                                  "batch_seconds": rows[name]["terminal_elapsed_seconds"]}),
                      flush=True)
                return True

            # Bounded diagnostic snapshots reveal slow native kernels without
            # publishing an incomplete batch as successful execution evidence.
            faulthandler.dump_traceback_later(120, repeat=True)
            try:
                if current_rss_mib() > args.max_rss_mib:
                    raise MemoryError(f"RSS already exceeds {args.max_rss_mib} MiB")
                with execution_deadline(
                        args.timeout_seconds * max(1, len(factors)), args.deadline_mode):
                    batch_result = (execute_default_batch(
                        engine, factors, sink,
                    ) if factors else {})
                # Preserve the engine's actual routing evidence, not a guessed
                # label inferred from the requested backend.
                if isinstance(batch_result, dict):
                    for factor_name, route in (batch_result.get("backend_paths") or {}).items():
                        if factor_name in rows:
                            rows[factor_name]["backend_path"] = route
                for row in rows.values():
                    if "status" not in row:
                        row.update(
                            status="NO_RESULT",
                            error_type="MissingTerminalResult",
                            error="run_many returned without invoking the terminal sink",
                        )
            except FactorTimeout as exc:
                for row in rows.values():
                    if "status" not in row:
                        row.update(status="TIMEOUT", error_type=type(exc).__name__, error=str(exc))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                unsunk = [factor for factor in factors if "status" not in rows[factor.name]]
                print(json.dumps({"event": "batch_abort", "unsunk": len(unsunk),
                                  "error_type": type(exc).__name__,
                                  "error": str(exc)[:MAX_ERROR_CHARS]}), flush=True)
                if args.batch_only:
                    mark_batch_aborted(rows, exc)
                elif len(factors) > 1:
                    # A single bad terminal aborts run_many before later sinks.
                    # Retry each unsunk factor exactly once so the batch exception
                    # is never falsely attributed to unrelated catalog rows.
                    for factor in unsunk:
                        row = rows[factor.name]
                        retry_started = time.monotonic()
                        print(json.dumps({"event": "factor_retry_started",
                                          "source_row": row["source_row"],
                                          "reason": type(exc).__name__}), flush=True)
                        try:
                            with execution_deadline(args.timeout_seconds, args.deadline_mode):
                                summary = retry_factor_summary(
                                    engine, factor, requested_backend=args.backend,
                                )
                            status = "EXECUTED" if summary["finite_count"] else "EXECUTED_ALL_NONFINITE"
                            row.update(status=status, retry_after_batch_abort=True,
                                       batch_abort_error_type=type(exc).__name__,
                                       batch_abort_error=str(exc)[:MAX_ERROR_CHARS],
                                       terminal_elapsed_seconds=round(time.monotonic() - retry_started, 6),
                                       **summary)
                        except Exception as single_exc:
                            row.update(
                                status="EXECUTION_FAILED",
                                retry_after_batch_abort=True,
                                error_type=type(single_exc).__name__,
                                error=str(single_exc)[:MAX_ERROR_CHARS],
                                batch_abort_error_type=type(exc).__name__,
                                batch_abort_error=str(exc)[:MAX_ERROR_CHARS],
                            )
                else:
                    for factor in unsunk:
                        rows[factor.name].update(
                            status="EXECUTION_FAILED", error_type=type(exc).__name__,
                            error=str(exc)[:MAX_ERROR_CHARS],
                        )
            finally:
                faulthandler.cancel_dump_traceback_later()
            elapsed = time.monotonic() - batch_started
            for item in chunk:
                row = rows[f"source_row_{item['record']['source_row']}"]
                row["elapsed_seconds"] = round(elapsed, 6)
                row["batch_size"] = len(chunk)
                row["rss_mib"] = round(current_rss_mib(), 3)
                counts[row["status"]] += 1
                emitted += 1
                dst.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            dst.flush()
            # Financial SourceRef children are execution-scoped and can own wide
            # Arrow/DuckDB buffers.  Release them between bounded batches so a
            # 114k-row catalog does not accumulate prior batches in RSS.
            source.clear_cache()
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass

    os.replace(temp_output_path, output_path)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "processed": emitted,
        "offset": args.offset,
        "next_offset": (
            args.offset + source_records_consumed
            if args.daily_only and emitted == len(prepared)
            else (args.offset + emitted if not args.daily_only else None)
        ),
        "complete": emitted == len(prepared) and effective_limit == args.limit,
        "stopped_reason": (stopped_reason or
                           ("PREPARE_CHUNK_LIMIT" if effective_limit < args.limit else None)),
        "counts": dict(counts),
        "seconds": time.monotonic() - started,
        "window": [args.start, args.end],
        "symbols": symbols,
        "dataset": SUPPORTED_DATASET,
        "max_rss_mib": args.max_rss_mib,
        "timeout_seconds": args.timeout_seconds,
        "deadline_kind": args.deadline_mode,
        "requested_backend": args.backend,
        "batch_size": args.batch_size,
        "batch_only": args.batch_only,
        "performance_options": "public_defaults",
        "requested_limit": args.limit,
        "prepare_chunk": args.prepare_chunk,
        "daily_only": args.daily_only,
        "input_records_scanned": source_records_consumed,
    }
    summary_path = output_path.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
