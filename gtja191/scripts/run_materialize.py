#!/usr/bin/env python3
"""GTJA-191 全量落值：185 条因子 → factor_engine 因子湖（Parquet）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.catalog import deliverable_names  # noqa: E402
from lib.materialize_config import CONFIG_DIR, default_lake_root, materialize_config_path  # noqa: E402
from lib.paths import resolve_factor_engine_root  # noqa: E402
from generate_materialize_configs import generate_configs  # noqa: E402


def _ensure_factor_engine() -> Path:
    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        raise SystemExit(
            "factor_engine not found; set FACTOR_ENGINE_ROOT or place sibling ../factor_engine"
        )
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))
    return fe_root


def _collect_config_paths(
    *,
    factor: str | None,
    limit: int | None,
) -> list[Path]:
    names = deliverable_names()
    if factor:
        if factor not in names:
            raise SystemExit(f"unknown deliverable factor: {factor}")
        names = [factor]
    if limit is not None:
        names = names[:limit]
    return [materialize_config_path(name) for name in names]


def _factor_already_materialized(factor_name: str, lake_root: Path) -> bool:
    factor_dir = lake_root / "factors" / factor_name
    if not factor_dir.is_dir():
        return False
    return any(factor_dir.glob("year=*/data.parquet"))


def run_materialize_sequential(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lake_root: str | None = None,
    write_target: str = "local",
    limit: int | None = None,
    resume: bool = False,
    stop_on_error: bool = False,
) -> dict:
    """逐因子落值，避免 batch_run 同时持有多因子中间结果导致 OOM。"""
    _ensure_factor_engine()
    from runtime.engine import FactorEngine  # noqa: WPS433

    root = Path(lake_root or default_lake_root())
    names = deliverable_names()
    if limit is not None:
        names = names[:limit]

    generate_configs(
        start_date=start_date,
        end_date=end_date,
        lake_root=str(root),
        write_target=write_target,
    )

    t0 = time.perf_counter()
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[str] = []

    for idx, name in enumerate(names, start=1):
        if resume and _factor_already_materialized(name, root):
            skipped.append(name)
            continue
        path = materialize_config_path(name)
        try:
            print(f"[{idx}/{len(names)}] materializing {name}...", flush=True)
            out = FactorEngine.materialize_from_config(path, lake_root=str(root))
            mat = out["materialization"]
            rec = {
                "factor": name,
                "factor_id": mat.get("factor_id"),
                "rows_written": mat.get("rows_written"),
                "partitions": mat.get("partitions"),
                "lake_root": mat.get("lake_root"),
                "index": idx,
            }
            succeeded.append(rec)
            progress_path = PACKAGE_ROOT / "dsl" / "materialize_progress.json"
            progress_path.write_text(
                json.dumps(
                    {
                        "last_factor": name,
                        "index": idx,
                        "total": len(names),
                        "succeeded": len(succeeded),
                        "failed": len(failed),
                        "skipped": len(skipped),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            failed.append({"factor": name, "error": str(exc), "index": idx})
            if stop_on_error:
                break

    elapsed = time.perf_counter() - t0
    return {
        "ok": not failed,
        "mode": "sequential",
        "factor_count": len(names),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "elapsed_seconds": round(elapsed, 3),
        "lake_root": str(root),
        "results": succeeded,
        "errors": failed,
        "skipped_factors": skipped,
    }


def run_materialize(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lake_root: str | None = None,
    write_target: str = "local",
    factor: str | None = None,
    limit: int | None = None,
    batch_run: bool = True,
    stop_on_error: bool = False,
) -> dict:
    _ensure_factor_engine()
    from runtime.engine import FactorEngine  # noqa: WPS433

    generate_configs(
        start_date=start_date,
        end_date=end_date,
        lake_root=lake_root,
        write_target=write_target,
        factor=factor,
    )
    paths = _collect_config_paths(factor=factor, limit=limit)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing materialize configs: {missing[0]}")

    t0 = time.perf_counter()
    if len(paths) == 1:
        out = FactorEngine.materialize_from_config(
            paths[0],
            lake_root=lake_root,
        )
        materializations = {out["factor"].name: out}
    else:
        batch = FactorEngine.materialize_many_from_config(
            paths,
            batch_run=batch_run,
        )
        materializations = batch.get("materializations", batch)

    elapsed = time.perf_counter() - t0
    succeeded: list[dict] = []
    failed: list[dict] = []
    for name, out in materializations.items():
        mat = out.get("materialization") if isinstance(out, dict) else None
        if mat and mat.get("rows_written", 0) >= 0 and "error" not in out:
            succeeded.append(
                {
                    "factor": name,
                    "factor_id": mat.get("factor_id"),
                    "rows_written": mat.get("rows_written"),
                    "partitions": mat.get("partitions"),
                    "lake_root": mat.get("lake_root"),
                }
            )
        else:
            failed.append({"factor": name, "error": str(out)})

    if stop_on_error and failed:
        raise SystemExit(f"materialize failed for {failed[0]['factor']}: {failed[0]['error']}")

    return {
        "ok": not failed,
        "factor_count": len(paths),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "elapsed_seconds": round(elapsed, 3),
        "lake_root": str(lake_root or default_lake_root()),
        "config_dir": str(CONFIG_DIR),
        "results": succeeded,
        "errors": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize GTJA-191 deliverable factors (185)")
    parser.add_argument("--start-date", default=None, help="override data_source.start_date")
    parser.add_argument("--end-date", default=None, help="override data_source.end_date")
    parser.add_argument("--lake-root", default=None, help="factor lake root (default ../data/factors/lake/gtja191)")
    parser.add_argument("--write-target", default="local", help="local|staging|both|clickhouse|staging_clickhouse")
    parser.add_argument("--factor", default=None, help="single factor e.g. gtja191_alpha_001")
    parser.add_argument("--limit", type=int, default=None, help="only first N deliverable factors")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="short window 2016-01-04..2016-01-10 + limit 3 (quick parity check)",
    )
    parser.add_argument("--no-batch-run", action="store_true", help="disable shared run_many batch")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="one factor at a time (low memory; recommended for full 185 run)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --sequential: skip factors already in lake with all expected partitions",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--report", default=None, help="write JSON summary path")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    limit = args.limit
    if args.smoke:
        start_date = start_date or "2016-01-04"
        end_date = end_date or "2016-01-10"
        limit = limit or 3

    try:
        use_sequential = args.sequential or (
            args.factor is None and limit is None and not args.smoke
        )
        if use_sequential:
            summary = run_materialize_sequential(
                start_date=start_date,
                end_date=end_date,
                lake_root=args.lake_root,
                write_target=args.write_target,
                limit=limit,
                resume=args.resume,
                stop_on_error=args.stop_on_error,
            )
        else:
            summary = run_materialize(
                start_date=start_date,
                end_date=end_date,
                lake_root=args.lake_root,
                write_target=args.write_target,
                factor=args.factor,
                limit=limit,
                batch_run=not args.no_batch_run,
                stop_on_error=args.stop_on_error,
            )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    report_path = Path(args.report) if args.report else PACKAGE_ROOT / "dsl" / "materialize_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
