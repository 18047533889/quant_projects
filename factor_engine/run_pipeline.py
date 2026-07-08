from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logging_utils import configure_logging, get_logger
from pipeline import run_config_directory, run_from_config

logger = get_logger("run_pipeline")


def _run_reconcile(args: argparse.Namespace) -> dict:
    if args.reconcile_command == "dual-write":
        from runtime.dual_write_reconcile import (
            list_open_dual_write_failures,
            reconcile_all_dual_write_states,
            reconcile_dual_write_state,
            repair_dual_write_clickhouse,
        )

        lake = args.lake_root
        compensate_staging = not args.no_compensate_staging
        if args.factor_id:
            if args.repair:
                return repair_dual_write_clickhouse(
                    factor_id=args.factor_id,
                    lake_root=lake,
                    clickhouse_table=args.clickhouse_table,
                    prefer_source=args.prefer_source,
                    compensate_staging=compensate_staging,
                    compensate_after=args.compensate_after,
                )
            return reconcile_dual_write_state(factor_id=args.factor_id, lake_root=lake)

        failures = list_open_dual_write_failures(
            lake_root=lake,
            limit=args.limit,
        )
        if args.repair:
            results = []
            ok = True
            for row in failures:
                fid = row["factor_id"]
                if not fid:
                    continue
                out = repair_dual_write_clickhouse(
                    factor_id=str(fid),
                    lake_root=lake,
                    clickhouse_table=args.clickhouse_table,
                    prefer_source=args.prefer_source,
                    compensate_staging=compensate_staging,
                    compensate_after=args.compensate_after,
                )
                results.append(out)
                ok = ok and out.get("ok", False)
            return {"ok": ok, "mode": "repair_batch", "results": results}

        if failures:
            return reconcile_all_dual_write_states(lake_root=lake, limit=args.limit)
        return {"ok": True, "lake_root": str(lake), "factors_with_failures": 0, "reports": []}

    if args.reconcile_command == "snapshot":
        from runtime.config import load_config
        from runtime.config_runtime import build_data_source_config
        from runtime.snapshot_reconcile import reconcile_data_snapshot

        data_source_config = None
        if args.config is not None:
            cfg = load_config(args.config)
            data_source_config = build_data_source_config(cfg)
        return reconcile_data_snapshot(
            factor_id=args.factor_id,
            lake_root=args.lake_root,
            data_source_config=data_source_config,
        )

    raise SystemExit(f"Unknown reconcile subcommand: {args.reconcile_command}")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=None, help="Override pipeline output root")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--materialize",
        dest="materialize",
        action="store_true",
        help="Force materialize mode even if the config does not specify it",
    )
    mode_group.add_argument(
        "--run-only",
        dest="materialize",
        action="store_false",
        help="Force run mode even if the config contains materialization settings",
    )
    parser.set_defaults(materialize=None)
    parser.add_argument("--preview-rows", type=int, default=5, help="Preview rows to keep in result JSON")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental materialize: read watermark, load lookback window, upsert tail",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Incremental lower bound (override watermark / config incremental.since)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Incremental upper bound (override config incremental.end_date)",
    )
    parser.add_argument(
        "--lookback-extra",
        type=int,
        default=None,
        help="Extra lookback bars beyond IR analysis lookback (default from config or 5)",
    )
    parser.add_argument(
        "--recompute-tail-bars",
        type=int,
        default=None,
        help="Bars to recompute from watermark backward (default: analysis lookback + 1)",
    )
    parser.add_argument(
        "--strict-dq",
        action="store_true",
        help="Enable data quality gates; fail the run on DQ violations",
    )
    parser.add_argument(
        "--input-dq",
        action="store_true",
        help="Check input column coverage before factor execution",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry failed config runs up to N times (default 0)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Merge examples/profiles/{profile}.yaml before config (dev/staging/prod)",
    )
    parser.add_argument(
        "--resume-materialize",
        action="store_true",
        help="Resume materialize from partition checkpoints (skip successful years)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for config-dir mode (default 1 = serial)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index for multi-host config-dir dispatch (0-based)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count when using --shard-index",
    )
    parser.add_argument(
        "--push-otlp-endpoint",
        default=None,
        help="Optional OTLP/HTTP endpoint, e.g. http://127.0.0.1:4318",
    )
    parser.add_argument(
        "--no-strict-dq",
        dest="dq_strict",
        action="store_false",
        help="With --strict-dq: log DQ violations instead of failing the run",
    )
    parser.set_defaults(dq_strict=True)
    parser.add_argument(
        "--write-target",
        default=None,
        choices=["local", "staging", "both", "clickhouse", "staging_clickhouse"],
        help="Override materialization.target from config/profile",
    )
    parser.add_argument(
        "--preserve-invalid-rows",
        dest="preserve_invalid_rows",
        action="store_true",
        help="Keep inf/NaN rows with is_valid=0 in factor lake output",
    )
    parser.add_argument(
        "--no-preserve-invalid-rows",
        dest="preserve_invalid_rows",
        action="store_false",
        help="Drop invalid rows during materialize (overrides config)",
    )
    parser.set_defaults(preserve_invalid_rows=None)
    parser.add_argument("--log-level", default="INFO", help="Log level for factor_engine logs")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional file path for logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor_engine pipeline from a config file or config directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Run pipeline for a single YAML config")
    config_parser.add_argument("config_path", type=Path, help="YAML config path")
    _add_common_options(config_parser)

    config_dir_parser = subparsers.add_parser("config-dir", help="Run pipeline for all YAML configs in a directory")
    config_dir_parser.add_argument("config_dir", type=Path, help="Directory containing YAML configs")
    config_dir_parser.add_argument("--pattern", default="*.yaml", help="Glob pattern to select config files")
    config_dir_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one config fails; default is continue and summarize failures.",
    )
    _add_common_options(config_dir_parser)

    reconcile_parser = subparsers.add_parser("reconcile", help="对账与双写修复")
    reconcile_sub = reconcile_parser.add_subparsers(dest="reconcile_command", required=True)

    dw_parser = reconcile_sub.add_parser("dual-write", help="检查 staging+ClickHouse 双写失败")
    dw_parser.add_argument("--lake-root", type=Path, required=True, help="因子湖根目录")
    dw_parser.add_argument("--factor-id", default=None, help="单个因子 ID；省略则扫描全部失败")
    dw_parser.add_argument("--repair", action="store_true", help="从 local/staging 补写 ClickHouse")
    dw_parser.add_argument(
        "--no-compensate-staging",
        action="store_true",
        help="repair 时不删除 staging 中 watermark 之后的脏行（默认会补偿删除）",
    )
    dw_parser.add_argument(
        "--compensate-after",
        default=None,
        help="staging 行级删除下界（默认 watermark end_date）",
    )
    dw_parser.add_argument(
        "--prefer-source",
        choices=["local", "staging"],
        default=None,
        help="repair 时优先数据源（默认 local → staging 回退）",
    )
    dw_parser.add_argument("--clickhouse-table", default="factor_values")
    dw_parser.add_argument("--limit", type=int, default=50)

    snap_parser = reconcile_sub.add_parser("snapshot", help="data_snapshot_id 对账")
    snap_parser.add_argument("--lake-root", type=Path, required=True)
    snap_parser.add_argument("--factor-id", required=True)
    snap_parser.add_argument("--config", type=Path, default=None, help="可选 YAML 以比对 config hash")

    queue_parser = subparsers.add_parser("queue", help="分布式任务队列 worker")
    queue_sub = queue_parser.add_subparsers(dest="queue_command", required=True)
    worker_parser = queue_sub.add_parser("worker", help="消费 file/redis 任务队列")
    worker_parser.add_argument(
        "--backend",
        choices=["file", "redis", "object_store"],
        default="file",
        help="队列 backend（file / redis / object_store；queue-root 为 s3:// 时自动 object_store）",
    )
    worker_parser.add_argument("--queue-root", type=Path, default=None, help="file 队列根目录")
    worker_parser.add_argument("--redis-url", default=None, help="Redis URL（backend=redis）")
    worker_parser.add_argument("--max-jobs", type=int, default=0, help="最多处理 N 个任务；0=无限")
    _add_common_options(worker_parser)

    enqueue_parser = queue_sub.add_parser("enqueue", help="将 config 目录批量入队")
    enqueue_parser.add_argument("config_dir", type=Path, help="YAML 配置目录")
    enqueue_parser.add_argument("--pattern", default="*.yaml")
    enqueue_parser.add_argument(
        "--backend",
        choices=["file", "redis", "object_store"],
        default="file",
    )
    enqueue_parser.add_argument("--queue-root", type=Path, default=None)
    enqueue_parser.add_argument("--redis-url", default=None)

    for p in (reconcile_parser, dw_parser, snap_parser):
        p.add_argument("--log-level", default="INFO")
        p.add_argument("--log-file", type=Path, default=None)
    queue_parser.add_argument("--log-level", default="INFO")
    queue_parser.add_argument("--log-file", type=Path, default=None)

    return parser.parse_args()


def _build_queue_from_args(args: argparse.Namespace):
    from runtime.task_queue import build_task_queue

    if args.backend == "redis":
        return build_task_queue(
            backend="redis",
            redis_url=args.redis_url or "redis://127.0.0.1:6379/0",
        )
    if args.backend == "object_store" or (
        args.queue_root is not None and "://" in str(args.queue_root)
    ):
        if args.queue_root is None:
            raise SystemExit("object_store backend 需要 --queue-root（如 s3://bucket/prefix/queue）")
        return build_task_queue(backend="object_store", root=args.queue_root)
    if args.queue_root is None:
        raise SystemExit("--queue-root required for file backend")
    return build_task_queue(backend="file", root=args.queue_root)


def _run_queue_enqueue(args: argparse.Namespace) -> dict:
    from runtime.task_queue import enqueue_config_directory

    queue = _build_queue_from_args(args)
    jobs = enqueue_config_directory(queue, args.config_dir, pattern=args.pattern)
    return {"ok": True, "enqueued": len(jobs), "jobs": [j.job_id for j in jobs]}


def _run_queue_worker(args: argparse.Namespace) -> dict:
    queue = _build_queue_from_args(args)

    processed = 0
    results: list[dict] = []
    while True:
        if args.max_jobs and processed >= args.max_jobs:
            break
        job = queue.claim()
        if job is None:
            break
        try:
            out = run_from_config(
                Path(job.config_path),
                materialize=True,
                profile=args.profile,
                dq_check=args.strict_dq,
                dq_strict=args.dq_strict,
                incremental=args.incremental,
            )
            queue.complete(job.job_id, result={"ok": True, "summary": out})
            results.append({"job_id": job.job_id, "ok": True})
        except Exception as exc:
            queue.fail(job.job_id, error=str(exc))
            results.append({"job_id": job.job_id, "ok": False, "error": str(exc)})
        processed += 1

    return {"ok": all(r.get("ok") for r in results), "processed": processed, "results": results}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, log_file=args.log_file)

    if args.command == "reconcile":
        payload = _run_reconcile(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload.get("ok", True):
            raise SystemExit(1)
        return

    if args.command == "queue":
        if args.queue_command == "enqueue":
            payload = _run_queue_enqueue(args)
        elif args.queue_command == "worker":
            payload = _run_queue_worker(args)
        else:
            raise SystemExit("unknown queue subcommand")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload.get("ok", True):
            raise SystemExit(1)
        return

    if args.incremental and args.materialize is False:
        raise SystemExit("--incremental requires materialize mode (omit --run-only)")

    if args.command == "config":
        output = run_from_config(
            args.config_path,
            output_root=args.output_root,
            materialize=args.materialize,
            preview_rows=args.preview_rows,
            incremental=args.incremental,
            dq_check=args.strict_dq,
            dq_strict=args.dq_strict,
            input_dq_check=args.input_dq,
            input_dq_strict=args.dq_strict,
            max_retries=args.max_retries,
            profile=args.profile,
            resume_materialize=args.resume_materialize,
            write_target=args.write_target,
            preserve_invalid_rows=args.preserve_invalid_rows,
            since=args.since,
            end_date=args.end_date,
            lookback_extra=args.lookback_extra,
            recompute_tail_bars=args.recompute_tail_bars,
        )
        payload = {
            "mode": "config",
            "output_root": output["output_root"],
            "run_summary_path": output["run_summary_path"],
            "metrics_path": output.get("metrics_path"),
            "config_snapshot": output.get("config_snapshot"),
            "summary": output["summary"],
        }
    else:
        output = run_config_directory(
            args.config_dir,
            pattern=args.pattern,
            output_root=args.output_root,
            materialize=args.materialize,
            preview_rows=args.preview_rows,
            stop_on_error=args.stop_on_error,
            incremental=args.incremental,
            dq_check=args.strict_dq,
            dq_strict=args.dq_strict,
            input_dq_check=args.input_dq,
            input_dq_strict=args.dq_strict,
            max_retries=args.max_retries,
            profile=args.profile,
            resume_materialize=args.resume_materialize,
            n_jobs=args.n_jobs,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            otlp_endpoint=args.push_otlp_endpoint,
            write_target=args.write_target,
            preserve_invalid_rows=args.preserve_invalid_rows,
            since=args.since,
            end_date=args.end_date,
            lookback_extra=args.lookback_extra,
            recompute_tail_bars=args.recompute_tail_bars,
        )
        payload = {
            "mode": "config-dir",
            "output_root": output["output_root"],
            "run_summary_path": output["run_summary_path"],
            "metrics_path": output.get("metrics_path"),
            "prometheus_metrics_path": output.get("prometheus_metrics_path"),
            "otlp_metrics_path": output.get("otlp_metrics_path"),
            "config_snapshots": output.get("config_snapshots", []),
            "summary": output["summary"],
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()