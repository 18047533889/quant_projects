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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, log_file=args.log_file)

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