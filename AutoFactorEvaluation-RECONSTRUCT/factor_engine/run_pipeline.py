from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logging_utils import configure_logging
from pipeline import run_config_directory, run_from_config


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

    if args.command == "config":
        output = run_from_config(
            args.config_path,
            output_root=args.output_root,
            materialize=args.materialize,
            preview_rows=args.preview_rows,
        )
        payload = {
            "mode": "config",
            "output_root": output["output_root"],
            "run_summary_path": output["run_summary_path"],
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
        )
        payload = {
            "mode": "config-dir",
            "output_root": output["output_root"],
            "run_summary_path": output["run_summary_path"],
            "config_snapshots": output.get("config_snapshots", []),
            "summary": output["summary"],
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()