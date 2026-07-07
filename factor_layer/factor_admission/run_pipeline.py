from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_MONOREPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))

from factor_layer.factor_admission.pipeline import run_config_directory, run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor_admission pipeline from a config file or a config directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Run admission pipeline for a single YAML config")
    config_parser.add_argument("config_path", type=Path, help="YAML config path")
    config_parser.add_argument("--output-root", type=Path, default=None, help="Override pipeline output root")

    config_dir_parser = subparsers.add_parser("config-dir", help="Run admission pipeline for all YAML configs in a directory")
    config_dir_parser.add_argument("config_dir", type=Path, help="Directory containing YAML configs")
    config_dir_parser.add_argument("--pattern", default="*.yaml", help="Glob pattern to select config files")
    config_dir_parser.add_argument("--output-root", type=Path, default=None, help="Override pipeline output root")
    config_dir_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one config fails; default is continue and summarize failures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "config":
        output = run_from_config(args.config_path, output_root=args.output_root)
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