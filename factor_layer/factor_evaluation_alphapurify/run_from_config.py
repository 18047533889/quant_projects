from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from factor_layer.factor_evaluation_alphapurify.pipeline import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run alphapurify Database -> Exposures -> FactorAnalyzer pipeline from YAML config."
    )
    parser.add_argument("config", type=Path, help="Path to alphapurify YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_from_config(args.config)
    print(
        json.dumps(
            {
                "output_root": result["output_root"],
                "run_summary_path": result["run_summary_path"],
                "summary": result["summary"],
                "config_snapshot": result["config_snapshot"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
