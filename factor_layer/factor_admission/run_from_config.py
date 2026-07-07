from __future__ import annotations

import argparse
from pathlib import Path
import sys

_MONOREPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))

from factor_layer.factor_admission.pipeline import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor admission from YAML config.")
    parser.add_argument("config", type=Path, help="Path to factor admission YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_from_config(args.config)
    result = output["results"][0]
    print(f"factor_id={result['factor_id']} run_id={result['run_id']} decision={result['decision']}")


if __name__ == "__main__":
    main()