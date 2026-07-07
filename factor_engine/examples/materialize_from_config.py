"""通过 YAML 配置执行因子并直接落盘。

运行示例
--------
::

    python examples/materialize_from_config.py examples/config_driven_materialize_factor.yaml
    python examples/materialize_from_config.py examples/config_driven_materialize_factor.yaml --lake-root /tmp/factor_lake
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logging_utils import configure_logging
from runtime.engine import FactorEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a factor config and materialize the result.")
    parser.add_argument("config", type=Path, help="YAML config path")
    parser.add_argument("--lake-root", type=Path, default=None, help="Override lake root")
    parser.add_argument("--factor-id", default=None, help="Override factor_id for materialization")
    parser.add_argument("--author", default=None, help="Override author for materialization")
    parser.add_argument("--frequency", default=None, help="Override frequency for materialization")
    parser.add_argument("--description", default=None, help="Override description for materialization")
    parser.add_argument("--expression", default=None, help="Override stored expression string")
    parser.add_argument("--log-level", default="INFO", help="Log level for factor_engine logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    output = FactorEngine.materialize_from_config(
        args.config,
        lake_root=args.lake_root,
        factor_id=args.factor_id,
        author=args.author,
        frequency=args.frequency,
        description=args.description,
        expression=args.expression,
    )
    summary = output["materialization"]
    print(
        json.dumps(
            {
                "factor_name": output["factor"].name,
                "factor_id": summary["factor_id"],
                "rows_written": summary["rows_written"],
                "partitions": summary["partitions"],
                "lake_root": summary["lake_root"],
                "watermark": summary["watermark"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()