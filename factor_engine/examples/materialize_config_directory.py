"""按配置目录批量执行因子并落盘。

运行示例
--------
::

    python examples/materialize_config_directory.py examples/configs/day_aggs_v1_price_volume
    python examples/materialize_config_directory.py examples/configs/day_aggs_v1_price_volume --lake-root ~/quant_projects/data/factors/lake --author whh
    python examples/materialize_config_directory.py examples/configs/day_aggs_v1_price_volume --log-file /tmp/factor_engine/materialize.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logging_utils import ProgressLogger, configure_logging, get_logger
from factor_engine.runtime.engine import FactorEngine


logger = get_logger("examples.materialize_config_directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize all YAML configs in a directory.")
    parser.add_argument("config_dir", type=Path, help="Directory containing YAML configs")
    parser.add_argument("--pattern", default="*.yaml", help="Glob pattern to select config files")
    parser.add_argument("--lake-root", type=Path, default=None, help="Override lake root for all configs")
    parser.add_argument("--author", default=None, help="Override author for all configs")
    parser.add_argument("--log-level", default="INFO", help="Log level for factor_engine logs")
    parser.add_argument("--log-file", type=Path, default=None, help="Write factor_engine logs to the specified file")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one config fails; default is continue and summarize failures.",
    )
    return parser.parse_args()


def materialize_config_directory(
    config_dir: Path,
    *,
    pattern: str = "*.yaml",
    lake_root: Path | None = None,
    author: str | None = None,
    stop_on_error: bool = False,
) -> dict:
    config_paths = sorted(path for path in config_dir.glob(pattern) if path.is_file())
    if not config_paths:
        raise FileNotFoundError(f"No config files matched under {config_dir} with pattern {pattern!r}")

    logger.info("开始批量物化配置目录: %s，匹配到 %d 个配置", config_dir, len(config_paths))

    summary: dict[str, object] = {
        "config_dir": str(config_dir),
        "config_count": len(config_paths),
        "succeeded": [],
        "failed": [],
    }
    progress = ProgressLogger(
        logger,
        desc="配置目录物化",
        total=len(config_paths),
        unit="config",
    )

    for path in config_paths:
        try:
            logger.info("开始处理配置: %s", path)
            out = FactorEngine.materialize_from_config(
                path,
                lake_root=lake_root,
                author=author,
            )
            materialization = out["materialization"]
            summary["succeeded"].append(
                {
                    "config": path.name,
                    "factor_id": materialization["factor_id"],
                    "rows_written": materialization["rows_written"],
                    "partitions": materialization["partitions"],
                    "lake_root": materialization["lake_root"],
                }
            )
            progress.advance(detail=f"{path.name} -> {materialization['factor_id']}")
        except Exception as exc:
            failure = {"config": path.name, "error": str(exc)}
            summary["failed"].append(failure)
            logger.exception("处理配置失败: %s", path)
            progress.advance(detail=f"{path.name} -> failed")
            if stop_on_error:
                raise

    logger.info(
        "配置目录物化完成: succeeded=%d, failed=%d",
        len(summary["succeeded"]),
        len(summary["failed"]),
    )
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, log_file=args.log_file)
    summary = materialize_config_directory(
        args.config_dir,
        pattern=args.pattern,
        lake_root=args.lake_root,
        author=args.author,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()