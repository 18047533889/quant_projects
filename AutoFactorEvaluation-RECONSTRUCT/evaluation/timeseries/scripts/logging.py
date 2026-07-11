"""时序模块日志初始化。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：将运行日志写入 database/log/evaluation/timeseries。
"""

from __future__ import annotations

import logging
from pathlib import Path


def build_run_logger(log_dir: Path, eval_run_id: str) -> logging.Logger:
    """为单次时序运行创建文件 logger。

    入参：
        log_dir: 运行日志目录。
        eval_run_id: 本次评估运行 ID。

    出参：
        logging.Logger：绑定到 `<log_dir>/<eval_run_id>.log` 的 logger。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"evaluation.timeseries.{eval_run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{eval_run_id}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger

