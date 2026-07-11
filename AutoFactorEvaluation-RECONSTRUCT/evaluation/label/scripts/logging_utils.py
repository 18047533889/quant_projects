"""evaluation.label 的日志工具。

模块: evaluation.label
职责:
1. 将模块 logger 绑定到配置指定的日志目录。
2. 保持日志初始化逻辑集中，避免各脚本重复配置 handler。
"""

from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "evaluation.label"
LOG_FILE_NAME = "label_pipeline.log"


def configure_label_logger(log_dir: str | Path) -> logging.Logger:
    """初始化并返回 evaluation.label 模块 logger。

    参数:
        log_dir: LabelModuleConfig.log_dir 指向的日志目录。

    返回:
        logging.Logger: 已绑定文件 handler 的模块 logger。
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = (log_path / LOG_FILE_NAME).resolve()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_file:
                return logger

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(file_handler)
    return logger


def get_label_logger() -> logging.Logger:
    """返回 evaluation.label 模块 logger。"""

    return logging.getLogger(LOGGER_NAME)
