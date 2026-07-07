from __future__ import annotations

import logging
import os
import time


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("FACTOR_ENGINE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _has_file_handler(logger: logging.Logger, log_file: str) -> bool:
    target = os.path.abspath(log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if getattr(handler, "baseFilename", None) == target:
                return True
    return False


def configure_logging(
    level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> logging.Logger:
    """为 factor_engine 命名空间配置一个最小可用日志输出。"""
    logger = logging.getLogger("factor_engine")
    logger.setLevel(_resolve_level(level))
    formatter = _build_formatter()

    root_logger = logging.getLogger()
    if not root_logger.handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    elif root_logger.handlers:
        logger.propagate = True

    if log_file is not None:
        resolved_path = os.fspath(log_file)
        os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
        if not _has_file_handler(logger, resolved_path):
            file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    if name.startswith("factor_engine"):
        return logging.getLogger(name)
    return logging.getLogger(f"factor_engine.{name}")


class ProgressLogger:
    """把进度条渲染成普通日志行，避免额外依赖。"""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        desc: str,
        total: int,
        unit: str = "step",
        width: int = 20,
        log_every: int | None = None,
        level: int = logging.INFO,
    ) -> None:
        self.logger = logger
        self.desc = desc
        self.total = max(int(total), 0)
        self.unit = unit
        self.width = max(int(width), 8)
        self.level = level
        self.current = 0
        self.started_at = time.perf_counter()
        self.log_every = log_every or self._default_log_every()
        self._last_logged = -1
        self.log(force=True)

    def _default_log_every(self) -> int:
        if self.total <= 10:
            return 1
        return max(1, self.total // 10)

    def _ratio(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(1.0, self.current / self.total)

    def _bar(self) -> str:
        ratio = self._ratio()
        filled = int(round(ratio * self.width))
        filled = min(self.width, max(0, filled))
        return f"[{'#' * filled}{'-' * (self.width - filled)}]"

    def log(self, *, detail: str | None = None, force: bool = False) -> None:
        if not force and self.current != self.total:
            if self.current == self._last_logged:
                return
            if self.current > 0 and self.current % self.log_every != 0:
                return

        elapsed = time.perf_counter() - self.started_at
        message = (
            f"{self.desc} {self._bar()} {self.current}/{self.total} {self.unit} "
            f"({self._ratio():.1%}, {elapsed:.2f}s)"
        )
        if detail:
            message = f"{message} - {detail}"
        self.logger.log(self.level, message)
        self._last_logged = self.current

    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        self.current = min(self.total, self.current + step)
        self.log(detail=detail, force=self.current >= self.total)

    def finish(self, *, detail: str | None = None) -> None:
        self.current = self.total
        self.log(detail=detail, force=True)