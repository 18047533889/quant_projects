"""factor_engine 日志：控制台/文件输出与无 tqdm 的进度条。"""

from __future__ import annotations

import logging
import os
import time


def _resolve_level(level: str | int | None) -> int:
    """把字符串级别（如 ``INFO``）或整数转为 ``logging`` 常量。"""
    if level is None:
        level = os.getenv("FACTOR_ENGINE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def _build_formatter() -> logging.Formatter:
    """统一日志格式：时间 + 级别 + logger 名 + 消息。"""
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _has_file_handler(logger: logging.Logger, log_file: str) -> bool:
    """检查 logger 是否已挂载指向同一绝对路径的 FileHandler（避免重复添加）。"""
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
    """为 ``factor_engine`` 命名空间配置最小可用日志输出。

    参数：
        level: 日志级别；默认读 ``FACTOR_ENGINE_LOG_LEVEL`` 或 INFO
        log_file: 若指定，追加 FileHandler（UTF-8）

    返回：
        名为 ``factor_engine`` 的根 logger
    """
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
    """获取子 logger；短名自动加 ``factor_engine.`` 前缀。"""
    if name.startswith("factor_engine"):
        return logging.getLogger(name)
    return logging.getLogger(f"factor_engine.{name}")


class ProgressLogger:
    """把进度条渲染成普通日志行，避免额外依赖 tqdm。"""

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
        """参数：
            desc: 进度描述前缀
            total: 总步数
            unit: 步单位文案（如 ``factor``）
            log_every: 每隔多少步打一条；默认 total/10
        """
        self.logger = logger
        self.desc = desc
        self.total = None if total is None else max(int(total), 0)
        self.unit = unit
        self.width = max(int(width), 8)
        self.level = level
        self.current = 0
        self.started_at = time.perf_counter()
        self.log_every = log_every or self._default_log_every()
        self._last_logged = -1
        self.log(force=True)

    def _default_log_every(self) -> int:
        """总步数少时每步都打；否则约 10 次日志。``total=None`` 时每步都打。"""
        if self.total is None:
            return 1
        if self.total <= 10:
            return 1
        return max(1, self.total // 10)

    def _ratio(self) -> float:
        """当前完成比例 [0, 1]；``total=None``（indeterminate）返回 0。"""
        if self.total is None or self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)

    def _bar(self) -> str:
        """ASCII 进度条，如 ``[####------]``。"""
        ratio = self._ratio()
        filled = int(round(ratio * self.width))
        filled = min(self.width, max(0, filled))
        return f"[{'#' * filled}{'-' * (self.width - filled)}]"

    def log(self, *, detail: str | None = None, force: bool = False) -> None:
        """输出一行进度；非 force 时按 ``log_every`` 节流。"""
        if not force and self.current != self.total:
            if self.current == self._last_logged:
                return
            if self.current > 0 and self.current % self.log_every != 0:
                return

        elapsed = time.perf_counter() - self.started_at
        total_str = "?" if self.total is None else str(self.total)
        message = (
            f"{self.desc} {self._bar()} {self.current}/{total_str} {self.unit} "
            f"({self._ratio():.1%}, {elapsed:.2f}s)"
        )
        if detail:
            message = f"{message} - {detail}"
        self.logger.log(self.level, message)
        self._last_logged = self.current

    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        """前进一步并尝试打日志；完成时 force 输出。``total=None`` 时不截断。"""
        if self.total is None:
            self.current += max(int(step), 0)
        else:
            self.current = min(self.total, self.current + step)
        done = self.total is not None and self.current >= self.total
        self.log(detail=detail, force=done)

    def finish(self, *, detail: str | None = None) -> None:
        """标记完成并打最终日志。"""
        if self.total is not None:
            self.current = self.total
        self.log(detail=detail, force=True)
