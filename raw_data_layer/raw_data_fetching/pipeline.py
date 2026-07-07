from __future__ import annotations

from typing import Any, Callable


def _load_download_all_history_runner() -> Callable[..., dict[str, Any]]:
    from raw_data_layer.raw_data_fetching.download_all_history import run_download_all_history

    return run_download_all_history


def _load_download_history_runner() -> Callable[..., dict[str, Any]]:
    from raw_data_layer.raw_data_fetching.download_history import run_download_history

    return run_download_history


def _load_validate_parquet_runner() -> Callable[..., dict[str, Any]]:
    from raw_data_layer.raw_data_fetching.validate_parquet import validate_directory

    return validate_directory


def download_all_history(*, emit_logs: bool = False, **kwargs: Any) -> dict[str, Any]:
    runner = _load_download_all_history_runner()
    return runner(emit_logs=emit_logs, **kwargs)


def download_history(*, emit_logs: bool = False, **kwargs: Any) -> dict[str, Any]:
    runner = _load_download_history_runner()
    return runner(emit_logs=emit_logs, **kwargs)


def validate_parquet(directory: str, *, workers: int, emit_logs: bool = False) -> dict[str, Any]:
    runner = _load_validate_parquet_runner()
    return runner(directory, workers=workers, emit_logs=emit_logs)


__all__ = ["download_all_history", "download_history", "validate_parquet"]