"""Parameter-grid batch backtesting built on top of the single-run engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd


_WORKER_TARGETS: dict[str, pd.DataFrame] = {}
_WORKER_MARKET: str = ""
_WORKER_SYMBOLS: Optional[list[str]] = None
_WORKER_START: Optional[str] = None
_WORKER_END: Optional[str] = None
_WORKER_KWARGS: dict[str, Any] = {}


def run_backtest(*args: Any, **kwargs: Any) -> Any:
    """Lazy, patchable compatibility boundary for the single-run engine."""
    from .runner import run_backtest as implementation

    return implementation(*args, **kwargs)


def compare_reports(*args: Any, **kwargs: Any) -> Any:
    """Lazy, patchable compatibility boundary for report comparison."""
    from .runner import compare_reports as implementation

    return implementation(*args, **kwargs)


def _set_process_data_root(market: str, data_root: Optional[str]) -> None:
    if data_root is None:
        return
    from ..data.adapter import set_data_root

    adapter_market = "ashare" if market.lower() == "ashare" else "us_stock"
    set_data_root(adapter_market, data_root)


def _initialize_batch_worker(
    targets: dict[str, pd.DataFrame],
    market: str,
    symbols: Optional[list[str]],
    start: Optional[str],
    end: Optional[str],
    kwargs: dict[str, Any],
    data_root: Optional[str],
) -> None:
    """Initialize immutable process-local state once instead of per run."""
    global _WORKER_TARGETS
    global _WORKER_MARKET
    global _WORKER_SYMBOLS
    global _WORKER_START
    global _WORKER_END
    global _WORKER_KWARGS

    _WORKER_TARGETS = targets
    _WORKER_MARKET = market
    _WORKER_SYMBOLS = symbols
    _WORKER_START = start
    _WORKER_END = end
    _WORKER_KWARGS = kwargs
    _set_process_data_root(market, data_root)


def _run_process_job(
    run_id: str,
    target_label: str,
    run_config: dict[str, Any],
) -> tuple[str, bytes, bytes]:
    """Execute one batch job inside a process-pool worker."""
    print(f"[batch/process {os.getpid()}] 开始 {run_id}")
    import dill
    portfolio = run_backtest(
        _WORKER_MARKET,
        _WORKER_TARGETS[target_label],
        symbols=_WORKER_SYMBOLS,
        start=_WORKER_START,
        end=_WORKER_END,
        config=run_config,
        **_WORKER_KWARGS,
    )
    print(f"[batch/process {os.getpid()}] 完成 {run_id}")
    portfolio_dumps = portfolio.dumps()
    qs_metadata = {
        name: value
        for name, value in vars(portfolio).items()
        if name.startswith("_qs_")
    }
    return run_id, portfolio_dumps, dill.dumps(qs_metadata)


def _restore_process_portfolio(
    portfolio_dumps: bytes,
    metadata_dumps: bytes,
) -> Any:
    import dill
    from .runner import vbt

    portfolio = vbt.Portfolio.loads(portfolio_dumps)
    for name, value in dill.loads(metadata_dumps).items():
        setattr(portfolio, name, value)
    return portfolio


def expand_parameter_grid(
    parameter_grid: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Expand an explicit parameter grid in deterministic Cartesian order."""
    if not isinstance(parameter_grid, Mapping):
        raise TypeError("parameter_grid 必须是参数名到列表的映射")
    if not parameter_grid:
        return [{}]

    keys: list[str] = []
    values: list[list[Any]] = []
    for raw_key, candidates in parameter_grid.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("batch 参数名不能为空")
        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            raise TypeError(f"batch.grid.{key} 必须是列表")
        candidates = list(candidates)
        if not candidates:
            raise ValueError(f"batch.grid.{key} 不能为空列表")
        keys.append(key)
        values.append(candidates)

    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def _set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path such as ``costs.commission`` on a copied config."""
    parts = path.split(".")
    if parts[0] == "backtest":
        parts = parts[1:]
    if not parts or any(not part for part in parts):
        raise ValueError(f"非法 batch 参数路径: {path!r}")

    cursor = config
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(
                f"无法设置 {path!r}：{part!r} 当前不是字典"
            )
        cursor = existing
    cursor[parts[-1]] = value


def apply_parameter_overrides(
    base_config: Optional[Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deep-copied config with dotted-path overrides applied."""
    config = deepcopy(dict(base_config or {}))
    for path, value in overrides.items():
        _set_nested(config, str(path), deepcopy(value))
    return config


def _normalize_targets(
    target_weights: pd.DataFrame | Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if isinstance(target_weights, pd.DataFrame):
        return {"positions": target_weights}
    if not isinstance(target_weights, Mapping) or not target_weights:
        raise TypeError(
            "target_weights 必须是 DataFrame 或非空的 {名称: DataFrame} 映射"
        )

    normalized: dict[str, pd.DataFrame] = {}
    for raw_label, weights in target_weights.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError("目标权重名称不能为空")
        if label in normalized:
            raise ValueError(f"目标权重名称重复: {label}")
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(f"目标权重 {label!r} 不是 pandas.DataFrame")
        normalized[label] = weights
    return normalized


@dataclass
class BatchBacktestResult:
    """In-memory results and exact parameter manifest for one batch."""

    portfolios: Dict[str, Any]
    parameters: pd.DataFrame
    configs: Dict[str, dict[str, Any]]
    errors: Dict[str, str]

    def compare(self) -> pd.DataFrame:
        """Return successful portfolio statistics indexed by run id."""
        if not self.portfolios:
            return pd.DataFrame(index=pd.Index([], name="run_id"))
        comparison = compare_reports(self.portfolios)
        comparison.index.name = "run_id"
        return comparison

    def performance_table(self) -> pd.DataFrame:
        """Return parameters, status and statistics in one table."""
        return self.parameters.join(self.compare(), how="left")

    def nav_curves(self) -> pd.DataFrame:
        """Return normalized strategy NAV curves for all successful runs."""
        curves: dict[str, pd.Series] = {}
        for run_id, portfolio in self.portfolios.items():
            value = portfolio.value()
            if isinstance(value, pd.DataFrame):
                value = value.sum(axis=1)
            value = pd.Series(value, dtype=float).sort_index()
            if value.empty:
                continue
            first = value.iloc[0]
            if not pd.notna(first) or first == 0:
                continue
            curves[run_id] = value / first
        result = pd.DataFrame(curves)
        result.index.name = "date"
        return result


def run_backtest_batch(
    market: str,
    target_weights: pd.DataFrame | Mapping[str, pd.DataFrame],
    parameter_grid: Mapping[str, Sequence[Any]],
    *,
    base_config: Optional[Mapping[str, Any]] = None,
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_runs: int = 256,
    on_error: str = "raise",
    workers: int = 1,
    data_root: Optional[str] = None,
    **kwargs: Any,
) -> BatchBacktestResult:
    """
    Run every target-weight/config combination in one deterministic batch.

    ``parameter_grid`` keys address backtest config fields. Dotted nested paths
    such as ``costs.commission`` are supported. A leading ``backtest.`` is
    accepted for readability and stripped before calling ``run_backtest``.
    """
    reserved_range_keys = {
        str(key).removeprefix("backtest.")
        for key in parameter_grid
    }.intersection({"start", "end"})
    if reserved_range_keys:
        raise ValueError(
            "start/end 是批次级日期边界，不能放入 parameter_grid；"
            "请使用 run_backtest_batch(start=..., end=...)"
        )
    targets = _normalize_targets(target_weights)
    combinations = expand_parameter_grid(parameter_grid)
    total_runs = len(targets) * len(combinations)
    if not isinstance(max_runs, int) or max_runs <= 0:
        raise ValueError("max_runs 必须是正整数")
    if total_runs > max_runs:
        raise ValueError(
            f"batch 将生成 {total_runs} 次回测，超过 max_runs={max_runs}"
        )
    on_error = str(on_error).lower()
    if on_error not in {"raise", "continue"}:
        raise ValueError("on_error 仅支持 raise 或 continue")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers 必须是正整数")

    portfolios_by_id: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    run_number = 0

    for target_label, weights in targets.items():
        for overrides in combinations:
            run_number += 1
            run_id = f"run_{run_number:04d}"
            run_config = apply_parameter_overrides(base_config, overrides)
            configs[run_id] = deepcopy(run_config)
            row = {
                "run_id": run_id,
                "positions": target_label,
                **overrides,
                "status": "success",
                "error": "",
            }
            rows_by_id[run_id] = row
            jobs.append((run_id, target_label, run_config))
            print(
                f"\n{'=' * 60}\n"
                f"  Batch {run_number}/{total_runs}: {run_id}"
                f" ({'queued' if workers > 1 else 'serial'})\n"
                f"  positions={target_label}, parameters={overrides}\n"
                f"{'=' * 60}"
            )

    data_root_text = str(data_root) if data_root is not None else None
    if workers == 1:
        _set_process_data_root(market, data_root_text)
        for run_id, target_label, run_config in jobs:
            row = rows_by_id[run_id]
            try:
                portfolios_by_id[run_id] = run_backtest(
                    market,
                    targets[target_label],
                    symbols=symbols,
                    start=start,
                    end=end,
                    config=run_config,
                    **kwargs,
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors[run_id] = row["error"]
                if on_error == "raise":
                    raise
                print(f"[batch] {run_id} 失败，继续下一组: {row['error']}")
    else:
        effective_workers = min(workers, total_runs)
        print(
            f"[batch] 启用多进程: workers={effective_workers}, "
            f"jobs={total_runs}"
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_initialize_batch_worker,
            initargs=(
                targets,
                market,
                symbols,
                start,
                end,
                dict(kwargs),
                data_root_text,
            ),
        ) as executor:
            future_to_job = {
                executor.submit(
                    _run_process_job,
                    run_id,
                    target_label,
                    run_config,
                ): (run_id, target_label)
                for run_id, target_label, run_config in jobs
            }
            for future in as_completed(future_to_job):
                run_id, _ = future_to_job[future]
                row = rows_by_id[run_id]
                try:
                    completed_id, portfolio_dumps, metadata_dumps = future.result()
                    portfolios_by_id[completed_id] = _restore_process_portfolio(
                        portfolio_dumps,
                        metadata_dumps,
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    errors[run_id] = row["error"]
                    if on_error == "raise":
                        for pending in future_to_job:
                            pending.cancel()
                        raise
                    print(
                        f"[batch] {run_id} 失败，继续其他进程: "
                        f"{row['error']}"
                    )

    portfolios = {
        run_id: portfolios_by_id[run_id]
        for run_id, _, _ in jobs
        if run_id in portfolios_by_id
    }
    rows = [rows_by_id[run_id] for run_id, _, _ in jobs]
    parameters = pd.DataFrame(rows).set_index("run_id")
    parameters.index.name = "run_id"
    return BatchBacktestResult(
        portfolios=portfolios,
        parameters=parameters,
        configs=configs,
        errors=errors,
    )
