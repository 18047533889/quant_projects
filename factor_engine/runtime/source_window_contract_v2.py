# -*- coding: utf-8 -*-
"""Install contract-preserving data-source window narrowing.

The legacy narrowing helper correctly changes date bounds but historically
reconstructed DataAccessSource without ``params/read_auto/lazy_scan`` and wrapped
logical sources in a generic load-column-only adapter.  This patch preserves
immutable read semantics while intentionally discarding caches and execution
state.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

_INSTALLED = False

_CONTRACT_ATTRIBUTES = (
    "full_history_start",
    "bar_freq",
    "session_open",
    "session_close",
    "session_minutes",
    "timestamp_convention",
)


def _copy_direct_contracts(source: Any, target: Any) -> None:
    params = dict(getattr(source, "params", {}) or {})
    if params:
        try:
            target.params = params
        except Exception:
            pass
    if hasattr(source, "read_auto"):
        try:
            target.read_auto = bool(source.read_auto)
        except Exception:
            pass
    if hasattr(source, "lazy_scan"):
        try:
            target._lazy_scan = bool(source.lazy_scan)
        except Exception:
            pass
    for attribute in _CONTRACT_ATTRIBUTES:
        if not hasattr(source, attribute):
            continue
        try:
            setattr(target, attribute, getattr(source, attribute))
        except Exception:
            pass


def _copy_recursive_contracts(source: Any, target: Any) -> Any:
    _copy_direct_contracts(source, target)
    source_children = getattr(source, "sources", None)
    target_children = getattr(target, "sources", None)
    if isinstance(source_children, dict) and isinstance(target_children, dict):
        for name, child in source_children.items():
            if name in target_children:
                _copy_recursive_contracts(child, target_children[name])
    return target


def _normalise_bound(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def install_source_window_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import storage.time_window as time_window

    original = time_window.narrow_data_source_for_window
    if getattr(original, "_contract_v2", False):
        _INSTALLED = True
        return

    def narrow(
        source,
        *,
        start_date=None,
        end_date=None,
        bar_freq=None,
    ):
        # Logical SourceRef resolution must remain available after narrowing.
        try:
            from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

            if isinstance(source, LQTPLogicalDataSource):
                narrowed_inner = narrow(
                    source.inner,
                    start_date=start_date,
                    end_date=end_date,
                    bar_freq=bar_freq,
                )
                output = LQTPLogicalDataSource(
                    narrowed_inner,
                    factor_freq=getattr(source, "factor_freq", "1d"),
                    factor_lake_root=getattr(source, "factor_lake_root", None),
                )
                _copy_direct_contracts(source, output)
                execution_id = getattr(source, "_execution_id", None)
                if execution_id:
                    output._execution_id = execution_id
                return output
        except ImportError:
            pass

        output = original(
            source,
            start_date=start_date,
            end_date=end_date,
            bar_freq=bar_freq,
        )
        _copy_recursive_contracts(source, output)

        # Generic WindowedDataSource has no public bounds.  Add them so cache,
        # lineage and subsequent warmup calculations see the narrowed scope.
        if output.__class__.__name__ == "WindowedDataSource":
            try:
                output.start_date = _normalise_bound(start_date) or getattr(
                    source, "start_date", None
                )
                output.end_date = _normalise_bound(end_date) or getattr(
                    source, "end_date", None
                )
                if bar_freq is not None:
                    output.bar_freq = str(bar_freq)
            except Exception:
                pass
        return output

    narrow._contract_v2 = True
    narrow.__name__ = original.__name__
    narrow.__doc__ = original.__doc__
    time_window.narrow_data_source_for_window = narrow
    _INSTALLED = True


__all__ = ["install_source_window_contract"]
