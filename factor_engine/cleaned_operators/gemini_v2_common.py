# -*- coding: utf-8 -*-
"""Shared helpers for the 2026-08-08 Gemini V2 operator round.

Every new operator in this round follows the same engineering contract:

* strict-PIT — only the current row and strictly earlier rows;
* deterministic — no random jitter, no Monte-Carlo, no seeded noise;
* shape-preserving pandas kernel (date x instrument) that maps each column /
  row independently (``NaN`` fail-closed);
* dual backend — a ``pandas_numpy`` kernel plus a ``polars`` bridge that
  converts ``pl.DataFrame`` -> pandas, calls the same kernel and rebuilds the
  ``pl.DataFrame`` (single source of truth -> bitwise parity);
* trailing-window NaN policy — ``trailing_contiguous_finite`` never compresses
  a missing value out of the time axis (a gap must not shorten a window and
  re-pair data that were never adjacent), which is the prefix-invariance and
  future-randomisation contract all new tests assert.

``register_dual`` binds the kernel as a *default argument* of ``calculate``
(``_fn=fn``) — a bare closure over the loop variable is late-bound and every
operator would call the last kernel.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.rolling_pack import frame_like

_SKIP = frozenset({"date", "stock_code"})

_EPS = 1e-12


def trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Longest trailing contiguous finite run (never re-connects across a gap).

    A missing value invalidates everything before it for the current row: the
    time structure is preserved and a gap never re-pairs data that were not
    actually adjacent.  Used by all trailing-window kernels in this round.
    """
    n = chunk.shape[0]
    end = n
    while end > 0 and not np.isfinite(chunk[end - 1]):
        end -= 1
    start = end
    while start > 0 and np.isfinite(chunk[start - 1]):
        start -= 1
    return chunk[start:end]


def trailing_contiguous_multi(*chunks: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """Longest trailing run where *every* series is finite, then return the run.

    Returns ``None`` when the current row is not finite in any input.  Used by
    multi-input operators (EDGE / Abdi-Ranaldo / cross-spectral) where one
    missing panel invalidates the aligned pair.
    """
    n = int(chunks[0].shape[0])
    end = n
    for _ in range(end):
        ok = True
        for ch in chunks:
            if not np.isfinite(ch[end - 1]):
                ok = False
                break
        if ok:
            break
        end -= 1
    if end <= 0:
        return None
    start = end
    for _ in range(start):
        ok = True
        for ch in chunks:
            if not np.isfinite(ch[start - 1]):
                ok = False
                break
        if not ok:
            break
        start -= 1
    return tuple(ch[start:end] for ch in chunks)


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        out.append(frame)
    return tuple(out)


def register_dual(
    canonical: str,
    fn: Callable[..., pd.DataFrame],
    params: Sequence[str],
    *,
    category: str,
    domain: str,
    unit: str,
    cost: int,
    source: str = "gemini_v2",
    tags_extra: Sequence[str] = (),
    input_units: Mapping[str, str] | None = None,
    output_unit: str | None = None,
    # R4-95: optional window-semantics label (meaning of the window-like params),
    # forwarded to the pandas OperatorMetadata.  Backward-compatible: None keeps
    # the previous behaviour for the other register_dual callers.
    window_semantics: str | None = None,
) -> None:
    """Register ``fn`` under ``canonical`` for both pandas_numpy and polars.

    The polars backend runs the *same* kernel (converted frame), guaranteeing
    numerical parity; it is registered ``backend_explicit`` so production
    hardening keeps it.
    """
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata

    tag_list = [
        "daily", "panel", "pit_safe", "causal", "deterministic",
        f"signature:{','.join(params)}->series",
        f"domain:{domain}", f"unit:{unit}", f"cost:{cost}",
        *tags_extra,
    ]
    units = dict(input_units) if input_units else None

    class _PandasOp(PandasOperator):
        metadata = PandasMetadata(
            name=canonical,
            category=category,
            description=canonical,
            examples=[],
            param_names=list(params),
            return_type="series",
            tags=tag_list,
            input_units=units,
            output_unit=output_unit,
            window_semantics=window_semantics,
        )

        def calculate(self, *args, _fn=fn, **kwargs):
            return _fn(*args, **kwargs)

    OperatorRegistry.register(
        _PandasOp(), canonical=canonical, backend="pandas_numpy",
        source=source, backend_explicit=True,
    )

    class _PolarsOp(PolarsSeriesOperator):
        metadata = PolarsMetadata(name=canonical, category=category, param_names=[])

        def _calculate_series(self, *frames, _fn=fn, **params):
            import polars as pl  # noqa: F401
            pdfs = [
                f.select([c for c in f.columns if c not in _SKIP]).to_pandas()
                for f in frames
            ]
            out = _fn(*pdfs, **params)
            base = frames[0]
            cols = [c for c in base.columns if c not in _SKIP]
            return base.with_columns(
                [pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols]
            )

    OperatorRegistry.register(
        _PolarsOp(), canonical=canonical, backend="polars",
        source=source + "_polars", backend_explicit=True,
    )


def union_extended(*canonicals: str) -> None:
    """Partition contract: register each name in EXTENDED_ONLY_CANONICALS."""
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(canonicals)
    )


def union_research(*canonicals: str) -> None:
    """Partition contract: register each name in RESEARCH_ONLY_CANONICALS."""
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | set(canonicals)
    )
