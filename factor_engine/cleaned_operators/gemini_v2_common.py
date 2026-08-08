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
  future-randomisation contract all new tests assert.  R5 P0-03: the *current*
  row is part of that contract — a NaN at the current row fails the window
  closed (empty block) instead of silently reusing the previous contiguous
  run, so a today-missing field never emits a stale yesterday factor.

``register_dual`` binds the kernel as a *default argument* of the backend
entry points (``_fn=fn``) — a bare closure over the loop variable is
late-bound and every operator would call the last kernel.  R5 P0-01: both
backends are ``SeriesOperator`` subclasses that implement
``_calculate_series`` (never a bare ``calculate`` override), so every call is
routed through the central logical-call validator
(``base.validate_operator_call``) — integer ``ParamSpec`` checks, panel-axis
alignment, ``validate_params`` and common parameter relations are enforced
uniformly and cannot be bypassed by a future module.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.rolling_pack import frame_like

# R6-155: the metadata/axis columns a polars wide frame may carry must be
# excluded from *both* the instrument feature set and the date-axis detection,
# or a frame using ``timestamp`` / ``trade_date`` / ``datetime`` as its time
# axis would feed the time column into the kernel as a numeric feature (and
# then rewrite it as an output column).  ``stock_code`` / ``instrument`` /
# ``symbol`` / ``session`` are identity / session columns and never features.
_AXIS_COLUMNS = ("__fe_time__", "date", "timestamp", "trade_date", "datetime")
_IDENTITY_COLUMNS = ("stock_code", "instrument", "symbol", "session")
_SKIP = frozenset((*_AXIS_COLUMNS, *_IDENTITY_COLUMNS))

_EPS = 1e-12


def trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Longest trailing contiguous finite run ending at the current row.

    A missing value invalidates everything before it for the current row: the
    time structure is preserved and a gap never re-pairs data that were not
    actually adjacent.  Used by all trailing-window kernels in this round.

    R5 P0-03 (fail-closed current row): a NaN at the current row returns the
    empty block so the caller emits NaN — a missing *today* value must never
    silently fall back to the previous contiguous historical run.
    """
    n = chunk.shape[0]
    if n == 0 or not np.isfinite(chunk[-1]):
        return chunk[:0]
    end = n
    while end > 0 and np.isfinite(chunk[end - 1]):
        end -= 1
    return chunk[end:]


def trailing_contiguous_multi(*chunks: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """Longest trailing run where *every* series is finite, then return the run.

    Returns ``None`` when the current row is not finite in any input.  Used by
    multi-input operators (EDGE / Abdi-Ranaldo / cross-spectral) where one
    missing panel invalidates the aligned pair.

    R5 P0-03: if the *current* row is NaN in any input, ``None`` is returned
    immediately (fail-closed) — a stale-history run is never produced.
    """
    n = int(chunks[0].shape[0])
    if n == 0 or any(not np.isfinite(ch[n - 1]) for ch in chunks):
        return None
    end = n
    for _ in range(end):
        if all(np.isfinite(ch[end - 1]) for ch in chunks):
            break
        end -= 1
    start = end
    for _ in range(start):
        if not all(np.isfinite(ch[start - 1]) for ch in chunks):
            break
        start -= 1
    return tuple(ch[start:end] for ch in chunks)


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Align composition parts, failing closed on any axis mismatch (R5 P1-14).

    Composition inputs are financial panels (asset / liability / flow data
    aligned on PubDate).  A silent ``reindex`` could re-pair a row to a
    different date (or an instrument column to a different name) after an
    upstream misalignment.  FactorEngine must fail closed instead.
    """
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for position, frame in enumerate(frames[1:], start=1):
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(
                "composition inputs are misaligned: panel %d has a different "
                "index/columns than panel 0 (fail-closed; no silent reindex)" % position
            )
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
    # R5 P1-01: authoritative per-parameter contracts.  A param without a spec
    # still resolves via the legacy name whitelist; a declared spec overrides it
    # in both directions (int stays validated, float stays float, bounds/choices
    # and ``searchable=False`` are enforced).
    param_specs: Mapping[str, Any] | None = None,
    # R6-24: declared cross-parameter feasibility constraints (RelationalParamSpec)
    # forwarded to the pandas metadata so the central validator rejects
    # guaranteed-NaN parameter combinations before search spends budget.
    relational_specs: Sequence[Any] | None = None,
) -> None:
    """Register ``fn`` under ``canonical`` for both pandas_numpy and polars.

    The polars backend runs the *same* kernel (converted frame), guaranteeing
    numerical parity; it is registered ``backend_explicit`` so production
    hardening keeps it.  R5 P0-02: the polars backend is a ``python_bridge``
    (materialises the full panel and is not a native polars execution), and it
    asserts that every input carries the same date axis and instrument columns
    before computing — a positionally-misaligned pair (``x_t`` against
    ``y_{t+1}`` or stock A against stock B) is rejected instead of silently
    paired.
    """
    from cleaned_operators.base import OperatorMetadata as PandasMetadata
    from cleaned_operators.base import SeriesOperator as PandasSeriesOperator

    tag_list = [
        "daily", "panel", "pit_safe", "causal", "deterministic",
        f"signature:{','.join(params)}->series",
        f"domain:{domain}", f"unit:{unit}", f"cost:{cost}",
        *tags_extra,
    ]
    units = dict(input_units) if input_units else None
    specs = dict(param_specs) if param_specs else None
    rel_specs = list(relational_specs) if relational_specs else None

    class _PandasOp(PandasSeriesOperator):
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
            param_specs=specs,
            relational_specs=rel_specs,
        )

        def _calculate_series(self, *args, _fn=fn, **kwargs):
            return _fn(*args, **kwargs)

    OperatorRegistry.register(
        _PandasOp(), canonical=canonical, backend="pandas_numpy",
        source=source, backend_explicit=True,
    )

    class _PolarsOp(PolarsSeriesOperator):
        metadata = PolarsMetadata(
            name=canonical, category=category, param_names=[],
            tags=["python_bridge", "materializes_full_panel"],
        )

        def _calculate_series(self, *frames, _fn=fn, **params):
            import polars as pl  # noqa: F401
            pdfs: list[pd.DataFrame] = []
            axes: list[tuple[np.ndarray | None, tuple[str, ...], int]] = []
            for f in frames:
                cols = [c for c in f.columns if c not in _SKIP]
                pdf = f.select(cols).to_pandas()
                date_vals: np.ndarray | None = None
                for name in _AXIS_COLUMNS:
                    if name in f.columns:
                        date_vals = np.asarray(f[name].to_numpy())
                        break
                axes.append((date_vals, tuple(cols), pdf.shape[0]))
                pdfs.append(pdf)
            base_date, base_cols, base_rows = axes[0]
            for j, (date_vals, cols, rows) in enumerate(axes[1:], start=1):
                if rows != base_rows:
                    raise ValueError(
                        f"{canonical}: polars input {j} has {rows} rows != base {base_rows}"
                    )
                if cols != base_cols:
                    raise ValueError(
                        f"{canonical}: polars input {j} instrument columns {cols} "
                        f"!= base {base_cols} (order matters)"
                    )
                if base_date is None and date_vals is not None:
                    raise ValueError(
                        f"{canonical}: polars input {j} carries a date axis while "
                        "the base input does not"
                    )
                if base_date is not None and date_vals is None:
                    raise ValueError(
                        f"{canonical}: polars input {j} has no date axis while the "
                        "base input carries one"
                    )
                if base_date is not None and not np.array_equal(base_date, date_vals):
                    raise ValueError(
                        f"{canonical}: polars input {j} date axis is misaligned with "
                        "the base input (a gap, an extra row, or a different order "
                        "would pair x_t with y_{t+1})"
                    )
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
    """Partition contract: register each name in EXTENDED_ONLY_CANONICALS.

    Uses the live ``extend_extended_only`` mutator (R5-50) — never a
    ``frozenset(set(old)|new)`` reassignment that would strand import-time
    snapshots.
    """
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(canonicals))


def union_research(*canonicals: str) -> None:
    """Partition contract: register each name in RESEARCH_ONLY_CANONICALS.

    Uses the live ``extend_research_only`` mutator (R5-50) — never a
    ``frozenset(set(old)|new)`` reassignment.
    """
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only(set(canonicals))
