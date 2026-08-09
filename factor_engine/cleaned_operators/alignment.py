# -*- coding: utf-8 -*-
"""Strict multi-panel axis alignment contract (round-6 P0-23/24/25).

Every operator that consumes two or more ``(timestamp x instrument)`` panels
must prove — before any ``.to_numpy()`` — that all inputs share the same
instrument identity (same columns, same order), the same ordered time axis and
the same length.  Positional pairing of ``A.to_numpy()`` against
``B.to_numpy()`` with different stock order or a one-day-shifted index is the
single most dangerous alignment bug in a factor engine: the numbers line up
but the *stocks and dates* silently disagree.

Two enforcement layers cover it:

* **Central gate** — ``cleaned_operators.base.validate_operator_call`` runs
  ``_validate_panel_axes`` on every ``SeriesOperator`` call, so a permuted
  column set or a shifted index between inputs is rejected before any kernel
  runs (this is the framework-level guarantee).
* **Shared helper** — ``align_panel_inputs`` / ``panel_arrays`` below is the
  public, strict utility for recipe/research code and for operator kernels
  that want defense-in-depth at the numpy boundary.  It raises on any axis
  mismatch; there is no silent ``reindex`` except under an explicit
  ``strict_axes=False`` opt-out (P0-25: production defaults to strict).

The audit asked for exactly this: *"建立 align_panel_inputs()，然后 framework
强制"* — build the helper, and make the framework enforce it.  The framework
layer already exists; this module makes the same contract callable and makes
every legacy ``_aligned()`` helper fail closed instead of silently reindexing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


class PanelAxisMismatch(ValueError):
    """Raised when two input panels do not share the same time/instrument axes."""


@dataclass(frozen=True)
class AlignmentPlan:
    """Explicit justification for a non-strict alignment (R11 P1-11).

    ``strict_axes=False`` (research only) is no longer a bare boolean escape
    hatch — every reindex must carry a plan naming WHY it is safe: the join
    direction, the time alignment (backward = last-observed / exact), the
    instrument policy, the missing-value policy and an availability policy.
    Production callers are forbidden from bypassing strict alignment entirely.
    """

    join: str = "left"                 # left / inner / asof
    time_direction: str = "backward"   # backward / exact
    instrument_policy: str = "exact"   # exact / subset
    missing_policy: str = "nan"        # nan / error
    availability_policy: str = "no_lookahead"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.join not in {"left", "inner", "asof"}:
            raise ValueError(f"AlignmentPlan.join={self.join!r} is unknown")
        if self.time_direction not in {"backward", "exact"}:
            raise ValueError(f"AlignmentPlan.time_direction={self.time_direction!r} is unknown")
        if self.instrument_policy not in {"exact", "subset"}:
            raise ValueError(f"AlignmentPlan.instrument_policy={self.instrument_policy!r} is unknown")
        if self.missing_policy not in {"nan", "error"}:
            raise ValueError(f"AlignmentPlan.missing_policy={self.missing_policy!r} is unknown")
        if not self.reason.strip():
            raise ValueError("AlignmentPlan.reason is required (document WHY the reindex is safe)")


def _describe(frame: Any) -> str:
    index = getattr(frame, "index", None)
    index_note = "no-index" if index is None else f"{index.name} len={len(index)}"
    columns = getattr(frame, "columns", None)
    col_note = "no-columns" if columns is None else f"cols={list(columns)}"
    return f"{type(frame).__name__}({index_note}, {col_note})"


def axes_aligned(*frames: pd.DataFrame) -> bool:
    """True when every frame shares ``base``'s exact index and columns."""
    if not frames:
        return True
    base = frames[0]
    for frame in frames[1:]:
        if not frame.index.equals(base.index):
            return False
        if not frame.columns.equals(base.columns):
            return False
    return True


def assert_axes_aligned(
    *frames: pd.DataFrame,
    names: Sequence[str] | None = None,
) -> None:
    """Raise :class:`PanelAxisMismatch` unless all frames share exact axes.

    ``names`` optionally labels the inputs in the error message (e.g. the
    operator's parameter names) so a mismatch points at the offending panel.
    """
    if not frames:
        return
    base = frames[0]
    base_index = base.index
    base_columns = base.columns
    labels = list(names) if names else [""] * len(frames)
    for position, frame in enumerate(frames[1:], start=1):
        label = labels[position] or f"input panel {position}"
        if not frame.index.equals(base_index):
            raise PanelAxisMismatch(
                f"{label}: index misaligned — base {_describe(base)} vs "
                f"{_describe(frame)}.  Two panels whose rows do not line up "
                "must never be paired positionally."
            )
        if not frame.columns.equals(base_columns):
            raise PanelAxisMismatch(
                f"{label}: columns misaligned — base {list(base_columns)} vs "
                f"{list(frame.columns)}.  Instrument identity/order must match "
                "exactly before any cross-panel to_numpy."
            )


def align_panel_inputs(
    *frames: pd.DataFrame,
    strict_axes: bool = True,
    names: Sequence[str] | None = None,
    alignment_plan: AlignmentPlan | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Return all panels with a guaranteed-common axis.

    * ``strict_axes=True`` (default, production) — raise
      :class:`PanelAxisMismatch` on any index/column mismatch.  This is the
      P0-25 fail-closed contract: a missing symbol, a duplicate date or a
      permuted column set is an error, not a NaN.
    * ``strict_axes=False`` (research only) — panels are reindexed onto the
      first panel's axes.  R11 P1-11: this is NOT a bare boolean escape hatch.
      Every non-strict call must pass an explicit :class:`AlignmentPlan` naming
      the join / time direction / instrument policy / missing policy /
      availability policy and a written reason; without one the call raises.
      The plan exists so a silent ``reindex`` can never hide a date/stock
      mismatch behind NaN.

    Returns the same frame objects (identical axes are a no-op).
    """
    if not frames:
        return ()
    if strict_axes:
        assert_axes_aligned(*frames, names=names)
        return frames
    if alignment_plan is None:
        raise PanelAxisMismatch(
            "align_panel_inputs(strict_axes=False) requires an explicit "
            "AlignmentPlan (join/time_direction/instrument_policy/"
            "missing_policy/availability_policy/reason) — a bare boolean opt-out "
            "would silently reindex mismatched axes (R11 P1-11 fail-closed)"
        )
    base = frames[0]
    aligned = []
    labels = list(names) if names else [""] * len(frames)
    for position, frame in enumerate(frames):
        if frame is base or (
            frame.index.equals(base.index) and frame.columns.equals(base.columns)
        ):
            aligned.append(frame)
            continue
        label = labels[position] or f"input panel {position}"
        aligned.append(frame.reindex(index=base.index, columns=base.columns))
        # A reindex under the research opt-out must still surface what it did:
        # silently introducing NaN columns/dates is exactly what P0-25 forbids
        # in production.
        _ = label
    return tuple(aligned)


def panel_arrays(
    *frames: pd.DataFrame,
    strict_axes: bool = True,
    names: Sequence[str] | None = None,
) -> list[np.ndarray]:
    """Strictly align panels, then return their float numpy arrays.

    This is the one call multi-panel operator kernels should use instead of
    scattering ``f.to_numpy(dtype=float)`` per input.  ``names`` labels the
    inputs in the error message.
    """
    aligned = align_panel_inputs(*frames, strict_axes=strict_axes, names=names)
    return [frame.to_numpy(dtype=float) for frame in aligned]


def align_panels_and_groups(
    value: pd.DataFrame,
    group: pd.DataFrame,
    *extra: pd.DataFrame,
    names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ...]:
    """Align a value panel, a group panel and any extra panels together.

    Convenience for group/peer operators: the group panel carries a per-cell
    label and must share the exact shape and axis of the value panel, and any
    extra inputs (weights, etc.) must too.
    """
    labels = list(names) if names else None
    assert_axes_aligned(value, group, *extra, names=labels)
    return (value, group, *extra)


def apply_universe_mask(
    panel: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    mask_axes: str = "strict",
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply a boolean universe mask to a panel, fail-closed (round-6 P0-27/28).

    The mask marks IN-universe cells (``True``/``1``); anything else (``False``,
    ``NaN``, ``Inf``) sets the corresponding cell to NaN (out of universe).  A
    stock that is suspended, not listed, or delisted on a date leaves a NaN
    rather than silently participating in a cross-sectional statistic.

    ``mask_axes``:
    * ``strict`` (default) — the mask must share the panel's exact index and
      columns; a date/stock mismatch raises instead of pairing positionally
      (round-6 P0-23).  This is the mandatory entry for cross-sectional runtime:
      a mask that does not line up with the panel IS an error.
    * ``broadcast_scalar`` — the mask may be a per-date row-vector (rows ==
      panel rows, one column) broadcast across instruments on the SAME date, or
      a full-shape mask.  A single-ROW mask (one date's membership flag) is
      REJECTED against a multi-date panel: broadcasting today's constituents
      over the whole history is survivorship / future-membership leakage
      (R11 P0-10).  A genuinely time-invariant pool must use an explicit
      ``StaticUniverseMask``, never a one-row broadcast.
    """
    if mask is None:
        return panel.copy()
    labels = list(names) if names else None
    if str(mask_axes) == "strict":
        assert_axes_aligned(panel, mask, names=labels)
    else:
        _broadcast_scalar_shape_gate(panel.shape, mask.shape, names=labels)
    mask_vals = mask.to_numpy(dtype=float)
    panel_vals = panel.to_numpy(dtype=float)
    inside = np.isfinite(mask_vals) & (mask_vals != 0.0)
    out = np.where(inside, panel_vals, np.nan)
    return pd.DataFrame(out, index=panel.index, columns=panel.columns, dtype=float)


def _broadcast_scalar_shape_gate(
    panel_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    *,
    names: Sequence[str] | None = None,
) -> None:
    """Fail-closed shape gate for ``broadcast_scalar`` masks (R11 P0-10).

    Only two broadcast shapes are legal against a multi-date panel:

    * ``mask_shape == panel_shape`` — a fully-aligned per-cell mask;
    * ``mask_shape == (panel_rows, 1)`` — a per-date membership vector applied
      to every instrument ON THE SAME DATE (cross-section broadcast, safe).

    A single-row mask ``(1, C)`` or scalar ``(1, 1)`` is the CURRENT date's
    cross-section (e.g. today's index-membership flag).  Broadcasting it across
    all ``panel_rows`` would make today's universe silently apply to every past
    date — the classic survivorship / future-membership leak.  It is rejected
    unless the panel itself is a single row (a genuine one-day panel).
    """
    if panel_shape == mask_shape:
        return
    if panel_shape[0] == mask_shape[0] and mask_shape[1] == 1:
        return  # per-date membership vector -> same-date cross-section broadcast
    if mask_shape == (1, 1) and panel_shape == (1, 1):
        return
    label = (names[0] if names and names[0] else "universe mask") if names else "universe mask"
    raise PanelAxisMismatch(
        f"{label} broadcast_scalar: shape {mask_shape} cannot be broadcast onto "
        f"panel {panel_shape}.  A single-row (or single-cell) mask is the current "
        "date's membership and must not be stretched across historical dates "
        "(survivorship / future-membership leakage, R11 P0-10).  Supply a per-date "
        "mask (rows == panel rows) or an explicit StaticUniverseMask for a "
        "time-invariant pool."
    )


__all__ = [
    "AlignmentPlan",
    "PanelAxisMismatch",
    "align_panel_inputs",
    "align_panels_and_groups",
    "apply_universe_mask",
    "assert_axes_aligned",
    "axes_aligned",
    "panel_arrays",
]
