# -*- coding: utf-8 -*-
"""Update-clock operators for sparse non-price fields (2026-08 market language, P1).

Report / shareholder / capital / index-weight fields are forward-filled daily,
so a raw ``ts_path_efficiency`` on the filled series sees ``0,0,0,0,update,...``
and measures the fill, not the economics.  These operators run the same
path/anomaly machinery on the **real update nodes only** (``update_event`` marks
true update days) and return a result as-of the current day:

* ``update_path_efficiency``         — net change / total path over the last
  ``n_updates`` real updates (persistent direction vs repeated revision).
* ``update_acceleration``            — last delta vs its own recent spread
  (improvement *speeding up*?).
* ``update_surprise``                — last delta vs the median/MAD of past
  deltas (innovation magnitude of this particular update).
* ``update_direction_persistence``   — magnitude-weighted fraction of same-sign
  deltas.

``update_event`` is the user's own "true update day" boolean (e.g. the raw
field is not a fill).  All kernels are prefix-causal and fail closed to NaN when
the window holds too few real updates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12


class UpdateMissingEventPolicy:
    """How an UNKNOWN update observation (NaN ``update_event``) is handled (#44).

    * ``BREAK`` — an unknown provider day BREAKS the update-node chain (fail
      closed): the output on that day is NaN and the two real update nodes on
      either side can never be joined into one trailing window.  Default.
    * ``SKIP``  — legacy opt-in: a NaN ``update_event`` is treated as a
      non-event, so unknown days are transparently skipped (this is exactly the
      join-across-unknown-day behaviour #44 forbids as a silent default).
    """

    BREAK = "BREAK"
    SKIP = "SKIP"
    EXECUTABLE = frozenset({BREAK, SKIP})


def _normalise_update_policy(policy: Any) -> str:
    p = str(policy).upper()
    if p not in UpdateMissingEventPolicy.EXECUTABLE:
        raise ValueError(
            f"invalid update missing-event policy {policy!r}; choose from "
            f"{sorted(UpdateMissingEventPolicy.EXECUTABLE)}"
        )
    return p


def _observed_mask(ev: np.ndarray) -> np.ndarray:
    """Rows where the provider's update status is KNOWN (``update_event`` finite)."""
    return np.isfinite(ev)


def _censored_boundaries(ev: np.ndarray, upd_idx: np.ndarray) -> np.ndarray:
    """Per-update-node censor flags (#44).

    For each real update node ``j >= 1`` the flag is True when the gap from the
    previous real update node ``j-1`` to node ``j`` crosses an UNKNOWN
    observation (a NaN ``update_event`` row).  A censored boundary means the two
    nodes are NOT adjacent in observation space — a trailing ``n``-update window
    may never span a censored boundary, so an unknown provider day cannot join
    two update nodes into one synthetic interval.
    """
    m = upd_idx.size
    censored = np.zeros(m, dtype=bool)
    for j in range(1, m):
        lo = upd_idx[j - 1] + 1
        hi = upd_idx[j]
        if np.any(~np.isfinite(ev[lo:hi])):
            censored[j] = True
    return censored


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="update_clock",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "update_clock", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:fundamental",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=param_specs or {},
    )


def _validate_update_event(ev: np.ndarray, canonical: str) -> None:
    """``update_event`` must be a strict boolean indicator {0, 1} or NaN (review P1-48a)."""
    finite = ev[np.isfinite(ev)]
    bad = finite[(finite != 0.0) & (finite != 1.0)]
    if bad.size:
        raise ValueError(
            f"{canonical} requires update_event to be a strict boolean indicator "
            f"(0/1 or NaN); found non-boolean finite value {float(bad[0])!r}"
        )


_UPDATE_MISSING_POLICY_SPEC = ParamSpec(
    dtype=str,
    choices=("BREAK", "SKIP"),
    default=UpdateMissingEventPolicy.BREAK,
)


def _update_kernel(canonical: str, min_updates: int, fn) -> SeriesOperator:
    def _calculate_series(
        self,
        x: pd.DataFrame,
        update_event: pd.DataFrame,
        n_updates: int = 5,
        missing_policy: str = UpdateMissingEventPolicy.BREAK,
        **_: Any,
    ) -> pd.DataFrame:
        n = int(n_updates)
        if n < min_updates:
            raise ValueError(f"{canonical} requires n_updates >= {min_updates}")
        policy = _normalise_update_policy(missing_policy)

        xv = x.to_numpy(dtype=float)
        ev = update_event.to_numpy(dtype=float)
        _validate_update_event(ev, canonical)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            # Audit #44: an unknown provider day (NaN update_event) must not be
            # transparently skipped — it would join two real update nodes into a
            # synthetic interval.  With the default BREAK policy the observed
            # mask leaves unknown days NaN and a trailing ``n``-update window may
            # never span a censored boundary.
            observed = _observed_mask(ev[:, c])
            upd_idx = np.flatnonzero(ev[:, c] == 1.0)
            if upd_idx.size < n:
                continue
            if policy == UpdateMissingEventPolicy.BREAK:
                censored = _censored_boundaries(ev[:, c], upd_idx)
            else:  # SKIP: legacy transparent skip — nothing is censored.
                censored = np.zeros(upd_idx.size, dtype=bool)
            for r in range(rows):
                if policy == UpdateMissingEventPolicy.BREAK and not observed[r]:
                    continue
                # Each row binary-searches for the last ``n`` *real* updates by
                # ordinal (PIT: only updates <= r).  There is deliberately NO
                # hidden max-lookback horizon — sparse events (annual reports,
                # rare capital events) must still reach the last ``n`` updates
                # however far back they are (P0-11).  NaN only when the whole
                # history holds fewer than ``n`` updates or the chain is broken.
                j1 = int(np.searchsorted(upd_idx, r + 1, side="left"))
                if j1 < n:
                    continue
                # The window's internal boundaries (between consecutive update
                # nodes) must all be observed-clean — otherwise two updates on
                # either side of an unknown day would be joined.
                if np.any(censored[j1 - n + 1 : j1]):
                    continue
                vals = xv[upd_idx[j1 - n : j1], c]
                if not np.all(np.isfinite(vals)):
                    continue
                out[r, c] = fn(vals)
        return frame_like(x, out)

    return register_operator(
        name=canonical,
        category="update_clock",
        business_category="update_clock",
        canonical=canonical,
        source="update_clock",
    )(
        type(
            canonical.replace("_", " ").title().replace(" ", "") + "Op",
            (SeriesOperator,),
            {
                "metadata": _metadata(
                    canonical,
                    _DESCRIPTIONS[canonical],
                    ["x", "update_event", "n_updates", "missing_policy"],
                    unit=_UNITS[canonical],
                    cost=4,
                    param_specs={"missing_policy": _UPDATE_MISSING_POLICY_SPEC},
                ),
                "_calculate_series": _calculate_series,
                "__module__": __name__,
            },
        )
    )


_DESCRIPTIONS = {
    "update_path_efficiency": "真实 update 节点上的路径效率 |Δ_K|/Σ|Δ_j|（方向一致性）。",
    "update_acceleration": "最近两次 update 差的归一化 (d_K - d_{K-1})/MAD(d_{≤K-1})（改善加速；基线排除当前 delta）。",
    "update_surprise": "本次 update 相对其自身历史的中位数 z 分（innovation）。",
    "update_direction_persistence": "幅度加权方向保持率 Σ|Δ_j|·1[sign(Δ_j)=sign(net)]/Σ|Δ_j|。",
}
_UNITS = {
    "update_path_efficiency": "ratio",
    "update_acceleration": "ratio",
    "update_surprise": "zscore",
    "update_direction_persistence": "probability",
}


def _path_eff(vals: np.ndarray) -> float:
    d = np.diff(vals)
    total = float(np.sum(np.abs(d)))
    if total <= _EPS:
        return np.nan
    return float(abs(vals[-1] - vals[0]) / total)


def _acceleration(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 3:
        return np.nan
    # Audit #45: the baseline scale that scores the CURRENT delta must use
    # t-1 and earlier — a large current delta would otherwise inflate its own
    # denominator.  ``past`` excludes ``d[-1]``, so the latest jump is scored
    # against the spread of the deltas that preceded it.
    past = d[:-1]
    med = float(np.median(past))
    mad = float(np.median(np.abs(past - med)))
    if mad <= _EPS:
        return np.nan
    return float((d[-1] - d[-2]) / mad)


def _surprise(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 4:
        return np.nan
    past = d[:-1]
    med = float(np.median(past))
    mad = float(np.median(np.abs(past - med)))
    if mad <= _EPS:
        return np.nan
    # MAD-normalised innovation (1.4826 puts MAD on the sigma scale).
    return float((d[-1] - med) / (1.4826 * mad))


def _direction_persist(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 2:
        return np.nan
    net = vals[-1] - vals[0]
    if abs(net) <= _EPS:
        return np.nan
    # Magnitude-weighted direction persistence (docstring said "magnitude
    # weighted" but the old body was unweighted sign consistency, review P1-48b):
    #   P = Σ|Δ_i|·1[sign(Δ_i) == sign(net)] / Σ|Δ_i|
    # Each update delta votes, weighted by its own size, on whether it agrees
    # with the direction of the overall move.  Zero deltas carry |Δ|=0 so they
    # no longer distort the denominator.  This stays distinct from path
    # efficiency: the old magnitude form |ΣΔ|/Σ|Δ| was algebraically identical
    # to ``_path_eff`` and duplicated that canonical (P0-10), whereas this is a
    # fraction in [0, 1] of the total update mass pointing the net way.
    s = np.sign(net)
    w = np.abs(d)
    total = float(w.sum())
    if total <= _EPS:
        return np.nan
    same = w[np.sign(d) == s]
    return float(same.sum() / total)


TsUpdatePathEfficiency = _update_kernel("update_path_efficiency", 3, _path_eff)
TsUpdateAcceleration = _update_kernel("update_acceleration", 4, _acceleration)
TsUpdateSurprise = _update_kernel("update_surprise", 5, _surprise)
TsUpdateDirectionPersistence = _update_kernel("update_direction_persistence", 3, _direction_persist)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "update_path_efficiency",
            "update_acceleration",
            "update_surprise",
            "update_direction_persistence",
        })
    for _canon in (
        "update_path_efficiency",
        "update_acceleration",
        "update_surprise",
        "update_direction_persistence",
    ):
        register_polars_udf(_canon)


_register_surface()


# Audit #7: update-clock operators are EVENT-HISTORY state — each output searches
# back for the most recent ``n_updates`` REAL update events with no fixed bar
# lookback.  A bar-window warmup must NOT take over, so the declared history is
# ``event_count`` (observations, not trading bars).  The operators have no
# segmented restore, so chunking is an honest full-history replay.
#
# P0-10: the event-count history must DERIVE from the ``n_updates`` parameter,
# not a stale constant.  ``n_updates`` is a public knob (default 5) and may be
# set to 10; a declared ``history_count=5`` would under-declare the required
# event nodes.  ``history_count`` is therefore a callable ``(canonical, params)
# -> int`` (Round-14 dynamic HistoryRequirement) that resolves ``n_updates`` and
# floors it at the kernel's ``min_updates`` — the same floor ``_update_kernel``
# enforces at runtime.
_MIN_UPDATES = {
    "update_path_efficiency": 3,
    "update_acceleration": 4,
    "update_surprise": 5,
    "update_direction_persistence": 3,
}
_DEFAULT_N_UPDATES = 5


def _update_clock_history_count(min_updates: int):
    """HistoryRequirement count = max(n_updates, min_updates) for a canonical."""

    def _fn(canonical: str, params: Any) -> int:
        n = (params or {}).get("n_updates")
        if n is None:
            n = _DEFAULT_N_UPDATES
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = _DEFAULT_N_UPDATES
        return max(n, min_updates)

    return _fn


def _declare_stateful_contracts() -> None:
    from factor_engine.runtime.execution_contract import declare_stateful

    for _canon, _min in _MIN_UPDATES.items():
        declare_stateful(
            _canon,
            state_model="recursive",
            chunking="required_full_history",
            history_kind="event_count",
            history_count=_update_clock_history_count(_min),
        )


_declare_stateful_contracts()
