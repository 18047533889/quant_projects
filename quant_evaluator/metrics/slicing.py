"""Metric x Slice x GroupBy engine: sub-universe slicing primitives.

This module provides fail-closed slicing primitives that partition a
factor panel into sub-universes without copying data until a view is
materialized:

- :func:`slice_by_time`    — contiguous time windows (whole / rolling /
  expanding / explicit ``[(start, end), ...]``).
- :func:`slice_by_group`   — arbitrary asset grouping (dict or aligned
  array) with strict label-coverage validation.
- :func:`slice_by_quantile` — cross-sectional factor-quantile
  sub-universes per time (reuses :mod:`quant_evaluator.metrics.quantile`
  tie policy).
- :func:`slice_by_mask`    — explicit boolean selection.

Every primitive returns :class:`SliceView` objects that carry *indices
only*; data is subset (and, for ragged per-time views, NaN-padded) when
``take``/``sub_batch``/``sub_labels`` materialize the view.

Invalid, empty, or ambiguous specs raise :class:`InvalidContractError`
(fail closed) — never silently return an empty or full-panel slice.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import MetricArtifact
from quant_evaluator.metrics.quantile import assign_quantiles

__all__ = [
    "UNLABELED",
    "SliceView",
    "slice_by_time",
    "slice_by_group",
    "slice_by_quantile",
    "slice_by_mask",
    "SliceEngine",
]

#: Bucket name used when ``allow_unlabeled=True`` and some assets carry
#: no group label.
UNLABELED = "__unlabeled__"

_TIME_MODES = ("whole", "rolling", "expanding", "explicit")


def _hash_content(payload: Mapping[str, Any]) -> str:
    """Deterministic sha256 over a JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _axis_info(axis: Any, name: str) -> Tuple[int, Optional[np.ndarray]]:
    """Return (size, values|None) for an AxisRef or 1-D array-like axis."""
    if isinstance(axis, AxisRef):
        vals = None if axis.values is None else np.asarray(axis.values)
        if vals is not None and len(vals) != axis.size:
            raise InvalidContractError(
                f"{name} declared size {axis.size} but values has {len(vals)}"
            )
        return axis.size, vals
    arr = np.asarray(axis)
    if arr.ndim != 1:
        raise InvalidContractError(
            f"{name} must be 1-D or an AxisRef, got ndim={arr.ndim}"
        )
    return arr.shape[0], arr


@dataclass(frozen=True)
class SliceView:
    """Indices-only description of one sub-universe.

    Attributes:
        slice_kind: One of ``"time"``, ``"group"``, ``"quantile"``, ``"mask"``.
        slice_key: Unique human-readable key (used as the engine result key).
        description: Free-form provenance string.
        params: Immutable parameter snapshot hashed into ``spec_id``.
        time_indices: Positions into the panel time axis.
        asset_indices: Positions into the panel asset axis; may be empty for
            pure time slices, in which case materialization uses *all* assets.
        per_time_asset_indices: Optional ragged refinement aligned with
            ``time_indices``; when set, asset membership varies per time and
            absent cells are padded (NaN / False) on materialization.
        spec_id: Content-addressed sha256 of kind/key/params/indices.
    """

    slice_kind: str
    slice_key: str
    description: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    time_indices: Tuple[int, ...] = ()
    asset_indices: Tuple[int, ...] = ()
    per_time_asset_indices: Optional[Tuple[Tuple[int, ...], ...]] = None
    spec_id: str = ""

    def __post_init__(self) -> None:
        if self.slice_kind not in ("time", "group", "quantile", "mask"):
            raise InvalidContractError(f"Unknown slice_kind: {self.slice_kind!r}")
        if not str(self.slice_key).strip():
            raise InvalidContractError("slice_key must be non-empty")
        if not self.time_indices and self.slice_kind not in ("group", "mask"):
            raise InvalidContractError(
                f"SliceView {self.slice_key!r} has empty time_indices"
            )
        if not self.asset_indices and self.slice_kind != "time":
            raise InvalidContractError(
                f"SliceView {self.slice_key!r} has empty asset_indices"
            )
        if self.per_time_asset_indices is not None:
            if len(self.per_time_asset_indices) != len(self.time_indices):
                raise InvalidContractError(
                    f"SliceView {self.slice_key!r}: per_time_asset_indices "
                    f"length {len(self.per_time_asset_indices)} != "
                    f"time_indices length {len(self.time_indices)}"
                )
            for i, members in enumerate(self.per_time_asset_indices):
                if not members:
                    raise InvalidContractError(
                        f"SliceView {self.slice_key!r}: empty asset set at "
                        f"time position {i}"
                    )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self,
            "spec_id",
            _hash_content(
                {
                    "kind": self.slice_kind,
                    "key": self.slice_key,
                    "params": dict(self.params),
                    "time": list(self.time_indices),
                    "assets": list(self.asset_indices),
                    "per_time": (
                        None
                        if self.per_time_asset_indices is None
                        else [list(m) for m in self.per_time_asset_indices]
                    ),
                }
            ),
        )

    @property
    def num_times(self) -> int:
        return len(self.time_indices)

    @property
    def num_assets(self) -> int:
        return len(self.asset_indices)

    def take(self, values: np.ndarray, num_assets: int = 0, fill: Any = np.nan) -> np.ndarray:
        """Materialize this view against a time-leading array.

        ``values`` may be ``(T,)``, ``(T, N)`` or ``(T, N, F)``. Ragged
        per-time views produce a rectangular result padded with ``fill``.
        ``num_assets`` supplies the asset count for pure time slices when
        the array is 1-D or the view carries no asset indices.
        """
        values = np.asarray(values)
        t_idx = np.asarray(self.time_indices, dtype=np.intp)
        if self.per_time_asset_indices is not None:
            if values.ndim < 2:
                raise InvalidContractError(
                    "Ragged SliceView.take requires a (T, N, ...) array, "
                    f"got ndim={values.ndim}"
                )
            a_idx = np.asarray(self.asset_indices, dtype=np.intp)
            col_of = {int(a): int(c) for c, a in enumerate(a_idx)}
            out = np.full(
                (len(t_idx), len(a_idx)) + values.shape[2:],
                fill,
                dtype=np.result_type(values.dtype, np.asarray(fill).dtype),
            )
            for i, members in enumerate(self.per_time_asset_indices):
                cols = [col_of[int(a)] for a in members]
                out[i, cols] = values[t_idx[i]][np.asarray(members, dtype=np.intp)]
            return out
        sub = values[t_idx]
        if self.asset_indices:
            a_idx = np.asarray(self.asset_indices, dtype=np.intp)
            return sub[:, a_idx]
        if sub.ndim >= 2:
            return sub
        return sub

    def _sub_axis(self, axis: Any, name: str, indices: Tuple[int, ...]) -> AxisRef:
        if isinstance(axis, AxisRef):
            vals = None
            if axis.values is not None:
                vals = np.asarray(axis.values)[np.asarray(indices, dtype=np.intp)]
            return AxisRef(name=axis.name, dtype=axis.dtype, size=len(indices), values=vals)
        return AxisRef(name=name, dtype="int64", size=len(indices))

    def sub_batch(self, batch: FactorBatch) -> FactorBatch:
        """Build a sub-panel :class:`FactorBatch` for this view."""
        values = self.take(
            batch.values,
            num_assets=batch.num_assets if not self.asset_indices else 0,
            fill=np.nan,
        )
        validity = None
        if batch.validity is not None:
            validity = self.take(
                batch.validity,
                num_assets=batch.num_assets if not self.asset_indices else 0,
                fill=False,
            )
        return FactorBatch(
            factor_ids=tuple(batch.factor_ids),
            time_axis=self._sub_axis(batch.time_axis, "time", self.time_indices),
            asset_axis=self._sub_axis(batch.asset_axis, "asset", self.asset_indices or tuple(range(batch.num_assets))),
            values=values,
            validity=validity,
            layout=batch.layout,
            dtype=batch.dtype,
            context_refs=dict(batch.context_refs),
        )

    def sub_labels(self, labels: LabelBundle) -> LabelBundle:
        """Build a row-subset :class:`LabelBundle` for this view."""
        label_values = np.asarray(labels.values)
        if label_values.ndim == 2 and self.per_time_asset_indices is not None:
            label_values = self.take(label_values, fill=np.nan)
        elif label_values.ndim == 2 and self.asset_indices:
            t_idx = np.asarray(self.time_indices, dtype=np.intp)
            a_idx = np.asarray(self.asset_indices, dtype=np.intp)
            label_values = label_values[np.ix_(t_idx, a_idx)]
        else:
            label_values = label_values[np.asarray(self.time_indices, dtype=np.intp)]
        validity = None
        if labels.validity is not None:
            lv = np.asarray(labels.validity)
            if lv.ndim == 2 and self.per_time_asset_indices is not None:
                validity = self.take(lv, fill=False)
            elif lv.ndim == 2 and self.asset_indices:
                t_idx = np.asarray(self.time_indices, dtype=np.intp)
                a_idx = np.asarray(self.asset_indices, dtype=np.intp)
                validity = lv[np.ix_(t_idx, a_idx)]
            else:
                validity = lv[np.asarray(self.time_indices, dtype=np.intp)]

        def _rows(seq: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
            if seq is None or not seq:
                return tuple(seq or ())
            return tuple(seq[i] for i in self.time_indices)

        return LabelBundle(
            target_id=labels.target_id,
            values=label_values,
            horizon=labels.horizon,
            execution_delay=labels.execution_delay,
            decision_time=_rows(labels.decision_time),
            execution_time=_rows(labels.execution_time),
            signal_available_time=_rows(labels.signal_available_time),
            label_start_time=_rows(labels.label_start_time),
            label_end_time=_rows(labels.label_end_time),
            validity=validity,
            source_ref=labels.source_ref,
            calendar_ref=labels.calendar_ref,
            metadata=dict(labels.metadata),
        )


def _resolve_time_position(
    bound: Any,
    time_values: Optional[np.ndarray],
    T: int,
    allow_end_sentinel: bool = False,
) -> int:
    """Resolve an integer position or a timestamp to a time-axis position.

    Resolution order: an exact match against ``time_values`` wins (so an
    integer that happens to equal an axis timestamp is treated as a
    timestamp); otherwise plain integers are positions, where an integer
    equal to ``T`` (or a timestamp past the last axis value) is accepted
    as the exclusive end-of-axis sentinel when
    ``allow_end_sentinel=True``.
    """
    if isinstance(bound, (bool, np.bool_)):
        raise InvalidContractError(f"Invalid time bound: {bound!r}")
    if time_values is not None:
        matches = np.nonzero(np.asarray(time_values) == bound)[0]
        if len(matches) == 1:
            return int(matches[0])
        if len(matches) == 0 and allow_end_sentinel and not isinstance(
            bound, (int, np.integer)
        ):
            time_arr = np.asarray(time_values)
            if time_arr.ndim == 1 and time_arr.size > 0:
                try:
                    if time_arr[-1] < bound:
                        return T
                except TypeError:
                    pass
    if isinstance(bound, (int, np.integer)):
        pos = int(bound)
        upper = T if allow_end_sentinel else T - 1
        if not 0 <= pos <= upper:
            raise InvalidContractError(
                f"Time position {pos} out of bounds for time axis of size {T}"
            )
        return pos
    raise InvalidContractError(
        f"Cannot resolve time bound {bound!r}: time axis carries no values"
    )


def slice_by_time(
    values: np.ndarray,
    time_axis: Any,
    spec: Mapping[str, Any],
) -> List[SliceView]:
    """Contiguous time windows over the panel time axis.

    ``spec`` keys:
      - ``"mode"``: ``"whole"`` | ``"rolling"`` | ``"expanding"`` | ``"explicit"``
      - rolling/expanding: ``"window"`` (int > 0), optional ``"step"`` (int > 0)
      - explicit: ``"windows"`` = sequence of ``(start, end)`` (positions or
        timestamps, end exclusive)

    Returns one :class:`SliceView` per window (keys like
    ``time:rolling[0:3]``). Fails closed on unknown modes, non-positive
    windows, windows exceeding the axis, and invalid explicit ranges.
    """
    if not isinstance(spec, Mapping) or not spec:
        raise InvalidContractError("slice_by_time spec must be a non-empty mapping")
    mode = spec.get("mode")
    if mode not in _TIME_MODES:
        raise InvalidContractError(
            f"slice_by_time mode must be one of {_TIME_MODES}, got {mode!r}"
        )
    T, time_values = _axis_info(time_axis, "time_axis")
    arr = np.asarray(values)
    if arr.ndim < 1 or arr.shape[0] != T:
        raise InvalidContractError(
            f"values time axis ({arr.shape[0] if arr.ndim else 0}) does not "
            f"match time_axis size {T}"
        )
    all_assets = tuple(range(arr.shape[1])) if arr.ndim >= 2 else ()

    windows: List[Tuple[int, int]] = []
    if mode == "whole":
        windows = [(0, T)]
    elif mode in ("rolling", "expanding"):
        window = spec.get("window")
        if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
            raise InvalidContractError(
                f"{mode} slice requires integer 'window', got {window!r}"
            )
        window = int(window)
        step = spec.get("step", 1)
        if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
            raise InvalidContractError(f"{mode} slice 'step' must be an integer")
        step = int(step)
        if window <= 0 or step <= 0:
            raise InvalidContractError(
                f"{mode} slice requires window>0 and step>0, got "
                f"window={window}, step={step}"
            )
        if window > T:
            raise InvalidContractError(
                f"{mode} window {window} exceeds time axis size {T}"
            )
        if mode == "rolling":
            windows = [(s, s + window) for s in range(0, T - window + 1, step)]
        else:  # expanding
            anchors = list(range(window, T + 1, step))
            if anchors[-1] != T:
                anchors.append(T)
            windows = [(0, e) for e in anchors]
        if not windows:
            raise InvalidContractError(
                f"{mode} spec produced no windows (window={window}, step={step}, T={T})"
            )
    else:  # explicit
        raw_windows = spec.get("windows")
        if not raw_windows or not isinstance(raw_windows, (list, tuple)):
            raise InvalidContractError(
                "explicit time slice requires a non-empty 'windows' list"
            )
        for pair in raw_windows:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise InvalidContractError(
                    f"Explicit windows must be (start, end) pairs, got {pair!r}"
                )
            start = _resolve_time_position(pair[0], time_values, T)
            end = _resolve_time_position(pair[1], time_values, T, allow_end_sentinel=True)
            if end <= start:
                raise InvalidContractError(
                    f"Explicit window end ({end}) must exceed start ({start})"
                )
            windows.append((start, end))

    views: List[SliceView] = []
    seen_keys = set()
    for start, end in windows:
        key = f"time:{mode}[{start}:{end}]"
        if key in seen_keys:
            raise InvalidContractError(f"Duplicate time slice key: {key}")
        seen_keys.add(key)
        views.append(
            SliceView(
                slice_kind="time",
                slice_key=key,
                description=f"contiguous time window [{start}, {end})",
                params={"mode": mode, "start": start, "end": end},
                time_indices=tuple(range(start, end)),
                asset_indices=all_assets,
            )
        )
    return views


def slice_by_group(
    asset_axis: Any,
    group_labels: Any,
    allow_unlabeled: bool = False,
) -> List[SliceView]:
    """Asset grouping into sub-universes.

    ``group_labels`` is either:
      - a mapping ``{group_name: asset_id or sequence of asset_ids}`` (asset
        ids resolved against ``asset_axis.values``), or
      - a 1-D array-like aligned with the asset axis (one label per asset).

    Label coverage is validated: every asset must receive exactly one
    label unless ``allow_unlabeled=True`` (unlabeled assets then land in
    the explicit ``UNLABELED`` bucket). An asset appearing in two groups,
    an empty group, an unknown asset id, or (without
    ``allow_unlabeled``) an unlabeled asset fails closed.
    """
    N, asset_values = _axis_info(asset_axis, "asset_axis")

    membership: Dict[str, List[int]] = {}
    if isinstance(group_labels, Mapping):
        id_to_pos: Dict[Any, int] = {}
        if asset_values is not None:
            id_to_pos = {a: i for i, a in enumerate(asset_values.tolist())}
        for group_name, members in group_labels.items():
            if not str(group_name).strip():
                raise InvalidContractError("Group labels must be non-empty strings")
            if isinstance(members, (int, np.integer, str, np.floating)) and not isinstance(members, bool):
                member_seq: Sequence[Any] = [members]
            elif isinstance(members, (list, tuple, np.ndarray)):
                member_seq = list(members)
            else:
                raise InvalidContractError(
                    f"Group {group_name!r} members must be an asset id or a "
                    f"sequence of ids, got {type(members).__name__}"
                )
            positions: List[int] = []
            for asset_id in member_seq:
                if asset_id in id_to_pos:
                    pos = id_to_pos[asset_id]
                elif (
                    asset_values is None
                    and isinstance(asset_id, (int, np.integer))
                    and not isinstance(asset_id, bool)
                    and 0 <= int(asset_id) < N
                ):
                    # Axis carries no ids: integers are accepted as positions.
                    pos = int(asset_id)
                else:
                    raise InvalidContractError(
                        f"Group {group_name!r} references unknown asset id {asset_id!r}"
                    )
                positions.append(pos)
            if not positions:
                raise InvalidContractError(f"Group {group_name!r} is empty")
            membership[str(group_name)] = sorted(set(positions))
    else:
        arr = np.asarray(group_labels)
        if arr.ndim != 1:
            raise InvalidContractError(
                f"Aligned group_labels must be 1-D, got ndim={arr.ndim}"
            )
        if arr.shape[0] != N:
            raise InvalidContractError(
                f"Aligned group_labels length {arr.shape[0]} does not match "
                f"asset axis size {N}"
            )
        for pos, label in enumerate(arr.tolist()):
            if label is None or (isinstance(label, float) and np.isnan(label)):
                continue
            membership.setdefault(str(label), []).append(pos)
        if not membership:
            raise InvalidContractError("group_labels contained no usable labels")
        for name in membership:
            membership[name] = sorted(set(membership[name]))

    # Coverage validation.
    covered = [pos for positions in membership.values() for pos in positions]
    counts = np.bincount(covered, minlength=N) if covered else np.zeros(N, dtype=int)
    if np.any(counts > 1):
        dupes = np.nonzero(counts > 1)[0].tolist()
        raise InvalidContractError(
            f"Assets {dupes} appear in more than one group"
        )
    unlabeled = np.nonzero(counts == 0)[0]
    if len(unlabeled) > 0:
        if not allow_unlabeled:
            raise InvalidContractError(
                f"{len(unlabeled)} assets carry no group label "
                f"(positions {unlabeled.tolist()[:10]}...); pass "
                f"allow_unlabeled=True to bucket them explicitly"
            )
        membership[UNLABELED] = sorted(int(p) for p in unlabeled)

    if not membership:
        raise InvalidContractError("group_labels produced no groups")

    views: List[SliceView] = []
    for group_name in sorted(membership):
        positions = tuple(membership[group_name])
        views.append(
            SliceView(
                slice_kind="group",
                slice_key=f"group:{group_name}",
                description=f"asset group {group_name!r}",
                params={"group": group_name, "n_members": len(positions)},
                time_indices=(),  # patched below
                asset_indices=positions,
            )
        )
    return views


def slice_by_quantile(
    factor_values: np.ndarray,
    quantiles: int = 5,
    method: str = "max",
) -> List[SliceView]:
    """Cross-sectional factor-quantile sub-universes per time.

    ``factor_values`` is a ``(T, N)`` panel (one factor). Bins reuse
    :func:`quant_evaluator.metrics.quantile.assign_quantiles` (percentile
    boundaries + the QE tie policy), so membership matches the quantile
    metrics exactly. Each quantile q becomes one ragged :class:`SliceView`
    whose asset members may differ per time; times where the bin is empty
    are dropped from that view.

    Fails closed on non-2-D input, ``quantiles < 2``, or a factor panel
    with no finite values.
    """
    arr = np.asarray(factor_values, dtype=float)
    if arr.ndim != 2:
        raise InvalidContractError(
            f"slice_by_quantile requires a (T, N) factor panel, got shape {arr.shape}"
        )
    if not isinstance(quantiles, (int, np.integer)) or isinstance(quantiles, bool):        raise InvalidContractError(f"quantiles must be an integer, got {quantiles!r}")
    quantiles = int(quantiles)
    if quantiles < 2:
        raise InvalidContractError(f"quantiles must be >= 2, got {quantiles}")
    T, N = arr.shape
    if T < 1 or N < 1:
        raise InvalidContractError(
            f"factor panel must be non-empty, got shape {arr.shape}"
        )
    if not np.any(np.isfinite(arr)):
        raise InvalidContractError("factor panel contains no finite values")

    assignments = assign_quantiles(arr, n_quantiles=quantiles, method=method)

    views: List[SliceView] = []
    for q in range(quantiles):
        per_time: List[Tuple[int, Tuple[int, ...]]] = []
        for t in range(T):
            members = tuple(int(i) for i in np.nonzero(assignments[t] == q)[0])
            if members:
                per_time.append((t, members))
        if not per_time:
            raise InvalidContractError(
                f"Quantile {q} is empty at every time; cannot form a slice"
            )
        time_indices = tuple(t for t, _ in per_time)
        members_union = sorted({a for _, members in per_time for a in members})
        views.append(
            SliceView(
                slice_kind="quantile",
                slice_key=f"quantile:Q{q}",
                description=f"factor-quantile {q} sub-universe per time",
                params={
                    "quantile": q,
                    "n_quantiles": quantiles,
                    "method": method,
                    "members_per_time": [len(m) for _, m in per_time],
                },
                time_indices=time_indices,
                asset_indices=tuple(members_union),
                per_time_asset_indices=tuple(m for _, m in per_time),
            )
        )
    return views


def slice_by_mask(boolean: np.ndarray) -> List[SliceView]:
    """Single explicit boolean asset selection.

    ``boolean`` must be a 1-D boolean array aligned with the asset axis.
    An all-False mask fails closed (empty sub-universe is never silently
    returned). Returns a one-element list with key ``mask:selected``.
    """
    arr = np.asarray(boolean)
    if arr.dtype != bool:
        raise InvalidContractError(
            f"slice_by_mask requires a boolean array, got dtype {arr.dtype}"
        )
    if arr.ndim != 1:
        raise InvalidContractError(
            f"slice_by_mask requires a 1-D mask, got ndim={arr.ndim}"
        )
    selected = tuple(int(i) for i in np.nonzero(arr)[0])
    if not selected:
        raise InvalidContractError("Boolean mask selects no assets")
    return [
        SliceView(
            slice_kind="mask",
            slice_key="mask:selected",
            description="explicit boolean asset selection",
            params={"n_selected": len(selected), "mask_sha": _hash_content(arr.astype(np.uint8).tolist())},
            time_indices=(0,),
            asset_indices=selected,
        )
    ]


def _patch_all_times(views: Sequence[SliceView], T: int) -> List[SliceView]:
    """Give group/mask views (which have no time spec) the full time axis."""
    patched = []
    for v in views:
        if v.time_indices:
            patched.append(v)
        else:
            patched.append(
                SliceView(
                    slice_kind=v.slice_kind,
                    slice_key=v.slice_key,
                    description=v.description,
                    params=v.params,
                    time_indices=tuple(range(T)),
                    asset_indices=v.asset_indices,
                    per_time_asset_indices=v.per_time_asset_indices,
                )
            )
    return patched


class SliceEngine:
    """Applies a metric compute function per :class:`SliceView`.

    The engine materializes each view against the batch and labels
    (indices-only until then), calls ``metric_fn(sub_batch, sub_labels)``,
    and wraps the result in the :class:`MetricArtifact` family. The
    result is keyed by ``slice_key`` and carries the view's ``spec_id``
    plus a metric-id prefix in provenance for reproducibility.
    """

    def __init__(self, min_slice_times: int = 1, min_slice_assets: int = 2) -> None:
        if min_slice_times < 1 or min_slice_assets < 1:
            raise InvalidContractError("Slice engine thresholds must be >= 1")
        self.min_slice_times = int(min_slice_times)
        self.min_slice_assets = int(min_slice_assets)

    def compute(
        self,
        metric_fn: Callable[[FactorBatch, LabelBundle], MetricArtifact],
        batch: FactorBatch,
        labels: LabelBundle,
        slices: Sequence[SliceView],
    ) -> Dict[str, MetricArtifact]:
        """Run ``metric_fn`` on every slice view of ``batch``/``labels``.

        Returns ``{slice_key: MetricArtifact}``. Fails closed if any view
        is too small (fewer than ``min_slice_times`` times or
        ``min_slice_assets`` assets) or if duplicate slice keys collide.
        """
        if not slices:
            raise InvalidContractError("SliceEngine.compute requires at least one slice")
        views = _patch_all_times(list(slices), batch.num_times)
        results: Dict[str, MetricArtifact] = {}
        for view in views:
            if view.num_times < self.min_slice_times:
                raise InvalidContractError(
                    f"Slice {view.slice_key!r} has {view.num_times} times "
                    f"(< {self.min_slice_times})"
                )
            n_assets = view.num_assets or batch.num_assets
            if n_assets < self.min_slice_assets:
                raise InvalidContractError(
                    f"Slice {view.slice_key!r} has {n_assets} assets "
                    f"(< {self.min_slice_assets})"
                )
            if view.slice_key in results:
                raise InvalidContractError(
                    f"Duplicate slice key {view.slice_key!r} in engine input"
                )
            sub_batch = view.sub_batch(batch)
            sub_labels = view.sub_labels(labels)
            artifact = metric_fn(sub_batch, sub_labels)
            if not isinstance(artifact, MetricArtifact):
                raise InvalidContractError(
                    f"metric_fn must return a MetricArtifact, got "
                    f"{type(artifact).__name__} for slice {view.slice_key!r}"
                )
            provenance = dict(artifact.provenance)
            provenance["slice_kind"] = view.slice_kind
            provenance["slice_spec_id"] = view.spec_id
            provenance["slice_description"] = view.description
            patched = type(artifact)(
                metric_id=artifact.metric_id,
                domain=artifact.domain,
                artifact_kind=artifact.artifact_kind,
                provenance=provenance,
                created_from=artifact.created_from,
                **artifact._payload_dict(),
            )
            results[view.slice_key] = patched
        return results
