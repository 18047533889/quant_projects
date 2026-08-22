"""
Sealed-test split contract (QE-side leakage gate).

Closes the R21-LEAKAGE-PIT-AUDIT Q5 gap: quant_evaluator previously had no
sealed/snapshot/split concept at all, so an evaluation could be pointed at a
label split overlapping the factor/label information boundary with nothing
stopping it (the factor_optimizer-side ``freeze_for_sealed_test`` guard
rejects overlapping plans at search time, but QE itself never re-checked).

This module adds the minimal, backward-compatible half of that gate:

- :class:`SealedSplitRef` — an immutable reference carrying a split
  identifier plus the data window it covers (``start_time`` / ``end_time``
  inclusive, or a single ``as_of`` for point-in-time snapshots).
- :func:`check_sealed_split_overlap` — a fail-closed overlap guard: when an
  evaluation request carries a ``split_ref`` whose window overlaps the
  factor/label information boundary of the inputs it is evaluated against, a
  typed :class:`SealedSplitOverlapError` is raised instead of silently
  evaluating.  ``split_ref=None`` means *no check* (preserves every existing
  caller); strict only when a reference is provided.

Overlap semantics (fail-closed):

1. A ``split_ref`` window that touches or contains any factor or label
   decision time is an overlap.  The sealed segment must be strictly AFTER
   every decision time — matching ``_sealed_test_disjoint`` in
   factor_optimizer/search/runner.py, which rejects a sealed test mask
   sharing any position with the search-time train/validation masks.
2. Forward-label reach is honored: a label whose window
   ``(label_start_time, label_end_time)`` ends after the split ``end_time``
   is a leak (the label saw future information relative to the sealed
   segment) and is rejected.
3. A ``split_ref`` that has no orderable times, or a window that is
   empty/inverted, is rejected outright — malformed references never
   silently disable the gate.

Reference points are chosen from explicit timestamps already required by
the input contracts (``LabelBundle.decision_time`` /
``label_start_time`` / ``label_end_time``, ``FactorBatch.time_axis.values``)
rather than array offsets, so positional re-alignment cannot mask overlap.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from quant_evaluator.contracts.errors import SealedSplitOverlapError


@dataclass(frozen=True)
class SealedSplitRef:
    """
    Immutable reference to a sealed test split and the data window it covers.

    Parameters
    ----------
    split_id:
        Identifier of the sealed split (mirrors the factor_optimizer
        ``SplitPlan.split_id`` / ``SealedTestHandle.split_id`` identity).
    start_time / end_time:
        Inclusive window the sealed split covers on the data's time axis.
        Either both or neither must be supplied; at least one of the
        window/as_of forms is required.
    as_of:
        Point-in-time boundary for a snapshot-style reference (an ``as_of``
        ``t`` behaves exactly like ``(start_time, end_time) == (t, t)``).
    metadata:
        Optional free-form provenance (e.g. the search session id the seal
        came from).
    """

    split_id: str
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    as_of: Optional[Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.split_id, str) or not self.split_id.strip():
            raise ValueError("split_id must be a non-empty string")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.as_of is None and (self.start_time is None or self.end_time is None):
            raise ValueError(
                "SealedSplitRef requires a window (start_time + end_time) or as_of"
            )
        if self.as_of is not None and (self.start_time is not None or self.end_time is not None):
            raise ValueError(
                "SealedSplitRef accepts a window (start_time/end_time) OR as_of, not both"
            )
        if self.start_time is not None:
            try:
                if self.start_time > self.end_time:
                    raise ValueError(
                        f"sealed split window is inverted: start {self.start_time!r} > "
                        f"end {self.end_time!r}"
                    )
            except TypeError as exc:
                raise ValueError("sealed split window endpoints must be mutually orderable") from exc

    @property
    def window(self) -> Tuple[Any, Any]:
        """Inclusive (start, end) window the reference covers."""
        if self.as_of is not None:
            return (self.as_of, self.as_of)
        return (self.start_time, self.end_time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict."""
        payload = {
            "split_id": self.split_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "as_of": self.as_of,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SealedSplitRef":
        """Rebuild from :meth:`to_dict` output."""
        if not isinstance(payload, dict):
            raise TypeError("SealedSplitRef.from_dict requires a dict")
        return cls(
            split_id=payload["split_id"],
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
            as_of=payload.get("as_of"),
            metadata=payload.get("metadata"),
        )

    @classmethod
    def _default_dict(cls) -> Dict[str, Any]:
        """Serialized form for the default ``split_ref=None`` (no check)."""
        return None


def check_sealed_split_overlap(
    split_ref: Optional[SealedSplitRef],
    decision_times: Tuple[Any, ...],
    label_start_times: Tuple[Any, ...],
    label_end_times: Tuple[Any, ...],
    factor_times: Optional[Tuple[Any, ...]] = None,
) -> None:
    """
    Fail-closed overlap guard for a sealed split reference.

    ``split_ref=None`` returns immediately (backward compatibility: no
    check).  When a reference is provided, ANY of the following raises
    :class:`SealedSplitOverlapError` instead of silently evaluating:

    - a factor or label decision time at or after the split start
      (``time >= start``), or a factor/label time equal to an ``as_of``;
    - a label whose window ends after the split end (``label_end > end``),
      which is forward-label leakage into the sealed segment;
    - a reference with no orderable time points or an empty window
      (malformed references never disarm the gate).

    The check is deliberately conservative: the sealed segment is required
    to be strictly after ALL decision times (matching the FO
    ``_sealed_test_disjoint`` contract) and strictly after every label
    window's end.
    """
    if split_ref is None:
        return

    if isinstance(split_ref, dict):
        split_ref = SealedSplitRef.from_dict(split_ref)

    start, end = split_ref.window

    # Fail-closed on unorderable reference times: a reference that cannot be
    # compared cannot be proven disjoint.
    try:
        for t in decision_times:
            if t >= start:
                raise _overlap(split_ref, "decision time", t, "at or after sealed split start", start)
        for t in label_start_times:
            if t >= start:
                raise _overlap(split_ref, "label start time", t, "at or after sealed split start", start)
        for t in label_end_times:
            if t > end:
                raise _overlap(split_ref, "label end time", t, "after sealed split end", end)
        if factor_times is not None:
            for t in factor_times:
                if t >= start:
                    raise _overlap(split_ref, "factor time", t, "at or after sealed split start", start)
    except TypeError as exc:
        raise _unorderable(split_ref, exc) from exc


def _overlap(split_ref: SealedSplitRef, kind: str, value: Any, relation: str, boundary: Any) -> SealedSplitOverlapError:
    return SealedSplitOverlapError(
        f"sealed split '{split_ref.split_id}' overlaps the factor/label "
        f"information boundary: {kind} {value!r} is {relation} "
        f"{boundary!r}; refusing to evaluate against a split that is not "
        f"strictly after all decision/label information"
    )


def _unorderable(split_ref: SealedSplitRef, exc: TypeError) -> SealedSplitOverlapError:
    return SealedSplitOverlapError(
        f"sealed split '{split_ref.split_id}' window times are not orderable "
        f"against the label/factor time axis ({exc}); refusing to evaluate "
        f"with an unverifiable split boundary"
    )


__all__ = ["SealedSplitRef", "check_sealed_split_overlap", "SealedSplitOverlapError"]
