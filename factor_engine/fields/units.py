"""Canonical unit names and source-to-canonical normalization rules."""
from __future__ import annotations

from dataclasses import dataclass


UNIT_DIMENSIONLESS = "dimensionless"
UNIT_BOOLEAN = "boolean"
UNIT_DATE = "date"
UNIT_DATETIME = "datetime"
UNIT_IDENTIFIER = "identifier"
UNIT_TEXT = "text"
UNIT_CNY = "CNY"
UNIT_CNY_10K = "CNY_10K"
UNIT_SHARE = "share"
UNIT_SHARE_10K = "share_10K"
UNIT_RATIO = "ratio"
UNIT_PERCENT = "percent"
UNIT_BASIS_POINT = "basis_point"
# R17-012: prices / EPS / cash dividend are CNY *per share* (not a bare CNY
# amount); stock dividend / stock transfer are dimensionless per-share ratios
# (NOT share counts).  These spellings keep the dimension distinct from a money
# amount / share count while mapping to the same scalar scale.
UNIT_CNY_PER_SHARE = "CNY/share"
UNIT_SHARE_RATIO = "share_ratio"

_UNIT_ALIASES = {
    "": UNIT_DIMENSIONLESS,
    "1": UNIT_DIMENSIONLESS,
    "none": UNIT_DIMENSIONLESS,
    "unitless": UNIT_DIMENSIONLESS,
    "bool": UNIT_BOOLEAN,
    "boolean": UNIT_BOOLEAN,
    "date": UNIT_DATE,
    "datetime": UNIT_DATETIME,
    "timestamp": UNIT_DATETIME,
    "id": UNIT_IDENTIFIER,
    "identifier": UNIT_IDENTIFIER,
    "cny/share": UNIT_CNY_PER_SHARE,
    "cny per share": UNIT_CNY_PER_SHARE,
    "yuan/share": UNIT_CNY_PER_SHARE,
    # R40 #185: the canonical per-share ratio constant must resolve to itself.
    "share_ratio": UNIT_SHARE_RATIO,
    "string": UNIT_TEXT,
    "text": UNIT_TEXT,
    "cny": UNIT_CNY,
    "yuan": UNIT_CNY,
    "rmb": UNIT_CNY,
    "cny_10k": UNIT_CNY_10K,
    "wan_yuan": UNIT_CNY_10K,
    "share": UNIT_SHARE,
    "shares": UNIT_SHARE,
    "share_10k": UNIT_SHARE_10K,
    "wan_share": UNIT_SHARE_10K,
    "ratio": UNIT_RATIO,
    "fraction": UNIT_RATIO,
    "percent": UNIT_PERCENT,
    "%": UNIT_PERCENT,
    "percentage_point": UNIT_PERCENT,
    "bp": UNIT_BASIS_POINT,
    "bps": UNIT_BASIS_POINT,
    "basis_point": UNIT_BASIS_POINT,
}


@dataclass(frozen=True)
class UnitNormalization:
    source_unit: str
    target_unit: str
    multiplier: float

    def apply(self, value):
        return value * self.multiplier


class KnownUnitId(str):
    """A canonical, registry-backed unit identifier (R40 #184).

    ``KnownUnitId`` is a ``str`` so it remains a drop-in for the legacy string
    vocabulary (``canonical_unit``/``unit_normalization`` compare by value), but
    it is only produced by :func:`canonical_unit_known`, which verifies the
    spelling exists in the canonical alias table.  An unknown spelling raises
    :class:`UnknownUnitError` instead of silently passing the raw string through
    (which let a typo like ``"cny/shar"`` flow into the unit algebra and
    ``unit_normalization`` as a distinct "known" unit).
    """


class UnknownUnitError(ValueError):
    """An unknown unit spelling was passed to a strict unit entry point."""


@dataclass(frozen=True)
class OpaqueUnit:
    """A research-only unit that does NOT participate in unit algebra (R40 #184).

    ``OpaqueUnit(label)`` is the explicit opt-in for an unregistered spelling in
    research mode: it is carried as a descriptive label, is never coerced into a
    ``KnownUnitId``, and any arithmetic that would mix it with a known unit or
    scale it numerically raises ``UnknownUnitError``.  Production never
    constructs one.
    """

    label: str


def _canonical_unit_or_none(unit: str | None) -> str | None:
    """Return the canonical spelling, or ``None`` when it is unknown."""
    if unit is None:
        return UNIT_DIMENSIONLESS
    raw = str(unit).strip()
    return _UNIT_ALIASES.get(raw.lower())


def canonical_unit_known(unit: str | None) -> KnownUnitId:
    """Production-strict canonical unit lookup.

    Returns a :class:`KnownUnitId` for any spelling that resolves in the alias
    table; raises :class:`UnknownUnitError` otherwise (R40 #184).  ``None`` maps
    to ``UNIT_DIMENSIONLESS`` (a valid known unit).  Callers in production paths
    MUST use this entry point so an unknown spelling fails closed instead of
    being silently carried as its own "canonical" unit.
    """
    canonical = _canonical_unit_or_none(unit)
    if canonical is None:
        raise UnknownUnitError(
            f"unknown unit spelling {str(unit)!r} has no canonical mapping; "
            "production requires a registered unit (use OpaqueUnit only in "
            "explicit research mode)"
        )
    return KnownUnitId(canonical)


def canonical_unit(unit: str | None) -> str:
    """Return the stable catalog spelling for a unit name.

    Backward-compatible lenient entry point: an unknown spelling is returned
    unchanged (legacy callers that treat the string as an opaque label keep
    working).  New code / production paths must use :func:`canonical_unit_known`
    so unknown spellings fail closed.
    """

    if unit is None:
        return UNIT_DIMENSIONLESS
    raw = str(unit).strip()
    return _UNIT_ALIASES.get(raw.lower(), raw)


def unit_normalization(source_unit: str | None, target_unit: str | None) -> UnitNormalization:
    """Describe a lossless scalar conversion between supported units.

    Ratios use ``1.0 == 100%``; a basis point is ``0.0001`` ratio. Currency
    and share counts are canonicalized to their base units.
    """

    source = canonical_unit(source_unit)
    target = canonical_unit(target_unit)
    if source == target:
        return UnitNormalization(source, target, 1.0)
    to_base = {
        UNIT_CNY: ("currency", 1.0),
        UNIT_CNY_10K: ("currency", 10_000.0),
        # R17-012: CNY/share is a per-share money dimension (scale 1.0 vs CNY —
        # a "CNY" amount and "CNY/share" have the same numeric scale, but the
        # dimension is distinct so price * amount is not silently conflated).
        UNIT_CNY_PER_SHARE: ("currency", 1.0),
        UNIT_SHARE: ("count", 1.0),
        UNIT_SHARE_10K: ("count", 10_000.0),
        UNIT_RATIO: ("ratio", 1.0),
        UNIT_SHARE_RATIO: ("ratio", 1.0),
        UNIT_PERCENT: ("ratio", 0.01),
        UNIT_BASIS_POINT: ("ratio", 0.0001),
    }
    if source not in to_base or target not in to_base:
        raise ValueError(f"incompatible units: {source!r} -> {target!r}")
    source_dimension, source_scale = to_base[source]
    target_dimension, target_scale = to_base[target]
    if source_dimension != target_dimension:
        raise ValueError(f"incompatible units: {source!r} -> {target!r}")
    return UnitNormalization(source, target, source_scale / target_scale)


def normalize_unit_value(value, source_unit: str | None, target_unit: str | None):
    """Normalize a scalar, NumPy object, Series, or DataFrame by multiplication."""

    return unit_normalization(source_unit, target_unit).apply(value)


__all__ = [
    "KnownUnitId",
    "OpaqueUnit",
    "UNIT_BASIS_POINT", "UNIT_BOOLEAN", "UNIT_CNY", "UNIT_CNY_10K",
    "UNIT_CNY_PER_SHARE", "UNIT_SHARE_RATIO",
    "UNIT_DATE", "UNIT_DATETIME", "UNIT_DIMENSIONLESS", "UNIT_IDENTIFIER",
    "UNIT_PERCENT", "UNIT_RATIO", "UNIT_SHARE", "UNIT_SHARE_10K", "UNIT_TEXT",
    "UnitNormalization", "UnknownUnitError", "canonical_unit",
    "canonical_unit_known", "normalize_unit_value", "unit_normalization",
]
