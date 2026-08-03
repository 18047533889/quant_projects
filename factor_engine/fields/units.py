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


def canonical_unit(unit: str | None) -> str:
    """Return the stable catalog spelling for a unit name."""

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
        UNIT_SHARE: ("count", 1.0),
        UNIT_SHARE_10K: ("count", 10_000.0),
        UNIT_RATIO: ("ratio", 1.0),
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
    "UNIT_BASIS_POINT", "UNIT_BOOLEAN", "UNIT_CNY", "UNIT_CNY_10K",
    "UNIT_DATE", "UNIT_DATETIME", "UNIT_DIMENSIONLESS", "UNIT_IDENTIFIER",
    "UNIT_PERCENT", "UNIT_RATIO", "UNIT_SHARE", "UNIT_SHARE_10K", "UNIT_TEXT",
    "UnitNormalization", "canonical_unit", "normalize_unit_value", "unit_normalization",
]
