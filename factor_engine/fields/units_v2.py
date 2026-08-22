"""Currency/denominator-aware unit specifications (additive v2 layer).

The legacy ``fields/units.py`` string vocabulary (``CNY``, ``share``, ``ratio``,
``percent``, ``basis_point``) is preserved untouched for the existing DSL/IR.
This module adds a structural ``UnitSpec`` that carries the dimensions the
multi-market plan requires (spec §12-§14):

- ``money`` with a ``currency`` (CNY / USD) — cross-currency money must not be
  added directly;
- ``price`` with ``currency`` + ``denominator`` (price-per-share);
- ``count`` (shares), ``ratio``, ``boolean``, ``date``, ``datetime``, etc.

``percent`` and ``basis_point`` are *source* units only: at the canonical
semantic layer every return/ratio is a ``ratio`` (``0.05 == 5%``).  The legacy
string -> v2 mapping is provided by :func:`unit_spec_from_legacy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .units import canonical_unit as _canonical_unit

# Dimensions
DIM_RATIO = "ratio"
DIM_MONEY = "money"
DIM_PRICE = "price"
DIM_COUNT = "count"
DIM_BOOLEAN = "boolean"
DIM_DATE = "date"
DIM_DATETIME = "datetime"
DIM_IDENTIFIER = "identifier"
DIM_TEXT = "text"
DIM_DIMENSIONLESS = "dimensionless"

# Denominators / commodities
DENOM_SHARE = "share"

# Known currencies (the markets FactorEngine currently serves).
CURRENCY_CNY = "CNY"
CURRENCY_USD = "USD"
_KNOWN_CURRENCIES = frozenset({CURRENCY_CNY, CURRENCY_USD})


@dataclass(frozen=True)
class UnitSpec:
    """Structural unit: ``(dimension, currency?, denominator?)``.

    ``currency`` is required for ``money`` and ``price`` EXCEPT for the
    market-local wildcard (``currency=None``, e.g. ``LOCAL_MONEY``), whose
    concrete currency is resolved from the ``MarketContext`` by
    :func:`resolve_unit` — a canonical concept must never hardcode CNY while US
    bindings declare USD.  ``denominator`` is required for ``price``.  ``scale``
    is the legacy multiplier for source-only spellings (percent=0.01,
    basis_point=0.0001) — canonical units use scale=1.
    """

    dimension: str
    currency: str | None = None
    denominator: str | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension == DIM_PRICE:
            if self.denominator is None:
                raise ValueError("price unit requires a denominator")
        if self.currency is not None:
            object.__setattr__(self, "currency", str(self.currency).upper())
        if self.denominator is not None:
            object.__setattr__(self, "denominator", str(self.denominator).lower())
        if self.scale <= 0:
            raise ValueError(f"unit scale must be positive, got {self.scale}")

    # -- constructors ------------------------------------------------------
    @classmethod
    def ratio(cls, scale: float = 1.0) -> "UnitSpec":
        return cls(dimension=DIM_RATIO, scale=scale)

    @classmethod
    def money(cls, currency: str | None, scale: float = 1.0) -> "UnitSpec":
        """Money in ``currency``, or market-local when ``currency`` is None."""
        return cls(dimension=DIM_MONEY, currency=currency, scale=scale)

    @classmethod
    def price(cls, currency: str | None, scale: float = 1.0) -> "UnitSpec":
        """Price per share in ``currency``, or market-local when None."""
        return cls(
            dimension=DIM_PRICE, currency=currency, denominator=DENOM_SHARE,
            scale=scale,
        )

    @classmethod
    def count(cls, scale: float = 1.0) -> "UnitSpec":
        return cls(dimension=DIM_COUNT, scale=scale)

    # -- predicates --------------------------------------------------------
    @property
    def is_ratio(self) -> bool:
        return self.dimension == DIM_RATIO

    @property
    def is_money(self) -> bool:
        return self.dimension == DIM_MONEY

    @property
    def is_price(self) -> bool:
        return self.dimension == DIM_PRICE

    @property
    def is_count(self) -> bool:
        return self.dimension == DIM_COUNT

    @property
    def is_local(self) -> bool:
        """True for the market-local wildcard (currency resolved from context)."""
        return self.dimension in (DIM_MONEY, DIM_PRICE) and self.currency is None

    def is_compatible_with(self, other: "UnitSpec") -> bool:
        """Same dimension AND same currency/denominator for money/price.

        A CNY ``money`` is NOT compatible with a USD ``money``; a CNY/share
        price is NOT compatible with a USD/share price.  A market-local unit
        (``currency=None``) is compatible with any concrete currency — the
        canonical layer never hardcodes CNY.  Ratios/counts compare on dimension
        alone (canonical ratios have scale=1).
        """
        if not isinstance(other, UnitSpec):
            return False
        if self.dimension != other.dimension:
            return False
        if self.dimension in (DIM_MONEY, DIM_PRICE):
            if self.currency != other.currency and self.currency is not None and other.currency is not None:
                return False
            if self.dimension == DIM_PRICE and self.denominator != other.denominator:
                return False
        return True

    def assert_compatible_with(self, other: "UnitSpec") -> None:
        """Raise a descriptive error when units must not be combined."""
        if not isinstance(other, UnitSpec):
            raise TypeError(
                f"expected a UnitSpec, got {type(other).__name__}"
            )
        if self.dimension != other.dimension:
            raise ValueError(
                f"incompatible unit dimensions: {self} vs {other}"
            )
        if self.dimension in (DIM_MONEY, DIM_PRICE):
            if (
                self.currency != other.currency
                and self.currency is not None
                and other.currency is not None
            ):
                raise ValueError(
                    f"cross-currency unit arithmetic not allowed: {self} vs {other}"
                )
            if (
                self.dimension == DIM_PRICE
                and self.denominator != other.denominator
            ):
                raise ValueError(
                    f"cross-denominator price arithmetic not allowed: {self} vs {other}"
                )

    def __str__(self) -> str:
        if self.dimension == DIM_MONEY:
            return self.currency or "local_money"
        if self.dimension == DIM_PRICE:
            return f"{(self.currency or 'local')}/{self.denominator}"
        return self.dimension

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "currency": self.currency,
            "denominator": self.denominator,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "UnitSpec":
        return cls(
            dimension=str(raw["dimension"]),
            currency=raw.get("currency"),
            denominator=raw.get("denominator"),
            scale=float(raw.get("scale", 1.0)),
        )


# ---------------------------------------------------------------------------
# UnitExpr — dimensionless-exponent unit algebra (R40 #186).
#
# ``UnitSpec`` is a *structural* unit (dimension + currency + denominator +
# scale); ``UnitExpr`` adds the exponent algebra the compiler needs to prove
# dimensional legality of derived formulas:
#   * MUL   -> unit(x) * unit(y)       (exponents add per base dimension)
#   * DIV   -> unit(x) / unit(y)       (exponents subtract per base dimension)
#   * POW   -> base^exponent           (exponent * 2 for x**2, 1/2 for sqrt)
#   * LOG   -> requires a dimensionless operand
#   * RANK  -> ratio / dimensionless
# A ``UnitExpr`` is expressed as a mapping ``dimension -> exponent`` over the
# primitive dimensions.  ``UnitSpec`` converts to/from ``UnitExpr`` losslessly
# for the primitive dimensions (currency/denominator are carried through the
# structural spec only; a cross-currency DIV already fails in ``UnitSpec``).
# ---------------------------------------------------------------------------
class UnitExpr:
    """A product of primitive dimensions to rational powers.

    ``factors`` maps a primitive dimension key to an exponent (``int`` or
    ``float``).  ``pow(UnitExpr, 1/2)`` yields a ``sqrt`` (exponent 0.5).
    The dimensionless expression is the empty mapping.
    """

    __slots__ = ("factors",)

    def __init__(self, factors: dict[str, float | int] | None = None):
        object.__setattr__  # no-op; class uses __slots__
        self.factors = {k: float(v) for k, v in (factors or {}).items() if float(v) != 0.0}

    @property
    def is_dimensionless(self) -> bool:
        return not self.factors

    def __mul__(self, other: "UnitExpr") -> "UnitExpr":
        merged = dict(self.factors)
        for dim, exp in other.factors.items():
            merged[dim] = merged.get(dim, 0.0) + exp
        return UnitExpr({d: e for d, e in merged.items() if e != 0.0})

    def __truediv__(self, other: "UnitExpr") -> "UnitExpr":
        merged = dict(self.factors)
        for dim, exp in other.factors.items():
            merged[dim] = merged.get(dim, 0.0) - exp
        return UnitExpr({d: e for d, e in merged.items() if e != 0.0})

    def __pow__(self, exponent: float) -> "UnitExpr":
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
            raise TypeError(f"UnitExpr.__pow__ needs a numeric exponent, got {exponent!r}")
        return UnitExpr({d: e * float(exponent) for d, e in self.factors.items()})

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnitExpr):
            return NotImplemented
        return self.factors == other.factors

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.factors.items())))

    def __repr__(self) -> str:
        if not self.factors:
            return "UnitExpr(dimensionless)"
        parts = []
        for dim in sorted(self.factors):
            exp = self.factors[dim]
            if exp == 1.0:
                parts.append(dim)
            elif exp == int(exp):
                parts.append(f"{dim}^{int(exp)}")
            else:
                parts.append(f"{dim}^{exp:g}")
        return "UnitExpr(" + "*".join(parts) + ")"

    # -- construction from UnitSpec ---------------------------------------
    @classmethod
    def from_spec(cls, spec: UnitSpec | None) -> "UnitExpr":
        """Lossless primitive-dimension view of a :class:`UnitSpec`.

        ``price`` maps to ``money^1 * share^-1``; ``money`` to ``money^1``;
        ``count`` to ``count^1``; a ratio/dimensionless to the empty mapping.
        """
        if spec is None:
            return cls()
        dim = spec.dimension
        if dim == DIM_PRICE:
            return cls({"money": 1.0, "count": -1.0})
        if dim in (DIM_MONEY,):
            return cls({"money": 1.0})
        if dim == DIM_COUNT:
            return cls({"count": 1.0})
        if dim in (DIM_RATIO, DIM_DIMENSIONLESS):
            return cls()
        # Non-numeric units (boolean/date/datetime/identifier/text) have no
        # numeric dimension; treated as dimensionless for algebra but flagged
        # by the compiler's legality pass when mixed with money/count.
        return cls()


UNIT_EXPR_DIMENSIONLESS = UnitExpr()
UNIT_EXPR_MONEY = UnitExpr({"money": 1.0})
UNIT_EXPR_COUNT = UnitExpr({"count": 1.0})
UNIT_EXPR_RATIO = UnitExpr()  # canonical ratio is dimensionless


def unit_algebra_kind(expr: UnitExpr) -> str:
    """Classify a ``UnitExpr`` for the operator legality pass (R40 #186).

    ``dimensionless`` -> the empty product (also pure ratio);
    ``money`` / ``money_count`` (price) / ``count`` -> the primitive economic
    dimensions; ``mixed`` -> anything else (a ``price^2`` energy-like quantity
    is legal but must be explicitly declared, never silently produced by an
    elementwise MUL of two prices).
    """
    keys = set(expr.factors)
    if not keys:
        return "dimensionless"
    if keys == {"money"} and expr.factors["money"] == 1.0:
        return "money"
    if keys == {"money", "count"} and expr.factors["money"] == 1.0 and expr.factors["count"] == -1.0:
        return "price"
    if keys == {"count"} and expr.factors["count"] == 1.0:
        return "count"
    return "mixed"


class UnitAlgebraError(ValueError):
    """An operator would combine units illegally (R40 #186).

    Raised by :func:`assert_log_operand_dimensionless` and the operator
    legality pass when e.g. ``log(close)`` is attempted (a price has dimension
    ``money*count^-1`` — the logarithm is only defined for a dimensionless
    operand).
    """


def assert_log_operand_dimensionless(operand: UnitExpr, *, operator: str = "") -> None:
    """Reject ``log``/``log1p``/``log10`` of a dimensioned operand (R40 #186).

    ``log`` requires a dimensionless operand; ``log(price)`` is economically
    meaningless and is rejected at compile time by the operator legality pass.
    A ratio / pure dimensionless operand passes.
    """
    if operand.is_dimensionless:
        return
    raise UnitAlgebraError(
        f"{operator or 'log'} requires a dimensionless operand, got {operand!r} "
        "(a price/amount/count cannot be passed through a logarithm — use a "
        "return/ratio or a protected log that declares the unit transformation)"
    )


def assert_rank_produces_dimensionless() -> UnitExpr:
    """A rank/bucket operator produces a dimensionless (ratio) expression."""
    return UNIT_EXPR_RATIO


# ---------------------------------------------------------------------------
# Legacy string vocabulary -> v2 UnitSpec (source units keep their scale).
# ---------------------------------------------------------------------------
def unit_spec_from_legacy(unit: str | None, *, canonical: bool = False) -> UnitSpec:
    """Map a legacy unit string to a structural :class:`UnitSpec`.

    With ``canonical=True`` percent/basis_point collapse to a scale-1 ratio
    (the canonical semantic layer always uses ``0.05 == 5%``).
    """
    key = _canonical_unit(unit)  # returns the stable catalog spelling
    if key in ("percent", "basis_point"):
        scale = 1.0 if canonical else (0.01 if key == "percent" else 0.0001)
        return UnitSpec.ratio(scale=scale)
    if key == "ratio":
        return UnitSpec.ratio()
    if key == "dimensionless":
        return UnitSpec(dimension=DIM_DIMENSIONLESS)
    if key == "boolean":
        return UnitSpec(dimension=DIM_BOOLEAN)
    if key in ("date",):
        return UnitSpec(dimension=DIM_DATE)
    if key in ("datetime",):
        return UnitSpec(dimension=DIM_DATETIME)
    if key in ("identifier",):
        return UnitSpec(dimension=DIM_IDENTIFIER)
    if key in ("text",):
        return UnitSpec(dimension=DIM_TEXT)
    if key in ("share", "share_10K"):
        return UnitSpec.count(scale=1.0 if key == "share" else 10_000.0)
    if key in ("CNY", "CNY_10K"):
        return UnitSpec.money(
            CURRENCY_CNY, scale=1.0 if key == "CNY" else 10_000.0
        )
    if key in ("USD", "USD_10K"):
        return UnitSpec.money(
            CURRENCY_USD, scale=1.0 if key == "USD" else 10_000.0
        )
    if key.endswith("/share") or key in ("price",):
        return UnitSpec.price(CURRENCY_USD if "usd" in key.lower() else CURRENCY_CNY)
    # Fallback: treat an uppercase currency code as money.
    if key.upper() in _KNOWN_CURRENCIES:
        return UnitSpec.money(key.upper())
    return UnitSpec(dimension=DIM_DIMENSIONLESS)


def legacy_unit_string(spec: UnitSpec) -> str:
    """Best-effort v2 -> legacy string (for manifest export)."""
    if spec.dimension == DIM_RATIO:
        if abs(spec.scale - 0.01) < 1e-12:
            return "percent"
        if abs(spec.scale - 0.0001) < 1e-12:
            return "basis_point"
        return "ratio"
    if spec.dimension == DIM_MONEY:
        if spec.currency is None:
            return "local_money"
        if spec.currency == CURRENCY_CNY:
            return "CNY" if abs(spec.scale - 1.0) < 1e-12 else "CNY_10K"
        return "USD" if abs(spec.scale - 1.0) < 1e-12 else "USD_10K"
    if spec.dimension == DIM_PRICE:
        return f"{(spec.currency or 'local')}/{spec.denominator}"
    if spec.dimension == DIM_COUNT:
        return "share" if abs(spec.scale - 1.0) < 1e-12 else "share_10K"
    if spec.dimension == DIM_BOOLEAN:
        return "boolean"
    if spec.dimension == DIM_DATE:
        return "date"
    if spec.dimension == DIM_DATETIME:
        return "datetime"
    if spec.dimension == DIM_IDENTIFIER:
        return "identifier"
    if spec.dimension == DIM_TEXT:
        return "text"
    return "dimensionless"


# ---------------------------------------------------------------------------
# Convenience singletons.
# ---------------------------------------------------------------------------
RATIO = UnitSpec.ratio()
CNY = UnitSpec.money(CURRENCY_CNY)
USD = UnitSpec.money(CURRENCY_USD)
CNY_PER_SHARE = UnitSpec.price(CURRENCY_CNY)
USD_PER_SHARE = UnitSpec.price(CURRENCY_USD)
SHARES = UnitSpec.count()
BOOLEAN = UnitSpec(dimension=DIM_BOOLEAN)
DATE = UnitSpec(dimension=DIM_DATE)
DATETIME = UnitSpec(dimension=DIM_DATETIME)
IDENTIFIER = UnitSpec(dimension=DIM_IDENTIFIER)
TEXT = UnitSpec(dimension=DIM_TEXT)
DIMENSIONLESS = UnitSpec(dimension=DIM_DIMENSIONLESS)
# Market-local wildcards: the canonical concept layer must not hardcode CNY
# while US bindings declare USD — the concrete currency is resolved from the
# MarketContext (see ``resolve_unit``).
LOCAL_MONEY = UnitSpec.money(None)
LOCAL_PRICE_PER_SHARE = UnitSpec.price(None)
LOCAL_MONEY_10K = UnitSpec.money(None, scale=10_000.0)


def resolve_unit(spec: UnitSpec, market: str | None = None, *, currency: str | None = None) -> UnitSpec:
    """Fill a market-local unit's currency from the market context.

    ``currency=None`` resolves from ``market`` (ashare -> CNY, us -> USD); an
    explicit ``currency`` wins.  Concrete units are returned unchanged.
    """
    if spec is None:
        return spec
    if spec.dimension not in (DIM_MONEY, DIM_PRICE) or spec.currency is not None:
        return spec
    cur = currency or (CURRENCY_CNY if (market or "").strip().lower() == "ashare" else CURRENCY_USD)
    return UnitSpec(
        dimension=spec.dimension,
        currency=cur,
        denominator=spec.denominator,
        scale=spec.scale,
    )


__all__ = [
    "BOOLEAN",
    "CNY",
    "CNY_PER_SHARE",
    "CURRENCY_CNY",
    "CURRENCY_USD",
    "DATE",
    "DATETIME",
    "DIM_BOOLEAN",
    "DIM_COUNT",
    "DIM_DATE",
    "DIM_DATETIME",
    "DIM_DIMENSIONLESS",
    "DIM_IDENTIFIER",
    "DIM_MONEY",
    "DIM_PRICE",
    "DIM_RATIO",
    "DIM_TEXT",
    "DIMENSIONLESS",
    "DENOM_SHARE",
    "IDENTIFIER",
    "LOCAL_MONEY",
    "LOCAL_MONEY_10K",
    "LOCAL_PRICE_PER_SHARE",
    "RATIO",
    "SHARES",
    "TEXT",
    "UNIT_EXPR_COUNT",
    "UNIT_EXPR_DIMENSIONLESS",
    "UNIT_EXPR_MONEY",
    "UNIT_EXPR_RATIO",
    "UnitAlgebraError",
    "UnitExpr",
    "UnitSpec",
    "assert_log_operand_dimensionless",
    "assert_rank_produces_dimensionless",
    "legacy_unit_string",
    "resolve_unit",
    "unit_algebra_kind",
    "unit_spec_from_legacy",
]
