"""Price-interval inputs, distinct from forward prediction labels.

These contracts authorize research probes only. They are not a replacement
for a share/cash/fill ledger or a certificate of executable market neutrality.
"""
from dataclasses import dataclass
import numpy as np
from .factor_batch import AxisRef
from .metric_artifacts import _freeze_array
from ._hashutil import stable_content_hex
from .portfolio_schedule import cohort_schedule


@dataclass(frozen=True)
class HoldingReturnPanel:
    values: np.ndarray
    time_axis: AxisRef
    asset_axis: AxisRef
    source_ref: str
    price_basis: str

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.shape != (self.time_axis.size, self.asset_axis.size):
            raise ValueError("HoldingReturnPanel requires named (time, asset) axes")
        if self.time_axis.values is None or self.asset_axis.values is None:
            raise ValueError("HoldingReturnPanel requires explicit time/asset coordinates")
        if not self.source_ref or not self.price_basis:
            raise ValueError("HoldingReturnPanel requires price source and basis")
        if np.any(values[np.isfinite(values)] < -1):
            raise ValueError("holding returns below -100% require a negative-capital contract")
        object.__setattr__(self, "values", _freeze_array(values, "values"))

    @classmethod
    def from_prices(cls, prices, *, time_axis, asset_axis, source_ref, price_basis):
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape != (time_axis.size, asset_axis.size):
            raise ValueError("Price panel does not match named axes")
        if np.any(prices[np.isfinite(prices)] < 0):
            raise ValueError("negative prices are unsupported")
        returns = np.full_like(prices, np.nan)
        if len(prices) > 1:
            np.divide(prices[1:], prices[:-1], out=returns[1:], where=prices[:-1] > 0)
            returns[1:] -= 1
        return cls(returns, time_axis, asset_axis, source_ref, price_basis)

    @property
    def content_hash(self):
        return stable_content_hex(tag="HoldingReturnPanel.v1", fields={
            "values": self.values, "times": self.time_axis.values,
            "assets": self.asset_axis.values, "source_ref": self.source_ref,
            "price_basis": self.price_basis, "interval": "P_t/P_previous-1"})


@dataclass(frozen=True)
class PortfolioSpec:
    holding: int = 20
    n_quantiles: int = 10
    min_bucket_size: int = 1
    long_weight: float = .5
    short_weight: float = -.5
    per_side_cost: float = 0.
    purpose: str = "RESEARCH_PROBE"
    missing_return_policy: str = "unknown"
    terminal_position_policy: str = "ongoing"

    def __post_init__(self):
        cohort_schedule(0, self.holding)
        for name, floor in (("n_quantiles", 2), ("min_bucket_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{name} must be an integer >= {floor}")
        if not (np.isfinite(self.per_side_cost) and 0 <= self.per_side_cost < 1):
            raise ValueError("per_side_cost must be a finite per-side fraction in [0,1)")
        if not (np.isfinite(self.long_weight) and self.long_weight >= 0
                and np.isfinite(self.short_weight) and self.short_weight <= 0):
            raise ValueError("invalid long/short weight signs")
        if self.purpose != "RESEARCH_PROBE" or self.missing_return_policy != "unknown":
            raise ValueError("This probe does not implement executable trading or missing-price imputation")
        if self.terminal_position_policy not in {"ongoing", "liquidate_at_end"}:
            raise ValueError("terminal_position_policy must be ongoing or liquidate_at_end")

    def to_dict(self):
        return dict(vars(self))

    @property
    def content_hash(self):
        return stable_content_hex(tag="PortfolioSpec.v3.terminal_position_policy", fields=self.to_dict())


@dataclass(frozen=True)
class TradeEligibilityPanel:
    """Side-specific, decision-visible permissions; never inferred from labels."""
    can_buy: np.ndarray
    can_sell: np.ndarray
    borrowable: np.ndarray
    coverable: np.ndarray
    available_time: np.ndarray
    time_axis: AxisRef
    asset_axis: AxisRef
    source_ref: str
    universe_snapshot_ref: str

    def __post_init__(self):
        shape = (self.time_axis.size, self.asset_axis.size)
        if self.time_axis.values is None or self.asset_axis.values is None:
            raise ValueError("TradeEligibilityPanel requires explicit coordinates")
        for name in ("can_buy", "can_sell", "borrowable", "coverable"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != np.dtype(bool):
                raise ValueError(f"{name} requires a named boolean (T,N) panel")
            object.__setattr__(self, name, _freeze_array(value, name))
        available = np.asarray(self.available_time)
        if available.shape != shape:
            raise ValueError("eligibility available_time must have shape (T,N)")
        try:
            if not np.all(available <= self.time_axis.values[:, None]):
                raise ValueError("trade eligibility is not known by the decision time")
        except TypeError as exc:
            raise ValueError("incompatible eligibility and decision clocks") from exc
        if not self.source_ref or not self.universe_snapshot_ref:
            raise ValueError("eligibility requires resolvable source and universe snapshot refs")
        object.__setattr__(self, "available_time", _freeze_array(available, "available_time"))

    @property
    def content_hash(self):
        return stable_content_hex(tag="TradeEligibilityPanel.v1", fields={
            **{name: getattr(self, name) for name in ("can_buy", "can_sell", "borrowable", "coverable", "available_time")},
            "time": self.time_axis.values, "assets": self.asset_axis.values,
            "source_ref": self.source_ref, "universe_snapshot_ref": self.universe_snapshot_ref})
