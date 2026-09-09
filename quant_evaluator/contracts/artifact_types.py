"""Formal artifact type contracts for derived metric inputs.

Registry metrics previously declared their derived inputs with magic strings
(e.g. ``requires=["ic_series"]``) that no type system recognised. This module
introduces *formal* artifact contracts: a derived metric declares the class
name of the artifact it consumes (e.g. ``ICSeriesArtifact``), and the public
facade constructs that artifact from the raw factor/label contracts and hands
it to the consumer kernel.

Each artifact is a frozen, self-describing container: payload array, the
coordinate metadata a consumer needs (time index / factor ids / correlation
method / quantile count / exposure type), and provenance. Arrays are copied
and marked read-only on construction (QE-P0-02 ownership), and ``provenance``
is a recursively-immutable :class:`FrozenMapping` (QE-P0-01).  ``to_dict`` /
``from_dict`` use the lossless ndarray codec (QE-P0-04).

Shape conventions (F = number of factors, always the LAST axis):
    - ICSeriesArtifact:         values (T, F)
    - QuantileReturnArtifact:   values (n_quantiles, F)
    - ProbePortfolioArtifact:   values (T, F)
    - ExposureArtifact:         values (K, F)
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Tuple
import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex, stable_hash
from quant_evaluator.contracts._ndarray_codec import decode_value, encode_value
from quant_evaluator.contracts.metric_artifacts import FrozenMapping

__all__ = [
    "ICSeriesArtifact",
    "QuantileReturnArtifact",
    "DailyQuantileReturnArtifact",
    "ProbePortfolioArtifact",
    "ExposureArtifact",
]


def _freeze_array(value: Any, name: str) -> np.ndarray:
    """Coerce ``value`` to a read-only ndarray the artifact owns (copy-on-write)."""
    array = np.array(value, copy=True, order="C")
    array.flags.writeable = False
    return array


def _freeze_tuple(value: Any) -> Tuple[Any, ...]:
    return tuple(value) if value is not None else ()


@dataclass(frozen=True, eq=False)
class ICSeriesArtifact:
    """Daily per-factor IC series, shape (T, F).

    The formal input contract for IC-summary derived metrics (ic.rank.mean,
    ic.rank.ir, hac_tstat/hac_pvalue, ic_median, ic_autocorr_lag1,
    half_life, subsample_stability, block_bootstrap_ci, ic_std). Produced
    from factor + label via the daily-IC kernel
    (``quant_evaluator.metrics.ic.compute_daily_ic``); ``ic_method`` records
    the correlation method so consumers never misread a Spearman series as
    Pearson (or vice versa).
    """

    values: np.ndarray
    time_index: Tuple[Any, ...] = ()
    ic_method: str = "pearson"
    factor_ids: Tuple[str, ...] = ()
    metric_id: str = "ic.daily"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise ValueError(
                f"ICSeriesArtifact.values must be (T, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_index", _freeze_tuple(self.time_index))
        object.__setattr__(self, "factor_ids", _freeze_tuple(self.factor_ids))
        if self.ic_method not in ("pearson", "spearman"):
            raise ValueError(
                f"ICSeriesArtifact.ic_method must be 'pearson' or 'spearman', "
                f"got {self.ic_method!r}"
            )
        if self.time_index and len(self.time_index) != values.shape[0]:
            raise ValueError(
                f"ICSeriesArtifact.time_index length {len(self.time_index)} "
                f"does not match T={values.shape[0]}"
            )
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return bool(
            np.array_equal(self.values, other.values, equal_nan=True)
            and self.time_index == other.time_index
            and self.ic_method == other.ic_method
            and self.factor_ids == other.factor_ids
            and self.metric_id == other.metric_id
            and dict(self.provenance) == dict(other.provenance)
        )

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED). Covers the payload bytes so equal-in-value but
        # differently-typed arrays still hash differently.
        return stable_hash(
            stable_content_hex(
                tag="ICSeriesArtifact",
                fields={
                    "values": self.values,
                    "time_index": self.time_index,
                    "ic_method": self.ic_method,
                    "factor_ids": self.factor_ids,
                    "metric_id": self.metric_id,
                    "provenance": self.provenance,
                },
            )
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        return {
            "values": encode_value(self.values),
            "time_index": encode_value(self.time_index),
            "ic_method": self.ic_method,
            "factor_ids": list(self.factor_ids),
            "metric_id": self.metric_id,
            "provenance": encode_value(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ICSeriesArtifact":
        """Deserialize from the payload produced by :meth:`to_dict`."""
        return cls(
            values=decode_value(data["values"]),
            time_index=tuple(decode_value(data.get("time_index", ()))),
            ic_method=data.get("ic_method", "pearson"),
            factor_ids=tuple(data.get("factor_ids", ())),
            metric_id=data.get("metric_id", "ic.daily"),
            provenance=decode_value(data.get("provenance", {})),
        )


@dataclass(frozen=True, eq=False)
class QuantileReturnArtifact:
    """Per-quantile time-averaged returns, shape (n_quantiles, F).

    The formal input contract for quantile-family derived metrics. Unlike a
    scalar spread, this carries the full per-quantile return vector so the
    consumer knows the quantile dimension explicitly.
    """

    values: np.ndarray
    n_quantiles: int
    factor_ids: Tuple[str, ...] = ()
    metric_id: str = "quantile_returns"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise ValueError(
                f"QuantileReturnArtifact.values must be (n_quantiles, F), "
                f"got shape {values.shape}"
            )
        if values.shape[0] != self.n_quantiles:
            raise ValueError(
                f"QuantileReturnArtifact.values row count {values.shape[0]} "
                f"does not match n_quantiles={self.n_quantiles}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "factor_ids", _freeze_tuple(self.factor_ids))
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return bool(
            np.array_equal(self.values, other.values, equal_nan=True)
            and self.n_quantiles == other.n_quantiles
            and self.factor_ids == other.factor_ids
            and self.metric_id == other.metric_id
            and dict(self.provenance) == dict(other.provenance)
        )

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED). Covers the payload bytes so equal-in-value but
        # differently-typed arrays still hash differently.
        return stable_hash(
            stable_content_hex(
                tag="QuantileReturnArtifact",
                fields={
                    "values": self.values,
                    "n_quantiles": self.n_quantiles,
                    "factor_ids": self.factor_ids,
                    "metric_id": self.metric_id,
                    "provenance": self.provenance,
                },
            )
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        return {
            "values": encode_value(self.values),
            "n_quantiles": self.n_quantiles,
            "factor_ids": list(self.factor_ids),
            "metric_id": self.metric_id,
            "provenance": encode_value(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantileReturnArtifact":
        """Deserialize from the payload produced by :meth:`to_dict`."""
        return cls(
            values=decode_value(data["values"]),
            n_quantiles=int(data["n_quantiles"]),
            factor_ids=tuple(data.get("factor_ids", ())),
            metric_id=data.get("metric_id", "quantile_returns"),
            provenance=decode_value(data.get("provenance", {})),
        )


@dataclass(frozen=True, eq=False)
class DailyQuantileReturnArtifact:
    """Daily quantile returns and counts, shape ``(T, Q, F)``.

    This intentionally does not replace the time-averaged QF artifact.
    Certification additionally requires external tie, tradability and risk
    evidence references; their absence never changes the numeric payload.
    """

    artifact_kind: ClassVar[str] = "daily_quantile"

    values: np.ndarray
    counts: np.ndarray
    valid_mask: np.ndarray
    time_axis: Tuple[Any, ...]
    quantile_axis: Tuple[int, ...]
    factor_axis: Tuple[str, ...]
    tie_status_ref: str | None = None
    tradability_ref: str | None = None
    risk_exposure_ref: str | None = None
    producer_version: str = "1.0.0"
    metric_id: str = "quantile_returns.daily"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _freeze_array(self.values, "values")
        counts = _freeze_array(self.counts, "counts")
        raw_valid = np.asarray(self.valid_mask)
        if raw_valid.dtype != np.bool_:
            raise ValueError("daily quantile valid_mask must have strict bool dtype")
        valid = _freeze_array(raw_valid, "valid_mask")
        if values.ndim != 3 or counts.shape != values.shape or valid.shape != values.shape:
            raise ValueError("daily quantile values/counts/valid_mask must share (T,Q,F) shape")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("daily quantile counts must be non-negative integers")
        axes = (tuple(self.time_axis), tuple(self.quantile_axis), tuple(self.factor_axis))
        if tuple(map(len, axes)) != values.shape:
            raise ValueError("daily quantile axes do not match (T,Q,F) payload")
        if tuple(self.quantile_axis) != tuple(range(values.shape[1])):
            raise ValueError("quantile_axis must be canonical zero-based bucket ids")
        if len(set(self.time_axis)) != len(self.time_axis) or len(set(self.factor_axis)) != len(self.factor_axis):
            raise ValueError("daily quantile time and factor axes must be unique")
        if np.any(valid & (~np.isfinite(values) | (counts <= 0))):
            raise ValueError("valid daily quantiles require finite values and positive counts")
        for name in ("producer_version", "metric_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "time_axis", axes[0])
        object.__setattr__(self, "quantile_axis", axes[1])
        object.__setattr__(self, "factor_axis", axes[2])
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))

    @property
    def provenance_refs_present(self) -> bool:
        return all(
            isinstance(ref, str) and bool(ref.strip())
            for ref in (self.tie_status_ref, self.tradability_ref, self.risk_exposure_ref)
        )

    def require_certification_ready(self, resolver) -> None:
        """Require every opaque ref to resolve through the caller's authority."""
        if not self.provenance_refs_present:
            raise ValueError("daily quantile artifact lacks tie/tradability/risk evidence refs")
        if not callable(resolver) or not all(
            resolver(ref) is True
            for ref in (self.tie_status_ref, self.tradability_ref, self.risk_exposure_ref)
        ):
            raise ValueError("daily quantile evidence refs are not authority-resolved")

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other): return NotImplemented
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        return {
            "values": encode_value(self.values), "counts": encode_value(self.counts),
            "valid_mask": encode_value(self.valid_mask), "time_axis": encode_value(self.time_axis),
            "quantile_axis": list(self.quantile_axis), "factor_axis": list(self.factor_axis),
            "tie_status_ref": self.tie_status_ref, "tradability_ref": self.tradability_ref,
            "risk_exposure_ref": self.risk_exposure_ref, "producer_version": self.producer_version,
            "metric_id": self.metric_id, "provenance": encode_value(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DailyQuantileReturnArtifact":
        return cls(
            values=decode_value(data["values"]), counts=decode_value(data["counts"]),
            valid_mask=decode_value(data["valid_mask"]),
            time_axis=tuple(decode_value(data["time_axis"])),
            quantile_axis=tuple(data["quantile_axis"]), factor_axis=tuple(data["factor_axis"]),
            tie_status_ref=data.get("tie_status_ref"), tradability_ref=data.get("tradability_ref"),
            risk_exposure_ref=data.get("risk_exposure_ref"),
            producer_version=data.get("producer_version", "1.0.0"),
            metric_id=data.get("metric_id", "quantile_returns.daily"),
            provenance=decode_value(data.get("provenance", {})),
        )


@dataclass(frozen=True, eq=False)
class ProbePortfolioArtifact:
    """Probe-portfolio returns (and positions) per factor, values shape (T, F).

    The formal input contract for portfolio-backtest derived metrics. The
    payload is a per-factor probe portfolio return time series; optional
    position weights may be attached via ``provenance``.
    """

    values: np.ndarray
    time_index: Tuple[Any, ...] = ()
    factor_ids: Tuple[str, ...] = ()
    metric_id: str = "probe_portfolio"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise ValueError(
                f"ProbePortfolioArtifact.values must be (T, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_index", _freeze_tuple(self.time_index))
        object.__setattr__(self, "factor_ids", _freeze_tuple(self.factor_ids))
        if self.time_index and len(self.time_index) != values.shape[0]:
            raise ValueError(
                f"ProbePortfolioArtifact.time_index length {len(self.time_index)} "
                f"does not match T={values.shape[0]}"
            )
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return bool(
            np.array_equal(self.values, other.values, equal_nan=True)
            and self.time_index == other.time_index
            and self.factor_ids == other.factor_ids
            and self.metric_id == other.metric_id
            and dict(self.provenance) == dict(other.provenance)
        )

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED). Covers the payload bytes so equal-in-value but
        # differently-typed arrays still hash differently.
        return stable_hash(
            stable_content_hex(
                tag="ProbePortfolioArtifact",
                fields={
                    "values": self.values,
                    "time_index": self.time_index,
                    "factor_ids": self.factor_ids,
                    "metric_id": self.metric_id,
                    "provenance": self.provenance,
                },
            )
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        return {
            "values": encode_value(self.values),
            "time_index": encode_value(self.time_index),
            "factor_ids": list(self.factor_ids),
            "metric_id": self.metric_id,
            "provenance": encode_value(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProbePortfolioArtifact":
        """Deserialize from the payload produced by :meth:`to_dict`."""
        return cls(
            values=decode_value(data["values"]),
            time_index=tuple(decode_value(data.get("time_index", ()))),
            factor_ids=tuple(data.get("factor_ids", ())),
            metric_id=data.get("metric_id", "probe_portfolio"),
            provenance=decode_value(data.get("provenance", {})),
        )


@dataclass(frozen=True, eq=False)
class ExposureArtifact:
    """Factor exposure decomposition (loadings / sector / style), values (K, F).

    The formal input contract for exposure-family derived metrics.
    ``exposure_type`` records which decomposition produced the payload so a
    consumer never misreads factor loadings as sector exposure.
    """

    values: np.ndarray
    exposure_type: str = "loadings"  # loadings | sector | style | concentration
    factor_ids: Tuple[str, ...] = ()
    metric_id: str = "exposure"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise ValueError(
                f"ExposureArtifact.values must be (K, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "factor_ids", _freeze_tuple(self.factor_ids))
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return bool(
            np.array_equal(self.values, other.values, equal_nan=True)
            and self.exposure_type == other.exposure_type
            and self.factor_ids == other.factor_ids
            and self.metric_id == other.metric_id
            and dict(self.provenance) == dict(other.provenance)
        )

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED). Covers the payload bytes so equal-in-value but
        # differently-typed arrays still hash differently.
        return stable_hash(
            stable_content_hex(
                tag="ExposureArtifact",
                fields={
                    "values": self.values,
                    "exposure_type": self.exposure_type,
                    "factor_ids": self.factor_ids,
                    "metric_id": self.metric_id,
                    "provenance": self.provenance,
                },
            )
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        return {
            "values": encode_value(self.values),
            "exposure_type": self.exposure_type,
            "factor_ids": list(self.factor_ids),
            "metric_id": self.metric_id,
            "provenance": encode_value(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExposureArtifact":
        """Deserialize from the payload produced by :meth:`to_dict`."""
        return cls(
            values=decode_value(data["values"]),
            exposure_type=data.get("exposure_type", "loadings"),
            factor_ids=tuple(data.get("factor_ids", ())),
            metric_id=data.get("metric_id", "exposure"),
            provenance=decode_value(data.get("provenance", {})),
        )
