"""
Per-style exposure / purity evidence kernels (R61-FI-024, plan §13.11 + §16.2).

These kernels compute *evidence about a factor's style exposure* from an
injected exposure panel.  The exposure panel is the **DataAccess-authoritative
ref** (``ExposurePanel`` below): QE never fabricates exposure data, and the
panel's ``source_ref`` / ``provider`` metadata record where the exposures
came from (production wiring injects a DataAccess adapter; tests inject a
synthetic panel and never touch COS).

Family (each style gets its own typed field — no single scalar "exposure"
API):

    - ``ExposurePanel`` (frozen dataclass): (T, N, K) exposure values per
      style factor (industry / size / beta / liquidity / volatility /
      momentum / ...), ``style_names`` (K,), ``source_ref`` (DataAccess
      authoritative ref, e.g. a dataset/table ref), ``provider``
      (injectable provenance tag) and ``date_index`` (optional).
    - ``StyleExposureEvidence`` (frozen dataclass): per-style mean exposure
      ``values`` (K,) + per-style cardinality ``counts`` (K,) + the
      ``source_ref`` / ``provider`` provenance.  Missing exposures are NaN
      (explicit), never 0.
    - ``ExposureStyle`` enum: the closed style vocabulary
      (INDUSTRY / SIZE / BETA / LIQUIDITY / VOLATILITY / MOMENTUM /
      UNKNOWN).  This is the *style dimension vocabulary* — it is NOT a new
      evidence-status token (the status vocabulary stays the existing
      8-token ``EvidenceStatus`` in ``contracts/evidence_status.py``).
    - ``compute_style_exposure_evidence`` — time-averaged mean per-style exposure
      (absolute or signed), delegating to the existing
      ``metrics/exposure.compute_factor_loadings`` for the signed regression
      loadings where the caller supplies a (T, N) factor panel.
    - ``compute_max_absolute_style_exposure`` — the style dimension with the
      largest mean *absolute* exposure.
    - ``compute_exposure_drift`` — mean absolute change of the per-style
      exposure series between adjacent periods (a persistence measure).
    - ``compute_purity_ratio`` — 1 - (total style exposure variance share
      explained) style-driven part; higher = cleaner (factor's own signal
      dominates its style footprint).
    - ``compute_neutralized_rank_ic`` — cross-sectional OLS residual IC:
      regress the factor on the exposure panel per date, then Spearman IC of
      the residual against forward returns.  CPU reference implementation;
      GPU parity path is optional and exposed separately.
    - ``compute_residual_rank_ic`` — alias of the neutralized rank IC for
      the ``residual_rank_ic`` metric id (kept as a named function so both
      ids bind distinct compute_fn entries in the registry).

GPU reuse: the R60 ``kernels/gpu/exposure.py`` batch kernels
(``batched_concentration_hhi`` / ``batched_sector_exposure`` /
``batched_factor_loadings`` / ``batched_style_exposure``) are the parity
reference.  The CPU kernels here are the authoritative parity counterpart
(``metrics/exposure.py``); a CPU-vs-GPU parity test in
``tests/test_exposure_evidence.py`` runs when CuPy is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quant_evaluator.metrics.exposure import compute_factor_loadings
from quant_evaluator.metrics.ic import _spearman_rank_correlation  # rank-IC kernel (stable)
__all__ = [
    "ExposureStyle",
    "ExposurePanel",
    "StyleExposureEvidence",
    "compute_style_exposure_evidence",
    "compute_max_absolute_style_exposure",
    "compute_exposure_drift",
    "compute_purity_ratio",
    "compute_neutralized_rank_ic",
    "compute_residual_rank_ic",
    "compute_industry_exposure",
    "compute_size_exposure",
    "compute_beta_exposure",
    "compute_liquidity_exposure",
    "compute_volatility_exposure",
    "compute_momentum_exposure",
    "compute_exposure_evidence",
]

EPS = 1e-12
_DEFAULT_STYLE_NAMES = ("industry", "size", "beta", "liquidity", "volatility", "momentum")


class ExposureStyle(str, Enum):
    """Closed style-dimension vocabulary (NOT an evidence-status token).

    The status vocabulary remains the existing 8-token ``EvidenceStatus`` in
    ``quant_evaluator.contracts.evidence_status``; this enum only names the
    style dimensions a per-style exposure can target.
    """

    INDUSTRY = "industry"
    SIZE = "size"
    BETA = "beta"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: object) -> "ExposureStyle":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                return cls.UNKNOWN
        raise TypeError(
            f"ExposureStyle.from_value expects str or ExposureStyle, got "
            f"{type(value).__name__}"
        )


@dataclass(frozen=True)
class ExposurePanel:
    """A DataAccess-authoritative exposure panel (T, N, K) per style.

    QE does NOT fabricate exposure data.  ``source_ref`` declares where the
    panel came from (a DataAccess dataset/table ref — the producer column /
    style builder authority); ``provider`` records the injectable provenance
    tag (production wires a DataAccess adapter; tests inject a synthetic panel
    and never send a real COS request).

    ``values`` is copied and frozen on construction (QE-P0-02 ownership).
    Missing cells stay NaN (explicit — never 0).
    """

    values: np.ndarray
    style_names: Tuple[str, ...] = _DEFAULT_STYLE_NAMES
    source_ref: str = ""
    provider: str = ""
    date_index: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(
                f"ExposurePanel.values must be (T, N, K), got shape {arr.shape}"
            )
        arr = np.array(arr, copy=True, order="C")
        arr.flags.writeable = False
        object.__setattr__(self, "values", arr)
        if arr.shape[2] != len(self.style_names):
            raise ValueError(
                f"ExposurePanel.style_names length {len(self.style_names)} must "
                f"match K={arr.shape[2]}"
            )
        object.__setattr__(self, "style_names", tuple(self.style_names))
        object.__setattr__(self, "date_index", tuple(self.date_index))
        if self.date_index and len(self.date_index) != arr.shape[0]:
            raise ValueError(
                f"ExposurePanel.date_index length {len(self.date_index)} does not "
                f"match T={arr.shape[0]}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        from quant_evaluator.contracts._ndarray_codec import encode_value

        return {
            "values": encode_value(self.values),
            "style_names": list(self.style_names),
            "source_ref": self.source_ref,
            "provider": self.provider,
            "date_index": list(self.date_index),
        }


@dataclass(frozen=True)
class StyleExposureEvidence:
    """Per-style exposure evidence (one typed field per style, never a scalar).

    ``values`` is (K,) mean exposure per style (signed or absolute, depending
    on the ``absolute`` flag of the producing kernel); ``counts`` (K,) is the
    per-style number of finite observations.  ``source_ref`` / ``provider``
    carry the DataAccess-authoritative provenance.  Missing styles are NaN —
    explicit evidence, never a fabricated 0.
    """

    values: np.ndarray
    style_names: Tuple[str, ...]
    source_ref: str = ""
    provider: str = ""
    counts: np.ndarray = ()

    def __post_init__(self) -> None:
        vals = np.asarray(self.values, dtype=np.float64)
        if vals.ndim != 1:
            raise ValueError(
                f"StyleExposureEvidence.values must be (K,), got shape {vals.shape}"
            )
        if len(self.style_names) != vals.shape[0]:
            raise ValueError(
                f"StyleExposureEvidence.style_names length {len(self.style_names)} "
                f"must match K={vals.shape[0]}"
            )
        vals = np.array(vals, copy=True, order="C")
        vals.flags.writeable = False
        object.__setattr__(self, "values", vals)
        cnt = np.asarray(self.counts, dtype=np.int64)
        if cnt.size == 0:
            cnt = np.zeros(vals.shape[0], dtype=np.int64)
        cnt = np.array(cnt, copy=True, order="C")
        object.__setattr__(self, "counts", cnt)
        object.__setattr__(self, "style_names", tuple(self.style_names))

    def to_dict(self) -> Dict[str, Any]:
        from quant_evaluator.contracts._ndarray_codec import encode_value

        return {
            "values": encode_value(self.values),
            "style_names": list(self.style_names),
            "counts": encode_value(self.counts),
            "source_ref": self.source_ref,
            "provider": self.provider,
        }


def _panel_arrays(panel: ExposurePanel) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a panel and return (values, finite-mask) — (T, N, K)."""
    arr = np.asarray(panel.values, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"ExposurePanel.values must be (T, N, K), got {arr.ndim}D"
        )
    return arr, np.isfinite(arr)


def compute_style_exposure_evidence(
    panel: ExposurePanel,
    absolute: bool = False,
) -> StyleExposureEvidence:
    """Time-averaged mean per-style exposure from an injected panel.

    ``absolute=False`` returns the signed mean exposure per style (a positive
    value means the factor loads positively on that style dimension across
    the sample); ``absolute=True`` returns the mean of the absolute exposures
    (magnitude — how much of the factor's variance lives on the style).

    When a (T, N) ``factor_values`` panel is also needed for regression-based
    loadings, use ``metrics/exposure.compute_style_exposure`` directly; this
    kernel is the panel-evidence entry point (mean of the panel values).
    """
    arr, _ = _panel_arrays(panel)
    T, N, K = arr.shape
    out = np.full(K, np.nan)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        vals = arr[:, :, k]
        finite = vals[np.isfinite(vals)]
        counts[k] = finite.size
        if finite.size == 0:
            continue
        out[k] = float(np.mean(np.abs(finite))) if absolute else float(np.mean(finite))
    return StyleExposureEvidence(
        values=out,
        style_names=tuple(panel.style_names),
        source_ref=panel.source_ref,
        provider=panel.provider,
        counts=counts,
    )


def compute_max_absolute_style_exposure(
    panel: ExposurePanel,
    min_finite: int = 5,
) -> Dict[str, Any]:
    """The style dimension with the largest mean *absolute* exposure.

    Returns a plain dict (``style`` / ``value`` / ``absolute_mean`` /
    ``counts``) — the ``max_absolute_style_exposure`` evidence payload.
    ``value`` is the signed mean exposure of the winning style; NaN when no
    style has at least ``min_finite`` finite observations.
    """
    ev = compute_style_exposure_evidence(panel, absolute=True)
    valid = np.isfinite(ev.values) & (ev.counts >= min_finite)
    if not np.any(valid):
        return {
            "style": ExposureStyle.UNKNOWN.value,
            "value": np.nan,
            "absolute_mean": np.nan,
            "counts": 0,
        }
    idx = int(np.argmax(np.where(valid, ev.values, -np.inf)))
    signed = compute_style_exposure_evidence(panel, absolute=False)
    return {
        "style": str(panel.style_names[idx]),
        "value": float(signed.values[idx]),
        "absolute_mean": float(ev.values[idx]),
        "counts": int(ev.counts[idx]),
    }


def compute_exposure_drift(panel: ExposurePanel) -> float:
    """Mean absolute change of the per-style exposure series between adjacent
    periods (a persistence / stability measure).

    ``drift = mean_{t,k} |panel[t+1,k] - panel[t,k]|`` over jointly-finite
    adjacent cells.  NaN when the panel has fewer than 2 periods or no finite
    adjacent pair.
    """
    arr, _ = _panel_arrays(panel)
    T, N, K = arr.shape
    if T < 2:
        return np.nan
    diffs: list[float] = []
    for t in range(T - 1):
        a = arr[t]
        b = arr[t + 1]
        joint = np.isfinite(a) & np.isfinite(b)
        if not np.any(joint):
            continue
        diffs.append(float(np.mean(np.abs(a[joint] - b[joint]))))
    if not diffs:
        return np.nan
    return float(np.mean(diffs))


def compute_purity_ratio(panel: ExposurePanel, min_finite: int = 5) -> float:
    """Purity ratio: 1 - style-explained share of total exposure dispersion.

    For each style k the cross-period mean exposure ``mu_k`` and its
    contribution ``mu_k^2``; the style-driven share is
    ``sum(mu_k^2) / sum(mu_k^2 + var_residual)`` where ``var_residual`` is the
    pooled within-style variance of the exposure residuals around the style
    mean.  Higher = cleaner (the factor's own signal dominates its style
    footprint); NaN when the panel has no valid style.

    This is a QE-side definition (plan §13.11); it does not replace FA's
    canonical purity/health grading, it merely surfaces the style-dispersion
    view from the exposure panel.
    """
    arr, _ = _panel_arrays(panel)
    T, N, K = arr.shape
    mu: list[float] = []
    resid: list[float] = []
    for k in range(K):
        vals = arr[:, :, k]
        finite = vals[np.isfinite(vals)]
        if finite.size < min_finite:
            continue
        m = float(np.mean(finite))
        mu.append(m)
        resid.append(float(np.mean((finite - m) ** 2)))
    if not mu:
        return np.nan
    style_part = float(np.sum(np.square(mu)))
    resid_part = float(np.sum(resid))
    denom = style_part + resid_part
    if denom <= EPS:
        return np.nan
    return float(1.0 - style_part / denom)


def compute_neutralized_rank_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    panel: ExposurePanel,
    min_obs: int = 10,
) -> float:
    """Cross-sectional residual (neutralized) rank IC.

    Per date t: regress the factor cross-section on the exposure panel
    (intercept + K styles) via ``metrics/exposure.compute_factor_loadings``,
    take the OLS residuals, then Spearman-rank-correlate the residuals with
    forward returns.  The time-mean of the daily residual IC is the
    neutralized rank IC.  A factor whose IC survives neutralization has alpha
    orthogonal to the style exposures.

    CPU reference implementation (GPU optional; parity checked in tests).
    NaN when fewer than ``min_obs`` jointly-finite assets on every date, or
    when the factor/label/panel shapes are inconsistent.
    """
    fv = np.asarray(factor_values, dtype=np.float64)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    if fv.ndim != 2 or fwd.ndim != 2:
        raise ValueError("factor_values / forward_returns must be (T, N)")
    if fv.shape != fwd.shape:
        raise ValueError(
            f"factor_values {fv.shape} and forward_returns {fwd.shape} must match"
        )
    arr = np.asarray(panel.values, dtype=np.float64)
    T, N, K = arr.shape
    if (T, N) != fv.shape:
        raise ValueError(
            f"panel (T,N)=({T},{N}) must match factor (T,N)={fv.shape}"
        )
    # Regress per date with intercept; residuals (T, N) — NaN where invalid.
    _, _, residuals = compute_factor_loadings(fv, arr, intercept=True, min_obs=min_obs)

    daily_ics: list[float] = []
    for t in range(T):
        y = fwd[t]
        resid = residuals[t]
        joint = np.isfinite(y) & np.isfinite(resid)
        if np.sum(joint) < 2:
            continue
        rho = _spearman_rank_correlation(resid[joint], y[joint], min_obs=max(2, min_obs // 2))
        if np.isfinite(rho):
            daily_ics.append(float(rho))
    if not daily_ics:
        return np.nan
    return float(np.mean(daily_ics))


def compute_residual_rank_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    panel: ExposurePanel,
    min_obs: int = 10,
) -> float:
    """Named alias for ``compute_neutralized_rank_ic`` (residual_rank_ic id)."""
    return compute_neutralized_rank_ic(
        factor_values, forward_returns, panel, min_obs=min_obs
    )


def _select_style(panel: ExposurePanel, style: str) -> float:
    """Signed mean exposure of one style dimension (NaN when absent)."""
    ev = compute_style_exposure_evidence(panel, absolute=False)
    names = list(ev.style_names)
    if style not in names:
        return float("nan")
    return float(ev.values[names.index(style)])


def compute_industry_exposure(panel: ExposurePanel) -> float:
    """Signed mean industry-style exposure of the factor (one typed field)."""
    return _select_style(panel, "industry")


def compute_size_exposure(panel: ExposurePanel) -> float:
    """Signed mean size-style exposure of the factor (one typed field)."""
    return _select_style(panel, "size")


def compute_beta_exposure(panel: ExposurePanel) -> float:
    """Signed mean beta-style exposure of the factor (one typed field)."""
    return _select_style(panel, "beta")


def compute_liquidity_exposure(panel: ExposurePanel) -> float:
    """Signed mean liquidity-style exposure of the factor (one typed field)."""
    return _select_style(panel, "liquidity")


def compute_volatility_exposure(panel: ExposurePanel) -> float:
    """Signed mean volatility-style exposure of the factor (one typed field)."""
    return _select_style(panel, "volatility")


def compute_momentum_exposure(panel: ExposurePanel) -> float:
    """Signed mean momentum-style exposure of the factor (one typed field)."""
    return _select_style(panel, "momentum")


def compute_exposure_evidence(
    panel: ExposurePanel,
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    min_obs: int = 10,
) -> Dict[str, Any]:
    """Bundle all per-style exposure evidence for a factor into a plain dict.

    Keys mirror the registry metric ids in ``registry/metrics.py``:
        industry_exposure / size_exposure / beta_exposure /
        liquidity_exposure / volatility_exposure / momentum_exposure /
        max_absolute_style_exposure / exposure_drift /
        neutralized_rank_ic / residual_rank_ic / purity_ratio
    """
    ev = compute_style_exposure_evidence(panel, absolute=False)
    by_name = dict(zip(ev.style_names, ev.values))
    max_abs = compute_max_absolute_style_exposure(panel)
    neutralized = compute_neutralized_rank_ic(
        factor_values, forward_returns, panel, min_obs=min_obs
    )
    return {
        "industry_exposure": by_name.get("industry", np.nan),
        "size_exposure": by_name.get("size", np.nan),
        "beta_exposure": by_name.get("beta", np.nan),
        "liquidity_exposure": by_name.get("liquidity", np.nan),
        "volatility_exposure": by_name.get("volatility", np.nan),
        "momentum_exposure": by_name.get("momentum", np.nan),
        "max_absolute_style_exposure": max_abs,
        "exposure_drift": compute_exposure_drift(panel),
        "neutralized_rank_ic": neutralized,
        "residual_rank_ic": neutralized,
        "purity_ratio": compute_purity_ratio(panel),
    }