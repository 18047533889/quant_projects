"""
Per-style exposure / purity evidence kernels (R61-FI-024, plan §13.11 + §16.2).

These kernels compute factor-specific evidence by regressing an explicitly
supplied factor on an injected risk panel. The panel alone is not a factor
loading. The exposure panel is the **DataAccess-authoritative
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
    - ``FactorLoadingSeries`` — standardized WLS coefficients, raw coefficients,
      same-fit R-squared, joint-support counts and provenance.
    - ``PortfolioExposureSeries`` — actual supplied portfolio weights times risk.
    - ``compute_style_exposure_evidence`` — time mean standardized WLS loading,
      absolute or signed, from a FactorLoadingSeries or panel plus factor values.
    - ``compute_max_absolute_style_exposure`` — the style dimension with the
      largest mean *absolute* exposure.
    - ``compute_exposure_drift`` — mean absolute change of the per-style
      exposure series between adjacent periods (a persistence measure).
    - ``compute_purity_ratio`` — time mean of same-fit 1-R-squared; descriptive
      residual variance share, not out-of-sample predictive utility.
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
from types import MappingProxyType
from hashlib import sha256

import numpy as np

from quant_evaluator.metrics.exposure import compute_factor_loadings
from quant_evaluator.metrics.ic import _spearman_rank_correlation  # rank-IC kernel (stable)
__all__ = [
    "ExposureStyle",
    "ExposurePanel",
    "SecurityExposurePanel",
    "FactorLoadingSeries",
    "PortfolioExposureSeries",
    "build_factor_loading_series",
    "compute_portfolio_exposure",
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
    security_ids: Tuple[Any, ...] = ()
    factor_ids: Tuple[str, ...] = ()
    universe_snapshot_ref: str = ""
    validity: Optional[np.ndarray] = None
    regression_weights: Optional[np.ndarray] = None
    weight_ref: str = ""

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
        object.__setattr__(self, "security_ids", tuple(self.security_ids))
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        if len(set(self.style_names))!=len(self.style_names) or not self.style_names:
            raise ValueError("risk style names must be nonempty and unique")
        for name in ("date_index","security_ids","factor_ids"):
            axis=getattr(self,name)
            if len(set(axis))!=len(axis):
                raise ValueError(f"{name} must be unique")
        if self.date_index and len(self.date_index) != arr.shape[0]:
            raise ValueError(
                f"ExposurePanel.date_index length {len(self.date_index)} does not "
                f"match T={arr.shape[0]}"
            )
        if self.security_ids and len(self.security_ids) != arr.shape[1]:
            raise ValueError("ExposurePanel.security_ids must match N")
        validity = self.validity
        if validity is not None:
            validity = np.asarray(validity)
            if validity.dtype!=np.bool_:
                raise TypeError("ExposurePanel.validity must be strict boolean")
            if validity.shape != arr.shape:
                raise ValueError("ExposurePanel.validity must match values (T,N,K)")
            validity = np.array(validity, copy=True, order="C")
            validity.flags.writeable = False
        object.__setattr__(self, "validity", validity)
        if self.regression_weights is None and self.weight_ref:
            raise ValueError("weight_ref requires bound regression_weights")
        if self.regression_weights is not None:
            weights=np.asarray(self.regression_weights,dtype=float)
            if weights.shape!=arr.shape[:2] or not np.isfinite(weights).all() or np.any(weights<0):
                raise ValueError("regression weights must be finite nonnegative (T,N)")
            if not self.weight_ref:
                raise ValueError("explicit regression weights require weight_ref")
            weights=np.array(weights,copy=True); weights.flags.writeable=False
            object.__setattr__(self,"regression_weights",weights)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (lossless ndarray codec)."""
        from quant_evaluator.contracts._ndarray_codec import encode_value

        return {
            "values": encode_value(self.values),
            "style_names": list(self.style_names),
            "source_ref": self.source_ref,
            "provider": self.provider,
            "date_index": encode_value(np.asarray(self.date_index)),
            "security_ids": encode_value(self.security_ids),
            "factor_ids": list(self.factor_ids),
            "universe_snapshot_ref": self.universe_snapshot_ref,
            "validity": encode_value(self.validity),
            "regression_weights":encode_value(self.regression_weights),"weight_ref":self.weight_ref,
        }

    @classmethod
    def from_dict(cls, data):
        """Restore typed axes, masks and weight binding using the shared codec."""
        from quant_evaluator.contracts._ndarray_codec import decode_value
        return cls(**{key:decode_value(value) for key,value in data.items()})


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
        if cnt.shape!=vals.shape or np.any(cnt<0):
            raise ValueError("style counts must be nonnegative and align with styles")
        cnt.flags.writeable=False
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


@dataclass(frozen=True)
class FactorLoadingSeries:
    """Single-factor standardized WLS loadings (T,K), not security exposures.

    ``values`` are beta_k * sd(Z_k) / sd(factor) on the same weighted joint
    sample. Raw coefficients and same-fit R² are retained separately. This
    descriptive in-sample explanation is not OOS/model/portfolio evidence.
    """
    values: np.ndarray
    raw_loadings: np.ndarray
    r_squared: np.ndarray
    counts: np.ndarray
    style_names: Tuple[str,...]
    factor_id: str
    source_ref: str
    provider: str
    date_index: Tuple[Any,...]
    common_support_ref: str
    diagnostics: Tuple[Mapping[str,Any],...]
    factor_value_ref: str
    weight_ref: str = "equal_weight"
    estimation_scope: str = "SAME_DATE_DESCRIPTIVE"
    method_version: str = "factor_standardized_wls.v2"

    def __post_init__(self):
        if np.ndim(self.values)!=2:
            raise ValueError("factor loadings require (T,K) values")
        t,k=np.shape(self.values)
        if not self.factor_id or not self.common_support_ref or not self.factor_value_ref:
            raise ValueError("factor loading evidence requires factor and support identities")
        if self.estimation_scope!="SAME_DATE_DESCRIPTIVE" or self.method_version!="factor_standardized_wls.v2":
            raise ValueError("factor loading series cannot certify another estimation scope or method")
        if len(set(self.style_names))!=k or len(set(self.date_index))!=t:
            raise ValueError("factor loading axes must be unique")
        counts=np.asarray(self.counts)
        if counts.dtype.kind not in "iu" or np.any(counts<0):
            raise ValueError("factor loading counts must be nonnegative integers")
        r2=np.asarray(self.r_squared,dtype=float)
        if np.isinf(r2).any() or np.any(np.isfinite(r2)&((r2 < -1e-12)|(r2 > 1+1e-12))):
            raise ValueError("intercept regression R-squared must be missing or in [0,1]")
        if len(self.style_names)!=k or len(self.date_index)!=t or len(self.diagnostics)!=t:
            raise ValueError("factor loading series axes do not align")
        for name,shape in (("values",(t,k)),("raw_loadings",(t,k)),("r_squared",(t,)),("counts",(t,))):
            value=np.array(getattr(self,name),copy=True)
            if value.shape!=shape:
                raise ValueError(f"loading {name} axes do not align")
            value.flags.writeable=False; object.__setattr__(self,name,value)
        object.__setattr__(self,"style_names",tuple(self.style_names))
        object.__setattr__(self,"date_index",tuple(self.date_index))
        object.__setattr__(self,"diagnostics",tuple(MappingProxyType(dict(d)) for d in self.diagnostics))

    def to_dict(self):
        from quant_evaluator.contracts._ndarray_codec import encode_value
        return {name:encode_value(getattr(self,name)) for name in (
            "values","raw_loadings","r_squared","counts","style_names","factor_id",
            "source_ref","provider","date_index","common_support_ref","factor_value_ref",
            "weight_ref","estimation_scope","method_version")} | {
            "date_index":encode_value(np.asarray(self.date_index)),
            "diagnostics":[encode_value(dict(d)) for d in self.diagnostics]}


@dataclass(frozen=True)
class PortfolioExposureSeries:
    """Actual supplied weights transposed times security risk exposures (T,K)."""
    values: np.ndarray
    style_names: Tuple[str,...]
    date_index: Tuple[Any,...]
    portfolio_ref: str
    exposure_ref: str
    estimation_scope: str = "PORTFOLIO_WEIGHTED_EXPOSURE"

    def __post_init__(self):
        values=np.array(self.values,dtype=float,copy=True)
        if values.shape!=(len(self.date_index),len(self.style_names)) or not self.portfolio_ref:
            raise ValueError("portfolio exposure requires matching axes and a portfolio ref")
        values.flags.writeable=False; object.__setattr__(self,"values",values)
        object.__setattr__(self,"style_names",tuple(self.style_names))
        object.__setattr__(self,"date_index",tuple(self.date_index))


SecurityExposurePanel=ExposurePanel


def build_factor_loading_series(panel, factor_values, *, factor_id="research:single-factor", min_obs=10, weights=None):
    if not isinstance(panel,ExposurePanel):
        raise TypeError("factor regression requires a SecurityExposurePanel")
    values=np.asarray(factor_values,dtype=float)
    arr,valid=_panel_arrays(panel)
    if values.ndim!=2 or values.shape!=arr.shape[:2]:
        raise ValueError("factor_values must match risk time/security axes (T,N)")
    if weights is not None and panel.regression_weights is not None:
        raise ValueError("regression weights already bound by ExposurePanel")
    w=panel.regression_weights if weights is None else np.asarray(weights,dtype=float)
    if w is None: w=np.ones(values.shape)
    if w.shape!=values.shape or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError("weights must be finite nonnegative (T,N)")
    raw,r2,_,diagnostics=compute_factor_loadings(values,arr,min_obs=min_obs,weights=w,return_diagnostics=True)
    standardized=np.full_like(raw[:,1:],np.nan)
    joint=np.isfinite(values)&valid.all(axis=2)&(w>0)
    for t in range(len(values)):
        mask=joint[t]
        if not mask.any() or not np.isfinite(r2[t]): continue
        ww=w[t,mask]/np.max(w[t,mask]); ww=ww/ww.sum(); y=values[t,mask]; z=arr[t,mask]
        yc=y-(y[0]+np.sum(ww*(y-y[0]))); zc=z-(z[0]+np.sum(ww[:,None]*(z-z[0]),axis=0))
        sy=np.sqrt(np.sum(ww*yc*yc)); sz=np.sqrt(np.sum(ww[:,None]*zc*zc,axis=0))
        if sy>0: standardized[t]=np.where(sz>0,raw[t,1:]*sz/sy,np.nan)
    safe_diagnostics=tuple({k:v for k,v in d.items() if k!="coefficients"} for d in diagnostics)
    return FactorLoadingSeries(standardized,raw[:,1:],r2,joint.sum(axis=1),tuple(panel.style_names),
        factor_id,panel.source_ref,panel.provider,panel.date_index or tuple(range(len(values))),
        "support:"+sha256(joint.tobytes()+w.tobytes()).hexdigest(),safe_diagnostics,
        "factor-values:"+sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        panel.weight_ref or ("explicit_weight_array" if weights is not None else "equal_weight"))


def compute_portfolio_exposure(panel, portfolio_weights, *, portfolio_ref):
    arr,valid=_panel_arrays(panel); w=np.asarray(portfolio_weights,dtype=float)
    if w.shape!=arr.shape[:2] or not np.isfinite(w).all():
        raise ValueError("portfolio weights must be finite and axis-aligned (T,N)")
    active=w!=0
    known=(valid|~active[:,:,None]).all(axis=1)
    values=np.sum(np.where(active[:,:,None]&valid,w[:,:,None]*arr,0.),axis=1)
    values=np.where(known,values,np.nan)
    return PortfolioExposureSeries(values,panel.style_names,panel.date_index or tuple(range(len(w))),portfolio_ref,panel.source_ref)


def _as_factor_loadings(panel,factor_values=None,min_obs=10,weights=None):
    if isinstance(panel,FactorLoadingSeries):
        if factor_values is not None or weights is not None:
            raise ValueError("factor loading evidence already binds factor and weights")
        return panel
    if factor_values is None:
        raise TypeError("SecurityExposurePanel is not factor evidence; factor_values are required")
    return build_factor_loading_series(panel,factor_values,min_obs=min_obs,weights=weights)


def _panel_arrays(panel: ExposurePanel) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a panel and return (values, finite-mask) — (T, N, K)."""
    arr = np.asarray(panel.values, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"ExposurePanel.values must be (T, N, K), got {arr.ndim}D"
        )
    valid = np.isfinite(arr)
    if panel.validity is not None:
        valid &= panel.validity
        arr = np.where(valid, arr, np.nan)
    return arr, valid


def compute_style_exposure_evidence(
    panel: FactorLoadingSeries,
    absolute: bool = False,
    *, factor_values=None, min_obs=10, weights=None,
) -> StyleExposureEvidence:
    """Time-averaged standardized loading from factor-specific WLS evidence.

    ``absolute=False`` returns the signed mean exposure per style (a positive
    value means the factor loads positively on that style dimension across
    the sample); ``absolute=True`` returns the mean of the absolute exposures
    (magnitude — how much of the factor's variance lives on the style).

    A security risk panel requires explicit (T,N) factor_values. Never average
    the security risk panel itself as a proxy for the factor's style exposure.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    arr=panel.values
    T,K=arr.shape
    out = np.full(K, np.nan)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        vals = arr[:, k]
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
    panel: FactorLoadingSeries,
    min_finite: int = 5,
    *, factor_values=None, min_obs=10, weights=None,
) -> Dict[str, Any]:
    """The style dimension with the largest mean *absolute* exposure.

    Returns a plain dict (``style`` / ``value`` / ``absolute_mean`` /
    ``counts``) — the ``max_absolute_style_exposure`` evidence payload.
    ``value`` is the signed mean exposure of the winning style; NaN when no
    style has at least ``min_finite`` finite observations.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    if isinstance(min_finite,bool) or not isinstance(min_finite,(int,np.integer)) or min_finite<1:
        raise ValueError("min_finite must be a positive integer")
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


def compute_exposure_drift(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Mean absolute change of the per-style exposure series between adjacent
    periods (a persistence / stability measure).

    ``drift = mean_{t,k} |panel[t+1,k] - panel[t,k]|`` over jointly-finite
    adjacent cells.  NaN when the panel has fewer than 2 periods or no finite
    adjacent pair.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    arr=panel.values
    T,K=arr.shape
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


def compute_purity_ratio(panel: FactorLoadingSeries, min_finite: int = 5, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Time mean of residual/total weighted variance, exactly 1-R².

    Same factor, joint support, weights and intercept regression as its
    loadings. Constant factor, saturated model or no explanatory variation is
    undefined, not perfect purity. This is in-sample descriptive evidence.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    if isinstance(min_finite,bool) or not isinstance(min_finite,(int,np.integer)) or min_finite<1:
        raise ValueError("min_finite must be a positive integer")
    finite=panel.r_squared[np.isfinite(panel.r_squared)]
    return float(np.mean(1.-finite)) if len(finite)>=min_finite else float("nan")


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
    arr, _ = _panel_arrays(panel)
    T, N, K = arr.shape
    if (T, N) != fv.shape:
        raise ValueError(
            f"panel (T,N)=({T},{N}) must match factor (T,N)={fv.shape}"
        )
    # Regress per date with intercept; residuals (T, N) — NaN where invalid.
    _, _, residuals = compute_factor_loadings(fv, arr, intercept=True, min_obs=min_obs,weights=panel.regression_weights)

    daily_ics: list[float] = []
    for t in range(T):
        y = fwd[t]
        resid = residuals[t]
        joint = np.isfinite(y) & np.isfinite(resid)
        if np.sum(joint) < 2:
            continue
        # ``min_obs`` is the declared per-date evidence floor for this metric,
        # and applies to the final residual/label pair as well as the OLS fit.
        rho = _spearman_rank_correlation(
            resid[joint], y[joint], min_obs=min_obs
        )
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


def _select_style(panel: FactorLoadingSeries, style: str, **kwargs) -> float:
    """Signed mean exposure of one style dimension (NaN when absent)."""
    ev = compute_style_exposure_evidence(panel, absolute=False,**kwargs)
    names = list(ev.style_names)
    if style not in names:
        return float("nan")
    return float(ev.values[names.index(style)])


def compute_industry_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean industry-style exposure of the factor (one typed field)."""
    return _select_style(panel, "industry",factor_values=factor_values,min_obs=min_obs,weights=weights)


def compute_size_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean size-style exposure of the factor (one typed field)."""
    return _select_style(panel, "size",factor_values=factor_values,min_obs=min_obs,weights=weights)


def compute_beta_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean beta-style exposure of the factor (one typed field)."""
    return _select_style(panel, "beta",factor_values=factor_values,min_obs=min_obs,weights=weights)


def compute_liquidity_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean liquidity-style exposure of the factor (one typed field)."""
    return _select_style(panel, "liquidity",factor_values=factor_values,min_obs=min_obs,weights=weights)


def compute_volatility_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean volatility-style exposure of the factor (one typed field)."""
    return _select_style(panel, "volatility",factor_values=factor_values,min_obs=min_obs,weights=weights)


def compute_momentum_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean momentum-style exposure of the factor (one typed field)."""
    return _select_style(panel, "momentum",factor_values=factor_values,min_obs=min_obs,weights=weights)


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
    loadings=build_factor_loading_series(panel,factor_values,min_obs=min_obs)
    ev = compute_style_exposure_evidence(loadings, absolute=False)
    by_name = dict(zip(ev.style_names, ev.values))
    max_abs = compute_max_absolute_style_exposure(loadings)
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
        "exposure_drift": compute_exposure_drift(loadings),
        "neutralized_rank_ic": neutralized,
        "residual_rank_ic": neutralized,
        "purity_ratio": compute_purity_ratio(loadings),
        "factor_loading_series":loadings,
        "estimation_scope":loadings.estimation_scope,
        "method_version":loadings.method_version,
    }
