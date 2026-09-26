"""Registered IC decay over explicit, matured Pearson horizon evaluations."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quant_evaluator.api.horizons import HorizonEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.metrics.ic_summary import compute_ic_decay_from_mean_ics


@dataclass(frozen=True, eq=False)
class ICDecayResult:
    """Horizon × factor finite-day mean Pearson IC with bound axis identities."""

    horizons: tuple[int, ...]
    factor_ids: tuple[str, ...]
    values: np.ndarray
    as_of: Any
    sample_policy: str
    label_content_refs: Mapping[int, str]
    maturity_counts: Mapping[int, int]
    sample_counts: Mapping[int, int]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        horizons = tuple(self.horizons)
        factor_ids = tuple(self.factor_ids)
        if len(horizons) < 2 or any(type(h) is not int or h <= 0 for h in horizons):
            raise ValueError("horizons must be at least two positive integers")
        if horizons != tuple(sorted(set(horizons))) or not factor_ids or len(set(factor_ids)) != len(factor_ids):
            raise ValueError("horizon and factor axes must be ordered and unique")
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.shape != (len(horizons), len(factor_ids)):
            raise ValueError("values must have shape (horizons, factors)")
        if self.sample_policy not in {"common", "per_horizon"} or self.as_of is None:
            raise ValueError("sample_policy and as_of must be explicit")
        refs = dict(self.label_content_refs)
        maturity = dict(self.maturity_counts)
        samples = dict(self.sample_counts)
        if any(set(mapping) != set(horizons) for mapping in (refs, maturity, samples)):
            raise ValueError("provenance mappings must cover every horizon")
        values.flags.writeable = False
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "label_content_refs", MappingProxyType(refs))
        object.__setattr__(self, "maturity_counts", MappingProxyType(maturity))
        object.__setattr__(self, "sample_counts", MappingProxyType(samples))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "content_hash", stable_content_hex(
            tag="ICDecayResult.v1",
            fields={"horizons": horizons, "factor_ids": factor_ids, "values": values,
                    "as_of": self.as_of, "sample_policy": self.sample_policy,
                    "label_content_refs": {str(k): v for k, v in refs.items()},
                    "maturity_counts": {str(k): v for k, v in maturity.items()},
                    "sample_counts": {str(k): v for k, v in samples.items()}, "provenance": dict(self.provenance)},
        ))


def compute_ic_decay(bundle: HorizonEvaluationBundle) -> ICDecayResult:
    """Compute registered ic_decay from explicit matured Pearson daily ICs.

    The one-label evaluate facade has no multi-horizon input contract. Call
    evaluate_horizons(..., ic_method="pearson") with explicit LabelBundles,
    then pass its result here. Spearman bundles fail closed.
    """
    if not isinstance(bundle, HorizonEvaluationBundle):
        raise TypeError("bundle must be a HorizonEvaluationBundle")
    if bundle.ic_method != "pearson":
        raise ValueError("ic_decay requires a Pearson horizon evaluation")
    means = []
    for horizon in bundle.horizons:
        artifact = bundle.daily_ic_artifacts[horizon]
        if artifact.metric_id != "pearson_ic_series":
            raise ValueError("every horizon requires pearson_ic_series")
        values = np.asarray(artifact.values, dtype=np.float64)
        finite = np.isfinite(values)
        counts = finite.sum(axis=0)
        sums = np.where(finite, values, 0.0).sum(axis=0)
        means.append(np.divide(
            sums, counts, out=np.full(len(bundle.factor_ids), np.nan),
            where=counts > 0,
        ))
    mean_ics = np.stack(means, axis=0)
    decay = compute_ic_decay_from_mean_ics(mean_ics, horizons=bundle.horizons)
    return ICDecayResult(
        horizons=bundle.horizons, factor_ids=bundle.factor_ids, values=decay,
        as_of=bundle.as_of, sample_policy=bundle.sample_policy,
        label_content_refs=bundle.label_content_refs,
        maturity_counts=bundle.maturity_counts,
        sample_counts=bundle.sample_counts,
        provenance={"metric_id": "ic_decay", "ic_method": "pearson",
                    "min_assets": bundle.provenance.get("min_assets"),
                    "factor_value_hash": bundle.provenance.get("factor_value_hash")},
    )


__all__ = ["ICDecayResult", "compute_ic_decay"]
