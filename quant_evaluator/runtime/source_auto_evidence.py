"""Deterministic evidence registry for the bounded factor-source auto router.

These records encode only already measured source-route predicates. They are
not a performance model and must not be widened without new end-to-end
evidence. Device capability, precision and current-memory admission remain
separate runtime gates in :mod:`quant_evaluator.api.factor_source`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SOURCE_AUTO_EVIDENCE_VERSION = "source_routes_20261001_v4"
SOURCE_F61_ALL_SHAPES = frozenset({(2586, 5461, 61), (2400, 5000, 61)})
SOURCE_F8_RANK_PAIR_SHAPE = (2586, 5461, 8)

_RANK_PAIR = frozenset({"rank_ic", "rank_ic_series"})
_MIXED_THREE = frozenset({"rank_ic", "quantile_spread", "factor_turnover_rate"})
_PEARSON_CHAIN = frozenset({"pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"})
_SOURCE_ALL_METRICS = frozenset({
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
})
SOURCE_AUTO_METRICS = _SOURCE_ALL_METRICS


@dataclass(frozen=True)
class SourceAutoEvidence:
    """One measured route envelope; shape/tile predicates are intentionally explicit."""

    evidence_id: str
    shape: tuple[int, int, int] | frozenset[tuple[int, int, int]]
    metrics: frozenset[str]
    minimum_requested_tile: int = 1
    exact_requested_tile: Optional[int] = None
    certified_tile_widths: tuple[int, ...] = ()
    legacy_reason: str = ""
    tile_reasons: tuple[tuple[int, str], ...] = ()
    evidence_artifacts: tuple[str, ...] = ()
    minimum_effective_vram_bytes: int = 14 * 1024 ** 3
    evidence_status: str = "measured_source_ab"

    def matches(self, shape, metrics, requested_tile_width) -> bool:
        shapes = self.shape if isinstance(self.shape, frozenset) else (self.shape,)
        return (
            tuple(shape) in shapes
            and frozenset(metrics) == self.metrics
            and len(tuple(metrics)) == len(self.metrics)
            and requested_tile_width >= self.minimum_requested_tile
            and (self.exact_requested_tile is None
                 or requested_tile_width == self.exact_requested_tile)
        )


# Exact evidence envelopes from the existing source API route table. F61 mixed
# three has measured tile widths 8 and 16; it selects the largest certified
# width no greater than the caller's cap. No other route is interpolated.
SOURCE_AUTO_EVIDENCE = (
    SourceAutoEvidence(
        "real_cos_f8_rank_pair", SOURCE_F8_RANK_PAIR_SHAPE, _RANK_PAIR,
        minimum_requested_tile=2, exact_requested_tile=2,
        certified_tile_widths=(2,),
        legacy_reason="bounded_f8_rank_pair_gpu",
        evidence_artifacts=("quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_cpu_first_20261001.json",
                            "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_cuda_first_20261001.json"),
        evidence_status="measured_source_ab"),
    SourceAutoEvidence(
        "real_cos_f8_rank_pair_cap8_tile2", SOURCE_F8_RANK_PAIR_SHAPE, _RANK_PAIR,
        minimum_requested_tile=8, exact_requested_tile=8,
        certified_tile_widths=(2,),
        legacy_reason="bounded_f8_rank_pair_gpu_cap8_tile2",
        evidence_artifacts=(
            "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_tile2_fresh_cpu_first_20261001.json",
            "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_tile2_fresh_cuda_first_20261001.json",
            "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_tile8_cpu_first_20261001.json",
            "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_tile8_cuda_first_20261001.json",
            "quant_evaluator/docs/benchmarks/real_cos_f8_source_auto_cap8_tile2_20261001.json",
        ),
        evidence_status="measured_source_ab"),
    SourceAutoEvidence("synthetic_f32_rank_pair_tile2", (2586, 5461, 32), _RANK_PAIR,
                       exact_requested_tile=2, certified_tile_widths=(2,),
                       legacy_reason="bounded_f32_rank_pair_gpu",
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/synthetic_f32_source_aba_20260929.json",)),
    SourceAutoEvidence("synthetic_f32_mixed_three_tile2", (2586, 5461, 32), _MIXED_THREE,
                       exact_requested_tile=2, certified_tile_widths=(2,),
                       legacy_reason="bounded_f32_mixed_three_gpu",
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/synthetic_f32_source_mixed_three_aba_20260929.json",)),
    SourceAutoEvidence("real_cos_f61_mixed_three_tile8_16", (2586, 5461, 61), _MIXED_THREE,
                       minimum_requested_tile=8, certified_tile_widths=(8, 16),
                       tile_reasons=((8, "bounded_f61_mixed_three_gpu"),
                                     (16, "bounded_f61_mixed_three_gpu_tile16")),
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/real_cos_f61_source_auto_tile16_20260929.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_gpu_tile_width_ab_20260929.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_mixed_three_source_ab_20260930.json")),
    SourceAutoEvidence("real_cos_f61_pearson_single_tile16", (2586, 5461, 61),
                       frozenset({"pearson_ic"}), minimum_requested_tile=16,
                       certified_tile_widths=(16,),
                       legacy_reason="bounded_f61_pearson_ic_gpu_tile16",
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/real_cos_f61_pearson_source_ab_20260930.json",)),
    SourceAutoEvidence("real_cos_f61_pearson_chain_tile16", (2586, 5461, 61),
                       _PEARSON_CHAIN, minimum_requested_tile=16,
                       certified_tile_widths=(16,),
                       legacy_reason="bounded_f61_pearson_chain_gpu_tile16",
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/real_cos_f61_pearson_chain_source_ab_20260930.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_pearson_chain_source_ab_reverse_20260930.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_consuming_source_pearson_ab_20260930.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_prefetch4_pearson_ab_20260930.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_prefetch4_reverse_pearson_ab_20260930.json")),
    SourceAutoEvidence("real_cos_f61_all_metrics_tile16",
                       SOURCE_F61_ALL_SHAPES,
                       _SOURCE_ALL_METRICS, minimum_requested_tile=16,
                       certified_tile_widths=(16,),
                       legacy_reason="bounded_f61_all_source_15_gpu_tile16",
                       evidence_artifacts=("quant_evaluator/docs/benchmarks/real_cos_f61_all_source_15_auto_20260930.json",
                                           "quant_evaluator/docs/benchmarks/real_cos_f61_all_source_15_2400d_5000a_ab_20260930.json")),
)


@dataclass(frozen=True)
class SourceAutoRoute:
    evidence_id: str
    legacy_reason: str
    effective_tile_width: int
    minimum_effective_vram_bytes: int
    evidence_artifacts: tuple[str, ...]
    evidence_status: str


def select_source_auto_route(*, shape, metrics, source_dtype, label_dtype,
                             requested_tile_width) -> Optional[SourceAutoRoute]:
    """Return a route only for an exact measured envelope; otherwise CPU fallback."""
    metrics = tuple(metrics)
    if source_dtype != "float64" or label_dtype != "float64":
        return None
    for item in SOURCE_AUTO_EVIDENCE:
        # Keep legacy evidence inspectable, but do not let it authorize
        # default GPU routing until a source-API A/B receipt is registered.
        if item.evidence_status != "measured_source_ab":
            continue
        if not item.matches(shape, metrics, requested_tile_width):
            continue
        if item.certified_tile_widths:
            tile_width = max(width for width in item.certified_tile_widths
                             if width <= requested_tile_width)
            reason = dict(item.tile_reasons).get(tile_width, item.legacy_reason)
        else:
            # F8 preserves its certified pass-through of the caller's cap.
            tile_width, reason = requested_tile_width, item.legacy_reason
        return SourceAutoRoute(item.evidence_id, reason, tile_width,
                               item.minimum_effective_vram_bytes, item.evidence_artifacts,
                               item.evidence_status)
    return None


__all__ = ("SOURCE_AUTO_EVIDENCE", "SOURCE_AUTO_EVIDENCE_VERSION", "SOURCE_AUTO_METRICS", "SOURCE_F61_ALL_SHAPES",
           "SourceAutoEvidence", "SourceAutoRoute", "select_source_auto_route")
