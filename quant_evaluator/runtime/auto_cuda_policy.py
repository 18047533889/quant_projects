"""Certified auto-CUDA routing policy data and pure shape/VRAM helpers.

The request selector remains in :mod:`quant_evaluator.runtime.evaluator` so
its long-standing evaluator-module monkeypatch points stay intact.
"""

# The 701-day x 5314-stock x 2-factor registered A-share panel showed full
# CPU/CUDA output parity and a CUDA advantage in cold and alternating warm
# public-facade runs for these metrics (2026-09-26). Other metrics retain CPU.
_AUTO_CUDA_POLICY_VERSION = "ashare_public_routes_20260929_v28"
_AUTO_SMALL_PROFILE = "ashare_701d_20260926"
_AUTO_LARGE_PROFILE = "real_cos_region_20260927"
_AUTO_LARGE_EXTRAP_PROFILE = "real_cos_bounded_headroom_20260927"
_AUTO_REAL_COS_F5_PROFILE = "real_cos_f5_exact_20260929"
_AUTO_REAL_COS_F5_SHAPE = (2586, 5461, 5)
_AUTO_REAL_COS_F5_MIXED_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
# The independent eight-factor COS A/B supports only this exact single-panel
# shape; do not widen the gate based on the two-factor COS region.
_AUTO_REAL_COS_F8_PROFILE = "real_cos_f8_exact_20260927"
_AUTO_REAL_COS_F8_SHAPE = (2586, 5461, 8)
_AUTO_REAL_COS_F8_METRICS = frozenset({
    "rank_ic", "rank_ic_series", "ic_ir", "quantile_spread", "factor_turnover_rate",
    "quantile_returns_daily", "quantile_returns_full",
})
_AUTO_REAL_COS_F8_RANK_FAMILY = frozenset({"rank_ic", "rank_ic_series", "ic_ir"})
_AUTO_REAL_COS_F8_RANK_MIN_EFFECTIVE_VRAM_BYTES = 13_999_136_256
_AUTO_REAL_COS_F8_OTHER_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F8_BATCH_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F8_BATCH_NAMES = frozenset({"rank_chain", "quantile_chain"})
_AUTO_REAL_COS_F8_QUANTILE_BATCH_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F12_PROFILE = "real_cos_f12_exact_20260927"
_AUTO_REAL_COS_F12_SHAPE = (2586, 5461, 12)
_AUTO_REAL_COS_F12_RANK_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F12_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F12_TURNOVER_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F12_PEARSON_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F12_SINGLE_METRICS = frozenset({"rank_ic", "quantile_spread", "factor_turnover_rate", "pearson_ic"})
_AUTO_REAL_COS_F12_BATCH_NAMES = frozenset({"rank_chain", "quantile_chain"})
_AUTO_REAL_COS_F13_PROFILE = "real_cos_f13_exact_20260927"
_AUTO_REAL_COS_F13_SHAPE = (2586, 5461, 13)
_AUTO_REAL_COS_F13_BATCH_NAMES = frozenset({"rank_chain", "quantile_chain"})
_AUTO_REAL_COS_F13_SINGLE_METRICS = frozenset({"factor_turnover_rate", "rank_ic_positive_ratio"})
_AUTO_REAL_COS_F13_RANK_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F13_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F13_RANK_POSITIVE_RATIO_PEAK_VRAM_BYTES = 9_332_757_504
_AUTO_REAL_COS_F13_RANK_POSITIVE_RATIO_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR = frozenset({"rank_ic", "rank_ic_positive_ratio"})
_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR_PEAK_VRAM_BYTES = 13_263_441_920
_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F24_PROFILE = "real_cos_f24_exact_20260927"
_AUTO_REAL_COS_F24_SHAPE = (2586, 5461, 24)
_AUTO_REAL_COS_F24_BATCH_NAMES = frozenset({"rank_chain", "quantile_chain"})
_AUTO_REAL_COS_F24_RANK_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F24_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
# The 2026-09-29 real-COS mixed-three A/B peaked at 16,050,598,912 bytes.
# Require 20 GiB effective free VRAM to retain headroom over that observation.
_AUTO_REAL_COS_F24_MIXED_THREE_MIN_EFFECTIVE_VRAM_BYTES = 20 * 1024 ** 3
_AUTO_REAL_COS_F32_PROFILE = "real_cos_f32_exact_20260928"
_AUTO_REAL_COS_F32_SHAPE = (2586, 5461, 32)
_AUTO_REAL_COS_F32_BATCH_NAMES = frozenset({
    "rank_chain", "quantile_chain", "real_cos_mixed_three", "pearson_chain",
    "coverage_mixed_four", "default_five"})
_AUTO_REAL_COS_F32_COVERAGE_MIN_EFFECTIVE_VRAM_BYTES = 2 * 1024 ** 3
_AUTO_REAL_COS_F32_COVERAGE_MIXED_FOUR_MIN_EFFECTIVE_VRAM_BYTES = 12 * 1024 ** 3
_AUTO_REAL_COS_F32_DEFAULT_FIVE_MIN_EFFECTIVE_VRAM_BYTES = 12 * 1024 ** 3
_AUTO_REAL_COS_F32_COVERAGE_MIXED_FOUR = frozenset(
    ("rank_ic", "quantile_spread", "factor_turnover_rate", "coverage"))
_AUTO_REAL_COS_F32_DEFAULT_FIVE = frozenset(
    ("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage"))
_AUTO_REAL_COS_F32_PEARSON_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F32_RANK_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F32_RANK_SERIES_MIN_EFFECTIVE_VRAM_BYTES = 17 * 1024 ** 3
_AUTO_REAL_COS_F32_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_REAL_COS_F32_SINGLE_METRICS = frozenset(
    {"rank_ic", "rank_ic_series", "quantile_returns_full", "quantile_returns_daily", "quantile_spread", "factor_turnover_rate", "coverage"})
_AUTO_REAL_COS_F2_RANK_SERIES_SHAPE = (2586, 5461, 2)
_AUTO_REAL_COS_F2_RANK_SERIES_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F2_BATCH_SHAPE = (2586, 5461, 2)
_AUTO_REAL_COS_F2_RANK_BATCH_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3
_AUTO_REAL_COS_F2_QUANTILE_BATCH_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3
_AUTO_LARGE_METRICS = frozenset({"rank_ic", "quantile_spread"})
_AUTO_LARGE_MIN_TIMES, _AUTO_LARGE_MAX_TIMES = 1000, 2600
_AUTO_LARGE_MIN_ASSETS, _AUTO_LARGE_MAX_ASSETS = 5000, 5500
_AUTO_LARGE_HEADROOM_MAX_TIMES, _AUTO_LARGE_HEADROOM_MAX_ASSETS = 3200, 6000
# Observed rank peaks were under 512 bytes/cell for F2 and 672 for F1.
# The 1.5x effective-free margins are rounded upward to 768/1024.
_AUTO_LARGE_RANK_F2_VRAM_BYTES_PER_CELL = 768
_AUTO_LARGE_RANK_F1_VRAM_BYTES_PER_CELL = 1024
_AUTO_CUDA_METRICS = frozenset({
    "daily_quantile_monotonicity_rate",
    "daily_quantile_monotonicity_series",
    "ic_ir", "ic_median", "ic_std",
    "quantile_monotonicity", "quantile_returns_daily",
    "quantile_returns_full", "quantile_spread",
    "rank_ic", "rank_ic_series", "turnover",
})
# Exact canonical metric sets; order and aliases do not affect shared GPU
# intermediates.  These three public batches passed cold/warm A/B and 8/8
# complete parity checks on the registered A-share panel and NVIDIA L20.
_AUTO_CUDA_BATCHES = {
    frozenset(("rank_ic", "rank_ic_series", "ic_std", "ic_ir")): "rank_chain",
    frozenset(("quantile_returns_full", "quantile_returns_daily",
               "quantile_spread", "quantile_monotonicity",
               "daily_quantile_monotonicity_rate")): "quantile_chain",
    frozenset(("rank_ic", "rank_ic_series", "ic_ir", "quantile_spread",
               "turnover", "factor_turnover_rate")): "mixed_core",
}
# This chain has only the exact F32 public-facade whole-request A/B certificate.
_AUTO_REAL_COS_F32_PEARSON_CHAIN = frozenset(
    ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"))
# This exact three-metric request passed whole-request CPU/CUDA A/B on the
# bound real COS panel. Its shape gate is narrower than the single-metric
# real-COS region and does not include headroom extrapolation.
_AUTO_REAL_COS_MIXED_THREE = frozenset(
    ("rank_ic", "quantile_spread", "factor_turnover_rate"))
_AUTO_REAL_COS_MIXED_THREE_SHAPE = (2586, 5461, 2)
_AUTO_BATCH_MIN_EFFECTIVE_VRAM_BYTES = 12 * 1024 ** 3
_AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES = 8 * 1024 ** 3

def _auto_public_shape_profile(factor_batch):
    """Identify a public-facade A/B-backed shape region."""
    shape = (factor_batch.num_times, factor_batch.num_assets, factor_batch.num_factors)
    if shape == _AUTO_REAL_COS_F32_SHAPE:
        return _AUTO_REAL_COS_F32_PROFILE
    if shape == _AUTO_REAL_COS_F24_SHAPE:
        return _AUTO_REAL_COS_F24_PROFILE
    if shape == _AUTO_REAL_COS_F5_SHAPE:
        return _AUTO_REAL_COS_F5_PROFILE
    if shape == _AUTO_REAL_COS_F13_SHAPE:
        return _AUTO_REAL_COS_F13_PROFILE
    if shape == _AUTO_REAL_COS_F12_SHAPE:
        return _AUTO_REAL_COS_F12_PROFILE
    if shape == _AUTO_REAL_COS_F8_SHAPE:
        return _AUTO_REAL_COS_F8_PROFILE
    if (600 <= factor_batch.num_times <= 800
            and 5000 <= factor_batch.num_assets <= 5500
            and factor_batch.num_factors == 2):
        return _AUTO_SMALL_PROFILE
    if (_AUTO_LARGE_MIN_TIMES <= factor_batch.num_times <= _AUTO_LARGE_HEADROOM_MAX_TIMES
            and _AUTO_LARGE_MIN_ASSETS <= factor_batch.num_assets <= _AUTO_LARGE_HEADROOM_MAX_ASSETS
            and (factor_batch.num_factors == 2 or
                 (factor_batch.num_factors == 1 and factor_batch.num_times >= 2000))):
        if (factor_batch.num_times <= _AUTO_LARGE_MAX_TIMES
                and factor_batch.num_assets <= _AUTO_LARGE_MAX_ASSETS):
            return _AUTO_LARGE_PROFILE
        return _AUTO_LARGE_EXTRAP_PROFILE
    return None


def _auto_large_min_effective_vram_bytes(factor_batch, metric_id):
    if metric_id == "rank_ic":
        bytes_per_cell = (_AUTO_LARGE_RANK_F1_VRAM_BYTES_PER_CELL if factor_batch.num_factors == 1
                          else _AUTO_LARGE_RANK_F2_VRAM_BYTES_PER_CELL)
        return max(_AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES,
                   bytes_per_cell * factor_batch.num_times
                   * factor_batch.num_assets * factor_batch.num_factors)
    return _AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES

__all__ = (
    '_AUTO_BATCH_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_CUDA_BATCHES',
    '_AUTO_CUDA_METRICS',
    '_AUTO_CUDA_POLICY_VERSION',
    '_AUTO_LARGE_EXTRAP_PROFILE',
    '_AUTO_LARGE_METRICS',
    '_AUTO_LARGE_HEADROOM_MAX_ASSETS',
    '_AUTO_LARGE_HEADROOM_MAX_TIMES',
    '_AUTO_LARGE_MAX_ASSETS',
    '_AUTO_LARGE_MAX_TIMES',
    '_AUTO_LARGE_MIN_ASSETS',
    '_AUTO_LARGE_MIN_TIMES',
    '_AUTO_LARGE_PROFILE',
    '_AUTO_LARGE_RANK_F1_VRAM_BYTES_PER_CELL',
    '_AUTO_LARGE_RANK_F2_VRAM_BYTES_PER_CELL',
    '_AUTO_REAL_COS_F12_BATCH_NAMES',
    '_AUTO_REAL_COS_F12_PEARSON_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F12_PROFILE',
    '_AUTO_REAL_COS_F12_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F12_RANK_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F12_SHAPE',
    '_AUTO_REAL_COS_F12_SINGLE_METRICS',
    '_AUTO_REAL_COS_F12_TURNOVER_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_BATCH_NAMES',
    '_AUTO_REAL_COS_F13_PROFILE',
    '_AUTO_REAL_COS_F13_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_RANK_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR',
    '_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_RANK_POSITIVE_PAIR_PEAK_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_RANK_POSITIVE_RATIO_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_RANK_POSITIVE_RATIO_PEAK_VRAM_BYTES',
    '_AUTO_REAL_COS_F13_SHAPE',
    '_AUTO_REAL_COS_F13_SINGLE_METRICS',
    '_AUTO_REAL_COS_F24_BATCH_NAMES',
    '_AUTO_REAL_COS_F24_MIXED_THREE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F24_PROFILE',
    '_AUTO_REAL_COS_F24_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F24_RANK_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F24_SHAPE',
    '_AUTO_REAL_COS_F2_BATCH_SHAPE',
    '_AUTO_REAL_COS_F2_QUANTILE_BATCH_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F2_RANK_BATCH_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F2_RANK_SERIES_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F2_RANK_SERIES_SHAPE',
    '_AUTO_REAL_COS_F32_BATCH_NAMES',
    '_AUTO_REAL_COS_F32_DEFAULT_FIVE',
    '_AUTO_REAL_COS_F32_DEFAULT_FIVE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_COVERAGE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_COVERAGE_MIXED_FOUR_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_COVERAGE_MIXED_FOUR',
    '_AUTO_REAL_COS_F32_PEARSON_CHAIN',
    '_AUTO_REAL_COS_F32_PEARSON_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_PROFILE',
    '_AUTO_REAL_COS_F32_QUANTILE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_RANK_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_RANK_SERIES_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F32_SHAPE',
    '_AUTO_REAL_COS_F32_SINGLE_METRICS',
    '_AUTO_REAL_COS_F5_MIXED_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F5_PROFILE',
    '_AUTO_REAL_COS_F5_SHAPE',
    '_AUTO_REAL_COS_F8_BATCH_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F8_BATCH_NAMES',
    '_AUTO_REAL_COS_F8_METRICS',
    '_AUTO_REAL_COS_F8_OTHER_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F8_PROFILE',
    '_AUTO_REAL_COS_F8_QUANTILE_BATCH_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F8_RANK_FAMILY',
    '_AUTO_REAL_COS_F8_RANK_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_REAL_COS_F8_SHAPE',
    '_AUTO_REAL_COS_MIXED_THREE',
    '_AUTO_REAL_COS_MIXED_THREE_SHAPE',
    '_AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES',
    '_AUTO_SMALL_PROFILE',
    '_auto_public_shape_profile',
    '_auto_large_min_effective_vram_bytes',
)
