"""Governed R20 aliases for causal expanding statistical diagnostics.

The legacy implementations are retained as analysis tools but their old
canonicals are removed by layer governance.  These reviewed canonicals reuse
the exact implementations while naming whether the emitted scalar is a
p-value or a statistic.  They remain research-only.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.common import statistics as _legacy
from factor_engine.cleaned_operators.base import register_operator
from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator
from factor_engine.cleaned_operators.composite_fastpath import (
    pd_vp_weighted_price, pd_vpmacd, pd_vpmacd_signal,
)
from factor_engine.cleaned_operators.microstructure import ops as _micro
from factor_engine.cleaned_operators.price_volume import ops as _price_volume

LEGACY_TO_REVIEWED = {
    "ACF": "ts_expanding_acf_statistic",
    "pacf": "ts_expanding_pacf_statistic",
    "bartlett_test": "ts_expanding_bartlett_pvalue",
    "chi_square_test": "ts_expanding_chi_square_pvalue",
    "corr_test": "ts_expanding_pearson_pvalue",
    "durbin_watson_test": "ts_expanding_durbin_watson_statistic",
    "granger_causality": "ts_expanding_granger_pvalue",
    "jarque_bera_test": "ts_expanding_jarque_bera_pvalue",
    "kendall_corr_test": "ts_expanding_kendall_pvalue",
    "kpss_test": "ts_expanding_kpss_pvalue",
    "ks_test": "ts_expanding_ks_pvalue",
    "levene_test": "ts_expanding_levene_pvalue",
    "lilliefors_test": "ts_expanding_lilliefors_pvalue",
    "spearman_corr_test": "ts_expanding_spearman_pvalue",
    "stationarity_test": "ts_expanding_adf_pvalue",
    "ttest_one_sample": "ts_expanding_ttest_one_sample_pvalue",
    "ttest_paired": "ts_expanding_ttest_paired_pvalue",
    "ttest_two_samples": "ts_expanding_ttest_two_sample_pvalue",
}

for _legacy_name, _canonical in LEGACY_TO_REVIEWED.items():
    _base = getattr(_legacy, _legacy_name)
    _metadata = copy.deepcopy(_base.metadata)
    _metadata.name = _canonical
    _adapter = type(
        "R20_" + _canonical,
        (_base,),
        {"metadata": _metadata, "__module__": __name__},
    )
    register_operator(
        name=_canonical,
        canonical=_canonical,
        backend="pandas_numpy",
        source="catalog_r20_statistical_adapters",
    )(_adapter)

def _vp_meta(name, last_param):
    return OperatorMetadata(name=name, category="experimental", description="FactorRecipeRegistry VP-MACD reviewed expansion", param_names=["close","volume","open","high","low",last_param], return_type="series", tags=["research","price_volume","pit_safe"])

@register_operator(name="recipe_vp_weighted_price", canonical="recipe_vp_weighted_price", backend="pandas_numpy", source="catalog_r20_statistical_adapters")
class RecipeVpWeightedPrice(SeriesOperator):
    metadata=_vp_meta("recipe_vp_weighted_price", "window")
    def _calculate_series(self, close, volume, open, high, low, window=20, **kwargs): return pd_vp_weighted_price(close,volume,open,high,low,window=window,min_periods=5)

@register_operator(name="recipe_vpmacd", canonical="recipe_vpmacd", backend="pandas_numpy", source="catalog_r20_statistical_adapters")
class RecipeVpMacd(SeriesOperator):
    metadata=_vp_meta("recipe_vpmacd", "lambda_param")
    def _calculate_series(self, close, volume, open, high, low, lambda_param=.9, **kwargs): return pd_vpmacd(close,volume,open,high,low,lambda_param)

@register_operator(name="recipe_vpmacd_signal", canonical="recipe_vpmacd_signal", backend="pandas_numpy", source="catalog_r20_statistical_adapters")
class RecipeVpMacdSignal(SeriesOperator):
    metadata=_vp_meta("recipe_vpmacd_signal", "lambda_param")
    def _calculate_series(self, close, volume, open, high, low, lambda_param=.9, **kwargs): return pd_vpmacd_signal(close,volume,open,high,low,lambda_param)


# The original microstructure canonicals are analysis tools moved out of the
# active factor registry by layer governance.  These reviewed adapters retain
# their exact, session-aware kernels under explicit factor-shaped names.
def _panelized_micro_calculate(name):
    def calculate(self, *args, **kwargs):
        first = next((x for x in args if hasattr(x, "columns")), None)
        if first is None:
            raise TypeError(f"{name} requires panel-shaped inputs")
        from factor_engine.cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session
        result = {}
        for column in first.columns:
            column_args = tuple(x[column] if hasattr(x, "columns") else x for x in args)
            if name == "recipe_micro_spread":
                high, low, close = column_args
                result[column] = (high - low) / close.replace(0, np.nan)
                continue
            close, volume = column_args[:2]
            ret = pct_change_by_session(pd.Series(close))
            if name == "recipe_micro_amihud_hf":
                result[column] = ret.abs() / (close * volume).replace(0, np.nan)
                continue
            window = int(column_args[2]) if len(column_args) > 2 else 20
            min_periods = int(column_args[3]) if len(column_args) > 3 else 2
            magnitude = ret.abs() if name == "recipe_micro_vpin" else np.sign(ret)
            numerator = rolling_by_session(magnitude * volume, window, "sum", min_periods=min_periods)
            denominator = rolling_by_session(volume, window, "sum", min_periods=min_periods).replace(0, np.nan)
            result[column] = numerator / denominator
        return first.__class__(result, index=first.index)
    return calculate

for _name, _base in {
    "recipe_micro_spread": _micro.MicroSpreadOp,
    "recipe_micro_trade_imbalance": _micro.MicroTradeImbalanceOp,
    "recipe_micro_vpin": _micro.MicroVpinOp,
    "recipe_micro_amihud_hf": _micro.MicroAmihudHfOp,
}.items():
    _metadata = copy.deepcopy(_base.metadata)
    _metadata.name = _name
    if _name == "recipe_micro_spread":
        _metadata.param_names = ["high", "low", "close"]
        _metadata.panel_params = ("high", "low", "close")
    elif _name in {"recipe_micro_trade_imbalance", "recipe_micro_vpin"}:
        _metadata.param_names = ["close", "volume", "window", "min_periods"]
        _metadata.panel_params = ("close", "volume")
        _metadata.scalar_params = ("window", "min_periods")
    else:
        _metadata.param_names = ["close", "volume"]
        _metadata.panel_params = ("close", "volume")
    _adapter = type(
        "R20_" + _name,
        (_base,),
        {
            "metadata": _metadata,
            "_calculate_series": _panelized_micro_calculate(_name),
            "__module__": __name__,
        },
    )
    register_operator(
        name=_name,
        canonical=_name,
        backend="pandas_numpy",
        source="catalog_r20_statistical_adapters",
    )(_adapter)

_downside_metadata = copy.deepcopy(_price_volume.LqtpDownsidebetaOp.metadata)
_downside_metadata.name = "recipe_downside_beta"

@register_operator(name="recipe_downside_beta", canonical="recipe_downside_beta", backend="pandas_numpy", source="catalog_r20_statistical_adapters")
class RecipeDownsideBeta(_price_volume.LqtpDownsidebetaOp):
    """Reviewed factor-shaped adapter retaining the exact downside-beta kernel."""
    metadata = _downside_metadata
