# -*- coding: utf-8 -*-
"""
日内与微观结构算子。

语义
----
高频/日内字段相关变换，例如真实换手率 ``real_turnover_rate`` 等。
数据频率与日线因子不同，使用时需匹配数据源与 ``Factor.freq`` 元数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)


@register_operator(name="real_turnover_rate", category="intraday_microstructure", business_category="intraday_microstructure", canonical="real_turnover_rate", source="factor_dsl_np")
class LqtpRealturnoverrateOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="real_turnover_rate",
        category="intraday_microstructure",
        description="真实流通盘换手率：volume / float",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import real_turnover_rate_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: real_turnover_rate_(s.values, **kwargs) if kwargs else real_turnover_rate_(s.values))
        return real_turnover_rate_(*args, **kwargs)

@register_operator(
    name="micro_realized_vol",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_realized_vol",
    source="factor_dsl_np",
)
class MicroRealizedVolOp(SeriesOperator):
    """滚动已实现波动率：sqrt(sum(r^2, window))，r 为逐 bar 收益。"""

    metadata = OperatorMetadata(
        name="micro_realized_vol",
        category="intraday_microstructure",
        description="已实现波动率 sqrt rolling sum of squared returns",
        param_names=["close"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, window: int = 20, **kwargs):
        if hasattr(close, "apply"):
            return close.apply(
                lambda s: pd.Series(s)
                .pct_change()
                .pow(2)
                .rolling(int(window), min_periods=1)
                .sum()
                .pow(0.5)
                .values
            )
        ret = pd.Series(close).pct_change()
        return ret.pow(2).rolling(int(window), min_periods=1).sum().pow(0.5).values


@register_operator(
    name="micro_spread",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_spread",
    source="factor_dsl_np",
)
class MicroSpreadOp(SeriesOperator):
    """相对价差代理：(high - low) / close。"""

    metadata = OperatorMetadata(
        name="micro_spread",
        category="intraday_microstructure",
        description="相对价差 (high-low)/close",
        param_names=["high", "low", "close"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, high, low, close, **kwargs):
        spread = (high - low) / close.replace(0, np.nan)
        if hasattr(spread, "apply"):
            return spread
        return spread


@register_operator(
    name="micro_amihud_hf",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_amihud_hf",
    source="factor_dsl_np",
)
class MicroAmihudHfOp(SeriesOperator):
    """Amihud 非流动性：|r| / (close × volume)。"""

    metadata = OperatorMetadata(
        name="micro_amihud_hf",
        category="intraday_microstructure",
        description="Amihud illiquidity |return|/(close*volume)",
        param_names=["close", "volume"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, volume, **kwargs):
        ret = close.pct_change().abs() if hasattr(close, "pct_change") else pd.Series(close).pct_change().abs()
        denom = close * volume
        out = ret / denom.replace(0, np.nan)
        return out


@register_operator(
    name="micro_mid_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_mid_return",
    source="factor_dsl_np",
)
class MicroMidReturnOp(SeriesOperator):
    """中间价收益：mid=(high+low)/2 的 pct_change。"""

    metadata = OperatorMetadata(
        name="micro_mid_return",
        category="intraday_microstructure",
        description="Mid-price return from (high+low)/2",
        param_names=["high", "low"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, high, low, **kwargs):
        mid = (high + low) / 2.0
        return mid.pct_change() if hasattr(mid, "pct_change") else pd.Series(mid).pct_change()


@register_operator(
    name="micro_bipower_var",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_bipower_var",
    source="factor_dsl_np",
)
class MicroBipowerVarOp(SeriesOperator):
    """Bipower variation：(pi/2) * rolling_mean(|r_t|*|r_{t-1}|)。"""

    metadata = OperatorMetadata(
        name="micro_bipower_var",
        category="intraday_microstructure",
        description="Bipower variation estimator",
        param_names=["close"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, window: int = 20, **kwargs):
        w = int(window)

        def _bv(s):
            r = pd.Series(s).pct_change()
            prod = r.abs() * r.abs().shift(1)
            return (np.pi / 2.0) * prod.rolling(w, min_periods=1).mean()

        if hasattr(close, "apply"):
            return close.apply(_bv)
        return _bv(close)


@register_operator(
    name="micro_jump_indicator",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_jump_indicator",
    source="factor_dsl_np",
)
class MicroJumpIndicatorOp(SeriesOperator):
    """跳跃指示：max(RV - BV, 0)，RV=rolling sum(r^2)。"""

    metadata = OperatorMetadata(
        name="micro_jump_indicator",
        category="intraday_microstructure",
        description="Jump indicator max(RV-BV,0)",
        param_names=["close"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, window: int = 20, **kwargs):
        w = int(window)

        def _jump(s):
            r = pd.Series(s).pct_change()
            rv = r.pow(2).rolling(w, min_periods=1).sum()
            bv = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(w, min_periods=1).sum()
            return (rv - bv).clip(lower=0.0)

        if hasattr(close, "apply"):
            return close.apply(_jump)
        return _jump(close)


@register_operator(
    name="micro_trade_imbalance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_trade_imbalance",
    source="factor_dsl_np",
)
class MicroTradeImbalanceOp(SeriesOperator):
    """成交不平衡代理：rolling sum(volume * sign(r)) / rolling sum(volume)。"""

    metadata = OperatorMetadata(
        name="micro_trade_imbalance",
        category="intraday_microstructure",
        description="Volume-weighted return sign imbalance",
        param_names=["close", "volume"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, volume, window: int = 20, **kwargs):
        w = max(1, int(window))
        ret = close.pct_change() if hasattr(close, "pct_change") else pd.Series(close).pct_change()
        signed = np.sign(ret) * volume
        num = signed.rolling(w, min_periods=1).sum()
        den = volume.rolling(w, min_periods=1).sum().replace(0, np.nan)
        return num / den


@register_operator(
    name="micro_vpin",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_vpin",
    source="factor_dsl_np",
)
class MicroVpinOp(SeriesOperator):
    """VPIN 代理：rolling sum(|r| * volume) / rolling sum(volume)。"""

    metadata = OperatorMetadata(
        name="micro_vpin",
        category="intraday_microstructure",
        description="Volume-synchronized |return| intensity proxy",
        param_names=["close", "volume"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, volume, window: int = 20, **kwargs):
        w = max(1, int(window))
        ret = close.pct_change().abs() if hasattr(close, "pct_change") else pd.Series(close).pct_change().abs()
        weighted = ret * volume
        return weighted.rolling(w, min_periods=1).sum() / volume.rolling(w, min_periods=1).sum().replace(
            0, np.nan
        )

# api stub placeholders — 未注册算子，仅供 catalog 占位；勿与 @register_operator 实现混用

def _make_stub(*args, **kwargs):
    raise NotImplementedError("stub: _make_stub")

def alt_8k_item_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_8k_item_stub")

def alt_app_rating_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_app_rating_stub")

def alt_carbon_intensity_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_carbon_intensity_stub")

def alt_credit_spread_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_credit_spread_stub")

def alt_customer_concentration_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_customer_concentration_stub")

def alt_earnings_call_tone_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_earnings_call_tone_stub")

def alt_esg_controversy_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_esg_controversy_stub")

def alt_esg_score_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_esg_score_stub")

def alt_job_posting_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_job_posting_stub")

def alt_litigation_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_litigation_stub")

def alt_news_sentiment_x_volume_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_news_sentiment_x_volume_stub")

def alt_news_volume_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_news_volume_stub")

def alt_patent_citation_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_patent_citation_stub")

def alt_satellite_activity_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_satellite_activity_stub")

def alt_sentiment_delta_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_sentiment_delta_stub")

def alt_sentiment_ema_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_sentiment_ema_stub")

def alt_sentiment_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_sentiment_stub")

def alt_sentiment_vol_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_sentiment_vol_stub")

def alt_social_buzz_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_social_buzz_stub")

def alt_supply_chain_exposure_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_supply_chain_exposure_stub")

def alt_web_traffic_stub(*args, **kwargs):
    raise NotImplementedError("stub: alt_web_traffic_stub")

def analyst_dispersion_stub(*args, **kwargs):
    raise NotImplementedError("stub: analyst_dispersion_stub")

def analyst_revision_30d_stub(*args, **kwargs):
    raise NotImplementedError("stub: analyst_revision_30d_stub")

def days_since_filing_stub(*args, **kwargs):
    raise NotImplementedError("stub: days_since_filing_stub")

def days_since_forecast_stub(*args, **kwargs):
    raise NotImplementedError("stub: days_since_forecast_stub")

def event_window_mask_stub(*args, **kwargs):
    raise NotImplementedError("stub: event_window_mask_stub")

def fundamental_accruals_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_accruals_stub")

def fundamental_altman_z_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_altman_z_stub")

def fundamental_asset_growth_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_asset_growth_stub")

def fundamental_cagr_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_cagr_stub")

def fundamental_cf_accruals_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_cf_accruals_stub")

def fundamental_current_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_current_ratio_stub")

def fundamental_goodwill_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_goodwill_ratio_stub")

def fundamental_gross_margin_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_gross_margin_stub")

def fundamental_interest_coverage_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_interest_coverage_stub")

def fundamental_inv_growth_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_inv_growth_stub")

def fundamental_lag_quarter_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_lag_quarter_stub")

def fundamental_leverage_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_leverage_stub")

def fundamental_net_margin_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_net_margin_stub")

def fundamental_no_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_no_stub")

def fundamental_oper_margin_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_oper_margin_stub")

def fundamental_payout_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_payout_stub")

def fundamental_qoq_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_qoq_stub")

def fundamental_quick_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_quick_ratio_stub")

def fundamental_rec_growth_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_rec_growth_stub")

def fundamental_report_delay_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_report_delay_stub")

def fundamental_revision_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_revision_stub")

def fundamental_rnd_intensity_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_rnd_intensity_stub")

def fundamental_roa_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_roa_stub")

def fundamental_roe_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_roe_stub")

def fundamental_surprise_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_surprise_stub")

def fundamental_tax_rate_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_tax_rate_stub")

def fundamental_ttm_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_ttm_stub")

def fundamental_yoy_stub(*args, **kwargs):
    raise NotImplementedError("stub: fundamental_yoy_stub")

def insider_net_buy_stub(*args, **kwargs):
    raise NotImplementedError("stub: insider_net_buy_stub")

def institutional_ownership_chg_stub(*args, **kwargs):
    raise NotImplementedError("stub: institutional_ownership_chg_stub")

def lob_ofi_stub(*args, **kwargs):
    raise NotImplementedError("stub: lob_ofi_stub")

def micro_avg_trade_size_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_avg_trade_size_stub")

def micro_book_slope_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_book_slope_stub")

def micro_cancel_trade_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_cancel_trade_ratio_stub")

def micro_depth_imbalance_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_depth_imbalance_stub")

def micro_effective_spread_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_effective_spread_stub")

def micro_kyle_lambda_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_kyle_lambda_stub")

def micro_large_trade_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_large_trade_ratio_stub")

def micro_quote_update_rate_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_quote_update_rate_stub")

def micro_tick_rule_agreement_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_tick_rule_agreement_stub")

def micro_trade_count_intensity_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_trade_count_intensity_stub")

def micro_trade_imbalance_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_trade_imbalance_stub")

def micro_vpin_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_vpin_stub")

def universe_reit_stub(*args, **kwargs):
    raise NotImplementedError("stub: universe_reit_stub")
