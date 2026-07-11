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


@register_operator(name="real_turnover_rate", category="intraday_microstructure", business_category="intraday_microstructure", canonical="real_turnover_rate", source="lqtp_numpy")
class LqtpRealturnoverrateOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="real_turnover_rate",
        category="intraday_microstructure",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import real_turnover_rate_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: real_turnover_rate_(s.values, **kwargs) if kwargs else real_turnover_rate_(s.values))
        return real_turnover_rate_(*args, **kwargs)

# api stub placeholders

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

def micro_amihud_hf_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_amihud_hf_stub")

def micro_avg_trade_size_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_avg_trade_size_stub")

def micro_bipower_var_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_bipower_var_stub")

def micro_book_slope_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_book_slope_stub")

def micro_cancel_trade_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_cancel_trade_ratio_stub")

def micro_depth_imbalance_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_depth_imbalance_stub")

def micro_effective_spread_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_effective_spread_stub")

def micro_jump_indicator_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_jump_indicator_stub")

def micro_kyle_lambda_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_kyle_lambda_stub")

def micro_large_trade_ratio_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_large_trade_ratio_stub")

def micro_mid_return_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_mid_return_stub")

def micro_quote_update_rate_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_quote_update_rate_stub")

def micro_realized_vol_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_realized_vol_stub")

def micro_spread_stub(*args, **kwargs):
    raise NotImplementedError("stub: micro_spread_stub")

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
