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


@register_operator(
    name="real_turnover_rate",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="real_turnover_rate",
    source="factor_dsl_np",
    backend="pandas_numpy",
    status="experimental",
)
class LqtpRealturnoverrateOp(TwoVarOperator):
    """真实流通盘换手率：volume / float_shares"""
    metadata = OperatorMetadata(
        name="real_turnover_rate",
        category="intraday_microstructure",
        description="真实流通盘换手率：volume / float_shares",
        param_names=["volume", "float_shares"],
        return_type="series",
        tags=["microstructure", "pit_safe"],
    )

    def _calculate_series(self, volume, float_shares, **kwargs):
        denom = float_shares.replace(0, np.nan) if hasattr(float_shares, "replace") else float_shares
        return volume / denom


@register_operator(
    name="intraday_vwap_deviation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_vwap_deviation",
    source="factor_dsl_np",
    backend="pandas_numpy",
    status="experimental",
)
class IntradayVwapDeviationOp(SeriesOperator):
    """收盘价相对 session 内累计 VWAP 的偏差。"""

    metadata = OperatorMetadata(
        name="intraday_vwap_deviation",
        category="intraday_microstructure",
        description="close / session_cum_vwap(price, volume) - 1（按日 reset）",
        param_names=["close", "price", "volume"],
        return_type="series",
        tags=["microstructure", "pit_safe", "session_aware"],
    )

    def _calculate_series(self, close, price, volume, **kwargs):
        from cleaned_operators.microstructure.session import session_vwap_deviation

        def _one(col: str) -> pd.Series:
            return session_vwap_deviation(close[col], price[col], volume[col])

        if hasattr(close, "columns"):
            out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
            for col in close.columns:
                out[col] = _one(col)
            return out
        return session_vwap_deviation(
            pd.Series(close), pd.Series(price), pd.Series(volume)
        )


@register_operator(
    name="micro_realized_vol",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_realized_vol",
    source="factor_dsl_np",
    backend="pandas_numpy",
    status="experimental",
)
class MicroRealizedVolOp(SeriesOperator):
    """滚动已实现波动率：sqrt(sum(r^2, window))，按 session 边界 reset。"""

    metadata = OperatorMetadata(
        name="micro_realized_vol",
        category="intraday_microstructure",
        description="已实现波动率 sqrt rolling sum of squared returns (session-aware)",
        param_names=["close"],
        return_type="series",
        tags=["microstructure", "pit_safe"],
    )

    def _calculate_series(self, close, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import rolling_by_session

        w = int(window)
        mp = max(2, int(min_periods))

        def _rv(col: pd.Series) -> pd.Series:
            ret = col.pct_change(fill_method=None)
            sq = ret.pow(2)
            rolled = rolling_by_session(sq, w, "sum", min_periods=mp)
            return rolled.pow(0.5)

        if hasattr(close, "apply"):
            return close.apply(_rv)
        return _rv(pd.Series(close))


@register_operator(
    name="micro_spread",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_spread",
    source="factor_dsl_np",
    status="experimental",
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
    status="experimental",
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
        from cleaned_operators.microstructure.session import pct_change_by_session

        if hasattr(close, "apply"):
            ret = close.apply(lambda s: pct_change_by_session(pd.Series(s)).abs())
        else:
            ret = pct_change_by_session(pd.Series(close)).abs()
        denom = close * volume
        return ret / denom.replace(0, np.nan)


@register_operator(
    name="micro_mid_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_mid_return",
    source="factor_dsl_np",
    status="experimental",
)
class MicroMidReturnOp(SeriesOperator):
    """中间价收益：mid=(high+low)/2 的 pct_change。"""

    metadata = OperatorMetadata(
        name="micro_mid_return",
        category="intraday_microstructure",
        description="Mid-price return from (high+low)/2",
        param_names=["high", "low"],
        return_type="series",
        tags=["microstructure", "pit_safe"],
    )

    def _calculate_series(self, high, low, **kwargs):
        from cleaned_operators.microstructure.session import pct_change_by_session

        mid = (high + low) / 2.0
        if hasattr(mid, "apply"):
            return mid.apply(lambda s: pct_change_by_session(pd.Series(s)))
        return pct_change_by_session(pd.Series(mid))


@register_operator(
    name="micro_bipower_var",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_bipower_var",
    source="factor_dsl_np",
    status="experimental",
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

    def _calculate_series(self, close, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session

        w = int(window)
        mp = max(2, int(min_periods))

        def _bv(s):
            r = pct_change_by_session(pd.Series(s))
            prod = r.abs() * r.abs().shift(1)
            return (np.pi / 2.0) * rolling_by_session(prod, w, "mean", min_periods=mp)

        if hasattr(close, "apply"):
            return close.apply(_bv)
        return _bv(close)


@register_operator(
    name="micro_jump_indicator",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_jump_indicator",
    source="factor_dsl_np",
    status="experimental",
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

    def _calculate_series(self, close, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session

        w = int(window)
        mp = max(2, int(min_periods))

        def _jump(s):
            r = pct_change_by_session(pd.Series(s))
            rv = rolling_by_session(r.pow(2), w, "sum", min_periods=mp)
            bv = (np.pi / 2.0) * rolling_by_session(
                r.abs() * r.abs().shift(1), w, "mean", min_periods=mp
            )
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
    status="experimental",
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

    def _calculate_series(self, close, volume, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session

        w = max(1, int(window))
        mp = max(1, int(min_periods))
        if hasattr(close, "apply"):
            ret = close.apply(lambda s: pct_change_by_session(pd.Series(s)))
        else:
            ret = pct_change_by_session(pd.Series(close))
        signed = np.sign(ret) * volume
        num = rolling_by_session(signed, w, "sum", min_periods=mp)
        den = rolling_by_session(volume, w, "sum", min_periods=mp).replace(0, np.nan)
        return num / den


@register_operator(
    name="micro_vpin",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_vpin",
    source="factor_dsl_np",
    status="experimental",
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

    def _calculate_series(self, close, volume, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session

        w = max(1, int(window))
        mp = max(1, int(min_periods))
        if hasattr(close, "apply"):
            ret = close.apply(lambda s: pct_change_by_session(pd.Series(s)).abs())
        else:
            ret = pct_change_by_session(pd.Series(close)).abs()
        weighted = ret * volume
        return rolling_by_session(weighted, w, "sum", min_periods=mp) / rolling_by_session(
            volume, w, "sum", min_periods=mp
        ).replace(0, np.nan)


@register_operator(
    name="micro_kyle_lambda",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_kyle_lambda",
    source="factor_dsl_np",
    status="experimental",
)
class MicroKyleLambdaOp(SeriesOperator):
    """Kyle lambda 代理：rolling Cov(|r|, volume) / Var(volume)。"""

    metadata = OperatorMetadata(
        name="micro_kyle_lambda",
        category="intraday_microstructure",
        description="Kyle lambda proxy Cov(|ret|,vol)/Var(vol)",
        param_names=["close", "volume"],
        return_type="series",
        tags=["microstructure"],
    )

    def _calculate_series(self, close, volume, window: int = 20, min_periods: int = 2, **kwargs):
        from cleaned_operators.microstructure.session import (
            pct_change_by_session,
            rolling_cov_by_session,
            rolling_var_by_session,
        )

        w = max(2, int(window))
        mp = max(2, int(min_periods))
        if hasattr(close, "apply"):
            out = close.copy() * np.nan
            for col in close.columns:
                r = pct_change_by_session(close[col]).abs()
                cov = rolling_cov_by_session(r, volume[col], w, min_periods=mp)
                var = rolling_var_by_session(volume[col], w, min_periods=mp).replace(0, np.nan)
                out[col] = cov / var
            return out
        r = pct_change_by_session(pd.Series(close)).abs()
        cov = rolling_cov_by_session(r, pd.Series(volume), w, min_periods=mp)
        var = rolling_var_by_session(pd.Series(volume), w, min_periods=mp).replace(0, np.nan)
        return cov / var

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
