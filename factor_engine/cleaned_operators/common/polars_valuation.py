# -*- coding: utf-8 -*-
"""Valuation operators - Polars native implementations.

Price-to-book, earnings yield, valuation gaps, credit scores (Altman Z, Zmijewski),
and quality scores (Piotroski F-score variants).
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Basic valuation ratios
# ---------------------------------------------------------------------------

@register_operator(
    name="book_to_price",
    category="valuation",
    business_category="valuation",
    canonical="book_to_price",
    source=_SRC,
    backend="polars",
)
class BookToPriceNative(SeriesOperator):
    """Book value / price; zero/negative price → null."""

    metadata = OperatorMetadata(
        name="book_to_price",
        category="valuation",
        description="账面市值比",
        param_names=["book_value", "price"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, book_value: pl.DataFrame, price: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(book_value)
        exprs = []
        for c in cols:
            bv = book_value[c]
            p = price[c] if c in price.columns else pl.lit(None)
            exprs.append(
                pl.when((p.is_null()) | (p <= 0))
                .then(None)
                .otherwise(bv / p)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, book_value)


@register_operator(
    name="earnings_yield",
    category="valuation",
    business_category="valuation",
    canonical="earnings_yield",
    source=_SRC,
    backend="polars",
)
class EarningsYieldNative(SeriesOperator):
    """Earnings / price; zero/negative price → null."""

    metadata = OperatorMetadata(
        name="earnings_yield",
        category="valuation",
        description="盈利收益率",
        param_names=["earnings", "price"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, earnings: pl.DataFrame, price: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(earnings)
        exprs = []
        for c in cols:
            e = earnings[c]
            p = price[c] if c in price.columns else pl.lit(None)
            exprs.append(
                pl.when((p.is_null()) | (p <= 0))
                .then(None)
                .otherwise(e / p)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, earnings)


# ---------------------------------------------------------------------------
# Valuation gap operators
# ---------------------------------------------------------------------------

@register_operator(
    name="valuation_pe_gap_positive",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pe_gap_positive",
    source=_SRC,
    backend="polars",
)
class ValuationPEGapPositiveNative(SeriesOperator):
    """max(0, pe_ttm - pe_lyr); negative gaps → 0."""

    metadata = OperatorMetadata(
        name="valuation_pe_gap_positive",
        category="valuation",
        description="PE TTM vs LYR 正向差距",
        param_names=["pe_ttm", "pe_lyr"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ttm: pl.DataFrame, pe_lyr: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ttm)
        exprs = []
        for c in cols:
            ttm = pe_ttm[c]
            lyr = pe_lyr[c] if c in pe_lyr.columns else pl.lit(None)
            exprs.append(pl.max_horizontal(pl.lit(0.0), ttm - lyr).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ttm)


@register_operator(
    name="valuation_pe_gap_signed_log",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pe_gap_signed_log",
    source=_SRC,
    backend="polars",
)
class ValuationPEGapSignedLogNative(SeriesOperator):
    """sign(pe_ttm - pe_lyr) * log(1 + |pe_ttm - pe_lyr|)."""

    metadata = OperatorMetadata(
        name="valuation_pe_gap_signed_log",
        category="valuation",
        description="PE TTM vs LYR 有符号对数差距",
        param_names=["pe_ttm", "pe_lyr"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ttm: pl.DataFrame, pe_lyr: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ttm)
        exprs = []
        for c in cols:
            ttm = pe_ttm[c]
            lyr = pe_lyr[c] if c in pe_lyr.columns else pl.lit(None)
            diff = ttm - lyr
            exprs.append(
                (diff.sign() * (pl.lit(1.0) + diff.abs()).log()).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ttm)


@register_operator(
    name="valuation_pe_ttm_lyr_gap",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pe_ttm_lyr_gap",
    source=_SRC,
    backend="polars",
)
class ValuationPETtmLyrGapNative(SeriesOperator):
    """(pe_ttm - pe_lyr) / |pe_lyr|; zero pe_lyr → null."""

    metadata = OperatorMetadata(
        name="valuation_pe_ttm_lyr_gap",
        category="valuation",
        description="PE TTM vs LYR 相对差距",
        param_names=["pe_ttm", "pe_lyr"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ttm: pl.DataFrame, pe_lyr: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ttm)
        exprs = []
        for c in cols:
            ttm = pe_ttm[c]
            lyr = pe_lyr[c] if c in pe_lyr.columns else pl.lit(None)
            exprs.append(
                pl.when((lyr.is_null()) | (lyr == 0))
                .then(None)
                .otherwise((ttm - lyr) / lyr.abs())
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ttm)


@register_operator(
    name="valuation_pcf_gap_positive",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pcf_gap_positive",
    source=_SRC,
    backend="polars",
)
class ValuationPCFGapPositiveNative(SeriesOperator):
    """max(0, pcf_operating - pcf_free); negative → 0."""

    metadata = OperatorMetadata(
        name="valuation_pcf_gap_positive",
        category="valuation",
        description="PCF 经营 vs 自由现金流正向差距",
        param_names=["pcf_operating", "pcf_free"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pcf_operating: pl.DataFrame, pcf_free: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pcf_operating)
        exprs = []
        for c in cols:
            op = pcf_operating[c]
            fr = pcf_free[c] if c in pcf_free.columns else pl.lit(None)
            exprs.append(pl.max_horizontal(pl.lit(0.0), op - fr).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, pcf_operating)


@register_operator(
    name="valuation_pcf_gap_signed_log",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pcf_gap_signed_log",
    source=_SRC,
    backend="polars",
)
class ValuationPCFGapSignedLogNative(SeriesOperator):
    """sign(pcf_operating - pcf_free) * log(1 + |gap|)."""

    metadata = OperatorMetadata(
        name="valuation_pcf_gap_signed_log",
        category="valuation",
        description="PCF 有符号对数差距",
        param_names=["pcf_operating", "pcf_free"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pcf_operating: pl.DataFrame, pcf_free: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pcf_operating)
        exprs = []
        for c in cols:
            op = pcf_operating[c]
            fr = pcf_free[c] if c in pcf_free.columns else pl.lit(None)
            diff = op - fr
            exprs.append(
                (diff.sign() * (pl.lit(1.0) + diff.abs()).log()).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, pcf_operating)


@register_operator(
    name="valuation_pcf_definition_gap",
    category="valuation",
    business_category="valuation",
    canonical="valuation_pcf_definition_gap",
    source=_SRC,
    backend="polars",
)
class ValuationPCFDefinitionGapNative(SeriesOperator):
    """(pcf_operating - pcf_free) / |pcf_operating|; zero → null."""

    metadata = OperatorMetadata(
        name="valuation_pcf_definition_gap",
        category="valuation",
        description="PCF 定义差距比率",
        param_names=["pcf_operating", "pcf_free"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pcf_operating: pl.DataFrame, pcf_free: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pcf_operating)
        exprs = []
        for c in cols:
            op = pcf_operating[c]
            fr = pcf_free[c] if c in pcf_free.columns else pl.lit(None)
            exprs.append(
                pl.when((op.is_null()) | (op == 0))
                .then(None)
                .otherwise((op - fr) / op.abs())
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, pcf_operating)


@register_operator(
    name="valuation_growth_mismatch",
    category="valuation",
    business_category="valuation",
    canonical="valuation_growth_mismatch",
    source=_SRC,
    backend="polars",
)
class ValuationGrowthMismatchNative(SeriesOperator):
    """pe_ratio - growth_rate; high PE low growth → positive."""

    metadata = OperatorMetadata(
        name="valuation_growth_mismatch",
        category="valuation",
        description="估值增长错配",
        param_names=["pe_ratio", "growth_rate"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ratio: pl.DataFrame, growth_rate: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ratio)
        exprs = []
        for c in cols:
            pe = pe_ratio[c]
            gr = growth_rate[c] if c in growth_rate.columns else pl.lit(None)
            exprs.append((pe - gr).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ratio)


@register_operator(
    name="valuation_quality_mismatch",
    category="valuation",
    business_category="valuation",
    canonical="valuation_quality_mismatch",
    source=_SRC,
    backend="polars",
)
class ValuationQualityMismatchNative(SeriesOperator):
    """pe_ratio - quality_score; high PE low quality → positive."""

    metadata = OperatorMetadata(
        name="valuation_quality_mismatch",
        category="valuation",
        description="估值质量错配",
        param_names=["pe_ratio", "quality_score"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ratio: pl.DataFrame, quality_score: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ratio)
        exprs = []
        for c in cols:
            pe = pe_ratio[c]
            qs = quality_score[c] if c in quality_score.columns else pl.lit(None)
            exprs.append((pe - qs).alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ratio)


@register_operator(
    name="valuation_cashflow_disagreement",
    category="valuation",
    business_category="valuation",
    canonical="valuation_cashflow_disagreement",
    source=_SRC,
    backend="polars",
)
class ValuationCashflowDisagreementNative(SeriesOperator):
    """|pe_ratio - pcf_ratio| / (pe_ratio + pcf_ratio); measures divergence."""

    metadata = OperatorMetadata(
        name="valuation_cashflow_disagreement",
        category="valuation",
        description="PE与PCF分歧度",
        param_names=["pe_ratio", "pcf_ratio"],
        return_type="series",
        tags=["valuation", "polars", "native"],
    )

    def _calculate_series(
        self, pe_ratio: pl.DataFrame, pcf_ratio: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(pe_ratio)
        exprs = []
        for c in cols:
            pe = pe_ratio[c]
            pcf = pcf_ratio[c] if c in pcf_ratio.columns else pl.lit(None)
            denom = pe + pcf
            exprs.append(
                pl.when((denom.is_null()) | (denom == 0))
                .then(None)
                .otherwise((pe - pcf).abs() / denom)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, pe_ratio)


# ---------------------------------------------------------------------------
# Credit risk scores
# ---------------------------------------------------------------------------

@register_operator(
    name="altman_z_score",
    category="valuation",
    business_category="credit_risk",
    canonical="altman_z_score",
    source=_SRC,
    backend="polars",
)
class AltmanZScoreNative(SeriesOperator):
    """Altman Z-score: 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*ME/TL + 1.0*Sales/TA."""

    metadata = OperatorMetadata(
        name="altman_z_score",
        category="valuation",
        description="Altman Z-Score破产风险",
        param_names=["working_capital", "retained_earnings", "ebit", "market_equity", "sales", "total_assets", "total_liabilities"],
        return_type="series",
        tags=["valuation", "credit", "polars", "native"],
    )

    def _calculate_series(
        self,
        working_capital: pl.DataFrame,
        retained_earnings: pl.DataFrame,
        ebit: pl.DataFrame,
        market_equity: pl.DataFrame,
        sales: pl.DataFrame,
        total_assets: pl.DataFrame,
        total_liabilities: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(working_capital)
        exprs = []
        for c in cols:
            wc = working_capital[c]
            re = retained_earnings[c] if c in retained_earnings.columns else pl.lit(None)
            eb = ebit[c] if c in ebit.columns else pl.lit(None)
            me = market_equity[c] if c in market_equity.columns else pl.lit(None)
            sl = sales[c] if c in sales.columns else pl.lit(None)
            ta = total_assets[c] if c in total_assets.columns else pl.lit(None)
            tl = total_liabilities[c] if c in total_liabilities.columns else pl.lit(None)

            z = (
                pl.lit(1.2) * wc / ta
                + pl.lit(1.4) * re / ta
                + pl.lit(3.3) * eb / ta
                + pl.lit(0.6) * me / tl
                + pl.lit(1.0) * sl / ta
            )
            exprs.append(z.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, working_capital)


@register_operator(
    name="zmijewski_score",
    category="valuation",
    business_category="credit_risk",
    canonical="zmijewski_score",
    source=_SRC,
    backend="polars",
)
class ZmijewskiScoreNative(SeriesOperator):
    """Zmijewski score: -4.3 - 4.5*NI/TA + 5.7*TL/TA - 0.004*CA/CL."""

    metadata = OperatorMetadata(
        name="zmijewski_score",
        category="valuation",
        description="Zmijewski破产概率分数",
        param_names=["net_income", "total_liabilities", "current_assets", "current_liabilities", "total_assets"],
        return_type="series",
        tags=["valuation", "credit", "polars", "native"],
    )

    def _calculate_series(
        self,
        net_income: pl.DataFrame,
        total_liabilities: pl.DataFrame,
        current_assets: pl.DataFrame,
        current_liabilities: pl.DataFrame,
        total_assets: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(net_income)
        exprs = []
        for c in cols:
            ni = net_income[c]
            tl = total_liabilities[c] if c in total_liabilities.columns else pl.lit(None)
            ca = current_assets[c] if c in current_assets.columns else pl.lit(None)
            cl = current_liabilities[c] if c in current_liabilities.columns else pl.lit(None)
            ta = total_assets[c] if c in total_assets.columns else pl.lit(None)

            z = (
                pl.lit(-4.3)
                - pl.lit(4.5) * ni / ta
                + pl.lit(5.7) * tl / ta
                - pl.lit(0.004) * ca / cl
            )
            exprs.append(z.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, net_income)


# ---------------------------------------------------------------------------
# Piotroski F-score variants
# ---------------------------------------------------------------------------

@register_operator(
    name="piotroski_f_score",
    category="valuation",
    business_category="quality",
    canonical="piotroski_f_score",
    source=_SRC,
    backend="polars",
)
class PiotroskiFScoreNative(SeriesOperator):
    """Full 9-component Piotroski F-score (0-9); missing → null."""

    metadata = OperatorMetadata(
        name="piotroski_f_score",
        category="valuation",
        description="Piotroski F-Score质量分数",
        param_names=["roa", "cfo", "delta_roa", "accruals", "delta_leverage", "delta_liquidity", "equity_offering", "delta_margin", "delta_turnover"],
        return_type="series",
        tags=["valuation", "quality", "polars", "native"],
    )

    def _calculate_series(
        self,
        roa: pl.DataFrame,
        cfo: pl.DataFrame,
        delta_roa: pl.DataFrame,
        accruals: pl.DataFrame,
        delta_leverage: pl.DataFrame,
        delta_liquidity: pl.DataFrame,
        equity_offering: pl.DataFrame,
        delta_margin: pl.DataFrame,
        delta_turnover: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(roa)
        exprs = []
        for c in cols:
            score = pl.lit(0)
            # Profitability (4 points)
            score = score + pl.when(roa[c] > 0).then(1).otherwise(0)
            score = score + pl.when(cfo[c] if c in cfo.columns else pl.lit(None) > 0).then(1).otherwise(0)
            score = score + pl.when(delta_roa[c] if c in delta_roa.columns else pl.lit(None) > 0).then(1).otherwise(0)
            score = score + pl.when((cfo[c] if c in cfo.columns else pl.lit(None)) > roa[c]).then(1).otherwise(0)
            # Leverage (3 points)
            score = score + pl.when((delta_leverage[c] if c in delta_leverage.columns else pl.lit(None)) < 0).then(1).otherwise(0)
            score = score + pl.when((delta_liquidity[c] if c in delta_liquidity.columns else pl.lit(None)) > 0).then(1).otherwise(0)
            score = score + pl.when((equity_offering[c] if c in equity_offering.columns else pl.lit(None)) == 0).then(1).otherwise(0)
            # Operating efficiency (2 points)
            score = score + pl.when((delta_margin[c] if c in delta_margin.columns else pl.lit(None)) > 0).then(1).otherwise(0)
            score = score + pl.when((delta_turnover[c] if c in delta_turnover.columns else pl.lit(None)) > 0).then(1).otherwise(0)
            exprs.append(score.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, roa)


@register_operator(
    name="piotroski_f_score_tolerant",
    category="valuation",
    business_category="quality",
    canonical="piotroski_f_score_tolerant",
    source=_SRC,
    backend="polars",
)
class PiotroskiFScoreTolerantNative(SeriesOperator):
    """Piotroski F-score treating null as 0 contribution."""

    metadata = OperatorMetadata(
        name="piotroski_f_score_tolerant",
        category="valuation",
        description="Piotroski F-Score (容忍缺失)",
        param_names=["roa", "cfo", "delta_roa", "accruals", "delta_leverage", "delta_liquidity", "equity_offering", "delta_margin", "delta_turnover"],
        return_type="series",
        tags=["valuation", "quality", "polars", "native"],
    )

    def _calculate_series(
        self,
        roa: pl.DataFrame,
        cfo: pl.DataFrame,
        delta_roa: pl.DataFrame,
        accruals: pl.DataFrame,
        delta_leverage: pl.DataFrame,
        delta_liquidity: pl.DataFrame,
        equity_offering: pl.DataFrame,
        delta_margin: pl.DataFrame,
        delta_turnover: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(roa)
        exprs = []
        for c in cols:
            score = pl.lit(0)
            score = score + pl.when(roa[c] > 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((cfo[c] if c in cfo.columns else pl.lit(None)) > 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((delta_roa[c] if c in delta_roa.columns else pl.lit(None)) > 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((cfo[c] if c in cfo.columns else pl.lit(None)) > roa[c]).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((delta_leverage[c] if c in delta_leverage.columns else pl.lit(None)) < 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((delta_liquidity[c] if c in delta_liquidity.columns else pl.lit(None)) > 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((equity_offering[c] if c in equity_offering.columns else pl.lit(None)) == 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((delta_margin[c] if c in delta_margin.columns else pl.lit(None)) > 0).then(1).otherwise(0).fill_null(0)
            score = score + pl.when((delta_turnover[c] if c in delta_turnover.columns else pl.lit(None)) > 0).then(1).otherwise(0).fill_null(0)
            exprs.append(score.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, roa)


@register_operator(
    name="piotroski_partial_score",
    category="valuation",
    business_category="quality",
    canonical="piotroski_partial_score",
    source=_SRC,
    backend="polars",
)
class PiotroskiPartialScoreNative(SeriesOperator):
    """Partial score from available components; normalized to [0,1]."""

    metadata = OperatorMetadata(
        name="piotroski_partial_score",
        category="valuation",
        description="Piotroski部分分数",
        param_names=["roa", "cfo", "delta_roa"],
        return_type="series",
        tags=["valuation", "quality", "polars", "native"],
    )

    def _calculate_series(
        self,
        roa: pl.DataFrame,
        cfo: pl.DataFrame | None = None,
        delta_roa: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(roa)
        exprs = []
        for c in cols:
            score = pl.lit(0)
            count = pl.lit(0)
            # ROA positive
            s1 = pl.when(roa[c] > 0).then(1).otherwise(0)
            score = score + pl.when(roa[c].is_not_null()).then(s1).otherwise(0)
            count = count + pl.when(roa[c].is_not_null()).then(1).otherwise(0)
            # CFO positive
            if cfo is not None and c in cfo.columns:
                s2 = pl.when(cfo[c] > 0).then(1).otherwise(0)
                score = score + pl.when(cfo[c].is_not_null()).then(s2).otherwise(0)
                count = count + pl.when(cfo[c].is_not_null()).then(1).otherwise(0)
            # Delta ROA positive
            if delta_roa is not None and c in delta_roa.columns:
                s3 = pl.when(delta_roa[c] > 0).then(1).otherwise(0)
                score = score + pl.when(delta_roa[c].is_not_null()).then(s3).otherwise(0)
                count = count + pl.when(delta_roa[c].is_not_null()).then(1).otherwise(0)
            exprs.append(
                pl.when(count == 0).then(None).otherwise(score.cast(pl.Float64) / count).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, roa)


@register_operator(
    name="piotroski_observed_count",
    category="valuation",
    business_category="quality",
    canonical="piotroski_observed_count",
    source=_SRC,
    backend="polars",
)
class PiotroskiObservedCountNative(SeriesOperator):
    """Count of non-null Piotroski components."""

    metadata = OperatorMetadata(
        name="piotroski_observed_count",
        category="valuation",
        description="Piotroski可观测组件数",
        param_names=["roa", "cfo", "delta_roa", "accruals", "delta_leverage", "delta_liquidity", "equity_offering", "delta_margin", "delta_turnover"],
        return_type="series",
        tags=["valuation", "quality", "polars", "native"],
    )

    def _calculate_series(
        self,
        roa: pl.DataFrame,
        cfo: pl.DataFrame | None = None,
        delta_roa: pl.DataFrame | None = None,
        accruals: pl.DataFrame | None = None,
        delta_leverage: pl.DataFrame | None = None,
        delta_liquidity: pl.DataFrame | None = None,
        equity_offering: pl.DataFrame | None = None,
        delta_margin: pl.DataFrame | None = None,
        delta_turnover: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(roa)
        exprs = []
        for c in cols:
            count = roa[c].is_not_null().cast(pl.Int32)
            if cfo is not None and c in cfo.columns:
                count = count + cfo[c].is_not_null().cast(pl.Int32)
            if delta_roa is not None and c in delta_roa.columns:
                count = count + delta_roa[c].is_not_null().cast(pl.Int32)
            if accruals is not None and c in accruals.columns:
                count = count + accruals[c].is_not_null().cast(pl.Int32)
            if delta_leverage is not None and c in delta_leverage.columns:
                count = count + delta_leverage[c].is_not_null().cast(pl.Int32)
            if delta_liquidity is not None and c in delta_liquidity.columns:
                count = count + delta_liquidity[c].is_not_null().cast(pl.Int32)
            if equity_offering is not None and c in equity_offering.columns:
                count = count + equity_offering[c].is_not_null().cast(pl.Int32)
            if delta_margin is not None and c in delta_margin.columns:
                count = count + delta_margin[c].is_not_null().cast(pl.Int32)
            if delta_turnover is not None and c in delta_turnover.columns:
                count = count + delta_turnover[c].is_not_null().cast(pl.Int32)
            exprs.append(count.alias(c))
        result = pl.DataFrame(exprs)
        return _with_meta(result, roa)
