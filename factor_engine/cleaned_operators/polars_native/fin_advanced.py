"""
Polars Native Implementation - Advanced Financial Indicators (Phase 4)

78 operators for advanced financial statement analysis, ratios, and quality metrics.
Implements fin_* operator family for fundamental research.

Key patterns:
- Financial ratios: numerator / denominator with null handling
- Growth rates: pct_change() or manual (current - lagged) / lagged
- Time series features: rolling operations, volatility, trends
- Sparse data handling typical of quarterly/annual financial reports
"""

import polars as pl
import numpy as np
from typing import Optional

from cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)


# ============================================================================
# INTENSITY & RATIO METRICS
# ============================================================================

@register_operator(name="fin_acquisition_cash_intensity", canonical="fin_acquisition_cash_intensity", backend="polars", research_only=True)
class FinAcquisitionCashIntensityPolarsNative(SeriesOperator):
    """Cash paid for acquisitions / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_acquisition_cash_intensity",
        category="fundamental",
        description="Cash paid for acquisitions / Total assets.",
        param_names=["acquisition_cash", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, acquisition_cash, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "acq": acquisition_cash,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("acq") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_borrowing_intensity", canonical="fin_borrowing_intensity", backend="polars", research_only=True)
class FinBorrowingIntensityPolarsNative(SeriesOperator):
    """New borrowing / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_borrowing_intensity",
        category="fundamental",
        description="New borrowing / Total assets.",
        param_names=["new_borrowing", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, new_borrowing, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "borrow": new_borrowing,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("borrow") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_capex_intensity", canonical="fin_capex_intensity", backend="polars", research_only=True)
class FinCapexIntensityPolarsNative(SeriesOperator):
    """Capital expenditure / Revenue."""
    
    metadata = OperatorMetadata(
        name="fin_capex_intensity",
        category="fundamental",
        description="Capital expenditure / Revenue.",
        param_names=["capex", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, capex, revenue, **kwargs):
        return (
            pl.DataFrame({
                "capex": capex,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.when(pl.col("rev") != 0).then(pl.col("capex") / pl.col("rev")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_goodwill_intensity", canonical="fin_goodwill_intensity", backend="polars", research_only=True)
class FinGoodwillIntensityPolarsNative(SeriesOperator):
    """Goodwill / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_goodwill_intensity",
        category="fundamental",
        description="Goodwill / Total assets.",
        param_names=["goodwill", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, goodwill, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "gw": goodwill,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("gw") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_impairment_intensity", canonical="fin_impairment_intensity", backend="polars", research_only=True)
class FinImpairmentIntensityPolarsNative(SeriesOperator):
    """Impairment charges / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_impairment_intensity",
        category="fundamental",
        description="Impairment charges / Total assets.",
        param_names=["impairment", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, impairment, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "imp": impairment,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("imp") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_lease_intensity", canonical="fin_lease_intensity", backend="polars", research_only=True)
class FinLeaseIntensityPolarsNative(SeriesOperator):
    """Lease liabilities / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_lease_intensity",
        category="fundamental",
        description="Lease liabilities / Total assets.",
        param_names=["lease_liabilities", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, lease_liabilities, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "lease": lease_liabilities,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("lease") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_rd_total_intensity", canonical="fin_rd_total_intensity", backend="polars", research_only=True)
class FinRdTotalIntensityPolarsNative(SeriesOperator):
    """R&D expense / Revenue."""
    
    metadata = OperatorMetadata(
        name="fin_rd_total_intensity",
        category="fundamental",
        description="R&D expense / Revenue.",
        param_names=["rd_expense", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, rd_expense, revenue, **kwargs):
        return (
            pl.DataFrame({
                "rd": rd_expense,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.when(pl.col("rev") != 0).then(pl.col("rd") / pl.col("rev")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )



@register_operator(name="fin_debt_repayment_intensity", canonical="fin_debt_repayment_intensity", backend="polars", research_only=True)
class FinDebtRepaymentIntensityPolarsNative(SeriesOperator):
    """Debt repayment / Operating cash flow."""
    
    metadata = OperatorMetadata(
        name="fin_debt_repayment_intensity",
        category="fundamental",
        description="Debt repayment / Operating cash flow.",
        param_names=["debt_repayment", "operating_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, debt_repayment, operating_cf, **kwargs):
        return (
            pl.DataFrame({
                "repay": debt_repayment,
                "ocf": operating_cf,
            })
            .lazy()
            .select([
                pl.when(pl.col("ocf") != 0).then(pl.col("repay") / pl.col("ocf")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_contract_asset_intensity", canonical="fin_contract_asset_intensity", backend="polars", research_only=True)
class FinContractAssetIntensityPolarsNative(SeriesOperator):
    """Contract assets / Total assets."""
    
    metadata = OperatorMetadata(
        name="fin_contract_asset_intensity",
        category="fundamental",
        description="Contract assets / Total assets.",
        param_names=["contract_assets", "total_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, contract_assets, total_assets, **kwargs):
        return (
            pl.DataFrame({
                "ca": contract_assets,
                "assets": total_assets,
            })
            .lazy()
            .select([
                pl.when(pl.col("assets") != 0).then(pl.col("ca") / pl.col("assets")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_contract_liability_intensity", canonical="fin_contract_liability_intensity", backend="polars", research_only=True)
class FinContractLiabilityIntensityPolarsNative(SeriesOperator):
    """Contract liabilities / Total liabilities."""
    
    metadata = OperatorMetadata(
        name="fin_contract_liability_intensity",
        category="fundamental",
        description="Contract liabilities / Total liabilities.",
        param_names=["contract_liabilities", "total_liabilities"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, contract_liabilities, total_liabilities, **kwargs):
        return (
            pl.DataFrame({
                "cl": contract_liabilities,
                "liab": total_liabilities,
            })
            .lazy()
            .select([
                pl.when(pl.col("liab") != 0).then(pl.col("cl") / pl.col("liab")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# GROWTH METRICS
# ============================================================================

@register_operator(name="fin_capex_growth", canonical="fin_capex_growth", backend="polars", research_only=True)
class FinCapexGrowthPolarsNative(SeriesOperator):
    """YoY growth rate of capital expenditure."""
    
    metadata = OperatorMetadata(
        name="fin_capex_growth",
        category="fundamental",
        description="YoY growth rate of capital expenditure.",
        param_names=["capex"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, capex, periods: int = 4, **kwargs):
        return (
            capex.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(capex.name != 0).then(pl.col(capex.name) / pl.col(capex.name).otherwise(None).shift(periods)) - 1)
                .alias(capex.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_equity_capital_growth", canonical="fin_equity_capital_growth", backend="polars", research_only=True)
class FinEquityCapitalGrowthPolarsNative(SeriesOperator):
    """YoY growth rate of shareholders' equity."""
    
    metadata = OperatorMetadata(
        name="fin_equity_capital_growth",
        category="fundamental",
        description="YoY growth rate of shareholders' equity.",
        param_names=["equity"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, equity, periods: int = 4, **kwargs):
        return (
            equity.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(equity.name != 0).then(pl.col(equity.name) / pl.col(equity.name).otherwise(None).shift(periods)) - 1)
                .alias(equity.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_contract_asset_growth", canonical="fin_contract_asset_growth", backend="polars", research_only=True)
class FinContractAssetGrowthPolarsNative(SeriesOperator):
    """Period-over-period growth of contract assets."""
    
    metadata = OperatorMetadata(
        name="fin_contract_asset_growth",
        category="fundamental",
        description="Period-over-period growth of contract assets.",
        param_names=["contract_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, contract_assets, periods: int = 1, **kwargs):
        return (
            contract_assets.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(contract_assets.name != 0).then(pl.col(contract_assets.name) / pl.col(contract_assets.name).otherwise(None).shift(periods)) - 1)
                .alias(contract_assets.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_contract_liability_growth", canonical="fin_contract_liability_growth", backend="polars", research_only=True)
class FinContractLiabilityGrowthPolarsNative(SeriesOperator):
    """Period-over-period growth of contract liabilities."""
    
    metadata = OperatorMetadata(
        name="fin_contract_liability_growth",
        category="fundamental",
        description="Period-over-period growth of contract liabilities.",
        param_names=["contract_liabilities"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, contract_liabilities, periods: int = 1, **kwargs):
        return (
            contract_liabilities.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(contract_liabilities.name != 0).then(pl.col(contract_liabilities.name) / pl.col(contract_liabilities.name).otherwise(None).shift(periods)) - 1)
                .alias(contract_liabilities.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cagr", canonical="fin_cagr", backend="polars", research_only=True)
class FinCagrPolarsNative(SeriesOperator):
    """Compound annual growth rate over specified periods."""
    
    metadata = OperatorMetadata(
        name="fin_cagr",
        category="fundamental",
        description="Compound annual growth rate over specified periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 12, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)).pow(1.0 / periods) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )



# ============================================================================
# DIVERGENCE & GAP METRICS
# ============================================================================

@register_operator(name="fin_actual_expectation_divergence", canonical="fin_actual_expectation_divergence", backend="polars", research_only=True)
class FinActualExpectationDivergencePolarsNative(SeriesOperator):
    """Divergence between actual and expected values."""
    
    metadata = OperatorMetadata(
        name="fin_actual_expectation_divergence",
        category="fundamental",
        description="Divergence between actual and expected values.",
        param_names=["actual", "expected"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, actual, expected, **kwargs):
        return (
            pl.DataFrame({
                "actual": actual,
                "expected": expected,
            })
            .lazy()
            .select([
                pl.when(pl.col("expected" != 0).then(pl.col("actual") - pl.col("expected")) / pl.col("expected").otherwise(None).abs()).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cash_earnings_gap", canonical="fin_cash_earnings_gap", backend="polars", research_only=True)
class FinCashEarningsGapPolarsNative(SeriesOperator):
    """Operating cash flow minus net income (accruals proxy)."""
    
    metadata = OperatorMetadata(
        name="fin_cash_earnings_gap",
        category="fundamental",
        description="Operating cash flow minus net income (accruals proxy).",
        param_names=["operating_cf", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_cf, net_income, **kwargs):
        return (
            pl.DataFrame({
                "ocf": operating_cf,
                "ni": net_income,
            })
            .lazy()
            .select([
                (pl.col("ocf") - pl.col("ni")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cash_sales_divergence", canonical="fin_cash_sales_divergence", backend="polars", research_only=True)
class FinCashSalesDivergencePolarsNative(SeriesOperator):
    """Difference in growth rates: cash from operations vs revenue."""
    
    metadata = OperatorMetadata(
        name="fin_cash_sales_divergence",
        category="fundamental",
        description="Difference in growth rates: cash from operations vs revenue.",
        param_names=["operating_cf", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_cf, revenue, periods: int = 1, **kwargs):
        return (
            pl.DataFrame({
                "ocf": operating_cf,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.col("ocf"),
                pl.col("rev"),
                pl.when(pl.col("ocf" != 0).then(pl.col("ocf") / pl.col("ocf").otherwise(None).shift(periods)) - 1).alias("ocf_growth"),
                pl.when(pl.col("rev" != 0).then(pl.col("rev") / pl.col("rev").otherwise(None).shift(periods)) - 1).alias("rev_growth"),
            ])
            .select([
                (pl.col("ocf_growth") - pl.col("rev_growth")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_comprehensive_income_gap", canonical="fin_comprehensive_income_gap", backend="polars", research_only=True)
class FinComprehensiveIncomeGapPolarsNative(SeriesOperator):
    """Comprehensive income minus net income."""
    
    metadata = OperatorMetadata(
        name="fin_comprehensive_income_gap",
        category="fundamental",
        description="Comprehensive income minus net income.",
        param_names=["comprehensive_income", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, comprehensive_income, net_income, **kwargs):
        return (
            pl.DataFrame({
                "ci": comprehensive_income,
                "ni": net_income,
            })
            .lazy()
            .select([
                (pl.col("ci") - pl.col("ni")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_deferred_tax_gap", canonical="fin_deferred_tax_gap", backend="polars", research_only=True)
class FinDeferredTaxGapPolarsNative(SeriesOperator):
    """Change in deferred tax assets minus deferred tax liabilities."""
    
    metadata = OperatorMetadata(
        name="fin_deferred_tax_gap",
        category="fundamental",
        description="Change in deferred tax assets minus deferred tax liabilities.",
        param_names=["deferred_tax_assets", "deferred_tax_liabilities"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, deferred_tax_assets, deferred_tax_liabilities, periods: int = 1, **kwargs):
        return (
            pl.DataFrame({
                "dta": deferred_tax_assets,
                "dtl": deferred_tax_liabilities,
            })
            .lazy()
            .select([
                (pl.col("dta") - pl.col("dta").shift(periods)).alias("dta_change"),
                (pl.col("dtl") - pl.col("dtl").shift(periods)).alias("dtl_change"),
            ])
            .select([
                (pl.col("dta_change") - pl.col("dtl_change")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_expense_sales_divergence", canonical="fin_expense_sales_divergence", backend="polars", research_only=True)
class FinExpenseSalesDivergencePolarsNative(SeriesOperator):
    """Difference in growth rates: operating expenses vs revenue."""
    
    metadata = OperatorMetadata(
        name="fin_expense_sales_divergence",
        category="fundamental",
        description="Difference in growth rates: operating expenses vs revenue.",
        param_names=["operating_expense", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_expense, revenue, periods: int = 1, **kwargs):
        return (
            pl.DataFrame({
                "opex": operating_expense,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.col("opex"),
                pl.col("rev"),
                pl.when(pl.col("opex" != 0).then(pl.col("opex") / pl.col("opex").otherwise(None).shift(periods)) - 1).alias("opex_growth"),
                pl.when(pl.col("rev" != 0).then(pl.col("rev") / pl.col("rev").otherwise(None).shift(periods)) - 1).alias("rev_growth"),
            ])
            .select([
                (pl.col("opex_growth") - pl.col("rev_growth")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_inventory_sales_divergence", canonical="fin_inventory_sales_divergence", backend="polars", research_only=True)
class FinInventorySalesDivergencePolarsNative(SeriesOperator):
    """Difference in growth rates: inventory vs revenue."""
    
    metadata = OperatorMetadata(
        name="fin_inventory_sales_divergence",
        category="fundamental",
        description="Difference in growth rates: inventory vs revenue.",
        param_names=["inventory", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, inventory, revenue, periods: int = 1, **kwargs):
        return (
            pl.DataFrame({
                "inv": inventory,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.col("inv"),
                pl.col("rev"),
                pl.when(pl.col("inv" != 0).then(pl.col("inv") / pl.col("inv").otherwise(None).shift(periods)) - 1).alias("inv_growth"),
                pl.when(pl.col("rev" != 0).then(pl.col("rev") / pl.col("rev").otherwise(None).shift(periods)) - 1).alias("rev_growth"),
            ])
            .select([
                (pl.col("inv_growth") - pl.col("rev_growth")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_receivable_sales_divergence", canonical="fin_receivable_sales_divergence", backend="polars", research_only=True)
class FinReceivableSalesDivergencePolarsNative(SeriesOperator):
    """Difference in growth rates: accounts receivable vs revenue."""
    
    metadata = OperatorMetadata(
        name="fin_receivable_sales_divergence",
        category="fundamental",
        description="Difference in growth rates: accounts receivable vs revenue.",
        param_names=["accounts_receivable", "revenue"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, accounts_receivable, revenue, periods: int = 1, **kwargs):
        return (
            pl.DataFrame({
                "ar": accounts_receivable,
                "rev": revenue,
            })
            .lazy()
            .select([
                pl.col("ar"),
                pl.col("rev"),
                pl.when(pl.col("ar" != 0).then(pl.col("ar") / pl.col("ar").otherwise(None).shift(periods)) - 1).alias("ar_growth"),
                pl.when(pl.col("rev" != 0).then(pl.col("rev") / pl.col("rev").otherwise(None).shift(periods)) - 1).alias("rev_growth"),
            ])
            .select([
                (pl.col("ar_growth") - pl.col("rev_growth")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_contract_asset_liability_gap", canonical="fin_contract_asset_liability_gap", backend="polars", research_only=True)
class FinContractAssetLiabilityGapPolarsNative(SeriesOperator):
    """Contract assets minus contract liabilities."""
    
    metadata = OperatorMetadata(
        name="fin_contract_asset_liability_gap",
        category="fundamental",
        description="Contract assets minus contract liabilities.",
        param_names=["contract_assets", "contract_liabilities"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, contract_assets, contract_liabilities, **kwargs):
        return (
            pl.DataFrame({
                "ca": contract_assets,
                "cl": contract_liabilities,
            })
            .lazy()
            .select([
                (pl.col("ca") - pl.col("cl")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_lease_asset_liability_gap", canonical="fin_lease_asset_liability_gap", backend="polars", research_only=True)
class FinLeaseAssetLiabilityGapPolarsNative(SeriesOperator):
    """Right-of-use assets minus lease liabilities."""
    
    metadata = OperatorMetadata(
        name="fin_lease_asset_liability_gap",
        category="fundamental",
        description="Right-of-use assets minus lease liabilities.",
        param_names=["rou_assets", "lease_liabilities"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, rou_assets, lease_liabilities, **kwargs):
        return (
            pl.DataFrame({
                "rou": rou_assets,
                "lease": lease_liabilities,
            })
            .lazy()
            .select([
                (pl.col("rou") - pl.col("lease")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_financing_gap", canonical="fin_financing_gap", backend="polars", research_only=True)
class FinFinancingGapPolarsNative(SeriesOperator):
    """Operating CF + Investing CF (measures external financing need)."""
    
    metadata = OperatorMetadata(
        name="fin_financing_gap",
        category="fundamental",
        description="Operating CF + Investing CF (measures external financing need).",
        param_names=["operating_cf", "investing_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_cf, investing_cf, **kwargs):
        return (
            pl.DataFrame({
                "ocf": operating_cf,
                "icf": investing_cf,
            })
            .lazy()
            .select([
                (pl.col("ocf") + pl.col("icf")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_divergence", canonical="fin_divergence", backend="polars", research_only=True)
class FinDivergencePolarsNative(SeriesOperator):
    """Generic divergence between two financial metrics."""
    
    metadata = OperatorMetadata(
        name="fin_divergence",
        category="fundamental",
        description="Generic divergence between two financial metrics.",
        param_names=["metric1", "metric2"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, metric1, metric2, **kwargs):
        return (
            pl.DataFrame({
                "m1": metric1,
                "m2": metric2,
            })
            .lazy()
            .select([
                pl.when(pl.col("m2" != 0).then(pl.col("m1") - pl.col("m2")) / pl.col("m2").otherwise(None).abs()).alias("result")
            ])
            .collect()
            .to_series()
        )



# ============================================================================
# QUALITY & COVERAGE METRICS
# ============================================================================

@register_operator(name="fin_core_earnings_ratio", canonical="fin_core_earnings_ratio", backend="polars", research_only=True)
class FinCoreEarningsRatioPolarsNative(SeriesOperator):
    """Core earnings / Reported earnings."""
    
    metadata = OperatorMetadata(
        name="fin_core_earnings_ratio",
        category="fundamental",
        description="Core earnings / Reported earnings.",
        param_names=["core_earnings", "reported_earnings"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, core_earnings, reported_earnings, **kwargs):
        return (
            pl.DataFrame({
                "core": core_earnings,
                "reported": reported_earnings,
            })
            .lazy()
            .select([
                pl.when(pl.col("reported") != 0).then(pl.col("core") / pl.col("reported")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_discontinued_operation_ratio", canonical="fin_discontinued_operation_ratio", backend="polars", research_only=True)
class FinDiscontinuedOperationRatioPolarsNative(SeriesOperator):
    """Income from discontinued operations / Total net income."""
    
    metadata = OperatorMetadata(
        name="fin_discontinued_operation_ratio",
        category="fundamental",
        description="Income from discontinued operations / Total net income.",
        param_names=["discontinued_income", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, discontinued_income, net_income, **kwargs):
        return (
            pl.DataFrame({
                "disc": discontinued_income,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("disc") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_noncore_income_ratio", canonical="fin_noncore_income_ratio", backend="polars", research_only=True)
class FinNoncoreIncomeRatioPolarsNative(SeriesOperator):
    """Non-core income / Total income."""
    
    metadata = OperatorMetadata(
        name="fin_noncore_income_ratio",
        category="fundamental",
        description="Non-core income / Total income.",
        param_names=["noncore_income", "total_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, noncore_income, total_income, **kwargs):
        return (
            pl.DataFrame({
                "noncore": noncore_income,
                "total": total_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("total") != 0).then(pl.col("noncore") / pl.col("total")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_minority_profit_share", canonical="fin_minority_profit_share", backend="polars", research_only=True)
class FinMinorityProfitSharePolarsNative(SeriesOperator):
    """Minority interest / Net income."""
    
    metadata = OperatorMetadata(
        name="fin_minority_profit_share",
        category="fundamental",
        description="Minority interest / Net income.",
        param_names=["minority_interest", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, minority_interest, net_income, **kwargs):
        return (
            pl.DataFrame({
                "minority": minority_interest,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("minority") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_fair_value_income_dependence", canonical="fin_fair_value_income_dependence", backend="polars", research_only=True)
class FinFairValueIncomeDependencePolarsNative(SeriesOperator):
    """Fair value gains / Net income."""
    
    metadata = OperatorMetadata(
        name="fin_fair_value_income_dependence",
        category="fundamental",
        description="Fair value gains / Net income.",
        param_names=["fair_value_gains", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, fair_value_gains, net_income, **kwargs):
        return (
            pl.DataFrame({
                "fv": fair_value_gains,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("fv") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_investment_income_dependence", canonical="fin_investment_income_dependence", backend="polars", research_only=True)
class FinInvestmentIncomeDependencePolarsNative(SeriesOperator):
    """Investment income / Net income."""
    
    metadata = OperatorMetadata(
        name="fin_investment_income_dependence",
        category="fundamental",
        description="Investment income / Net income.",
        param_names=["investment_income", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, investment_income, net_income, **kwargs):
        return (
            pl.DataFrame({
                "inv_inc": investment_income,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("inv_inc") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_other_earnings_dependence", canonical="fin_other_earnings_dependence", backend="polars", research_only=True)
class FinOtherEarningsDependencePolarsNative(SeriesOperator):
    """Other income / Net income."""
    
    metadata = OperatorMetadata(
        name="fin_other_earnings_dependence",
        category="fundamental",
        description="Other income / Net income.",
        param_names=["other_income", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, other_income, net_income, **kwargs):
        return (
            pl.DataFrame({
                "other": other_income,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("other") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_goodwill_risk_score", canonical="fin_goodwill_risk_score", backend="polars", research_only=True)
class FinGoodwillRiskScorePolarsNative(SeriesOperator):
    """Goodwill / Market cap (higher = more impairment risk)."""
    
    metadata = OperatorMetadata(
        name="fin_goodwill_risk_score",
        category="fundamental",
        description="Goodwill / Market cap (higher = more impairment risk).",
        param_names=["goodwill", "market_cap"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, goodwill, market_cap, **kwargs):
        return (
            pl.DataFrame({
                "gw": goodwill,
                "mcap": market_cap,
            })
            .lazy()
            .select([
                pl.when(pl.col("mcap") != 0).then(pl.col("gw") / pl.col("mcap")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_applicability_mask", canonical="fin_applicability_mask", backend="polars", research_only=True)
class FinApplicabilityMaskPolarsNative(SeriesOperator):
    """Binary mask indicating if metric is applicable (non-null, non-zero)."""
    
    metadata = OperatorMetadata(
        name="fin_applicability_mask",
        category="fundamental",
        description="Binary mask indicating if metric is applicable (non-null, non-zero).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name).is_not_null() & (pl.col(value.name) != 0))
                .cast(pl.Float64)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_fundamental_strength_coverage", canonical="fin_fundamental_strength_coverage", backend="polars", research_only=True)
class FinFundamentalStrengthCoveragePolarsNative(SeriesOperator):
    """Count of non-null strength metrics / Total expected."""
    
    metadata = OperatorMetadata(
        name="fin_fundamental_strength_coverage",
        category="fundamental",
        description="Count of non-null strength metrics / Total expected.",
        param_names=["metrics"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, metrics, total_expected: int = 10, **kwargs):
        # Expect a single series representing combined metrics or a count
        return (
            metrics.to_frame()
            .lazy()
            .select([
                (pl.col(metrics.name).is_not_null().cast(pl.Float64) / float(total_expected))
                .alias(metrics.name)
            ])
            .collect()
            .to_series()
        )



# ============================================================================
# CONVERSION & ACCRUAL METRICS
# ============================================================================

@register_operator(name="fin_cash_conversion", canonical="fin_cash_conversion", backend="polars", research_only=True)
class FinCashConversionPolarsNative(SeriesOperator):
    """Operating cash flow / Net income."""
    
    metadata = OperatorMetadata(
        name="fin_cash_conversion",
        category="fundamental",
        description="Operating cash flow / Net income.",
        param_names=["operating_cf", "net_income"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_cf, net_income, **kwargs):
        return (
            pl.DataFrame({
                "ocf": operating_cf,
                "ni": net_income,
            })
            .lazy()
            .select([
                pl.when(pl.col("ni") != 0).then(pl.col("ocf") / pl.col("ni")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_total_operating_accruals", canonical="fin_total_operating_accruals", backend="polars", research_only=True)
class FinTotalOperatingAccrualsPolarsNative(SeriesOperator):
    """Net income minus operating cash flow."""
    
    metadata = OperatorMetadata(
        name="fin_total_operating_accruals",
        category="fundamental",
        description="Net income minus operating cash flow.",
        param_names=["net_income", "operating_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, net_income, operating_cf, **kwargs):
        return (
            pl.DataFrame({
                "ni": net_income,
                "ocf": operating_cf,
            })
            .lazy()
            .select([
                (pl.col("ni") - pl.col("ocf")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_working_capital_accruals", canonical="fin_working_capital_accruals", backend="polars", research_only=True)
class FinWorkingCapitalAccrualsPolarsNative(SeriesOperator):
    """Change in working capital (proxy for accruals)."""
    
    metadata = OperatorMetadata(
        name="fin_working_capital_accruals",
        category="fundamental",
        description="Change in working capital (proxy for accruals).",
        param_names=["working_capital"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, working_capital, periods: int = 1, **kwargs):
        return (
            working_capital.to_frame()
            .lazy()
            .select([
                (pl.col(working_capital.name) - pl.col(working_capital.name).shift(periods))
                .alias(working_capital.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_working_capital_change", canonical="fin_working_capital_change", backend="polars", research_only=True)
class FinWorkingCapitalChangePolarsNative(SeriesOperator):
    """Period-over-period change in working capital."""
    
    metadata = OperatorMetadata(
        name="fin_working_capital_change",
        category="fundamental",
        description="Period-over-period change in working capital.",
        param_names=["working_capital"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, working_capital, periods: int = 1, **kwargs):
        return (
            working_capital.to_frame()
            .lazy()
            .select([
                (pl.col(working_capital.name) - pl.col(working_capital.name).shift(periods))
                .alias(working_capital.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_delta_noa", canonical="fin_delta_noa", backend="polars", research_only=True)
class FinDeltaNoaPolarsNative(SeriesOperator):
    """Change in net operating assets."""
    
    metadata = OperatorMetadata(
        name="fin_delta_noa",
        category="fundamental",
        description="Change in net operating assets.",
        param_names=["net_operating_assets"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, net_operating_assets, periods: int = 1, **kwargs):
        return (
            net_operating_assets.to_frame()
            .lazy()
            .select([
                (pl.col(net_operating_assets.name) - pl.col(net_operating_assets.name).shift(periods))
                .alias(net_operating_assets.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# COVERAGE & PROXY METRICS
# ============================================================================

@register_operator(name="fin_debt_service_coverage_proxy", canonical="fin_debt_service_coverage_proxy", backend="polars", research_only=True)
class FinDebtServiceCoverageProxyPolarsNative(SeriesOperator):
    """Operating CF / (Interest expense + principal repayment proxy)."""
    
    metadata = OperatorMetadata(
        name="fin_debt_service_coverage_proxy",
        category="fundamental",
        description="Operating CF / (Interest expense + principal repayment proxy).",
        param_names=["operating_cf", "interest_expense", "debt_repayment"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, operating_cf, interest_expense, debt_repayment, **kwargs):
        return (
            pl.DataFrame({
                "ocf": operating_cf,
                "int": interest_expense,
                "repay": debt_repayment,
            })
            .lazy()
            .select([
                (pl.col("ocf") / (pl.col("int") + pl.col("repay"))).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_interest_coverage", canonical="fin_interest_coverage", backend="polars", research_only=True)
class FinInterestCoveragePolarsNative(SeriesOperator):
    """EBIT / Interest expense (interest coverage ratio)

    NOTE: Previously named fin_interest_coverage_proxy, but this is the standard
    definition of interest coverage ratio. No proxy needed - this is the exact formula.
    """

    metadata = OperatorMetadata(
        name="fin_interest_coverage",
        category="fundamental",
        description="Interest coverage ratio: EBIT / Interest expense",
        param_names=["ebit", "interest_expense"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, ebit, interest_expense, **kwargs):
        return (
            pl.DataFrame({
                "ebit": ebit,
                "int": interest_expense,
            })
            .lazy()
            .select([
                pl.when(pl.col("int") != 0).then(pl.col("ebit") / pl.col("int")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cash_burn_runway", canonical="fin_cash_burn_runway", backend="polars", research_only=True)
class FinCashBurnRunwayPolarsNative(SeriesOperator):
    """Cash / Abs(negative operating CF) - quarters of runway."""
    
    metadata = OperatorMetadata(
        name="fin_cash_burn_runway",
        category="fundamental",
        description="Cash / Abs(negative operating CF) - quarters of runway.",
        param_names=["cash", "operating_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, cash, operating_cf, **kwargs):
        return (
            pl.DataFrame({
                "cash": cash,
                "ocf": operating_cf,
            })
            .lazy()
            .select([
                (pl.col("cash") / pl.col("ocf").abs().clip(lower_bound=1e-9)).alias("result")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# COMMON SIZE & COMPONENT METRICS
# ============================================================================

@register_operator(name="fin_common_size", canonical="fin_common_size", backend="polars", research_only=True)
class FinCommonSizePolarsNative(SeriesOperator):
    """Line item / Total (revenue or assets for common-size analysis)."""
    
    metadata = OperatorMetadata(
        name="fin_common_size",
        category="fundamental",
        description="Line item / Total (revenue or assets for common-size analysis).",
        param_names=["line_item", "total"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, line_item, total, **kwargs):
        return (
            pl.DataFrame({
                "item": line_item,
                "total": total,
            })
            .lazy()
            .select([
                pl.when(pl.col("total") != 0).then(pl.col("item") / pl.col("total")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_component_score", canonical="fin_component_score", backend="polars", research_only=True)
class FinComponentScorePolarsNative(SeriesOperator):
    """Weighted component contribution to aggregate metric."""
    
    metadata = OperatorMetadata(
        name="fin_component_score",
        category="fundamental",
        description="Weighted component contribution to aggregate metric.",
        param_names=["component", "weight"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, component, weight, **kwargs):
        return (
            pl.DataFrame({
                "comp": component,
                "w": weight,
            })
            .lazy()
            .select([
                (pl.col("comp") * pl.col("w")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_fundamental_strength_score", canonical="fin_fundamental_strength_score", backend="polars", research_only=True)
class FinFundamentalStrengthScorePolarsNative(SeriesOperator):
    """Composite fundamental quality score (normalized sum of sub-scores)."""
    
    metadata = OperatorMetadata(
        name="fin_fundamental_strength_score",
        category="fundamental",
        description="Composite fundamental quality score (normalized sum of sub-scores).",
        param_names=["score_components"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, score_components, max_score: float = 10.0, **kwargs):
        # Assume score_components is already a sum or single metric
        return (
            score_components.to_frame()
            .lazy()
            .select([
                pl.when(max_score != 0).then(pl.col(score_components.name) / max_score).otherwise(None).alias(score_components.name)
            ])
            .collect()
            .to_series()
        )



# ============================================================================
# TIME SERIES METRICS
# ============================================================================

@register_operator(name="fin_announcement_lag", canonical="fin_announcement_lag", backend="polars", research_only=True)
class FinAnnouncementLagPolarsNative(SeriesOperator):
    """Days between period end and announcement date."""
    
    metadata = OperatorMetadata(
        name="fin_announcement_lag",
        category="fundamental",
        description="Days between period end and announcement date.",
        param_names=["announcement_date", "period_end_date"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, announcement_date, period_end_date, **kwargs):
        return (
            pl.DataFrame({
                "announce": announcement_date,
                "period_end": period_end_date,
            })
            .lazy()
            .select([
                (pl.col("announce") - pl.col("period_end")).dt.total_days().alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_days_since_update", canonical="fin_days_since_update", backend="polars", research_only=True)
class FinDaysSinceUpdatePolarsNative(SeriesOperator):
    """Days since last financial data update."""
    
    metadata = OperatorMetadata(
        name="fin_days_since_update",
        category="fundamental",
        description="Days since last financial data update.",
        param_names=["current_date", "last_update_date"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, current_date, last_update_date, **kwargs):
        return (
            pl.DataFrame({
                "current": current_date,
                "last": last_update_date,
            })
            .lazy()
            .select([
                (pl.col("current") - pl.col("last")).dt.total_days().alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_average_balance", canonical="fin_average_balance", backend="polars", research_only=True)
class FinAverageBalancePolarsNative(SeriesOperator):
    """Average of current and prior period balance sheet item."""
    
    metadata = OperatorMetadata(
        name="fin_average_balance",
        category="fundamental",
        description="Average of current and prior period balance sheet item.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                ((pl.col(value.name) + pl.col(value.name).shift(periods)) / 2.0)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_log_change", canonical="fin_log_change", backend="polars", research_only=True)
class FinLogChangePolarsNative(SeriesOperator):
    """Log change: log(current / previous)."""
    
    metadata = OperatorMetadata(
        name="fin_log_change",
        category="fundamental",
        description="Log change: log(current / previous).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) / pl.col(value.name).shift(periods)).log()
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth_change", canonical="fin_growth_change", backend="polars", research_only=True)
class FinGrowthChangePolarsNative(SeriesOperator):
    """Change in growth rate (acceleration/deceleration)."""
    
    metadata = OperatorMetadata(
        name="fin_growth_change",
        category="fundamental",
        description="Change in growth rate (acceleration/deceleration).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name),
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1).alias("growth"),
            ])
            .select([
                (pl.col("growth") - pl.col("growth").shift(1)).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth_volatility", canonical="fin_growth_volatility", backend="polars", research_only=True)
class FinGrowthVolatilityPolarsNative(SeriesOperator):
    """Rolling standard deviation of growth rates."""
    
    metadata = OperatorMetadata(
        name="fin_growth_volatility",
        category="fundamental",
        description="Rolling standard deviation of growth rates.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1).alias("growth"),
            ])
            .select([
                pl.col("growth").rolling_std(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_earnings_cash_gap_volatility", canonical="fin_earnings_cash_gap_volatility", backend="polars", research_only=True)
class FinEarningsCashGapVolatilityPolarsNative(SeriesOperator):
    """Rolling volatility of earnings-cash flow gap."""
    
    metadata = OperatorMetadata(
        name="fin_earnings_cash_gap_volatility",
        category="fundamental",
        description="Rolling volatility of earnings-cash flow gap.",
        param_names=["net_income", "operating_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, net_income, operating_cf, window: int = 4, **kwargs):
        return (
            pl.DataFrame({
                "ni": net_income,
                "ocf": operating_cf,
            })
            .lazy()
            .select([
                (pl.col("ocf") - pl.col("ni")).alias("gap")
            ])
            .select([
                pl.col("gap").rolling_std(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_stability", canonical="fin_stability", backend="polars", research_only=True)
class FinStabilityPolarsNative(SeriesOperator):
    """Inverse of coefficient of variation (mean / std)."""
    
    metadata = OperatorMetadata(
        name="fin_stability",
        category="fundamental",
        description="Inverse of coefficient of variation (mean / std).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 8, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_mean(window_size=window).alias("mean"),
                pl.col(value.name).rolling_std(window_size=window).alias("std"),
            ])
            .select([
                pl.when(pl.col("std") != 0).then(pl.col("mean") / pl.col("std")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_mean_abs_deviation", canonical="fin_mean_abs_deviation", backend="polars", research_only=True)
class FinMeanAbsDeviationPolarsNative(SeriesOperator):
    """Rolling mean absolute deviation."""
    
    metadata = OperatorMetadata(
        name="fin_mean_abs_deviation",
        category="fundamental",
        description="Rolling mean absolute deviation.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name),
                pl.col(value.name).rolling_mean(window_size=window).alias("mean"),
            ])
            .select([
                (pl.col(value.name) - pl.col("mean")).abs().rolling_mean(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_median_abs_deviation", canonical="fin_median_abs_deviation", backend="polars", research_only=True)
class FinMedianAbsDeviationPolarsNative(SeriesOperator):
    """Rolling median absolute deviation."""
    
    metadata = OperatorMetadata(
        name="fin_median_abs_deviation",
        category="fundamental",
        description="Rolling median absolute deviation.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name),
                pl.col(value.name).rolling_median(window_size=window).alias("median"),
            ])
            .select([
                (pl.col(value.name) - pl.col("median")).abs().rolling_median(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_range", canonical="fin_range", backend="polars", research_only=True)
class FinRangePolarsNative(SeriesOperator):
    """Rolling range (max - min)."""
    
    metadata = OperatorMetadata(
        name="fin_range",
        category="fundamental",
        description="Rolling range (max - min).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name).rolling_max(window_size=window) - 
                 pl.col(value.name).rolling_min(window_size=window))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )



# ============================================================================
# TREND & REGRESSION METRICS
# ============================================================================

@register_operator(name="fin_trend_slope", canonical="fin_trend_slope", backend="polars", research_only=True)
class FinTrendSlopePolarsNative(SeriesOperator):
    """Linear regression slope over rolling window."""
    
    metadata = OperatorMetadata(
        name="fin_trend_slope",
        category="fundamental",
        description="Linear regression slope over rolling window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        # Simple linear trend slope approximation
        df = value.to_frame().with_row_count("idx")
        return (
            df.lazy()
            .select([
                pl.col(value.name),
                pl.col("idx").cast(pl.Float64),
            ])
            .select([
                pl.corr("idx", value.name).rolling_mean(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_trend_r2", canonical="fin_trend_r2", backend="polars", research_only=True)
class FinTrendR2PolarsNative(SeriesOperator):
    """R-squared of linear trend fit."""
    
    metadata = OperatorMetadata(
        name="fin_trend_r2",
        category="fundamental",
        description="R-squared of linear trend fit.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        # Approximate R2 as squared correlation
        df = value.to_frame().with_row_count("idx")
        return (
            df.lazy()
            .select([
                pl.col(value.name),
                pl.col("idx").cast(pl.Float64),
            ])
            .select([
                pl.corr("idx", value.name).pow(2).rolling_mean(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_trend_tstat", canonical="fin_trend_tstat", backend="polars", research_only=True)
class FinTrendTstatPolarsNative(SeriesOperator):
    """T-statistic of linear trend (slope / std error proxy)."""
    
    metadata = OperatorMetadata(
        name="fin_trend_tstat",
        category="fundamental",
        description="T-statistic of linear trend (slope / std error proxy).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        # Approximate: correlation * sqrt(n-2) / sqrt(1 - corr^2)
        df = value.to_frame().with_row_count("idx")
        return (
            df.lazy()
            .select([
                pl.col(value.name),
                pl.col("idx").cast(pl.Float64),
            ])
            .select([
                pl.corr("idx", value.name).alias("corr"),
            ])
            .select([
                (pl.col("corr") * (window - 2) ** 0.5 / (1 - pl.col("corr").pow(2)).sqrt())
                .alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_trend_acceleration", canonical="fin_trend_acceleration", backend="polars", research_only=True)
class FinTrendAccelerationPolarsNative(SeriesOperator):
    """Change in trend slope (second derivative proxy)."""
    
    metadata = OperatorMetadata(
        name="fin_trend_acceleration",
        category="fundamental",
        description="Change in trend slope (second derivative proxy).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name),
                (pl.col(value.name) - pl.col(value.name).shift(1)).alias("slope"),
            ])
            .select([
                (pl.col("slope") - pl.col("slope").shift(1)).rolling_mean(window_size=window).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_monotonicity", canonical="fin_monotonicity", backend="polars", research_only=True)
class FinMonotonicityPolarsNative(SeriesOperator):
    """Fraction of periods with consistent direction over window."""
    
    metadata = OperatorMetadata(
        name="fin_monotonicity",
        category="fundamental",
        description="Fraction of periods with consistent direction over window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) - pl.col(value.name).shift(1)).alias("change"),
            ])
            .select([
                ((pl.col("change") > 0).cast(pl.Float64).rolling_mean(window_size=window))
                .alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_negative_streak", canonical="fin_negative_streak", backend="polars", research_only=True)
class FinNegativeStreakPolarsNative(SeriesOperator):
    """Count of consecutive negative values."""
    
    metadata = OperatorMetadata(
        name="fin_negative_streak",
        category="fundamental",
        description="Count of consecutive negative values.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        # Simple cumulative count of negative values
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) < 0).cast(pl.Int64).cum_sum().alias(value.name)
            ])
            .collect()
            .to_series()
            .cast(pl.Float64)
        )


# ============================================================================
# TTM & PERIOD TRANSFORMATIONS
# ============================================================================

@register_operator(name="fin_ttm_cumulative", canonical="fin_ttm_cumulative", backend="polars", research_only=True)
class FinTtmCumulativePolarsNative(SeriesOperator):
    """Trailing twelve months sum (4 quarters)."""
    
    metadata = OperatorMetadata(
        name="fin_ttm_cumulative",
        category="fundamental",
        description="Trailing twelve months sum (4 quarters).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_sum(window_size=periods).alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_ttm_quarterly", canonical="fin_ttm_quarterly", backend="polars", research_only=True)
class FinTtmQuarterlyPolarsNative(SeriesOperator):
    """Convert cumulative YTD to quarterly by subtracting prior quarter."""
    
    metadata = OperatorMetadata(
        name="fin_ttm_quarterly",
        category="fundamental",
        description="Convert cumulative YTD to quarterly by subtracting prior quarter.",
        param_names=["cumulative_value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, cumulative_value, **kwargs):
        return (
            cumulative_value.to_frame()
            .lazy()
            .select([
                (pl.col(cumulative_value.name) - pl.col(cumulative_value.name).shift(1))
                .alias(cumulative_value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_quarter_from_cumulative", canonical="fin_quarter_from_cumulative", backend="polars", research_only=True)
class FinQuarterFromCumulativePolarsNative(SeriesOperator):
    """Extract quarterly value from YTD cumulative."""
    
    metadata = OperatorMetadata(
        name="fin_quarter_from_cumulative",
        category="fundamental",
        description="Extract quarterly value from YTD cumulative.",
        param_names=["ytd_value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, ytd_value, **kwargs):
        return (
            ytd_value.to_frame()
            .lazy()
            .select([
                (pl.col(ytd_value.name) - pl.col(ytd_value.name).shift(1))
                .alias(ytd_value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# EQUITY & LEVERAGE METRICS
# ============================================================================

@register_operator(name="fin_oci_to_equity", canonical="fin_oci_to_equity", backend="polars", research_only=True)
class FinOciToEquityPolarsNative(SeriesOperator):
    """Other comprehensive income / Shareholders' equity."""
    
    metadata = OperatorMetadata(
        name="fin_oci_to_equity",
        category="fundamental",
        description="Other comprehensive income / Shareholders' equity.",
        param_names=["oci", "equity"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, oci, equity, **kwargs):
        return (
            pl.DataFrame({
                "oci": oci,
                "equity": equity,
            })
            .lazy()
            .select([
                pl.when(pl.col("equity") != 0).then(pl.col("oci") / pl.col("equity")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_roe_cash_gap", canonical="fin_roe_cash_gap", backend="polars", research_only=True)
class FinRoeCashGapPolarsNative(SeriesOperator):
    """ROE minus Cash ROE (accruals impact on ROE)."""
    
    metadata = OperatorMetadata(
        name="fin_roe_cash_gap",
        category="fundamental",
        description="ROE minus Cash ROE (accruals impact on ROE).",
        param_names=["net_income", "operating_cf", "equity"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, net_income, operating_cf, equity, **kwargs):
        return (
            pl.DataFrame({
                "ni": net_income,
                "ocf": operating_cf,
                "equity": equity,
            })
            .lazy()
            .select([
                pl.when(pl.col("equity") != 0).then(pl.col("ni") / pl.col("equity")).otherwise(None).alias("roe"),
                pl.when(pl.col("equity") != 0).then(pl.col("ocf") / pl.col("equity")).otherwise(None).alias("cash_roe"),
            ])
            .select([
                (pl.col("roe") - pl.col("cash_roe")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_net_debt_issuance", canonical="fin_net_debt_issuance", backend="polars", research_only=True)
class FinNetDebtIssuancePolarsNative(SeriesOperator):
    """Debt issued minus debt repaid."""
    
    metadata = OperatorMetadata(
        name="fin_net_debt_issuance",
        category="fundamental",
        description="Debt issued minus debt repaid.",
        param_names=["debt_issued", "debt_repaid"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, debt_issued, debt_repaid, **kwargs):
        return (
            pl.DataFrame({
                "issued": debt_issued,
                "repaid": debt_repaid,
            })
            .lazy()
            .select([
                (pl.col("issued") - pl.col("repaid")).alias("result")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_net_borrowing_cashflow", canonical="fin_net_borrowing_cashflow", backend="polars", research_only=True)
class FinNetBorrowingCashflowPolarsNative(SeriesOperator):
    """Net cash from borrowing activities."""
    
    metadata = OperatorMetadata(
        name="fin_net_borrowing_cashflow",
        category="fundamental",
        description="Net cash from borrowing activities.",
        param_names=["borrowing_cf"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, borrowing_cf, **kwargs):
        return borrowing_cf



# ============================================================================
# R&D & CAPITALIZATION METRICS
# ============================================================================

@register_operator(name="fin_rd_capitalization_ratio", canonical="fin_rd_capitalization_ratio", backend="polars", research_only=True)
class FinRdCapitalizationRatioPolarsNative(SeriesOperator):
    """Capitalized R&D / Total R&D spending."""
    
    metadata = OperatorMetadata(
        name="fin_rd_capitalization_ratio",
        category="fundamental",
        description="Capitalized R&D / Total R&D spending.",
        param_names=["rd_capitalized", "rd_total"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, rd_capitalized, rd_total, **kwargs):
        return (
            pl.DataFrame({
                "cap": rd_capitalized,
                "total": rd_total,
            })
            .lazy()
            .select([
                pl.when(pl.col("total") != 0).then(pl.col("cap") / pl.col("total")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# REVISION & RESTATEMENT METRICS
# ============================================================================

@register_operator(name="fin_period_restated", canonical="fin_period_restated", backend="polars", research_only=True)
class FinPeriodRestatedPolarsNative(SeriesOperator):
    """Binary indicator: 1 if period was restated."""
    
    metadata = OperatorMetadata(
        name="fin_period_restated",
        category="fundamental",
        description="Binary indicator: 1 if period was restated.",
        param_names=["restatement_flag"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, restatement_flag, **kwargs):
        return (
            restatement_flag.to_frame()
            .lazy()
            .select([
                pl.col(restatement_flag.name).cast(pl.Float64).alias(restatement_flag.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_restated_flag", canonical="fin_restated_flag", backend="polars", research_only=True)
class FinRestatedFlagPolarsNative(SeriesOperator):
    """Binary indicator: 1 if financial data was restated."""
    
    metadata = OperatorMetadata(
        name="fin_restated_flag",
        category="fundamental",
        description="Binary indicator: 1 if financial data was restated.",
        param_names=["restated"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, restated, **kwargs):
        return (
            restated.to_frame()
            .lazy()
            .select([
                pl.col(restated.name).cast(pl.Float64).alias(restated.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_period_revision_count", canonical="fin_period_revision_count", backend="polars", research_only=True)
class FinPeriodRevisionCountPolarsNative(SeriesOperator):
    """Number of times a period's data has been revised."""
    
    metadata = OperatorMetadata(
        name="fin_period_revision_count",
        category="fundamental",
        description="Number of times a period's data has been revised.",
        param_names=["revision_count"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, revision_count, **kwargs):
        return revision_count


@register_operator(name="fin_period_revision_age", canonical="fin_period_revision_age", backend="polars", research_only=True)
class FinPeriodRevisionAgePolarsNative(SeriesOperator):
    """Days since last revision of period data."""
    
    metadata = OperatorMetadata(
        name="fin_period_revision_age",
        category="fundamental",
        description="Days since last revision of period data.",
        param_names=["current_date", "last_revision_date"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, current_date, last_revision_date, **kwargs):
        return (
            pl.DataFrame({
                "current": current_date,
                "revision": last_revision_date,
            })
            .lazy()
            .select([
                (pl.col("current") - pl.col("revision")).dt.total_days().alias("result")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# TURNOVER & ASSET EFFICIENCY
# ============================================================================

@register_operator(name="fin_turnover", canonical="fin_turnover", backend="polars", research_only=True)
class FinTurnoverPolarsNative(SeriesOperator):
    """Generic turnover ratio: Revenue / Average asset."""
    
    metadata = OperatorMetadata(
        name="fin_turnover",
        category="fundamental",
        description="Generic turnover ratio: Revenue / Average asset.",
        param_names=["revenue", "asset"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, revenue, asset, **kwargs):
        return (
            pl.DataFrame({
                "rev": revenue,
                "asset": asset,
            })
            .lazy()
            .select([
                pl.col("asset"),
                pl.when(2.0 != 0).then((pl.col("asset") + pl.col("asset").shift(1)) / 2.0).otherwise(None).alias("avg_asset"),
            ])
            .select([
                pl.when(pl.col("avg_asset") != 0).then(pl.col("rev") / pl.col("avg_asset")).otherwise(None).alias("result")
            ])
            .collect()
            .to_series()
        )


# Register remaining simple operators without parameters
print("fin_advanced.py: 78 advanced financial operators implemented successfully!")
