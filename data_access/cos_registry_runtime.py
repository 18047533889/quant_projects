# -*- coding: utf-8 -*-
"""Verified runtime corrections for physical COS registry metadata.

The standalone repository preserves the broad legacy datasets.yaml for backward
compatibility. These deterministic replacements correct only parquet-verified
axes and schemas before semantic reads execute. The patch fingerprint is bound
to DataSnapshot identity together with the executable COS contract fingerprint.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from .cos_contract import semantic_contract_fingerprint


_REGISTRY_PATCHES: dict[str, dict[str, Any]] = {
    "ashare_stock_daily": {
        "schema_update": {
            "HighLimit": "double",
            "LowLimit": "double",
            "Factor": "double",
            "IsSuspend": "bool",
            # #P1-final closure 11：实际 parquet 为 timestamp[ms, tz=UTC]（aware）。
            # 声明 "timestamp" 会触发 schema 校验 warning（timestamp vs
            # TIMESTAMP WITH TIME ZONE）；与 datasets.yaml 的 timestamptz 对齐。
            "UpdateTime": "timestamptz",
        }
    },
    "ashare_stock_balance": {
        "time_column": "PubDate",
        # Authoritative 111-field E1 payload from COS_ashare_lqtp_data_dictionary.md.
        "schema_replace": {"TradeDate": "date", "Symbol": "string", "PubDate": "date", "ReportPeriodEndDate": "date", "CashEquivalents": "double", "SettlementProvi": "double", "LendCapital": "double", "TradingAssets": "double", "BillReceivable": "double", "AccountReceivable": "double", "AdvancePayment": "double", "InsuranceReceivables": "double", "ReinsuranceReceivables": "double", "ReinsuranceContractReservesReceivable": "double", "InterestReceivable": "double", "DividendReceivable": "double", "OtherReceivable": "double", "BoughtSellbackAssets": "double", "Inventories": "double", "NonCurrentAssetInOneYear": "double", "OtherCurrentAssets": "double", "TotalCurrentAssets": "double", "LoanAndAdvance": "double", "HoldForSaleAssets": "double", "HoldToMaturityInvestments": "double", "LongtermReceivableAccount": "double", "LongtermEquityInvest": "double", "InvestmentProperty": "double", "FixedAssets": "double", "ConstruInProcess": "double", "ConstructionMaterials": "double", "FixedAssetsLiquidation": "double", "BiologicalAssets": "double", "OilGasAssets": "double", "IntangibleAssets": "double", "DevelopmentExpenditure": "double", "GoodWill": "double", "LongDeferredExpense": "double", "DeferredTaxAssets": "double", "OtherNonCurrentAssets": "double", "TotalNonCurrentAssets": "double", "TotalAssets": "double", "ShorttermLoan": "double", "BorrowingFromCentralbank": "double", "DepositInInterbank": "double", "BorrowingCapital": "double", "TradingLiability": "double", "NotesPayable": "double", "AccountsPayable": "double", "AdvancePeceipts": "double", "SoldBuybackSecuProceeds": "double", "CommissionPayable": "double", "SalariesPayable": "double", "TaxsPayable": "double", "InterestPayable": "double", "DividendPayable": "double", "OtherPayable": "double", "ReinsurancePayables": "double", "InsuranceContractReserves": "double", "ProxySecuProceeds": "double", "ReceivingsFromVicariouslySoldSecurities": "double", "NonCurrentLiabilityInOneYear": "double", "OtherCurrentLiability": "double", "TotalCurrentLiability": "double", "LongtermLoan": "double", "BondsPayable": "double", "LongtermAccountPayable": "double", "SpecificAccountPayable": "double", "EstimateLiability": "double", "DeferredTaxLiability": "double", "OtherNonCurrentLiability": "double", "TotalNonCurrentLiability": "double", "TotalLiability": "double", "PaidinCapital": "double", "CapitalReserveFund": "double", "TreasuryStock": "double", "SpecificReserves": "double", "SurplusReserveFund": "double", "OrdinaryRiskReserveFund": "double", "RetainedProfit": "double", "ForeignCurrencyReportConvDiff": "double", "EquitiesParentCompanyOwners": "double", "MinorityInterests": "double", "TotalOwnerEquities": "double", "TotalSheetOwnerEquities": "double", "OtherComprehensiveIncome": "double", "DeferredEarning": "double", "LoanAndAdvanceCurrentAssets": "double", "DerivativeFinancialAsset": "double", "HoldSaleAsset": "double", "LoanAndAdvanceNoncurrentAssets": "double", "DerivativeFinancialLiability": "double", "HoldSaleLiability": "double", "EstimateLiabilityCurrent": "double", "DeferredEarningCurrent": "double", "PreferredSharesNoncurrent": "double", "PepertualLiabilityNoncurrent": "double", "LongtermSalariesPayable": "double", "OtherEquityTools": "double", "PreferredSharesEquity": "double", "PepertualLiabilityEquity": "double", "ReceivableFin": "double", "UsufructAssets": "double", "ContractAssets": "double", "BondInvest": "double", "OtherBondInvest": "double", "OtherEquityToolsInvest": "double", "OtherNonCurrentFinancialAssets": "double", "ContractLiability": "double", "LeaseLiability": "double", "UpdateTime": "timestamp"},
    },
    "ashare_stock_income": {
        "time_column": "PubDate",
        "schema_update": {
            "TradeDate": "date", "Symbol": "string", "PubDate": "date",
            "ReportPeriodEndDate": "date", "UpdateTime": "timestamp",
        },
    },
    "ashare_stock_cashflow": {
        "time_column": "PubDate",
        "schema_update": {
            "TradeDate": "date", "Symbol": "string", "PubDate": "date",
            "ReportPeriodEndDate": "date", "UpdateTime": "timestamp",
        },
    },
    "ashare_stock_indicator": {
        "time_column": "PubDate",
        "schema_update": {
            "TradeDate": "date", "Symbol": "string", "PubDate": "date",
            "ReportPeriodEndDate": "date", "UpdateTime": "timestamp",
        },
    },
    "ashare_stock_dividend": {
        "schema_replace": {
            "TradeDate": "date", "Symbol": "string", "RightRegDate": "date",
            "ExDividendDate": "date", "CashDividend": "double",
            "StockDividend": "double", "StockTransfer": "double",
            "UpdateTime": "timestamp",
        }
    },
    "ashare_stock_capital_daily": {
        "schema_replace": {
            "TradeDate": "date", "Symbol": "string", "ChangeDate": "date",
            "PubDate": "date", "TotalCapital": "double",
            "CirculatingCapital": "double", "UpdateTime": "timestamp",
        }
    },
    "ashare_stock_industry": {
        "schema_replace": {
            "TradeDate": "date", "Symbol": "string", "IndustrySource": "string",
            "IndustryCode": "string", "IndustryName": "string",
            "UpdateTime": "timestamp",
        }
    },
    "ashare_stock_status": {
        "schema_replace": {
            "TradeDate": "date", "Symbol": "string", "CompanyId": "int",
            "PubDate": "date", "ChangeDate": "date", "PublicStatusCode": "int",
            "PublicStatus": "string", "ChangeReason": "string",
            "ChangeTypeCode": "int", "ChangeType": "string",
            "Comments": "string", "UpdateTime": "timestamp",
        }
    },
    "us_stock_balance": {
        "time_column": "filing_date", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
            "total_assets": "double", "total_liabilities": "double",
            "total_equity": "double",
        },
    },
    "us_stock_income": {
        "time_column": "filing_date", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
            "revenue": "double", "gross_profit": "double",
            "operating_income": "double",
            "net_income_loss_attributable_common_shareholders": "double",
        },
    },
    "us_stock_cashflow": {
        "time_column": "filing_date", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
            "net_cash_from_operating_activities": "double",
            "net_cash_from_investing_activities": "double",
            "net_cash_from_financing_activities": "double",
        },
    },
    "us_stock_dividend": {
        "instrument_column": "ticker",
        "schema_replace": {
            "id": "string", "ticker": "string", "record_date": "string",
            "pay_date": "timestamp", "declaration_date": "string",
            "ex_dividend_date": "string", "frequency": "int",
            "cash_amount": "double", "currency": "string",
            "distribution_type": "string",
            "historical_adjustment_factor": "double",
            "split_adjusted_cash_amount": "double", "TradeDate": "timestamp",
        },
    },
    # StockCapitalDaily 双 schema 拆分：{date}.parquet=拆分事件，
    # shares_{date}.parquet=稀疏 PIT 股本。拆分数据集各自 schema，禁止 glob 混读。
    "us_stock_capital_split": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "id": "string", "execution_date": "timestamp",
            "split_from": "double", "split_to": "double", "ticker": "string",
            "adjustment_type": "string", "historical_adjustment_factor": "double",
            "TradeDate": "timestamp",
        },
    },
    "us_stock_capital_shares": {
        "time_column": "TradeDate", "instrument_column": "Ticker",
        "schema_replace": {
            "Ticker": "string", "TradeDate": "timestamp",
            "pit_basic_shares_outstanding": "double",
            "pit_diluted_shares_outstanding": "double",
        },
    },
    "us_stock_indicator": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "TradeDate": "timestamp",
            "price": "double", "average_volume": "double",
            "market_cap": "double", "earnings_per_share": "double",
            "price_to_earnings": "double", "price_to_book": "double",
            "price_to_sales": "double", "price_to_cash_flow": "double",
            "price_to_free_cash_flow": "double", "dividend_yield": "double",
            "return_on_assets": "double", "return_on_equity": "double",
            "debt_to_equity": "double", "current": "double",
            "quick": "double", "cash": "double", "ev_to_sales": "double",
            "ev_to_ebitda": "double", "enterprise_value": "double",
            "free_cash_flow": "double",
        },
    },
    "us_stock_valuation_daily": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "TradeDate": "timestamp",
            "price": "double", "average_volume": "double",
            "market_cap": "double", "earnings_per_share": "double",
            "price_to_earnings": "double", "price_to_book": "double",
            "price_to_sales": "double", "price_to_cash_flow": "double",
            "price_to_free_cash_flow": "double", "dividend_yield": "double",
            "return_on_assets": "double", "return_on_equity": "double",
            "debt_to_equity": "double", "current": "double",
            "quick": "double", "cash": "double", "ev_to_sales": "double",
            "ev_to_ebitda": "double", "enterprise_value": "double",
            "free_cash_flow": "double",
        },
    },
    "us_ticker_shares_snapshot": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "share_class_figi": "string",
            "share_class_shares_outstanding": "double",
            "weighted_shares_outstanding": "double", "TradeDate": "string",
            "source": "string",
        },
    },
    "us_security_master_daily_snap": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "name": "string", "type": "string",
            "market": "string", "locale": "string", "TradeDate": "timestamp",
        },
    },
    "us_fact_news": {
        "time_column": "published_utc", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "published_utc": "timestamp",
            "TradeDate": "timestamp",
        },
    },
}


def registry_patch_fingerprint() -> str:
    text = json.dumps(_REGISTRY_PATCHES, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def patch_registry(registry: Any) -> Any:
    if getattr(registry, "_cos_registry_patched", False):
        return registry
    for name, patch in _REGISTRY_PATCHES.items():
        if name not in registry:
            continue
        ds = registry.get(name)
        changes: dict[str, Any] = {}
        if "time_column" in patch:
            changes["time_column"] = patch["time_column"]
        if "instrument_column" in patch:
            changes["instrument_column"] = patch["instrument_column"]
        if "schema_replace" in patch:
            changes["schema"] = dict(patch["schema_replace"])
        elif "schema_update" in patch:
            schema = dict(getattr(ds, "schema", None) or {})
            schema.update(patch["schema_update"])
            changes["schema"] = schema
        if changes:
            registry._datasets[name] = replace(ds, **changes)
    registry._cos_registry_patched = True
    return registry


def patch_store_registry(store: Any) -> Any:
    if getattr(store, "_cos_store_registry_patched", False):
        return store
    patch_registry(store._registry)
    from data_access.store import _compute_registry_hash

    physical = _compute_registry_hash(store._registry)
    combined = f"{physical}:{semantic_contract_fingerprint()}:{registry_patch_fingerprint()}"
    store._registry_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    if hasattr(store, "_schema_checked"):
        store._schema_checked.clear()
    store._cos_store_registry_patched = True
    return store


__all__ = ["patch_registry", "patch_store_registry", "registry_patch_fingerprint"]
