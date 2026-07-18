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
            "UpdateTime": "timestamp",
        }
    },
    "ashare_stock_balance": {
        "time_column": "PubDate",
        "schema_update": {
            "TradeDate": "date", "Symbol": "string", "PubDate": "date",
            "ReportPeriodEndDate": "date", "UpdateTime": "timestamp",
        },
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
            "ticker": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
        },
    },
    "us_stock_income": {
        "time_column": "filing_date", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
        },
    },
    "us_stock_cashflow": {
        "time_column": "filing_date", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "period_end": "string",
            "filing_date": "timestamp", "fiscal_quarter": "int",
            "fiscal_year": "int", "timeframe": "string",
        },
    },
    "us_stock_dividend": {
        "instrument_column": "ticker",
        "schema_replace": {
            "id": "string", "ticker": "string", "record_date": "string",
            "pay_date": "timestamp", "ex_dividend_date": "string",
            "frequency": "int", "cash_amount": "double", "currency": "string",
            "distribution_type": "string", "TradeDate": "timestamp",
        },
    },
    "us_stock_capital_daily": {
        "instrument_column": "ticker",
        "schema_replace": {
            "id": "string", "execution_date": "timestamp",
            "split_from": "double", "split_to": "double", "ticker": "string",
            "adjustment_type": "string", "historical_adjustment_factor": "double",
            "TradeDate": "timestamp",
        },
    },
    "us_stock_indicator": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "TradeDate": "timestamp",
            "price": "double", "market_cap": "double",
            "price_to_earnings": "double",
        },
    },
    "us_stock_valuation_daily": {
        "time_column": "TradeDate", "instrument_column": "ticker",
        "schema_replace": {
            "ticker": "string", "cik": "string", "TradeDate": "timestamp",
            "price": "double", "market_cap": "double",
            "price_to_earnings": "double",
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
