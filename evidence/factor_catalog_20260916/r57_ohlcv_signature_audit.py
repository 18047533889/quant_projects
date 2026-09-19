"""Audit which technical indicators accept the directory's unified OHLCV signature.

The factors.directory DSL calls every technical indicator as
    Indicator(Open, High, Low, Close, Volume)
Engine operators advertise their own param_names. Where they match the directory
convention the factor runs; where they do not, the call fails at runtime.
"""
from __future__ import annotations

import inspect
import json
import re
import sys

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

# indicators the directory calls with the unified 5-argument OHLCV signature
CONVENTION = {
    "ADX": 32, "RSI": 32, "MACD_hist": 32, "ATR": 32, "OBV": 32,
    "FisherTransform": 32, "KAMA": 32, "WilliamsR": 32,
    "CoppockCurve": 26, "RSX": 26,
    "ADXR": 7, "ALMA": 7, "AROON": 7, "AROON_up": 7, "AROON_down": 7,
    "ATR_WILDER": 7, "BollingerUpper": 7, "BollingerLower": 7, "CCI": 7,
    "DPO": 7, "ElderRay": 7, "HMA": 7, "MACD": 7, "MACD_line": 7,
    "MACD_signal": 7, "MOM": 7, "QQE": 7, "ROC": 7, "RSI_WILDER": 7,
    "StochasticK": 7, "StochasticD": 7, "TRIX": 7, "WMA": 7,
}


def resolve_op(canonical: str):
    for backend in ("pandas_numpy", "polars", "pandas"):
        try:
            op = OperatorRegistry.get(canonical, backend, mode="any")
        except Exception:  # noqa: BLE001
            op = None
        if op is not None:
            return op, backend
    return None, None


def main() -> int:
    load_all()
    registry_aliases = getattr(OperatorRegistry, "_aliases", {}) or {}
    rows = []
    for canonical, uses in sorted(CONVENTION.items(), key=lambda kv: -kv[1]):
        entry = OperatorRegistry._catalog.get(canonical)
        if entry is None:
            rows.append({
                "canonical": canonical, "uses": uses, "registered": False,
                "param_names": None, "signature": None, "status": None,
                "backend": None,
            })
            continue
        op, backend = resolve_op(canonical)
        md = getattr(op, "metadata", None)
        param_names = list(getattr(md, "param_names", None) or [])
        sig = None
        fn = getattr(op, "_calculate_series", None)
        if fn is not None:
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = None
        rows.append({
            "canonical": canonical, "uses": uses, "registered": True,
            "param_names": param_names, "signature": sig,
            "status": entry.get("status"), "backend": backend,
        })

    payload = {"rows": rows}
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
