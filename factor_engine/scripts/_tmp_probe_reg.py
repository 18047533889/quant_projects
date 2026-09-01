from factor_engine.cleaned_operators import load_all
load_all()
from factor_engine.cleaned_operators.registry import OperatorRegistry
for name in ("ts_ewm_mean", "ts_ema", "ts_ewm_corr", "ts_ewm_cov", "ewm_corr", "ewm_cov", "ts_sma", "sma", "ts_delay", "lag", "rank"):
    try:
        op = OperatorRegistry.get(name)
    except Exception as e:
        op = f"ERR {e}"
    print(name, "->", None if op is None else getattr(op, "name", op))
print("module loaded:", OperatorRegistry.get("ts_ewm_corr") is not None)
