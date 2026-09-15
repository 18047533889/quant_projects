"""Side-effect-free contracts and native cross-sectional winsorization kernels."""
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar

def scalar_contract(name):
    """Six authored public signatures, shared by their actual backend twins."""
    numeric = ParamRole.NUMERICAL
    resolution = ParamRole.ESTIMATOR_RESOLUTION
    contracts = {
        "constant": ((), {"c": ParamSpec(dtype=float, default=0.0, param_role=numeric)}),
        "lerp": (("a", "b"), {"fraction": ParamSpec(dtype=float, default=0.5, param_role=numeric)}),
        "round": (("x",), {"decimals": ParamSpec(dtype=int, min=-18, max=18, default=0, param_role=resolution)}),
        "truncate": (("x",), {"decimals": ParamSpec(dtype=int, min=-18, max=18, default=0, param_role=resolution)}),
        "winsorize": (("x",), {
            "lower": ParamSpec(dtype=float, min=0, max=1, default=0.05, param_role=resolution),
            "upper": ParamSpec(dtype=float, min=0, max=1, default=0.95, param_role=resolution)}),
        "winsorize_mean": (("x",), {"trim_pct": ParamSpec(dtype=float, min=0, max=0.49, default=0.1, param_role=resolution)}),
    }
    panels, specs = contracts[name]
    return dict(panel_params=panels, scalar_params=tuple(specs), param_specs=specs)

def quantile_bounds(lower, upper):
    lower = strict_finite_scalar(lower, "lower", minimum=0, maximum=1)
    upper = strict_finite_scalar(upper, "upper", minimum=0, maximum=1)
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return lower, upper

def polars_winsorize(x, lower=0.05, upper=0.95, *, mean=False):
    """Finite-only row quantiles; bounded input-sized native unpivot workspace."""
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    lower, upper = quantile_bounds(lower, upper)
    cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
    if not cols or not x.height:
        return x
    long = (x.select(cols).with_row_index("_r")
            .unpivot(index="_r", on=cols, variable_name="_c", value_name="_v")
            .with_columns(pl.when(pl.col("_v").is_finite()).then(pl.col("_v")).otherwise(None).alias("_v")))
    lo = pl.col("_v").quantile(lower, interpolation="linear").over("_r")
    hi = pl.col("_v").quantile(upper, interpolation="linear").over("_r")
    long = long.with_columns(pl.col("_v").clip(lo, hi).alias("_v"))
    if mean:
        long = long.with_columns(pl.col("_v").mean().over("_r").alias("_v"))
    wide = (long.pivot(values="_v", index="_r", on="_c", aggregate_function="first")
            .sort("_r").select(cols))
    return x.with_columns([wide[c] for c in cols])
