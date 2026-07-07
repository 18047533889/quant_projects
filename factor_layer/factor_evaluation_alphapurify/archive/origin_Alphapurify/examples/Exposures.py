# ============================================================
# Portfolio & Pure Exposure Analysis Example
# ============================================================
# This example demonstrates how to use:
#   1. PortfolioExposures  → quantile portfolio attribution
#   2. PureExposures       → factor-weighted pure exposure analysis
#
# The goal is to understand:
#   - Where your returns come from
#   - Whether your alpha is "real alpha"
#   - Or just exposure to known risk factors
# ============================================================


# ============================================================
# 1. You need to have a Dataframe
# ============================================================
# Load your Dataframe:
#
#           datetime       code   close    open     alpha  momentum_12_1    vol_60  beta_252
# 252     2011-01-20  000001.SZ  535.72  555.82 -0.185521      -0.248901  0.290062  0.703001
# 253     2011-01-21  000001.SZ  543.98  533.92 -0.413315      -0.264591  0.291075  0.705560
# 254     2011-01-24  000001.SZ  540.39  543.98 -0.054991      -0.261379  0.289297  0.704529
# 255     2011-01-25  000001.SZ  544.69  542.54  0.117437      -0.251519  0.289174  0.702129
# 256     2011-01-26  000001.SZ  547.93  543.98  0.025941      -0.269864  0.289312  0.706504
# ...            ...        ...     ...     ...       ...            ...       ...       ...
# 2897407 2024-12-25  688981.SH   97.99   96.48 -0.749708       0.691141  0.982570  1.066201
# 2897408 2024-12-26  688981.SH   96.73   98.00 -0.624982       0.688120  0.983486  1.065321
# 2897409 2024-12-27  688981.SH   97.51   96.78 -0.572219       0.694365  0.974394  1.065805
# 2897410 2024-12-30  688981.SH   99.29   96.60 -0.465605       0.767870  0.893125  1.064373
# 2897411 2024-12-31  688981.SH   94.62   99.00 -0.463487       0.707974  0.807936  1.074279 
#
# Your DataFrame must include a time column ("datetime"), an asset identifier column ("symbol"), 
# and your factor columns ("alpha", "momentum_12_1", "vol_60", "beta_252") to ensure proper usage.
#
# ============================================================


# ============================================================
# 3. Portfolio Exposure Analysis (Quantile Portfolio)
# ============================================================
# Construct long-short portfolio based on factor ranking
# and analyze exposure attribution
#
# >>> Initialize
# pe = PortfolioExposures(
#     base_df=df,
#     trade_date_col="datetime",
#     symbol_col="symbol",
#     price_col="close",
#     factor_name="alpha",
#     exposure_cols=["beta", "volatility"],
#     bins=5,
#     position="ls"   # long-short
# )

# >>> Run analysis
# pe.run()

# >>> Plot results
# pe.plot_portfolio_exposures()
# pe.plot_portfolio_returns()
# pe.plot_portfolio_exposures_and_returns()


# ============================================================
# 4. Pure Exposure Analysis (Factor-Weighted Portfolio)
# ============================================================
# Instead of quantile portfolios, we build a continuous
# factor-weighted portfolio:
#
#   weight_i = factor_i / sum(|factor_i|)
#
# This allows us to measure "true" exposure of the signal
#
# >>> Initialize
# pure = PureExposures(
#     base_df=df,
#     trade_date_col="datetime",
#     symbol_col="symbol",
#     price_col="close",
#     factor_name="alpha",
#     exposure_cols=["beta", "volatility"]
# )

# >>> Run analysis
# pure.run()

# >>> Plot results
# pure.plot_pure_exposures()
# pure.plot_pure_returns()
# pure.plot_pure_exposures_and_returns()

# >>> Correlation diagnostics
# pure.plot_correlations()


# ============================================================
# Summary
# ============================================================
# PortfolioExposures:
#   → "What does my portfolio behave like?"
#
# PureExposures:
#   → "What does my signal inherently load on?"
#
# Together:
#   → Full understanding of alpha vs risk exposures
# ============================================================

import pandas as pd
import numpy as np
from alphapurify.Exposures import PureExposures, PortfolioExposures

def df():
    np.random.seed(42)
    n_stocks = 100
    start_date = "2024-01-01"
    end_date = "2025-12-31"

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    symbols = [f"stock_{i}" for i in range(1, n_stocks + 1)]
    dfs = []

    for sym in symbols:
        n = len(dates)
        drift = np.random.uniform(0.0001, 0.0005)
        vol = np.random.uniform(0.01, 0.03)

        eps = np.random.randn(n)
        returns = drift + vol * eps

        price = 100 * np.exp(np.cumsum(returns))

        df_temp = pd.DataFrame({
            "datetime": dates,
            "symbol": sym,
            "close": price,
            "ret": returns
        })

        dfs.append(df_temp)

    df = pd.concat(dfs).sort_values(["datetime", "symbol"]).reset_index(drop=True)

    df["future_ret"] = df.groupby("symbol")["ret"].shift(-1)
    noise = np.random.randn(len(df)) * 0.02
    df["factor"] = 0.2 * df["future_ret"] + noise

    stock_beta = {
        sym: np.random.uniform(0.5, 1.5) for sym in df["symbol"].unique()
    }
    df["beta"] = df["symbol"].map(stock_beta) + np.random.randn(len(df)) * 0.05

    df["volatility"] = (
        df.groupby("symbol")["ret"]
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    df = df.drop(columns=["ret", "future_ret"])

    return df

pe = PortfolioExposures(
    base_df=df(),
    trade_date_col="datetime",
    symbol_col="symbol",
    price_col="close",
    factor_name="factor",
    exposure_cols=["beta", "volatility"],
    bins=5,
    position="ls"   
)

pe.run()
pe.plot_portfolio_exposures()
pe.plot_portfolio_returns()
pe.plot_portfolio_exposures_and_returns()

pure = PureExposures(
    base_df=df(),
    trade_date_col="datetime",
    symbol_col="symbol",
    price_col="close",
    factor_name="factor",
    exposure_cols=["beta", "volatility"]
)

# >>> Run analysis
pure.run()

# >>> Plot results
pure.plot_pure_exposures()
pure.plot_pure_returns()
pure.plot_pure_exposures_and_returns()

# >>> Correlation diagnostics
pure.plot_correlations()


