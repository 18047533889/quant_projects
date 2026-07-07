# ============================================================
#  Class AlphaPurifier Usage Example
# ============================================================
# This section demonstrates how to use AlphaPurifier
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
# 2. Explore Available Methods and Parameters
# ============================================================
# AlphaPurifier provides a built-in method to explore all supported
# preprocessing operations and their configurations.
#
# You can use `get_methods()` to:
#   - View all main method categories
#   - Inspect methods under a specific category
#   - Check detailed parameters for a specific method


# ------------------------------------------------------------
# 2.1 View all method categories
# ------------------------------------------------------------
# AlphaPurifier.get_methods()
#
# Output:
#
# Available main categories of methods:
#
#  - winsorize (12 implementations)
#  - neutralize (15 implementations)
#  - standardize (15 implementations)


# ------------------------------------------------------------
# 2.2 View methods under a specific category
# ------------------------------------------------------------
# AlphaPurifier.get_methods("neutralize")
#
# Output:
#
# Available methods under [neutralize]:
#
#  - multiOLS: Perform cross-sectional multi-factor OLS neutralization on a factor column.
#  - lasso: Perform cross-sectional factor neutralization using Lasso regression.
#  - ridge: Perform cross-sectional factor neutralization using Ridge regression.
#  - elasticnet: Perform cross-sectional factor neutralization using Elastic Net regression.
#  - polynomial: Perform cross-sectional factor neutralization using Polynomial Regression.
#  - kernelridge: Perform cross-sectional factor neutralization using Kernel Ridge Regression (KRR).
#  - huber: Perform robust cross-sectional factor neutralization using Huber Regression.
#  ...


# ------------------------------------------------------------
# 2.3 View detailed parameters of a specific method
# ------------------------------------------------------------
# AlphaPurifier.get_methods("neutralize", "polynomial")
#
# Output:
#
# Method: polynomial
#
# Description:
# Perform cross-sectional factor neutralization using Polynomial Regression.
#
# Parameters:
#
#  - neutralizer_cols : list of str
#      Continuous control variables used in the regression.
#
#  - dummy_cols : list of str, optional
#      Categorical columns used to generate dummy variables.
#
#  - scale_X : bool, default False
#      Whether to standardize continuous neutralizer variables before regression.
#
#  - degree : int, default 2
#      Degree of polynomial expansion.
#      (1 = linear, 2 = quadratic + interactions, 3 = cubic, etc.)
#
#  - interaction_only : bool, default False
#      If True, only interaction terms are generated (no squared terms).
#
#  - include_bias : bool, default True
#      Whether to include a constant (bias) term in the model.
#
# ------------------------------------------------------------
# This design allows users to easily explore and understand
# all available preprocessing tools before applying them.
# ------------------------------------------------------------


# ============================================================
# 3. Initialize AlphaPurifier and Apply Methods
# ============================================================
# In this step, we initialize AlphaPurifier and apply multiple
# preprocessing methods using method chaining.
#
#
# ------------------------------------------------------------
# 3.1 Initialize AlphaPurifier
# ------------------------------------------------------------
# >>> AP = AlphaPurifier(
# ...     base_df=df,
# ...     factor_name='alpha',
# ...     trade_date_col='datetime',
# ...     symbol_col='symbol'
# ... )
#
#
# ------------------------------------------------------------
# 3.2 Apply preprocessing methods (method chaining)
# ------------------------------------------------------------
# >>> df = (
# ...     AP
# ...     .winsorize('mad')
# ...     .neutralize(
# ...         'multiOLS',
# ...         neutralizer_cols=['ret_20', 'beta_60', 'liq_20'],
# ...         dummy_cols=['industry']
# ...     )
# ...     .standardize('zscore')
# ...     .to_result()
# ... )
# ============================================================
from alphapurify import AlphaPurifier
import pandas as pd
import numpy as np

AlphaPurifier.get_methods()
AlphaPurifier.get_methods("neutralize")
AlphaPurifier.get_methods("neutralize","polynomial")

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

def test(df):
    print(df)
    AP = AlphaPurifier(
        base_df=df,
        factor_name="factor",
        trade_date_col="datetime",
        symbol_col="symbol"
    )
    df = AP.winsorize().standardize().neutralize('multiOLS', ['beta','volatility']).to_result()
    print(df)

test(df())