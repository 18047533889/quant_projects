#!/usr/bin/env python
"""Fill NaN gaps in the feature matrix (SMART fill, no hard noise).

Approach per factor (independent, within its own valid date range):
1. Inner small gaps: groupby(asset) ffill (carry forward) + bfill (back-fill lead segment).
2. Remaining NaN (periods where the factor simply has no data):
   fill with that day's cross-sectional median as a fallback.
3. Drop columns that are entirely NaN (no data at all).

The fill is keyed on a (date, asset) MultiIndex. date/asset are stored as
string columns; we normalize date to datetime for correct time ordering and
re-emit date/asset columns at the end so downstream code keeps working.
"""
import time
import numpy as np
import pandas as pd

BUILD = "/home/sunhaiwei/quant_projects/lightgbm_qs/data/build"
IN = f"{BUILD}/features_all.parquet"
OUT = f"{BUILD}/features_filled.parquet"


def main():
    t0 = time.time()
    df = pd.read_parquet(IN)
    orig_shape = df.shape
    meta = df[["date", "asset"]].copy()
    factor_cols = [c for c in df.columns if c not in ("date", "asset")]

    before_nan = float(df[factor_cols].isna().mean().mean())

    # Normalize date to datetime and build a proper (date, asset) ordering.
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].astype(str)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)

    X = df[factor_cols].astype(np.float64)

    # 1) Within-asset carry forward / backfill for small internal gaps.
    #    groupby by asset preserves the (date-sorted) within-group order.
    X = X.groupby(df["asset"], sort=False).ffill()
    X = X.groupby(df["asset"], sort=False).bfill()

    # 2) Remaining NaN -> daily cross-sectional median fallback.
    daily_med = X.groupby(df["date"]).transform("median")
    X = X.fillna(daily_med)

    # 3) Drop fully-NaN columns (should not happen since we median-fill above,
    #    but keep the guard for factors with no cross-sectional median at all).
    nan_cols = X.columns[X.isna().all()].tolist()
    if nan_cols:
        X = X.drop(columns=nan_cols)

    result = pd.concat([df[["date", "asset"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    # Re-emit date as the original object dtype for consistency with inputs.
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    # Final ordering: original row order (sorted). Keep date string col order like input.
    result = result[["date", "asset"] + factor_cols]
    # drop dropped columns from final col list if any
    if nan_cols:
        result = result[[c for c in result.columns if c not in nan_cols]]

    result.to_parquet(OUT, index=False)

    out_nan = float(result.isna().mean().mean())
    dt = time.time() - t0

    print("=" * 60)
    print("input shape        :", orig_shape)
    print("output shape       :", result.shape)
    print("input  factor cols :", len(factor_cols))
    print("output factor cols :", len([c for c in result.columns if c not in ("date", "asset")]))
    print("dropped cols       :", len(nan_cols), nan_cols[:5])
    print("NaN before         : %.5f" % before_nan)
    print("NaN after          : %.5f" % out_nan)
    print("time               : %.1fs" % dt)
    print("saved              :", OUT)
    print("=" * 60)


if __name__ == "__main__":
    main()
