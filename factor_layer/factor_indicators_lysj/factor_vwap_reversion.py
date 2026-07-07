"""
VWAP Reversion Factor for NQ Futures (Intraday)
================================================
因子逻辑：
  价格偏离日内累计VWAP的程度 → 预测短期均值回归
  factor = -(close - vwap) / vwap  (负偏离 → 预期上涨, 正偏离 → 预期下跌)

日内IC计算：
  对每个交易日内，计算 factor 与 forward_ret 的 Spearman rank correlation
  汇报每日IC均值、IC标准差、IC_IR (= mean/std)、IC>0占比
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path


# ── 参数 ──────────────────────────────────────────────
FORWARD_MINUTES = 5          # 前瞻收益窗口（分钟）
DATA_DIR = Path(__file__).parent / "NQ"
OUTPUT_PATH = Path(__file__).parent / "factor_output.parquet"


def load_data() -> pd.DataFrame:
    """读取全量NQ分钟线数据"""
    df = pd.read_parquet(
        DATA_DIR,
        columns=["data", "open", "high", "low", "close", "volume", "average"],
    )
    df = df.sort_values("data").reset_index(drop=True)
    df["date"] = df["data"].dt.date
    return df


def compute_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    逐日计算因子值
    - cum_vwap: 日内累计 VWAP = cumsum(average * volume) / cumsum(volume)
    - factor:   -(close - cum_vwap) / cum_vwap  (VWAP偏离反转)
    """
    # 日内累计量
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_turnover"] = df.groupby("date").apply(
        lambda g: (g["average"] * g["volume"]).cumsum(), include_groups=False
    ).droplevel(0)

    df["cum_vwap"] = df["cum_turnover"] / df["cum_vol"]

    # 偏离度 (取负号 → 偏离越负=价格越低于VWAP → 预期反弹)
    df["factor"] = -(df["close"] - df["cum_vwap"]) / df["cum_vwap"]

    # 剔除开盘前几分钟累计量太小导致的异常
    mask_warmup = df.groupby("date").cumcount() < 5
    df.loc[mask_warmup, "factor"] = np.nan

    return df


def compute_forward_return(df: pd.DataFrame, n: int = FORWARD_MINUTES) -> pd.DataFrame:
    """计算 n 分钟前瞻收益率"""
    df["forward_ret"] = df.groupby("date")["close"].shift(-n) / df["close"] - 1
    return df


def calc_daily_ic(df: pd.DataFrame) -> pd.DataFrame:
    """逐日计算 Spearman Rank IC"""
    records = []
    for date, grp in df.groupby("date"):
        valid = grp[["factor", "forward_ret"]].dropna()
        if len(valid) < 30:
            continue
        ic, pval = spearmanr(valid["factor"], valid["forward_ret"])
        records.append({"date": date, "ic": ic, "pval": pval, "n_obs": len(valid)})
    return pd.DataFrame(records)


def report(ic_df: pd.DataFrame):
    """打印IC统计摘要"""
    ic_mean = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos_ratio = (ic_df["ic"] > 0).mean()

    print("=" * 60)
    print(f"  VWAP Reversion Factor  |  Forward = {FORWARD_MINUTES} min")
    print("=" * 60)
    print(f"  Days           : {len(ic_df)}")
    print(f"  IC Mean        : {ic_mean:+.4f}")
    print(f"  IC Std         : {ic_std:.4f}")
    print(f"  ICIR           : {ic_ir:+.4f}")
    print(f"  IC > 0 ratio   : {ic_pos_ratio:.2%}")
    print(f"  IC Median      : {ic_df['ic'].median():+.4f}")
    print("=" * 60)

    # 按年汇总
    ic_df = ic_df.copy()
    ic_df["year"] = pd.to_datetime(ic_df["date"]).dt.year
    yearly = ic_df.groupby("year")["ic"].agg(["mean", "std", "count"])
    yearly["icir"] = yearly["mean"] / yearly["std"]
    print("\n  Yearly breakdown:")
    print(yearly.to_string())
    print()


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} bars, {df['date'].nunique()} days")

    print("Computing factor...")
    df = compute_factor(df)

    print("Computing forward returns...")
    df = compute_forward_return(df)

    print("Calculating daily IC...")
    ic_df = calc_daily_ic(df)

    report(ic_df)

    # 保存因子列供后续使用
    out = df[["data", "factor"]].rename(columns={"data": "timestamp"}).copy()
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Factor output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
