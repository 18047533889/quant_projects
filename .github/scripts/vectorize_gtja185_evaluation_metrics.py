from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "AutoFactorEvaluation-RECONSTRUCT" / "evaluation" / "gtja185_batch.py"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    text = replace_exact(
        text,
        '''def _daily_ic(signal: pd.Series, returns: pd.Series, *, min_assets: int) -> pd.DataFrame:
    joined = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1)
    rows: list[dict[str, Any]] = []
    for date, group in joined.groupby(level="datetime", sort=True):
        valid = group.dropna()
        if len(valid) < min_assets:
            continue
        rows.append(
            {
                "datetime": pd.Timestamp(date),
                "n": int(len(valid)),
                "ic": _safe_corr(valid["signal"], valid["return"], "pearson"),
                "rank_ic": _safe_corr(valid["signal"], valid["return"], "spearman"),
            }
        )
    return pd.DataFrame(rows)
''',
        '''def _matrix_row_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pairwise-finite row correlation and valid observation counts."""
    left, right = left.align(right, join="outer", axis=0)
    left, right = left.align(right, join="outer", axis=1)
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=1).astype(np.int64)
    safe_counts = np.maximum(counts, 1)
    x_mean = np.where(valid, x, 0.0).sum(axis=1) / safe_counts
    y_mean = np.where(valid, y, 0.0).sum(axis=1) / safe_counts
    x_centered = np.where(valid, x - x_mean[:, None], 0.0)
    y_centered = np.where(valid, y - y_mean[:, None], 0.0)
    numerator = (x_centered * y_centered).sum(axis=1)
    denominator = np.sqrt(
        (x_centered * x_centered).sum(axis=1)
        * (y_centered * y_centered).sum(axis=1)
    )
    correlations = np.full(len(counts), np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=correlations,
        where=(counts >= 2) & np.isfinite(denominator) & (denominator > 0),
    )
    return correlations, counts


def _daily_ic(signal: pd.Series, returns: pd.Series, *, min_assets: int) -> pd.DataFrame:
    """Vectorized daily Pearson IC and average-tie RankIC."""
    joined = pd.concat(
        [
            pd.to_numeric(signal, errors="coerce").rename("signal"),
            pd.to_numeric(returns, errors="coerce").rename("return"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    ic, counts = _matrix_row_correlation(signal_panel, return_panel)
    rank_signal = signal_panel.rank(axis=1, method="average", na_option="keep")
    rank_return = return_panel.rank(axis=1, method="average", na_option="keep")
    rank_ic, _ = _matrix_row_correlation(rank_signal, rank_return)
    eligible = counts >= int(min_assets)
    if not eligible.any():
        return pd.DataFrame(columns=["datetime", "n", "ic", "rank_ic"])
    return pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(signal_panel.index[eligible]),
            "n": counts[eligible].astype(int),
            "ic": ic[eligible],
            "rank_ic": rank_ic[eligible],
        }
    )
''',
        "vectorized daily IC",
    )
    start = text.find("def _long_short_series(")
    end = text.find("\n\ndef _performance_stats", start)
    if start < 0 or end < 0:
        raise RuntimeError("long-short function anchors not found")
    replacement = '''def _long_short_series(
    signal: pd.Series,
    returns: pd.Series,
    *,
    min_assets: int,
    n_quantiles: int,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorized equal-weight gross/net long-short returns and weight turnover."""
    joined = pd.concat(
        [
            pd.to_numeric(signal, errors="coerce").rename("signal"),
            pd.to_numeric(returns, errors="coerce").rename("return"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    valid = signal_panel.notna() & return_panel.notna()
    counts = valid.sum(axis=1)
    ranked = signal_panel.where(valid).rank(
        axis=1,
        method="average",
        pct=True,
        na_option="keep",
    )
    low_cut = 1.0 / int(n_quantiles)
    high_cut = 1.0 - low_cut
    bottom = (ranked <= low_cut) & valid
    top = (ranked > high_cut) & valid
    top_counts = top.sum(axis=1)
    bottom_counts = bottom.sum(axis=1)
    eligible = (
        (counts >= max(int(min_assets), int(n_quantiles) * 2))
        & (top_counts > 0)
        & (bottom_counts > 0)
    )
    if not bool(eligible.any()):
        empty_index = pd.DatetimeIndex([], name="datetime")
        return (
            pd.Series(index=empty_index, dtype=float, name="long_short_gross"),
            pd.Series(index=empty_index, dtype=float, name="long_short_net"),
            pd.Series(index=empty_index, dtype=float, name="turnover"),
        )
    weights = top.astype(float).div(top_counts.replace(0, np.nan), axis=0)
    weights -= bottom.astype(float).div(bottom_counts.replace(0, np.nan), axis=0)
    weights = weights.loc[eligible].fillna(0.0)
    realized = return_panel.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    gross = (weights * realized).sum(axis=1).astype(float)
    weight_changes = weights.diff()
    weight_changes.iloc[0] = weights.iloc[0]
    turnover = 0.5 * weight_changes.abs().sum(axis=1)
    net = gross - turnover * float(cost_bps) / 10_000.0
    gross.index.name = "datetime"
    net.index.name = "datetime"
    turnover.index.name = "datetime"
    return (
        gross.rename("long_short_gross"),
        net.rename("long_short_net"),
        turnover.rename("turnover"),
    )
'''
    text = text[:start] + replacement + text[end:]
    EVALUATOR.write_text(text, encoding="utf-8")
    print("GTJA185 IC, RankIC, quantile portfolios and turnover vectorized")


if __name__ == "__main__":
    main()
