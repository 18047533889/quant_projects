from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"
DOCS = PROJECT / "docs" / "gtja185_batch_evaluation.md"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def patch_evaluator() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    start = text.find("def _performance_stats(")
    end = text.find("\n\ndef _route_factor", start)
    if start < 0 or end < 0:
        raise RuntimeError("performance stats anchors not found")
    replacement = '''def _performance_stats(
    returns: pd.Series,
    *,
    annualization: float,
    hac_lags: int = 0,
    holding_period: int = 1,
) -> dict[str, float | int | None]:
    """Performance metrics robust to overlapping holding-period observations.

    ``annualization`` is the non-overlapping observation count (for an h-day
    holding return, normally 252/h). The HAC denominator includes positive
    overlap autocovariance and therefore cannot be used with a second daily
    sqrt(252) multiplier without double-counting observation frequency.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "hac_sharpe": None,
            "hac_long_run_volatility": None,
            "max_drawdown": None,
            "max_drawdown_non_overlapping_worst": None,
            "hit_rate": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    annual_return = mean * annualization
    annual_vol = std * math.sqrt(annualization) if math.isfinite(std) else float("nan")
    sharpe = annual_return / annual_vol if math.isfinite(annual_vol) and annual_vol > 0 else float("nan")

    values = clean.to_numpy(dtype=float)
    demeaned = values - mean
    n = len(values)
    max_lag = min(max(int(hac_lags), 0), max(n - 1, 0))
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    long_run_std = math.sqrt(long_run_variance)
    hac_sharpe = (
        mean / long_run_std * math.sqrt(float(annualization))
        if math.isfinite(long_run_std) and long_run_std > 0
        else float("nan")
    )
    hac_long_run_vol = long_run_std * math.sqrt(float(annualization))

    period = max(int(holding_period), 1)
    sleeve_drawdowns: list[float] = []
    for offset in range(min(period, n)):
        sleeve = clean.iloc[offset::period]
        if sleeve.empty:
            continue
        wealth = (1.0 + sleeve.clip(lower=-0.999999)).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        sleeve_drawdowns.append(float(drawdown.min()))
    worst_drawdown = min(sleeve_drawdowns) if sleeve_drawdowns else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "annual_return": annual_return,
        "annual_volatility": annual_vol if math.isfinite(annual_vol) else None,
        "sharpe": sharpe if math.isfinite(sharpe) else None,
        "hac_sharpe": hac_sharpe if math.isfinite(hac_sharpe) else None,
        "hac_long_run_volatility": hac_long_run_vol if math.isfinite(hac_long_run_vol) else None,
        "max_drawdown": worst_drawdown if math.isfinite(worst_drawdown) else None,
        "max_drawdown_non_overlapping_worst": worst_drawdown
        if math.isfinite(worst_drawdown)
        else None,
        "hit_rate": float((clean > 0).mean()),
    }
'''
    text = text[:start] + replacement + text[end:]
    text = text.replace(
        '''    sharpe = ((validation.get("long_short_net") or {}).get("sharpe"))
''',
        '''    sharpe = ((validation.get("long_short_net") or {}).get("hac_sharpe"))
''',
        1,
    )
    text = text.replace(
        '''                "long_short_gross": _performance_stats(
                    long_short_gross,
                    annualization=annualization,
                ),
                "long_short_net": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                ),
                "long_short": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                ),
''',
        '''                "long_short_gross": _performance_stats(
                    long_short_gross,
                    annualization=annualization,
                    hac_lags=max(int(horizon) - 1, 0),
                    holding_period=int(horizon),
                ),
                "long_short_net": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                    hac_lags=max(int(horizon) - 1, 0),
                    holding_period=int(horizon),
                ),
                "long_short": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                    hac_lags=max(int(horizon) - 1, 0),
                    holding_period=int(horizon),
                ),
''',
        1,
    )
    text = text.replace(
        '''                "validation_long_short_net_sharpe": ((validation.get("long_short_net") or {}).get("sharpe")),
''',
        '''                "validation_long_short_net_sharpe": ((validation.get("long_short_net") or {}).get("sharpe")),
                "validation_long_short_net_hac_sharpe": ((validation.get("long_short_net") or {}).get("hac_sharpe")),
''',
        1,
    )
    text = text.replace(
        '''                "test_long_short_net_sharpe": ((test.get("long_short_net") or {}).get("sharpe")),
''',
        '''                "test_long_short_net_sharpe": ((test.get("long_short_net") or {}).get("sharpe")),
                "test_long_short_net_hac_sharpe": ((test.get("long_short_net") or {}).get("hac_sharpe")),
''',
        1,
    )
    text = text.replace(
        '''            row.get("validation_long_short_net_sharpe")
            if row.get("validation_long_short_net_sharpe") is not None
''',
        '''            row.get("validation_long_short_net_hac_sharpe")
            if row.get("validation_long_short_net_hac_sharpe") is not None
''',
        1,
    )
    EVALUATOR.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    text = text.replace(
        '''    _price_forward_returns,
''',
        '''    _performance_stats,
    _price_forward_returns,
''',
        1,
    )
    text += '''

def test_hac_sharpe_penalizes_positive_overlap_autocorrelation():
    rng = np.random.default_rng(7)
    innovations = rng.normal(0.001, 0.01, size=400)
    values = np.zeros_like(innovations)
    for index in range(1, len(values)):
        values[index] = 0.8 * values[index - 1] + innovations[index]
    stats = _performance_stats(
        pd.Series(values),
        annualization=252 / 5,
        hac_lags=4,
        holding_period=5,
    )
    assert stats["hac_sharpe"] is not None
    assert stats["sharpe"] is not None
    assert abs(stats["hac_sharpe"]) < abs(stats["sharpe"])
    assert stats["max_drawdown"] == stats["max_drawdown_non_overlapping_worst"]
'''
    TESTS.write_text(text, encoding="utf-8")


def patch_docs() -> None:
    text = DOCS.read_text(encoding="utf-8")
    text = text.replace(
        '''   net returns, weight turnover, annualized return/volatility, ordinary and HAC-adjusted
   Sharpe, non-overlapping worst-sleeve drawdown, hit rate and yearly stability for every
   configured horizon. Routing uses net HAC-adjusted performance.
''',
        '''   net returns, weight turnover, annualized return/volatility, ordinary and conservative
   HAC-adjusted Sharpe, non-overlapping worst-sleeve drawdown, hit rate and yearly stability
   for every configured horizon. Routing uses net HAC-adjusted performance.
''',
        1,
    )
    DOCS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_evaluator()
    patch_tests()
    patch_docs()
    print("GTJA185 overlapping-return performance metrics HAC-adjusted")


if __name__ == "__main__":
    main()
