"""生成阶段 4-2 本地开发所需的阶段 4-1 mock 输入。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    从 repo 根目录运行时，在 `database/mock/evaluation/indicator/stage4_1_output/`
    下生成符合阶段 4-2 输入契约的绩效时序 parquet、csv 样例和
    `performance_series_bundle.json`。
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from main_code.calculator import DEFAULT_METHOD_CONFIG, default_run_paths, ensure_dir, write_json, write_table


def generate_mock_stage4_1_data(
    input_dir: Path,
    factor_id: str = "FACTOR_DEMO_001",
    eval_run_id: str = "ER20260512_001",
    horizon: str = "5D",
    periods: int = 180,
    seed: int = 42,
    project_root: Path | None = None,
) -> dict:
    """生成本地测试用的阶段 4-1 模拟绩效序列。

    这个函数只服务于本地开发和测试，不属于阶段 4-2 的核心计算代码。
    """

    ensure_dir(input_dir)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-08-01", periods=periods)

    base = {
        "factor_id": factor_id,
        "eval_run_id": eval_run_id,
        "horizon": horizon,
    }

    rank_ic_t = rng.normal(0.042, 0.055, periods)
    rank_ic = pd.DataFrame(
        {
            "date": dates,
            **base,
            "rank_ic_t": rank_ic_t,
            "valid_asset_count": rng.integers(2800, 3500, periods),
        }
    )

    ic = pd.DataFrame(
        {
            "date": dates,
            **base,
            "ic_t": rank_ic_t * 0.85 + rng.normal(0.0, 0.025, periods),
        }
    )

    kendall_tau = pd.DataFrame(
        {
            "date": dates,
            **base,
            "kendall_tau_t": rank_ic_t * 0.70 + rng.normal(0.0, 0.020, periods),
        }
    )

    quantile_rows = []
    quantile_means = np.array([-0.0030, -0.0012, 0.0002, 0.0018, 0.0045])
    for dt in dates:
        day_noise = rng.normal(0, 0.001)
        for q, mean_ret in enumerate(quantile_means, start=1):
            gross = mean_ret + day_noise + rng.normal(0, 0.006)
            net = gross - 0.00025 - 0.00008 * abs(q - 3)
            quantile_rows.append(
                {
                    "date": dt,
                    **base,
                    "quantile_id": q,
                    "gross_return": gross,
                    "net_return": net,
                    "asset_count": int(rng.integers(550, 730)),
                }
            )
    quantile = pd.DataFrame(quantile_rows)

    q1 = quantile[quantile["quantile_id"] == 1][["date", "gross_return"]].rename(columns={"gross_return": "bottom_return"})
    q5 = quantile[quantile["quantile_id"] == 5][["date", "gross_return"]].rename(columns={"gross_return": "top_return"})
    top_bottom = q5.merge(q1, on="date")
    top_bottom["factor_id"] = factor_id
    top_bottom["eval_run_id"] = eval_run_id
    top_bottom["horizon"] = horizon
    top_bottom["top_minus_bottom_t"] = top_bottom["top_return"] - top_bottom["bottom_return"]
    top_bottom = top_bottom[["date", "factor_id", "eval_run_id", "horizon", "top_return", "bottom_return", "top_minus_bottom_t"]]

    gross_return = top_bottom["top_minus_bottom_t"].to_numpy() + rng.normal(0.0004, 0.002, periods)
    long_short = pd.DataFrame(
        {
            "date": dates,
            **base,
            "gross_return": gross_return,
            "net_return": gross_return - rng.normal(0.0006, 0.00015, periods).clip(0.0002, None),
        }
    )

    turnover = pd.DataFrame(
        {
            "date": dates,
            **base,
            "turnover_t": rng.beta(4, 8, periods).clip(0, 1),
        }
    )

    universe_count = rng.integers(3400, 3700, periods)
    valid_asset_count = (universe_count * rng.normal(0.965, 0.015, periods).clip(0.90, 0.99)).astype(int)
    coverage = pd.DataFrame(
        {
            "date": dates,
            **base,
            "valid_asset_count": valid_asset_count,
            "universe_count": universe_count,
            "coverage_t": valid_asset_count / universe_count,
        }
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            **base,
            "benchmark_return": rng.normal(0.0002, 0.010, periods),
        }
    )

    cost = pd.DataFrame(
        {
            "date": dates,
            **base,
            "impact_cost": rng.normal(0.00035, 0.00008, periods).clip(0, None),
            "borrow_fee": rng.normal(0.00005, 0.00002, periods).clip(0, None),
            "net_return_erosion": (long_short["gross_return"] - long_short["net_return"]).clip(0, None),
        }
    )

    style_rows = []
    for style_name in ["size", "value", "momentum", "volatility"]:
        for dt in dates:
            style_rows.append(
                {
                    "date": dt,
                    "factor_id": factor_id,
                    "eval_run_id": eval_run_id,
                    "style_name": style_name,
                    "exposure_value": rng.normal(0, 0.18),
                }
            )
    style = pd.DataFrame(style_rows)

    corr = pd.DataFrame(
        {
            "factor_id": factor_id,
            "other_factor_id": [f"LIB_FACTOR_{i:03d}" for i in range(1, 11)],
            "corr_value": rng.normal(0.12, 0.18, 10).clip(-0.85, 0.85),
        }
    )

    tables = {
        "rank_ic_series": rank_ic,
        "ic_series": ic,
        "kendall_tau_series": kendall_tau,
        "quantile_return_panel": quantile,
        "top_minus_bottom_series": top_bottom,
        "long_short_return_series": long_short,
        "turnover_series": turnover,
        "coverage_series": coverage,
        "benchmark_return_series": benchmark,
        "cost_series": cost,
        "style_exposure_panel": style,
        "library_corr_matrix": corr,
    }

    bundle = {}
    for name, df in tables.items():
        path = input_dir / f"{name}.parquet"
        write_table(path, df)
        write_table(input_dir / f"{name}.csv", df.head(20))
        if project_root is not None:
            bundle[name] = path.resolve().relative_to(project_root.resolve()).as_posix()
        else:
            bundle[name] = str(path)

    write_json(input_dir / "metric_method_config.json", dict(DEFAULT_METHOD_CONFIG))
    write_json(
        input_dir / "factor_metadata.json",
        {
            "factor_id": factor_id,
            "eval_run_id": eval_run_id,
            "track": "Track_A",
            "asset_class": "CN_STOCK",
            "frequency": "daily",
            "domain_root": "mock_local",
            "horizon": horizon,
        },
    )
    write_json(input_dir / "performance_series_bundle.json", bundle)
    return bundle


def main() -> None:
    """按默认根目录路径生成 mock 输入。"""

    paths = default_run_paths(ROOT)
    bundle = generate_mock_stage4_1_data(paths.input_dir, project_root=ROOT)
    print(f"Generated {len(bundle)} mock Stage 4-1 series in {paths.input_dir}")


if __name__ == "__main__":
    main()
