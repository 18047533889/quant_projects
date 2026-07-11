"""生成阶段 4-2 本地检查文档。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    读取当前 mock 输入、指标输出和校验报告，生成
    `docs/stage4_2_task_check.md`，便于 reviewer 人工检查输入输出路径、
    schema、关键公式和示例结果。
"""

from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from main_code.calculator import SERIES_REQUIRED_COLUMNS, default_run_paths, ensure_dir, read_json, read_table


def _rel(path: Path | str) -> str:
    """把展示路径转成相对项目根目录的写法，避免检查文档出现本机绝对路径。"""

    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    """读取默认产物并写出本地检查文档。"""

    paths = default_run_paths(ROOT)
    docs_dir = ROOT / "docs"
    ensure_dir(docs_dir)
    bundle = read_json(paths.bundle_path)
    metrics = read_table(paths.output_dir / "metrics_matrix.parquet")
    scorecard = read_json(paths.output_dir / "summary_scorecard.json")
    validation = read_json(paths.output_dir / "metric_validation_report.json")

    lines = [
        "# 阶段 4-2 A 轨基于时序的指标计算模块本地检查说明",
        "",
        "## 1. 本次实现边界",
        "",
        "本地代码只实现阶段 4-2：消费阶段 4-1 已生成的绩效序列，聚合输出指标矩阵、评分卡、指标包和校验报告。",
        "它不从原始因子值、行情数据或 forward return 重新生成 RankIC、分组收益、多空收益等上游序列。",
        "",
        "## 2. 生成的文件",
        "",
        "| 类型 | 路径 |",
        "|---|---|",
        f"| 模拟输入目录 | `{_rel(paths.input_dir)}` |",
        f"| 指标输出目录 | `{_rel(paths.output_dir)}` |",
        f"| 日志目录 | `{_rel(paths.log_dir)}` |",
        f"| 指标矩阵 | `{_rel(paths.output_dir / 'metrics_matrix.parquet')}` / `{_rel(paths.output_dir / 'metrics_matrix.json')}` |",
        f"| 评分卡 | `{_rel(paths.output_dir / 'summary_scorecard.json')}` |",
        f"| 指标包 | `{_rel(paths.output_dir / 'metrics_package.json')}` |",
        f"| 校验报告 | `{_rel(paths.output_dir / 'metric_validation_report.json')}` |",
        f"| manifest | `{_rel(paths.output_dir / 'metric_manifest.json')}` |",
        "",
        "## 3. 模拟输入数据接口",
        "",
        "| 序列名 | 文件 | 关键字段 |",
        "|---|---|---|",
    ]

    for name, path in bundle.items():
        cols = ", ".join(sorted(SERIES_REQUIRED_COLUMNS.get(name, [])))
        lines.append(f"| `{name}` | `{_rel(path)}` | `{cols}` |")

    lines.extend(
        [
            "",
            "## 4. 指标计算口径摘要",
            "",
            "| 指标 | 口径 |",
            "|---|---|",
            "| `rank_ic_mean` | `mean(rank_ic_t)` |",
            "| `effective_annual_factor` | `annual_factor / horizon_days`，例如 `5D -> 252 / 5 = 50.4` |",
            "| `rank_ic_ir_raw` | `mean(rank_ic_t) / std(rank_ic_t)`，分母使用 `eps` 保护 |",
            "| `rank_ic_ir` | `rank_ic_ir_raw * sqrt(effective_annual_factor)` |",
            "| `rank_ic_win_rate` | `rank_ic_t > win_rate_threshold` 的占比 |",
            "| `quantile_curve` | 各分位组 `net_return` 全样本均值 |",
            "| `quantile_monotonicity_score` | `Spearman(quantile_id, mean_quantile_return)` |",
            "| `top_minus_bottom_mean` | `mean(top_minus_bottom_t)` |",
            "| `long_short_ann_return` | 多空 `net_return` 复利年化收益 |",
            "| `long_short_sharpe` | 年化收益 / 年化波动 |",
            "| `max_drawdown` | 多空净值曲线最大回撤 |",
            "| `calmar` | 行业常用口径：年化收益 / 样本最大回撤绝对值；若最大回撤为 0，则输出 null |",
            "| `turnover` | `mean(turnover_t)` |",
            "| `universe_coverage_mean` | `mean(coverage_t)` |",
            "| `library_corr_max` | 库内相关系数绝对值最大值 |",
            "| `*_sample_count` / `*_valid_count` / `*_valid_ratio` | 原始样本数、有限值样本数、有效样本比例 |",
            "",
            "说明：`metrics_matrix.value` 在 JSON 与 Parquet 中统一为 string-or-null，`value_type` 标记 `number` / `json` / `string` / `null`。",
            "",
            "## 5. 输出样例",
            "",
            "### 评分卡摘要",
            "",
            "```json",
            str(scorecard).replace("'", '"')[:2000],
            "```",
            "",
            "### 指标矩阵前 10 行",
            "",
            metrics.head(10).to_markdown(index=False),
            "",
            "## 6. 测试覆盖",
            "",
            "- 输入 bundle 字段完整性与输出文件存在性。",
            "- P0/P1 指标是否生成且状态可读。",
            "- 核心指标数值是否与手工计算口径一致。",
            "- 缺少 P0 序列时是否显式产生 `missing_input` 与校验错误。",
            "- 样本不足时是否产生 warning。",
            "- 标准差为 0 时是否使用 `eps` 保护并写入 warning。",
            "- horizon 不一致时是否在校验报告中标记。",
            "- P1/P2 输入缺失时是否不阻断 P0 输出，并标记 `missing_input`。",
            "",
            "## 7. 当前校验事件摘要",
            "",
            "```json",
            str(validation).replace("'", '"')[:2000],
            "```",
        ]
    )

    out = docs_dir / "stage4_2_task_check.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
