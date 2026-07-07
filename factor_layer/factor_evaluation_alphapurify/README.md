# alphapurify

`alphapurify` 是因子层中的特化分析模块，核心链路为：

`Database` -> `Exposures` -> `FactorAnalyzer`

当前目录已经完成配置化 CLI 对接，支持按 YAML 一键批量运行，并稳定输出：

- 每个目标因子 1 份结构化 JSON
- 每个目标因子 11 张 PNG 图
- 汇总 `run_summary.json`
- 可选 `config_snapshot.yaml`

## 模块定位

- **目标**：把原始因子目录和行情目录适配为标准分析输入，并批量完成暴露分析与因子分析。
- **边界**：本模块只做因子分析与可视化，不负责策略持仓、交易执行或回测撮合。
- **入口**：`run_from_config.py`（CLI）或 `pipeline.run_from_config()`（API）。

## 目录与文件功能（含输入/输出）

### 运行入口与编排

| 文件 | 功能 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `run_from_config.py` | CLI 入口，解析配置路径后执行 pipeline | 命令行参数：`config` | 控制台运行摘要 JSON |
| `run.py` | 兼容入口，内部复用 `run_from_config.main` | 命令行参数：`config` | 同 `run_from_config.py` |
| `pipeline.py` | 全流程编排、按因子循环执行、落盘 JSON/PNG | `AlphaPurifyAdapterConfig` | `run_summary.json`、`json/*.json`、`images/**/*.png`、可选快照 |
| `module_selector.py` | 运行时模块选择层，默认选择根目录正式模块 | 配置中的模块版本字段 | 选中的 `Database/Exposures/FactorAnalyzer` 实现 |
| `config.py` | YAML 配置模型、加载和校验 | `run_config.yaml` | `AlphaPurifyAdapterConfig` 对象 |

### 核心分析实现

| 文件 | 功能 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `Database.py` | 数据读取/适配/按 symbol 落盘，提供 `DataBase` | `data_root`、目标因子目录、暴露因子目录、年份等 | 标准化 `panel_df`、`PathConfig`、`stocks_list` |
| `Exposures.py` | 组合暴露与纯暴露分析（Portfolio/Pure） | 包含 `datetime/symbol/close/factor/exposures` 的面板数据 | `result_df`、相关性表、可视化 Figure |
| `FactorAnalyzer.py` | 单因子统计分析与多面板图表 | 同上面板数据 + 研究配置 | `ls/l/s/ic` 统计面板、可视化 Figure |
| `AlphaPurifier.py` | 因子净化工具集合（winsorize/neutralize/standardize 等） | 因子面板数据 | 净化后的因子数据 |
| `APr_utils.py` | 工具函数集合（供核心类复用） | 依调用而定 | 依调用而定 |
| `__init__.py` | 对外导出公共 API | - | `DataBase`、`FactorAnalyzer`、`run_pipeline` 等 |

### 版本模块与归档

| 目录/文件 | 功能 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `history_module/` | 保存历史快照与候选开发版本，供显式版本验证使用 | 显式模块版本选择 | 受控开发与回归对照 |
| `history_module/Database_original.py` | Database 历史基线实现 | 历史格式输入 | 历史格式输出 |
| `history_module/Exposures_legacy.py` | Exposures 历史快照 | 标准面板输入 | 历史暴露分析结果 |
| `history_module/Exposures_0505_v_next.py` | Exposures 候选开发版本 | 标准面板输入 | 候选暴露分析结果 |
| `history_module/pipeline_legacy.py` | pipeline 历史快照 | 历史配置与输入 | 历史流程产物 |
| `archive/` | 归档参考层，不参与正式默认运行 | 上游快照、实验脚本 | 参考材料 |

### 文档与示例

| 文件 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `examples/run_config.example.yaml` | 配置示例 | - | 可直接运行的 YAML |
| `docs/README.md` | 文档索引 | - | 文档入口 |
| `docs/Database适配差异检查报告.md` | `Database` 与旧版差异说明 | - | 适配结论与检查记录 |
| `docs/Exposures_FactorAnalyzer抽象版差异报告.md` | 历史抽象版对比说明 | - | 差异分析记录 |
| `docs/FactorAnalyzer_Exposures_输出总览.md` | 输出字段与图表说明 | - | 输出字典与图表清单 |

> 草稿目录 `0431-测试文件` 中的报告文档已迁移并归档到 `factor_layer/factor_evaluation_alphapurify/docs/`。

## 输入数据要求

运行时配置中的 `data_root` 目录应至少包含：

- `daily_market_summary/`：行情 parquet 文件（如 `daily_market_summary_2025.parquet`）
- `factors/<factor_name>/year=YYYY/*.parquet`：目标因子与暴露因子的分年数据

关键字段约定（适配后标准列）：

- 时间列：`datetime`
- 标的列：`symbol`
- 价格列：`close`
- 目标因子列：`<target_factor_name>`
- 暴露列：`<exposure_factor_name>...`

## 配置说明（YAML）

`examples/run_config.example.yaml` 支持以下核心字段：

- `data_root`：测试数据根目录
- `output_root`：输出根目录
- `year`：测试年份；为空时自动检测（优先 2025）
- `min_symbols_per_day`：最小截面样本数过滤阈值
- `target_factors`：目标因子目录名列表
- `exposure_factors`：暴露因子目录名列表
- `save_config_snapshot`：是否保存配置快照
- `database_module`：可选，显式指定 `history_module` 中的 Database 版本；默认使用根目录正式模块
- `exposures_module`：可选，显式指定 `history_module` 中的 Exposures 版本；默认使用根目录正式模块
- `factor_analyzer_module`：可选，显式指定 `history_module` 中的 FactorAnalyzer 版本；默认使用根目录正式模块

## 输出目录结构

默认示例输出到：
`workspace_data/alphapurify/outputs_exposures_factoranalyzer/`

```text
outputs_exposures_factoranalyzer/
├── run_summary.json
├── config_snapshot.yaml
├── json/
│   ├── <factor_1>.json
│   ├── <factor_2>.json
│   └── <factor_3>.json
└── images/
    ├── <factor_1>/
    │   ├── portfolio_exposures/*.png
    │   ├── pure_exposures/*.png
    │   └── factor_analyzer/*.png
    ├── <factor_2>/...
    └── <factor_3>/...
```

单因子 JSON 主要包含：

- `database`：适配和读取统计
- `exposures.portfolio`：PortfolioExposures 结果
- `exposures.pure`：PureExposures 结果
- `factor_analyzer`：因子统计面板
- `images`：该因子全部图片相对路径
- `errors`：失败信息（成功时为空）

## 快速开始

在仓库根目录运行：

```bash
python factor_layer/factor_evaluation_alphapurify/run_from_config.py \
  factor_layer/factor_evaluation_alphapurify/examples/run_config.example.yaml
```

单因子示例输出（已落盘）可直接查看：

- 配置：`factor_layer/alphapurify/examples/run_config.single_factor.example.yaml`
- 产物目录：`factor_layer/alphapurify/examples/sample_output/sf1`

## 样例图片（全量）

以下为本次示例配置实际生成的全部样例图（3 个因子 x 11 张 = 33 张）：

### day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1

![fa_ic_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/factor_analyzer/fa_ic_sheet.png)
![fa_long_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/factor_analyzer/fa_long_sheet.png)
![fa_long_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/factor_analyzer/fa_long_short_sheet.png)
![fa_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/factor_analyzer/fa_short_sheet.png)
![portfolio_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures.png)
![portfolio_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures_and_returns.png)
![portfolio_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/portfolio_exposures/portfolio_returns.png)
![pure_correlations](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/pure_exposures/pure_correlations.png)
![pure_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/pure_exposures/pure_exposures.png)
![pure_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/pure_exposures/pure_exposures_and_returns.png)
![pure_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1/pure_exposures/pure_returns.png)

### day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1

![fa_ic_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/factor_analyzer/fa_ic_sheet.png)
![fa_long_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/factor_analyzer/fa_long_sheet.png)
![fa_long_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/factor_analyzer/fa_long_short_sheet.png)
![fa_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/factor_analyzer/fa_short_sheet.png)
![portfolio_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures.png)
![portfolio_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures_and_returns.png)
![portfolio_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/portfolio_exposures/portfolio_returns.png)
![pure_correlations](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/pure_exposures/pure_correlations.png)
![pure_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/pure_exposures/pure_exposures.png)
![pure_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/pure_exposures/pure_exposures_and_returns.png)
![pure_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1/pure_exposures/pure_returns.png)

### day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1

![fa_ic_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/factor_analyzer/fa_ic_sheet.png)
![fa_long_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/factor_analyzer/fa_long_sheet.png)
![fa_long_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/factor_analyzer/fa_long_short_sheet.png)
![fa_short_sheet](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/factor_analyzer/fa_short_sheet.png)
![portfolio_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures.png)
![portfolio_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/portfolio_exposures/portfolio_exposures_and_returns.png)
![portfolio_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/portfolio_exposures/portfolio_returns.png)
![pure_correlations](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/pure_exposures/pure_correlations.png)
![pure_exposures](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/pure_exposures/pure_exposures.png)
![pure_exposures_and_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/pure_exposures/pure_exposures_and_returns.png)
![pure_returns](../../workspace_data/alphapurify/outputs_exposures_factoranalyzer/images/day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1/pure_exposures/pure_returns.png)

## 常见问题

- **找不到输出目录**：默认产物在 `workspace_data/alphapurify/outputs_exposures_factoranalyzer`。
- **PNG 导出失败**：请确认安装 `kaleido`，并优先使用当前 `pipeline.py` 的导出 fallback 逻辑。
- **样本过少**：提高 `year` 覆盖范围或降低 `min_symbols_per_day`。

## 参考文档

- `docs/README.md`
- `docs/Database适配差异检查报告.md`
- `docs/Exposures_FactorAnalyzer抽象版差异报告.md`
- `docs/FactorAnalyzer_Exposures_输出总览.md`
