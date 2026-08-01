# FactorEngine 原生冷启动因子库

本目录面向因子挖掘、遗传搜索、LLM/Agent迭代和AutoFactorEvaluation。它与`gtja191/`、`week2_pv_factors/`并列，不修改历史公式包。

## 核心口径

`daily`现在表示：**由当前FactorEngine生产合同动态准入的日频输出因子库**。

它不再表示“公式只调用旧`DAILY_CANONICALS`”。一个原来归入Extended的技术指标、财务变换、分钟转日频因子或状态算子，只要满足当前生产合同，也可以进入`daily`；反过来，旧daily公式只要证据失效、参数超出认证域或生产路由不可用，就会被自动移出默认目录。

生产准入要求包括：

- 公式完整表达式能够由当前DSL构建；
- 全部算子调用通过`production`策略检查；
- 参数取值落在已认证参数域内；
- PIT、披露时间、SourceRef、交易时段、lookback和warmup合同可满足；
- stateful/full-history算子具备对应replay或checkpoint合同；
- 每个物理执行节点至少有一个当前证据支持的生产后端；
- 宏和FactorRecipe已经展开，并且其叶子算子均通过生产门禁；
- 运行时不得发生未计划、未认证的后端fallback；
- 不含负lag、lead、backfill、shuffle等未来或非确定性路径。

FactorEngine当前并不要求所有生产算子都同时具备Pandas、Polars和DuckDB三套实现。生产条件是：语义通过审核，并且至少存在一个独立认证的物理后端；路由只能使用已经认证的后端。三引擎同时认证属于更强的可移植性能力，不是所有算子的统一准入前提。

## 目录含义

| API目录 | 实际含义 | 默认使用 |
|---|---|---|
| `load_catalog(market, "daily")` | 当前证据和生产合同下可投递的日频输出因子 | 是 |
| `load_catalog(market, "extended")` | 未按生产准入过滤的研究候选档案 | 否 |
| `load_candidate_catalog(market, "daily")` | 历史daily authoring候选 | 否 |
| `load_candidate_catalog(market, "extended")` | 历史extended authoring候选和扩展种子 | 否 |

因此，`extended`不再参与“生产覆盖率”统计，也不会被默认AutoFactor provider加载。它只用于研究、补算子、定位生产门禁缺口和后续认证。

## 数据频率与生产状态分离

以下属于数据来源或输出频率，不是可用性等级：

- `daily`：日线输入、日频输出；
- `minute_to_daily`：分钟数据聚合为日频因子；
- `fundamental`：PIT财务报表和披露时间数据；
- `analyst`：分析师预期、修订和分歧数据；
- `benchmark`：指数或市场基准数据；
- `microstructure`：Trade/Quote/L2数据。

只要来源合同齐全并通过生产门禁，这些因子都可以进入默认日频输出目录。字段或SourceRef不足时，因子会在准入或抽样阶段被排除，而不是运行时临时伪造数据。

## 市场隔离

- A股使用A股字段、复权和交易日合同；
- 美股使用美股字段、复权、隔夜/日内和交易时段合同；
- 两个市场分别生成和准入；
- 美股没有可靠自由流通股本来源时，不准入`real_turnover_rate`；
- Quote/Trade/L2专属指标不能从分钟OHLCV近似冒充。

## 使用

### 默认生产冷启动目录

```python
from factor_cold_start import load_catalog

ashare_production = load_catalog("ashare", "daily")
us_production = load_catalog("us", "daily")
```

返回数量不是固定常数。它会随当前FactorEngine实现、参数合同和认证证据变化。准确数量以`reports/coverage.json`为准。

### 查看研究候选档案

```python
from factor_cold_start import load_candidate_catalog

ashare_extended_candidates = load_candidate_catalog("ashare", "extended")
```

候选档案中的公式能被解析，不代表能够生产执行。

### 单公式生产准入检查

```python
from factor_cold_start import admit_formula

result = admit_formula("ts_mean(ret, 20)")
print(result.eligible)
print(result.canonical_operators)
print(result.backend_map)
print(result.violations)
```

### 按真实字段抽样

```python
from factor_cold_start import sample_factors

batch = sample_factors(
    market="us",
    surface="daily",
    size=64,
    seed="experiment-001",
    available_fields={
        "open", "high", "low", "close", "pre_close", "volume", "amount",
        "ret", "vwap", "adj_factor", "ret__intra", "ret__overnight",
        "high__low__ratio", "upper__shadow__ratio", "vwap__close__dist",
    },
    availability_tiers=("core", "derived"),
    max_per_family=8,
)
```

### AutoFactorEvaluation

默认生产provider：

- `load_ashare_daily_pack`
- `load_us_daily_pack`

显式研究provider：

- `load_ashare_extended_pack`
- `load_us_extended_pack`

```bash
PYTHONPATH=.:factor_engine:AutoFactorEvaluation-RECONSTRUCT \
python -m evaluation.batch \
  --provider factor_cold_start.autofactor.provider:load_ashare_daily_pack \
  --output-dir output/factor_evaluation \
  --synthetic
```

## 校验

```bash
python factor_cold_start/scripts/validate_catalog.py --all
python factor_cold_start/scripts/report_coverage.py
pytest -q factor_cold_start/tests
```

永久CI会先校验FactorEngine的primitive、factor-operator和recipe evidence，再逐条检查默认`daily`目录的生产准入。生产覆盖率只针对当前真实可生产的公开canonical，不再针对旧daily/extended parser名单。

完整统计和被拒绝原因见：

- `reports/coverage.json`
- `reports/coverage.md`
