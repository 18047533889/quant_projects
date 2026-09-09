# factor_engine — DSL 因子计算引擎（1737 算子）

企业级 A 股横截面日频多因子量化项目的**因子计算层**：把因子公式写成 DSL
（如 `rank(ts_mean(close, 20))`），引擎从 **data_access** 读行情，在 `(日期 × 标的)`
网格上算出因子值（MultiIndex Series），并可**落盘到因子湖（factor_lake）**。

**版本:** 0.3.1 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_engine (私有)
**中文完整指南:** [`docs/FactorEngine完全指南.md`](docs/FactorEngine完全指南.md) ｜
**HTTP 服务:** [`service/README.md`](service/README.md) ｜
**算子白名单:** `docs/dsl_allowlist.json`（1421 名）｜
**算子审计表:** `cleaned_operators/docs/operators_catalog.json`（1737 canonical）｜
**后端覆盖（权威）:** `docs/BACKEND_COVERAGE.md`

---

## 它是什么 / 不是什么

**做什么：** DSL 解析 → AST → IR 分析 → 规划（Lowerer/Optimizer/CSE）→ 多后端执行
（SQL 下推 → Polars → Pandas）→ 因子值落盘因子湖。支持批量 `run_many`（共享子树只算一次）、
增量物化、shard 物化、因子矩阵物化、HTTP 服务、挖掘集成、公式身份。

**不做什么：** 不做数据清洗（data_access 提供）、不做因子评估（quant_evaluator 做）、
不训练模型（modeling 做）。它只负责**算因子 + 写因子湖**，且这是唯一实现。

## 安装与第一个因子

```bash
# Python >= 3.10；团队源码安装先准备匹配版本的内部依赖
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git
git clone https://github.com/HKUST-QUANT-SOCIETY/data_access.git
python -m pip install --no-deps -e ./data_access
cd factor_engine
python -m pip install -e .
```

`data-access>=0.10.2` 是运行时硬依赖。先用 `--no-deps` 绑定同级私有源码库，
可避免 pip 从公共索引解析内部同名包；随后安装 Factor Engine 时再由统一
constraints/bootstrap 解析外部依赖。

```python
from factor_engine.api import col, rank, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source

source = build_data_source({
    "type": "data_access",
    "dataset": "ashare_stock_daily",
    "fields": {"close": "Close"},
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
})
engine = FactorEngine(backend=build_backend("pandas"), data_source=source, run_mode="research")
factor = Factor(name="mom20_rank", expr=rank(ts_mean(col("close"), 20)))
out = engine.run(factor)
print(out["result"].head())   # MultiIndex Series (TradeDate, Symbol)
```

公共入口和配置的默认后端为 `pandas` 参考路径，数据源须支持列读取契约。
需要原生性能时可显式选择 `polars`，但数据源必须满足其长表扫描契约；
仅提供 `load_column` 的数据源不可假定兼容 Polars。
这只修复默认入口的可用性，不代表全部算子已认证，也不改变 production 的
PIT、DQ、物理计划及后端认证门禁。显式 `auto`/`hybrid` 不会静默 fallback：
当前公共 `run`、`run_many`、`run_many_iter`、`run_many_parallel`、
`run_many_stream` 尚未接通 auto 物理执行，会抛 `PhysicalPlanRequiredError`。

或 YAML 一键跑：`FactorEngine.run_from_config("examples/config_driven_factor.yaml")`。
或 HTTP 服务：`pip install -e ".[service]"` 后 `factor-engine-serve --port 8088`。

> ⚠️ **已知硬伤**：`examples/` 目录在仓库中整体缺失（含 `prod.yaml`、`simple_factor.py`、
> `config_driven_factor.yaml`），但 docs、完全指南、`runtime/config.py` 都引用它。
> 生产 profile 实际在 `runtime/quality/dq_profiles.yaml`（`research` 宽松 vs
> `us_equity_daily_prod` 生产门禁，min_coverage=0.85 / min_instruments_per_day=500 等）。
> 引用 `examples/profiles/prod.yaml` 的文档/示例待修复。

## DSL API（`factor_engine.api`）

`api/__init__.py` 显式导出 10 个符号：`Factor`、`col(name)`、`field(...)`、`rank`、
`ts_mean`、`ts_std`、`ts_std_dev`、`zscore`、`delay`、`make_cleaned_call_factory`。

其余全部算子（`ts_corr`、`group_rank`、`trade_when` 等）通过模块级 `__getattr__` 懒加载：
首次访问时校验名字在 `build_dsl_allowlist()`（白名单 1421 名）内，然后缓存 factory。
**api/ 只造表达式树不算数，runtime 全在 cleaned_operators。**

其他公共入口：
- `parse_expr(text, surface="daily", dialect="native", budget=ComplexityBudget) -> Expr` 与
  `parse_factor(...) -> Factor`（`dsl_parser.py`；基于 `ast.parse(mode="eval")`，
  `and`/`or`/`not` 需写 `and_`/`or_`/`not_`，裸列名自动视为列引用）
- `api/operator_registry.py`：`build_dsl_allowlist` / `build_authoring_allowlist` /
  `build_research_mining_allowlist` / `build_production_mining_allowlist`
  （mining/production 准入 **fail-closed**）
- `api/technical_macros_v2`（布林/唐奇安等复合宏）、`api/intraday_daily`（分钟→日 SourceRef 列）、
  `api/lqtp_*` 兼容层、`api/mining_integration`

## 架构

```
Factor(expr) → expr AST → ir.Analyzer → planner（Lowerer + Optimizer + CSE）
    → planning（多后端物理 region 切分）→ backend.execute
    （已准入区域按固定后端执行；公共默认 Pandas 参考路径）→ 因子值 → materialize → factor_lake
```

| 层 | 目录 | 作用 |
|---|---|---|
| DSL API | `api/` | `col()`, `rank()`, `ts_mean()`, `Factor`, `parse_expr`, 白名单构建 |
| AST | `expr/` | 仅 3 节点：`ColumnRef` / `Literal` / `CleanedCall`（`Expr` 与四则重载） |
| 分析 | `ir/` | `Analyzer.lower(expr) → AnalysisResult(ir, lookback, has_ts_op, has_cs_op, referenced_columns)` |
| 规划 | `planner/` | `Lowerer`、`Optimizer`、`CSE`、`DAGPlan`、成本模型、region 优化（52 .py） |
| 物理规划 | `planning/` | 多后端物理 region 规划（Pandas/Polars/DuckDB/q 区域切分） |
| 执行 | `backend/` | 11 个后端 + numba 内核 + SQL 下推 + q 后端（128 顶层 .py） |
| 算子 | `cleaned_operators/` | **唯一 runtime 算子库**（447 .py，1737 canonical） |
| 存储 | `storage/` | `build_data_source`、`DataSource`、parquet/CH/long_table 源、物化/落盘/发布门。分钟→日 intraday_feature 运行时（`storage/sources/intraday_feature_runtime_v2.py`）在 `_grouped_bars` 对**每个 (TradeDate, Symbol)** 做精确槽位 session 完整性校验（R61-P0 #60）：A 股 09:31–11:30 / 13:01–15:00 共 240 根 bar_end；缺 bar/重复/盘外(12:30 等) 单独报告；停牌日历命中 → 该组 NaN 不 abort；非停牌损坏组 → quarantine，`session_require_full=True`（production 默认）fail-closed。停牌源 = `ashare_stock_daily.IsSuspend` |
| 运行时 | `runtime/` | `FactorEngine`、批调度、多后端并行、dq_gates、生产策略（156 .py） |
| 服务 | `service/` | 薄 FastAPI 适配层（15 路由，端口 8088） |
| 挖掘 | `mining/` | `MiningRole` 分类、`get_mining_operators`、直接可用判定 |
| 身份 | `identity/` | `get_factor_identity`（AST 归一化 SHA-256，signal_equivalence_id） |
| 安全 | `security/` | 派生因子数据权限继承、factor_id 文件系统安全 domain 校验 |
| 字段 | `fields/` | `FieldRegistry`/`FieldSpec`/`TableSpec`；188 个规范字段（v17） |
| 市场 | `market/` | instrument/calendar/price_basis/return_semantic/adjustment_policy |
| 配方 | `factor_recipes/` | 受限表达式→生产 primitive 展开 + DAG 执行 |
| 导出 | `export/` | serializers/importers/converters/versioning（算子导出导入） |
| 研究面 | `research_operators/` `research_tools/` | 研究专用显式 opt-in 面（统计/矩阵/PCA/FFT/wavelet） |
| 可观测 | `telemetry/` | opt-in metrics/traces/health（prometheus/otel 桥） |

## 后端矩阵（`backend/factory.build_backend(type)`）

| 后端名 | 类 | 说明 |
|---|---|---|
| `pandas` / `pandas_modin` | `PandasBackend` | 完全委托 cleaned_operators |
| `polars` / `polars_lazy` | `PolarsBackend` | 宽表 / LazyFrame |
| `polars_long` / `polars_native` / `long_polars` | `PolarsLongBackend` | 长表 Polars 编译 |
| `auto_long` / `hybrid_long` | `HybridLongBackend` | DuckDB long 物化 + 原生 Polars long |
| `duckdb_sql` / `sql_pushdown` / `duckdb_pushdown` / `sql` | `DuckDBPushdownBackend` | SQL 子树下推 |
| `clickhouse_sql` / `ch_sql` / `clickhouse_pushdown` | `ClickHousePushdownBackend` | 同上 CH 方言 |
| `auto` / `hybrid` | `HybridBackend` | physical-only：仅供已准入 PhysicalRegionPlan 集成；公共 run 系列尚不支持，非默认 |
| `debug` | `DebugBackend` | 只打印计划不读数据 |
| `q_kdb` / `q` | `QBackend` | Q/KDB 物理执行 |

**覆盖（权威 = `docs/BACKEND_COVERAGE.md`，2026-08-20）：** total canonicals **1624**；
clickhouse_sql implemented 394 / selectable 385；duckdb_sql **394**；pandas_numpy **1622**；
polars **1623**；oracle_passed 296~1299；**production_admitted = 0（诚实 fail-closed）**。
⚠️ 旧 `backend_coverage.md` 数字过时，勿引用。

## 算子全景（1737 canonical）

`cleaned_operators/` 共 **447 个 .py**，按子目录（文件数）：common 67、intraday 25、
polars_native 24、fundamental 23、price_volume 19、technical 19、ts_model 11、
cross_section 9、stateful 9、overhaul 9、microstructure 7、closure 7、ashare 4、
relation 4、shareholder 4、index_listing 3、valuation 3。

**权威统计（`cleaned_operators/docs/operators_catalog.json`，schema v2）：**
- canonical 总数 **1737**（全部 `canonical=True`，名字唯一）
- surface：daily **1238** ／ extended **480** ／ research **8** ／ unsafe 7 ／ legacy 1 ／ internal 3
- 有别名 **136** 个；scope 分布：time_series **871**、elementwise **503**、
  fundamental_period **136**、cross_sectional **96**、group **59**、session_intraday **70**
- execution_kind：primitive 1564 / stateful 110 / composite 53 / external_kernel 10
- status（`operator_manifest.json` 2026-08-28）：experimental 1593 / production 137 / deprecated 1 / research 6

### 算子族代表（按族）

- **截面/排名/中性化（13）**：`rank` `zscore` `normalize` `winsorize` `cs_demean` `cs_mean`
  `cs_std` `cs_regression` `cs_resid` `cs_quantile` `cs_mad` `rank_corr` `group_neutralize`
- **时序 ts_（26）**：`ts_mean` `ts_std` `ts_rank` `ts_corr` `ts_delta` `ts_sum` `ts_min`
  `ts_max` `ts_median` `ts_skew` `ts_kurt` `ts_decay_linear` `ts_decay_exp_window`
  `ts_argmax` `ts_argmin` `ts_quantile` `ts_autocorr` `ts_beta` `ts_max_drawdown`
  `ts_topk_sum` `ts_ema` `ts_pct` `ts_days_since` `ts_true_streak` `ts_count_if` `ts_mean_if`
- **分组 group_（7）**：`group_rank` `group_zscore` `group_mean` `group_std` `group_sum`
  `group_neutralize` `group_quantile_spread`
- **均线（8）**：`ts_mean` `ts_ema` `WMA` `DEMA` `TEMA` `HMA` `KAMA` `ALMA`
- **技术指标/振荡（20）**：`RSI_WILDER` `MACD_line` `MACD_signal` `MACD_hist` `ATR_WILDER`
  `CMF` `CMO` `ForceIndex` `MFI` `PPO` `PSAR` `Supertrend` `TSI` `UltimateOscillator` `ADX`
  `DMI_plus` `KeltnerUpper` `bollinger_pct_b` `bollinger_width` `rolling_obv`
- **事件/条件/方向（8）**：`trade_when` `where` `event_frequency` `event_rate_pct`
  `event_decay_asof` `directional_change_state` `limit_up_close` `ts_true_streak`
  （另有 event_* 族 49 个：event_active_count / cluster_* / fano_* / hawkes_* 等）
- **量价/流动性（11）**：`adv` `amihud_illiquidity` `abnormal_volume` `abnormal_turnover`
  `vwap_deviation` `vwap_distance_pct` `volume_momentum` `turnover_momentum`
  `free_float_ratio` `float_share_ratio` `price_volume_divergence`
- **收益/波动估计（6）**：`ts_pct` `garman_klass_vol` `parkinson_vol` `yang_zhang_vol`
  `return_volume_corr` `return_per_turnover`
- **分钟级（5）**：`intraday_volatility` `intraday_vwap_deviation` `intraday_medrv`
  `intraday_realized_semivariance_balance` `session_event_recovery_score`（intraday_* 共 36）
- **基本面/PIT（9）**：`book_to_price` `altman_z_score` `holder_concentration`
  `fin_staleness` `fin_surprise_zscore` `yoy_by_period` `valuation_pe_ttm_lyr_gap`
  `ashare_limit_up_touch` `fin_beat_streak`（fundamental+holder+fin_* 族共 181）
- **基础数学/安全（14）**：`abs` `log` `sqrt` `power` `add` `subtract` `multiply` `divide`
  `protected_div` `safe_div_null` `clip` `floor` `ceil` `sign`

### 别名真相（写因子最常踩的坑）

| 想写 | 真名 |
|---|---|
| `SMA` / `ma` / `Mean` / `move` | → `ts_mean` |
| `EMA` | → `ts_ema` |
| `neutralize` | → `group_neutralize` |
| `ts_regression` | → `ts_regression_slope` |
| `if_else` / `IIF` | → `where` |
| `returns` | → `ts_pct` |
| `safe_div` | → `safe_div_null` |
| `RSI` | ⚠️ 无 canonical 无别名 → 用 `RSI_WILDER` |
| `OBV` | ⚠️ 无 → 用 `rolling_obv` |
| `ROC` / `TRIX` / `CCI` / `STOCH` / `pe_ttm` | ⚠️ 无 canonical 无别名，查 `dsl_allowlist.json` |

## 算子文档现状（重写目标：算子深度文档见下节链接）

| 文档 | 内容 | 缺口 |
|---|---|---|
| `docs/dsl_operators_reference.md` | 白名单**分类枚举**（31 行） | 无公式/无逐算子可用性 |
| `cleaned_operators/docs/算子全览.md` | 259 个算子节，含公式(LaTeX)/示例 | **严重过期**（抬头 469/256，真实 1737/1421） |
| `cleaned_operators/docs/operators_catalog.md` | 1737 行机器审计表（元数据） | 无语义 |
| `docs/operator_core_specs.yaml` | **86 条**核心 spec（24 字段/算子） | 仅基础算子，1651 个无 spec |
| `docs/算子与导入教程.md` | import/DSL 写法/角色必读表 | 教程非参考 |
| `docs/FactorEngine完全指南.md` | 面向新手的模块总览 | 无算子细节 |
| `docs/operators_semantics.md` / `docs/daily_panel_operators.md` | 语法约定 / 部分算子语义 | 非逐算子 |

**算子深度参考手册：** `docs/OPERATORS_DEEP_REFERENCE.md`（逐算子：标签/公式/可用条件/讲解）。

## 运行时 / 物化 / 落盘

- `FactorEngine.__init__(backend, data_source, cache=None, *, run_mode=None, production_fallback_policy=None)`
- `compile(factor, *, pit_enforce=False, pit_forbid_forward_fill=False)`
- `run(factor, *, plan=None, analysis=None, input_dq_check=False, input_dq_strict=True,
  auto_warmup=False, trim_warmup=True, market=None, pit_enforce=False, ...) -> dict`
- `run_many(factors, *, perf=None, enable_cse=None, result_policy="return", sink=None,
  warmup_clusters=False, ...)` — 先算共享子树再各因子根（CSE）
- `run_many_stream(factors, *, sink, wave_size=None, sink_queue_bytes=None, perf=None, ...)`
  — 有界波次计算与异步结果交付。调用方必须显式提供 `sink_queue_bytes`，或在
  `PerfConfig.result_budget_bytes` 中提供同一内存预算；该预算必须计入任务总资源租约。
  不再为未声明预算隐式分配队列。用户 sink 缺少写回执/幂等保证时，任何异常
  都不会自动重放；中途失败仍需由目标端以 manifest/提交协议证明完整性。
- `materialize(factor, *, lake_root=None, factor_id=None, author=None, frequency=None,
  value_dtype="float32", write_target="local", staging_dataset="factor_lake_staging",
  storage_format="long", resume_materialize=False, ...)` — 还有 `materialize_from_config`、
  `materialize_sharded`、`materialize_matrix`、`materialize_incremental*` 等

**factor_lake 落盘流程**：`run` → `materialize_service.execute_materialize` → 写
`factor_lake_staging` → `lake_publish.publish_factor_lake()`（**人工审批门**：
`QUANT_PUBLISH_APPROVED=1` 或 `approve=True`，否则 `PublishNotApprovedError`）。
write_target 支持 local/staging/clickhouse/staging_clickhouse；另有 factor_matrix 物化。

## HTTP 服务（`factor-engine-serve`，默认 8088）

| 方法 | 路径 |
|---|---|
| GET | `/health` `/livez` `/readyz` `/metrics` `/factor-engine/operators` |
| POST | `/factor-engine/validate-spec`（不跑数） |
| POST | `/factor-engine/research/compute`（可 sync） |
| POST | `/factor-engine/production/compute`（恒异步、需 API key） |
| POST | `/factor-engine/production/materialize`（需 MATERIALIZE 权限） |
| POST | `/factor-engine/jobs/compute` `/factor-engine/jobs/materialize` |
| GET | `/factor-engine/jobs/{run_id}` `/factor-engine/jobs/{run_id}/artifacts` |
| POST | `/factor-engine/jobs/{run_id}/cancel` `/factor-engine/jobs/{run_id}/retry` |

OpenAPI 在 `:8088/docs`。认证：API key→principal，production 路由无条件认证；
jobstore 支持 JSON manifests 或 sqlite（`FACTOR_ENGINE_SERVICE_DURABLE_STORE`）。

## 版本与测试

- 版本 `0.3.1`（pyproject）；changelog `docs/changelog_shw.md`（第 1~38 版历史）。
- 测试：1145 个测试文件 / 14408 个测试函数（含 stress/bench）；`tests/` 120 个子目录，
  operators 425 文件居首、backend 136、runtime 91。

```bash
cd factor_engine && PYTHONPATH=. pytest tests/ -q
```

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（`data_access.clickhouse.panel` 等读数）。
- **被谁调用**：`alphaprobe`（`fe_bridge`）、`quant_evaluator`（FactorBatchProvider，stub）、
  `factor_optimizer`（canonical hash/validate_mutation/complexity）、`factor_assets`（身份 hash）、
  `quant_platform`（storage adapter，guarded）。

## 相关仓库

- **data_access** — 数据层（唯一读通道）
- **quant_evaluator / factor_optimizer / factor_assets** — 下游评估/寻优/入库
- **alphaprobe** — 上游挖掘（表达式→DSL）
