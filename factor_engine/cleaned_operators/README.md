# cleaned_operators — Runtime 算子库

本目录是 factor_engine 的 **唯一算子实现层**：`OperatorRegistry` 注册、`SeriesOperator.calculate()` 在 **宽表 panel**（index=时间, columns=标的）上执行。

- **DSL / 工厂**：[`api/`](../api/README.md)（`from api import rank` → `CleanedCall`）  
- **执行桥**：[`backend/cleaned_bridge.py`](../backend/cleaned_bridge.py)（MultiIndex ↔ panel）  
- **算子文档**：[`docs/`](docs/README.md)（全览、catalog、语义公式 — **与代码分离**）  
- **挖掘侧**：一般 **不 import 本目录**；查 [`docs/operators_catalog.md`](docs/operators_catalog.md) 与 [`_aliases.py`](_aliases.py) 即可。

> monorepo 外还有一份源码副本：[`quantsociety/cleaned_operators/`](../../../../cleaned_operators/)。改算子时请 **同步** 到 `factor_engine/cleaned_operators/`。

---

## 1. 与 `api` 的分工（不重复）

| 问题 | 找谁 |
|------|------|
| 公式怎么写、怎么 import | `api` |
| 算子叫什么、有哪些别名 | [`docs/operators_catalog.md`](docs/operators_catalog.md)、[`_aliases.py`](_aliases.py) |
| 算子含义与公式 | [`docs/算子全览.md`](docs/算子全览.md) |
| 算子怎么算（rolling、rank 语义） | **本目录** `*.py` 实现 |
| 字符串能否 parse | `api.operator_registry.build_dsl_allowlist()` |

```text
api.make_cleaned_call_factory("rank")(col("x"))
    → expr.CleanedCall
    → ir (op="rank")
    → backend → OperatorRegistry.get("rank").calculate(panel)
```

---

## 2. 目录结构

```text
cleaned_operators/
├── registry.py, base.py, base_polars.py   # 注册中心与基类
├── _aliases.py, _dedupe.py              # 别名与去重
├── _numpy_kernels.py, _rolling_fast.py   # 内部加速内核
├── common/                              # 共通运算（不绑特定数据域）
│   ├── elementwise.py                   # 四则、log、矩阵、clip …
│   ├── time_series.py                   # ts_mean、ts_corr、ts_decay_linear …
│   ├── shift_cum.py                     # 滞后、差分、累计
│   ├── cross_sectional.py               # rank、zscore、scale …
│   ├── group.py                         # group_neutralize、group_rank …
│   ├── data_cleaning.py                 # fillna、winsorize、window_* …
│   ├── statistics.py                    # 相关、回归、假设检验
│   └── polars_ops.py                    # polars 高频算子扩展
├── price_volume/ops.py                  # 收益、波动、夏普、beta …
├── technical/signal.py                  # MACD、RSI、trade_when …
├── fundamental/ops.py                   # ttm、yoy、quarter …
├── microstructure/ops.py                # 日内微观结构
└── docs/                                # 算子文档（与代码分离）
```

根目录仍保留 **兼容 shim**（如 `elementwise_math.py` → `common/elementwise.py`），旧 import 路径可继续使用。

### 2.1 企业级运行时（Phase 11–12）

| 能力 | 入口 |
|------|------|
| auto_warmup / RunWindow | `runtime/run_window.py`、`engine.run(auto_warmup=True)` |
| DQ Profile | `runtime/dq_profiles.yaml`、`config.dq.profile` |
| PIT enforce | `runtime/pit_audit.py`、`config.pit.enforce` |
| 物化 target | `local` / `staging` / `both` / `clickhouse` / `staging_clickhouse` |
| 双写对账/修复 | `run_pipeline.py reconcile dual-write [--repair] [--prefer-source staging]` |
| 生产 profile | `examples/profiles/prod.yaml`（staging）、`prod_clickhouse.yaml`（staging_clickhouse） |
| Intraday 日频聚合 | `runtime/intraday_aggregator.py`、`data_source.type: intraday_daily` |
| 微观算子（已实现） | `real_turnover_rate`、`micro_realized_vol`、`micro_spread`、`micro_amihud_hf`、`micro_mid_return`、`micro_bipower_var`、`micro_jump_indicator`、`micro_trade_imbalance`、`micro_vpin`、`micro_kyle_lambda` |

> `microstructure/ops.py` 底部 `*_stub` 函数为 **catalog 占位**，未 `@register_operator`；与已实现算子同名 stub 已移除。

测试门禁：`tests/test_enterprise_readiness.py`、`test_run_window.py`、`test_enterprise_materialize_paths.py`、`test_microstructure_ops.py`。

---

## 2.2 命名规范（canonical vs 别名）

| 类型 | canonical 示例 | 常见别名（仍可计算） |
|------|----------------|----------------------|
| 时序滚动 | `ts_mean`, `ts_var`, `ts_pct` | `SMA`, `m_var`, `returns`, `pct_change` |
| 截面 | `rank`, `zscore` | `CS_RANK`, `standardize`, `panel_zscore` |
| 分组中性 | `group_neutralize` | `neutralize`, `group_demean`, `industry_neutralize` |
| 裁剪 | `clip` | `cap`, `clamp`, `CLIP` |
| 扩展统计 | `expanding_mean`, `expanding_zscore` | `cum_avg`, `cum_standardize` |
| 技术指标 | `RSI`, `MACD`, `WMA` | `ts_rsi`, `ts_macd`, `ts_wma` |

新增算子时优先选用上表 canonical 风格；旧 DSL 名在 [`_aliases.py`](_aliases.py) 登记即可。

---

## 2.2 执行 backend（哪个快用哪个）

| backend | 说明 |
|---------|------|
| `auto` / `hybrid` | **推荐**：部分 SQL 子树 → Polars → Pandas |
| `sql` / `duckdb_sql` | SqlBackend：maximal SQL 子树 + Python fallback |
| `polars` | 算子层 auto 优先 polars（**320+ canonical**） |
| `pandas` | 默认全 Python |

**混合执行（中期架构）**：
- `planner/sql_lowerer.py` 将 PlanNode 切分为 SQL 可编译子树 + Python 段
- SQL 子树预计算为 `materialized_series`，不可编译部分走 pandas/polars
- `OperatorRegistry` 支持 `backend="sql"` 元数据（与 emitter 同步）
- `LongTableDataSource` / `long_table: true`：数据源保持长表；**panel-native 仍启用**，算子链内宽表中间态、根节点一次 stack

Polars 新增覆盖：`ADX/AROON/KAMA`、`Slope/ts_regression`、`sharpe_ratio`、信号算子、`group_*` 全簇、`ewm_*`、`cs_regression/cs_resid`、CAPM 簇等；**2026-07**：`polars_batch_mirror` 桥接至 **325** canonical。企业门禁 Polars ≥ **320** / SQL ≥ **39**；剩余 **36** intentional pandas-only（FFT/矩阵/随机/CDF-PDF）。

SQL 白名单 **42** 个算子（含 `ts_ema`、`group_winsorize`、`ts_beta`、`ts_mad`、`ts_rank`、`ewm_mean`）；多子树 **WITH CSE 批执行**。

**企业级门禁**：`tests/test_enterprise_readiness.py`（覆盖阈值、dedupe 契约、P0 双 backend、env bootstrap）。

**ClickHouse 环境变量**：见 monorepo 根目录 [`.env.example`](../../.env.example)（`CLICKHOUSE_*`）；`FactorEngine` 启动时自动加载 `.env`（不覆盖已有环境变量）。

- **`FACTOR_ENGINE_OPERATOR_BACKEND=auto`**：Polars 路径下算子 auto 选择
- **`data_source.type: clickhouse`**：ClickHouse 长表只读 + SQL 下推（需 `clickhouse-connect`）
- **`data_source.type: data_access`**：DuckDB 读 parquet + 可选 `duckdb_sql`/`auto` 算因子
- **ClickHouse 写入**：`FactorEngine.materialize(write_target="clickhouse"|"staging_clickhouse")`（推荐）；兼容入口 `materialize_clickhouse()` 委托同一实现
- **Parquet → CH ETL**：`load_parquet_to_panel_table()`

SQL 已支持（MVP）：`ts_mean/std/sum/max/min/delay/delta/pct/zscore`、`ts_corr`、`rank/zscore/scale/cs_demean/group_neutralize`、`where`、四则、`abs/log/exp/sqrt/clip`。

---

## 3. 数据形态约定

- factor_engine 中间结果为 **`(timestamp, instrument)` MultiIndex Series**。  
- `cleaned_bridge` 在 **panel-native** 模式下中间结果保持宽表，仅在根节点 stack；否则逐算子 unstack/stack。  
- `long_table: true` 时列读取走 Series，`load_column_panel()` 按需 unstack 并缓存。  
- **SQL 下推**：`ts_rank` / `ewm_mean` 已加入白名单与 emitter（Tier-1 生产常用）
- **不走 SQL 下推**：多输入量价结构算子（如 `vp_weighted_price`、`vpmacd`）与全部 `micro_*` 簇——由 Polars/Pandas 路径执行
- **`ts_std`**：样本标准差 `ddof=1`（与 pandas 默认一致）

---

## 4. 新增 / 修改算子

1. 在合适子目录添加 `SeriesOperator` 子类（共通 → `common/`，量价 → `price_volume/`，等），使用 `@register_operator(...)`。  
2. 在 [`_aliases.py`](_aliases.py) 增加 DSL 别名（若需要）。  
3. 运行测试：  
   ```bash
   cd factor_engine
   PYTHONPATH=. pytest tests/test_cleaned_operators_comprehensive.py -q
   ```  
4. 若语义变化，更新 [`docs/operator_doc_semantics.py`](docs/operator_doc_semantics.py) 并重跑 [`generate_operators_guide.py`](../scripts/generate_operators_guide.py)。  
5. Gateway **`operator_whitelist.json`** 与 `build_dsl_allowlist()` 对齐（评估侧）。

**注册后自动生效的路径**：`build_dsl_allowlist()` → DSL；新建 `PandasBackend()` → 执行 kernel。

---

## 5. stub 与未实现算子

- `microstructure/ops.py` 内部分 `*_stub` **普通函数** 可能 **未** `register_operator`，**不会**进入 DSL 白名单。  
- 仅 **catalog**、无 pandas runtime 的名字不会出现在 `build_dsl_allowlist()`。  
- 仅 **polars** backend 的算子同样不进 pandas DSL 白名单。

投递 manifest 时只使用 **`parse_expr` 能解析** 且 **本地试算不报错** 的算子名。

---

## 6. 延伸阅读

- [`docs/算子全览.md`](docs/算子全览.md)  
- [`docs/算子与导入教程.md`](../docs/算子与导入教程.md)  
- [`docs/operators_semantics.md`](../docs/operators_semantics.md)  
- [`api/README.md`](../api/README.md)
