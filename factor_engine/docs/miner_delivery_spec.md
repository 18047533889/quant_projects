# 自动化因子挖掘 — 统一交付规范

> **文档版本**：`miner_delivery_spec.v2.15`
> **契约版本**：`schema_version = disk.v1`（对接 AutoFactorEvaluation Gateway）
> **适用对象**：所有自动化因子挖掘算法组员（QuantaAlpha、CogAlpha、AlphaProbe 等）
> **效力**：全员挖掘产出 **唯一** 按本文投递。
> **执行引擎（2025 起）**：**A 股与美股统一走 factor_engine**（同一套 DSL + `cleaned_operators`）；仅数据源与 canonical 字段不同。`platforms.lqtp` / `lqtp_dsl` 为历史 LQTP 平台路径，新 campaign **不必安装**。
> **字段表约定：** 「具体能填的值」= **完整合法取值**；写表外值 Gateway 会拒。无固定枚举的字段会写明格式规则。

---

## 0. 你要做什么

### 0.1 三个词先分清

| 词                     | 是什么                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **campaign**     | **一次完整挖掘任务**（同一 `market` + `domain_root` + `universe_id` + 一套配置）。跑完一轮后，下面会有**多个因子**目录。 |
| **campaign_id**  | 这次任务的文件夹名；整场共用一份`config.json`。                                                                                          |
| **candidate_id** | **单个因子**的文件夹名；每个因子一份 `manifest.json`（公式、指标、描述等）。                                                       |

### 0.2 谁写、什么时候写

**组员不手填 JSON。** 在各自挖掘框架的 yaml / 环境变量里配好一场 campaign 的参数，**开跑后由框架导出器自动生成**下面两个文件；组员只做校验与同步 COS。

| 时机                       | 产出                                               | 谁写                                                  |
| -------------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| 挖掘**开跑**         | `config.json` 主体（market、universe、日期段等） | 框架导出器（从 experiment 配置映射）                  |
| 每出一个**合格因子** | `{candidate_id}/manifest.json`                   | 框架导出器（公式、metrics、description 来自当次挖掘） |
| 挖掘**结束**         | `config.json` 里的 `mining_run_stats`          | 框架导出器（汇总 LLM 模型 id、Token、时长）           |
| **投递**             | 整目录拷到`candidate_pool/` 并同步 COS           | 组员 / CI（不改正文 JSON 字段含义）                   |

框架尚未接 disk.v1 导出时，可临时用脚本从 `factor_pool` / `factorlib` **转换**为 §0.3 布局；字段含义仍以本文为准。

### 0.3 交付物

目录布局见 §1.1。只投递 **`config.json`（整场 1 份）** 与 **每个合格因子的 `manifest.json`**。
**不要投递：** 因子值 parquet/h5 · tier 目录 · `afv.json` · manifest 内嵌套 `config`（§3.3）。

字段与示例：`config.json` → §2；`manifest.json` → §3。CogAlpha 中间态 `python` 投递前须转为 **factor_engine `dsl`**（§4.3）。

### 0.4 流程

配 yaml（§0.5）→ 框架导出 JSON → **投递前校验**（§5.4）→ `candidate_pool/` → COS → AFV。

### 0.5 框架 yaml 示例

#### QuantaAlpha（A 股价量示例）

**主配置** `quantaalpha/configs/experiment_compare_ashare.yaml`（或 `generated/model_compare_ashare/experiment_{model}.yaml`）：

```yaml
data:
  source: ashare
  ashare_data_dir: data/a_share/lqtp_data
symbolic:
  backtest_config_path: configs/backtest_compare_ashare.yaml
  factor_library_path: data/factorlib/all_factors_library.json
protocol:                              # enabled: true 时 ProtocolExporter 导出
  enabled: true
  campaign_id: "quantaalpha_ashare_pv_202606151500"
  generator_name: "quantaalpha"
  generator_version: "1.0.0"
  mined_by: "zhangborui"                    # §2.3 枚举
  market: "ashare"
  universe_id: "A_SHARE_ALL_A_EX_ST"
  domain_root: "price_volume"
  domain: "price_volume"
  frequency_bucket: "daily"
  signal_structure: "cross_sectional"
  asset_class: "equity"
  operator_policy: "lqtp_pv_daily"         # 未译 lqtp_dsl 前可暂填 afv_us_pv_daily
  output_dir: "data/results/protocol"
  config_output_filename: "config.json"
  factors_output_filename: "factors.json"
```

**日期段** `quantaalpha/configs/backtest_compare_ashare.yaml`（由 `protocol` 引用）：

```yaml
dataset:
  segments:
    train: ["2014-01-01", "2021-12-31"]
    valid: ["2022-01-01", "2023-12-31"]
    test:  ["2024-01-01", "2026-06-25"]
```

**LLM 型号**（→ `mining_run_stats.llm_model`）：`model_compare_ashare_models.json` 的 `chat_model`，例如：

```json
{
  "name": "deepseek_v4_pro",
  "chat_model": "deepseek-v4-pro",
  "reasoning_model": "deepseek-v4-pro",
  "base_url": "https://api.b.ai/v1"
}
```

填 API id（`deepseek-v4-pro`），不要填内部 slug。参考：`configs/experiment.yaml`、`configs/experiment_compare_ashare.yaml`。

#### CogAlpha（A 股价量示例）

投递字段与算法超参分开：后者在 `mvp.yaml`，不进 `mining_config`。

**① 日期** `cogalpha-dev/configs/baseline.yaml`：

```yaml
dataset: company_all_a
split:
  train_start: 2014-01-01
  train_end:   2021-12-31
  valid_start: 2022-01-01
  valid_end:   2023-12-31
  test_start:  2024-01-01
  test_end:    2026-06-25
```

**② LLM** `cogalpha-dev/KEY.md`：

```ini
COGALPHA_LLM_BASE_URL=https://api.deepseek.com
COGALPHA_LLM_MODEL=deepseek-v4-flash
```

**③ 投递块** `configs/delivery.yaml`（或并入 `baseline.yaml`）：

```yaml
delivery:
  campaign_id: "cogalpha_ashare_pv_202606151430"
  generator_name: "cogalpha"
  generator_version: "2.1.0"
  mined_by: "zhangjiayin"
  market: "ashare"
  universe_id: "A_SHARE_ALL_A_EX_ST"
  domain_root: "price_volume"
  domain: "price_volume"
  frequency_bucket: "daily"
  signal_structure: "cross_sectional"
  asset_class: "equity"
  operator_policy: "lqtp_pv_daily"
  mining_scope:
    primary_tables: ["StockDailyBar", "StockCapitalDaily"]
    auxiliary_tables: ["StockList", "StockStatus", "StockIndustry", "Calendar"]
    forbidden_tables: ["StockBalance", "StockIncome", "StockCashFlow"]
  data_source:
    local: "data/a_share/lqtp_data/"
    cos: "cos://qs-cold/clean_data/ashare/lqtp_data/"
  output_dir: "~/quant_projects/data/factor_pools/candidate_pool"
```

**④ 算法超参** `configs/mvp.yaml`：不进 disk.v1（如 `max_generations`、`parent_pool_size`）。

#### yaml → JSON 对照

| yaml / 环境变量                                     | 写入 disk.v1                                     |
| --------------------------------------------------- | ------------------------------------------------ |
| `protocol.campaign_id` / `delivery.campaign_id` | `config.campaign_id` + 文件夹名                |
| `protocol.market` / `delivery.market`           | `config.market` + 每个 `manifest.market`     |
| `dataset.segments.*` / `split.*`                | `config.mining_config.*_period`                |
| `protocol.universe_id` / `delivery.universe_id` | `config.universe_id` + manifest 拷贝           |
| `chat_model` / `COGALPHA_LLM_MODEL`             | `mining_run_stats.llm_model`                   |
| 框架内因子表达式、回测 IC                           | 每个`manifest.formula` · `manifest.metrics` |
| LLM rationale                                       | `manifest.description`                         |

---

## 1. 投递什么、放哪里

### 1.1 目录结构

- `{campaign_id}/` 直接挂在 `candidate_pool/` 下；**不要**再建 `ashare/`、`us_stock/` 等分类子目录。
- 一场 campaign 可有多个 `{candidate_id}/`；不同批次 → **新开** `{campaign_id}/`。

```text
candidate_pool/
├── cogalpha_ashare_pv_202606151430/
│   ├── config.json
│   ├── cogalpha_20260615150530_a1b2c3d4/manifest.json
│   └── cogalpha_20260615153218_f7e8d9c0/manifest.json
└── quantaalpha_us_stock_pv_202606151500/
    ├── config.json
    └── quantaalpha_20260615160530_b2c3d4e5/manifest.json
```

### 1.2 路径（本地 + COS）

| 环境           | 根路径                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------- |
| **本地** | `~/quant_projects/data/factor_pools/candidate_pool/`                                      |
| **COS**  | `cos://qs-cold/candidate_pool/candidate/{campaign_id}/...`（比本地多一层 `candidate/`） |

流程：本地落盘 → 校验 → 同步 COS → 通知 AFV。

### 1.3 命名

| 对象             | 格式                                                       |
| ---------------- | ---------------------------------------------------------- |
| `campaign_id`  | `{generator_name}_{market}_{domain_slug}_{YYYYMMDDHHMM}` |
| `candidate_id` | `{generator_name}_{YYYYMMDDHHMMSS}_{hash8}`              |

**`domain_slug`（与 `domain_root` 一一对应）：**

| `domain_slug` | `domain_root`    | 说明                                |
| --------------- | ------------------ | ----------------------------------- |
| `pv`          | `price_volume`   | 价量 / 收益（当前主力）             |
| `fundamental` | `fundamental`    | 基本面（财报 / 估值 / 指标 / 分红） |
| `alt`         | `alternative`    | 另类数据（股东 / 持仓等）           |
| `micro`       | `microstructure` | 微观结构（分钟 / L2）               |

**`market` 只能填：** `ashare` · `us_stock`

**`hash8` 算法（内容哈希，与公式绑定）：**

```python
import hashlib
normalized = " ".join(formula.split())  # 压缩空白
payload = f"{normalized}|{universe_id}|{frequency_bucket}"
hash8 = hashlib.sha256(payload.encode()).hexdigest()[:8]
# candidate_id = f"{generator_name}_{YYYYMMDDHHMMSS}_{hash8}"
```

### 1.6 `domain_root` 与 `domain`（数据子域）

**`domain_root` 只能填以下之一：**

| 值                 | 含义                                 |
| ------------------ | ------------------------------------ |
| `price_volume`   | 价量（当前主力）                     |
| `fundamental`    | 基本面                               |
| `alternative`    | 另类数据                             |
| `microstructure` | 微观结构                             |
| `mixed`          | 混合类型（需在`description` 说明） |

**`domain` 与 `domain_root` 配套取值（不是 SSH 域名）：**

| `domain_root`    | `domain` 只能填                              |
| ------------------ | ---------------------------------------------- |
| `price_volume`   | `price_volume`                               |
| `fundamental`    | `fundamental` · `financial_statement`     |
| `alternative`    | `alternative` · `sentiment`               |
| `microstructure` | `microstructure` · `order_flow`           |
| `mixed`          | 与`domain_root` 相同，或在上表范围内组合说明 |

不确定时 **`domain` 与 `domain_root` 填相同**；不同 `domain_root` **禁止**混在同一 campaign。表模板见 §2.5。

---

## 2. `config.json`（campaign 级）

### 2.1 示例（A 股价量 + CogAlpha 本地）

```json
{
  "schema_version": "disk.v1",
  "campaign_id": "cogalpha_ashare_pv_202606151430",
  "generator_name": "cogalpha",
  "generator_version": "2.1.0",
  "market": "ashare",
  "universe_id": "A_SHARE_ALL_A_EX_ST",
  "signal_structure": "cross_sectional",
  "asset_class": "equity",
  "frequency_bucket": "daily",
  "domain_root": "price_volume",
  "domain": "price_volume",
  "mining_scope": {
    "primary_tables": ["StockDailyBar", "StockCapitalDaily"],
    "auxiliary_tables": ["StockList", "StockStatus", "StockIndustry", "Calendar"],
    "forbidden_tables": ["StockBalance", "StockIncome", "StockCashFlow"]
  },
  "operator_policy": "lqtp_pv_daily",
  "data_source": {
    "local": "data/a_share/lqtp_data/",
    "cos": "cos://qs-cold/clean_data/ashare/lqtp_data/"
  },
  "mined_by": "zhangjiayin",
  "created_at": "2026-06-15T14:30:00Z",
  "mining_config": {
    "train_period": ["2014-01-01", "2021-12-31"],
    "valid_period": ["2022-01-01", "2023-12-31"],
    "test_period": ["2024-01-01", "2026-06-25"]
  },
  "mining_run_stats": {
    "llm_model": "deepseek-v4-flash",
    "llm_provider": "deepseek",
    "llm_base_url": "https://api.deepseek.com",
    "token_usage": {
      "prompt_tokens": 892400,
      "completion_tokens": 215600,
      "total_tokens": 1108000
    },
    "started_at": "2026-06-15T14:30:00Z",
    "finished_at": "2026-06-15T17:52:18Z",
    "duration_seconds": 12138,
    "candidates_generated": 48,
    "candidates_submitted": 12
  }
}
```

### 2.2 示例（美股价量 + QuantaAlpha）

```json
{
  "schema_version": "disk.v1",
  "campaign_id": "quantaalpha_us_stock_pv_202606151500",
  "generator_name": "quantaalpha",
  "generator_version": "1.0.0",
  "market": "us_stock",
  "universe_id": "US_MASSIVE_ALL",
  "signal_structure": "cross_sectional",
  "asset_class": "equity",
  "frequency_bucket": "daily",
  "domain_root": "price_volume",
  "domain": "price_volume",
  "mining_scope": {
    "primary_tables": ["StockDailyBar", "FactReturnsDaily", "PanelDaily"],
    "auxiliary_tables": ["DimCalendar", "DimSecurityMaster", "DimTickerMap", "DimTickerAlias"],
    "forbidden_tables": []
  },
  "operator_policy": "afv_us_pv_daily",
  "data_source": {
    "local": "data/us_stock/massive_data/",
    "cos": "cos://qs-cold/clean_data/us_stock/massive_data/"
  },
  "mined_by": "zhangborui",
  "created_at": "2026-06-15T15:00:00Z",
  "mining_config": {
    "train_period": ["2014-01-01", "2021-12-31"],
    "valid_period": ["2022-01-01", "2023-12-31"],
    "test_period": ["2024-01-01", "2026-06-25"]
  },
  "mining_run_stats": {
    "llm_model": "glm-5.2",
    "llm_provider": "bai",
    "llm_base_url": "https://api.b.ai/v1",
    "token_usage": {
      "prompt_tokens": 1250000,
      "completion_tokens": 380000,
      "total_tokens": 1630000
    },
    "started_at": "2026-06-15T15:00:00Z",
    "finished_at": "2026-06-15T18:45:22Z",
    "duration_seconds": 13522,
    "candidates_generated": 86,
    "candidates_submitted": 24
  }
}
```

### 2.3 字段表

| 字段                  | 必填 | 类型   | 具体能填的值                                                                                                                                                                                                                   |
| --------------------- | :--: | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schema_version`    |  ✅  | string | **只能填：** `disk.v1`                                                                                                                                                                                                 |
| `campaign_id`       |  ✅  | string | 小写`[a-z0-9_]`；**必须** = 上级文件夹名                                                                                                                                                                               |
| `generator_name`    |  ✅  | string | **只能填：** `quantaalpha` · `cogalpha` · `alphacfg` · `alphaagentevo` · `alphaprobe` · `alphasage` · `quantevolver` · `alphaqcm` · `rdagent` · `evoalpha` · `factorminer` · `manual` |
| `generator_version` |  ✅  | string | semver`MAJOR.MINOR.PATCH`；❌ 禁止 `latest` / `dev`                                                                                                                                                                      |
| `market`            |  ✅  | string | **只能填：** `ashare` · `us_stock`                                                                                                                                                                                  |
| `universe_id`       |  ✅  | string | 须与`market` 配对，见 §2.4                                                                                                                                                                                                  |
| `signal_structure`  |  ✅  | string | **只能填：** `cross_sectional` · `time_series` · `panel`                                                                                                                                                         |
| `asset_class`       |  ✅  | string | **只能填：** `equity`（当前数据仅支持此项）                                                                                                                                                                            |
| `frequency_bucket`  |  ✅  | string | **只能填：** `daily` · `minute`（❌ 禁止 `1d` 等别名；manifest 须显式填写）                                                                                                                                       |
| `domain_root`       |  ✅  | string | 见 §1.6                                                                                                                                                                                                                       |
| `domain`            |  ✅  | string | 见 §1.6 数据子域；不确定时与`domain_root` 填相同                                                                                                                                                                            |
| `mining_scope`      |  ✅  | object | 见 §2.5                                                                                                                                                                                                                       |
| `operator_policy`   |  ✅  | string | 见 §2.6                                                                                                                                                                                                                       |
| `data_source`       | 可选 | object | 仅 campaign config；见 §2.8（AFV 不读）                                                                                                                                                                                       |
| `mined_by`          |  ✅  | string | 见下方枚举                                                                                                                                                                                                                     |
| `created_at`        |  ✅  | string | ISO 8601 UTC；与`campaign_id` 末段 `YYYYMMDDHHMM` 同一分钟                                                                                                                                                                 |
| `mining_config`     |  ✅  | object | 只能含 §2.7 三个键                                                                                                                                                                                                            |
| `mining_run_stats`  | 建议 | object | 整场 LLM / 耗时统计，见 §2.9（AFV 不读；投递前填）                                                                                                                                                                            |

**`mined_by` 只能填：** `zhangborui` · `zhangjiayin` · `lizhuo` · `yangyisi` · `xiangyurui` · `yeyihan`

---

### 2.4 `universe_id`（须与 `market` 配对）

**`market = ashare` 只能填：**

| `universe_id`         | 含义                    | 本地/COS 数据               | 何时填                                            |
| ----------------------- | ------------------------- | --------------------------- | ------------------------------------------------- |
| `A_SHARE_LQTP`        | 全 A（历史名，仍可用）    | gRPC / ClickHouse 或本地    | **新 campaign 建议改** `A_SHARE_ALL_A_EX_ST` + factor_engine |
| `A_SHARE_ALL_A_EX_ST` | 本地 parquet 全 A 剔 ST | `data/a_share/lqtp_data/` | **A 股默认**；公式 `expression_type: dsl`，factor_engine 执行 |
| `CSI300`              | 沪深 300 成分           | 成分表                      | 仅 CSI300 内挖掘                                  |

❌ 没有 `A_SHARE_ALL`；本地全 A 剔 ST 用 **`A_SHARE_ALL_A_EX_ST`**。

**`market = us_stock` 只能填：**

| `universe_id`        | 含义                                 | 本地/COS 数据                   | 何时填                                  |
| ---------------------- | ------------------------------------ | ------------------------------- | --------------------------------------- |
| `US_MASSIVE_ALL`     | Massive 全量日线（~1.2万 ticker/天） | `data/us_stock/massive_data/` | **当前默认**；QuantaAlpha 本地 PV |
| `US_MAIN_STOCKS_PIT` | PIT 可投资子集                       | 从 massive 再筛                 | 非下载全量；勿误标                      |
| `US_SP500`           | 标普 500 成分                        | 成分表                          | 仅 SP500 内挖掘                         |

> `universe_id` = 这场挖掘的**股票池定义**，不是 COS 下载范围。

---

### 2.5 `mining_scope`

| 子字段               | 必填 | 说明                               |
| -------------------- | :--: | ---------------------------------- |
| `primary_tables`   |  ✅  | 公式**主信号**只能引用这些表 |
| `auxiliary_tables` | 建议 | 仅 filter / neutralize             |
| `forbidden_tables` | 建议 | 本场禁止引用                       |

**`primary_tables` 推荐模板（按 `domain_root`）：**

> **全量字段清单**（每张表每一列的 physical / canonical / **signal_domain** / **formula_usage**）：[`factor-pool-standard/enums/canonical_data_fields.json`](../../factor-pool-standard/enums/canonical_data_fields.json)（v3，由 [`../scripts/build_canonical_fields.py`](../scripts/build_canonical_fields.py) 生成）。
> **分域约束与可用主信号列索引**：同文件 `domain_scopes` 段 + [`factor-pool-standard/enums/domain_roots.yaml`](../../factor-pool-standard/enums/domain_roots.yaml)。
> 人类可读索引：[`canonical_data_fields.md`](canonical_data_fields.md)。

**硬规则（分域挖掘）：**

1. **一场 campaign 只有一个 `domain_root`**，不得混挖价量 + 基本面 + 另类。
2. 公式**主信号**只能引用该 `domain_root` 下 `signal_domain` 相同的列（见 JSON 每列字段）。
3. `primary_tables` 之外的逻辑表不得出现在主信号公式中；`auxiliary_tables` 仅用于 filter / neutralize。
4. `formula_usage=forbidden`（metadata/key）**禁止**出现在 `formula`；`filter_only` 列仅用于 filter 字符串。
5. **美股** `close` / `open` 等 OHLCV 在 `StockDailyBar` 上为 **`primary_signal`**（小写 canonical，勿写 `StockDailyBar.Close`）。
6. **A 股** 价量公式用 factor_engine **canonical 别名**（`close` / `volume` 等）；勿写 `StockDailyBar.Close` 表前缀（见 §4.2）。

| `domain_root`    | A 股`primary_tables`                                                                                                         | 美股`primary_tables`                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `price_volume`   | `StockDailyBar` · `StockCapitalDaily`                                                                                     | `StockDailyBar` · `FactReturnsDaily` · `PanelDaily` |
| `fundamental`    | `StockBalance` · `StockIncome` · `StockCashFlow` · `StockIndicator` · `StockValuationDaily` · `StockDividend` | —                                                          |
| `alternative`    | `StockTopTenShareholder` · `StockTopTenFloatShareholder`                                                                  | —                                                          |
| `microstructure` | `StockMinuteBar` · `StockTransaction`（本地未同步）                                                                       | —                                                          |

**`auxiliary_tables` 推荐：**

| `market`   | 表名                                                                               |
| ------------ | ---------------------------------------------------------------------------------- |
| `ashare`   | `StockList` · `StockStatus` · `StockIndustry` · `Calendar`              |
| `us_stock` | `DimCalendar` · `DimSecurityMaster` · `DimTickerMap` · `DimTickerAlias` |

### 2.6 `operator_policy`（算子白名单策略名）

| 值                  | 适用                   | 算子来源                                                        |
| ------------------- | ---------------------- | --------------------------------------------------------------- |
| `afv_us_pv_daily` | **A 股 / 美股 PV + `dsl`** | factor_engine `build_dsl_allowlist()` / [`dsl_allowlist.json`](dsl_allowlist.json) |
| `lqtp_pv_daily`   | 历史 A 股 manifest 兼容 | **与上相同**（标签遗留；新 campaign 请填 `afv_us_pv_daily`） |
| `lqtp_fs_event`   | 历史 A 股财报 manifest  | factor_engine 基本面算子白名单（`ttm` / `quarter` / `yoy` 等） |

**校验（投递前必跑，A 股 / 美股同一套）：**

| 步骤 | 命令                                                                                                                                         |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 字段 / domain | `python3 ~/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py path/to/manifest.json`（或 campaign 目录 `--recursive`） |
| **DSL 语法** | `cd ~/quant_projects/factor_engine && PYTHONPATH=. python scripts/validate_delivery_formula.py --manifest path/to/manifest.json` |

算子白名单 **唯一来源**：`build_dsl_allowlist()` / [`dsl_allowlist.json`](dsl_allowlist.json)。**不要**再跑 `platforms.lqtp.operators check`（LQTP 平台已非必需路径）。

### 2.7 `mining_config`（三段日期，闭区间 `YYYY-MM-DD`）

| 键               | 统一默认值                       |
| ---------------- | -------------------------------- |
| `train_period` | `["2014-01-01", "2021-12-31"]` |
| `valid_period` | `["2022-01-01", "2023-12-31"]` |
| `test_period`  | `["2024-01-01", "2026-06-25"]` |

❌ 禁止把 `max_depth` · `population_size` 等算法参数塞进 `mining_config`（放各框架自有 yaml）。

### 2.8 `data_source`（可选）

```json
"data_source": {
  "local": "data/us_stock/massive_data/",
  "cos": "cos://qs-cold/clean_data/us_stock/massive_data/"
}
```

仅 campaign `config.json`；AFV **不读**。`local` 相对仓库根目录。

### 2.9 `mining_run_stats`（建议投递前填写）

只写在 campaign `config.json`；AFV **不读**。记录 LLM 型号、Token、时长（便于对比成本）。

| 子字段                                                  | 必填 | 说明                                                                                                  |
| ------------------------------------------------------- | :--: | ----------------------------------------------------------------------------------------------------- |
| `llm_model`                                           |  ✅  | OpenAI-compatible API 的`model` 字符串（如 `deepseek-v4-pro`）；禁止 `chat` · `gpt-4` 等泛称 |
| `llm_provider` / `llm_base_url`                     | 建议 | 供应商与 Base URL                                                                                     |
| `token_usage.*`                                       |  ✅  | prompt / completion / total（total = 前两者之和）                                                     |
| `started_at` / `finished_at` / `duration_seconds` |  ✅  | ISO 8601 UTC；`duration_seconds = int(finished - started)`                                          |
| `candidates_generated` / `candidates_submitted`     | 建议 | 候选总数 / 最终投递数                                                                                 |

**配置来源：** CogAlpha → `KEY.md` 的 `COGALPHA_LLM_MODEL`；QuantaAlpha → `model_compare_ashare_models.json` 的 `chat_model`（非内部 slug）。

**多模型混用：** `llm_model` 填主跑 API id；轮换多个模型时加 `llm_models_used` 数组，`token_usage` 填合计。

| 内部 slug           | **`llm_model`（API id）** |
| ------------------- | --------------------------------- |
| `deepseek_v4_pro` | `deepseek-v4-pro`               |
| `glm_5_2`         | `glm-5.2`                       |
| `minimax_3`       | `minimax-m3`                    |
| `gpt_5_4`         | `gpt-5.4`                       |
| `claude_code_4_6` | `claude-opus-4.6`               |

---

## 3. `manifest.json`（因子级）

公共字段从 campaign `config.json` 拷贝；Gateway 可补全 manifest 缺失项（如 `signal_structure`）。

### 3.1 示例

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "quantaalpha_20260615150530_a1b2c3d4",
  "campaign_id": "quantaalpha_us_stock_pv_202606151500",
  "formula": "ts_covariance(rank(close - open), rank(close), 5) / (ts_std(close, 5) + 1e-8)",
  "expression_type": "dsl",
  "generator_name": "quantaalpha",
  "generator_version": "1.0.0",
  "mined_by": "zhangborui",
  "market": "us_stock",
  "universe_id": "US_MASSIVE_ALL",
  "domain_root": "price_volume",
  "domain": "price_volume",
  "frequency_bucket": "daily",
  "born_timestamp": "2026-06-15T15:05:30Z",
  "metrics": {
    "train": { "ic": 0.051, "icir": 2.38, "rank_ic": 0.048 },
    "valid": { "ic": 0.032, "icir": 1.85 },
    "test": { "ic": 0.028, "icir": 1.62 }
  },
  "description": "开盘缺口与收盘 rank 的 5 日协方差，经波动率标准化，捕捉短期价量联动。",
  "generation_stats": {
    "llm_model": "glm-5.2",
    "llm_provider": "bai",
    "token_usage": {
      "prompt_tokens": 4200,
      "completion_tokens": 980,
      "total_tokens": 5180
    },
    "generation_duration_seconds": 12
  }
}
```

### 3.2 顶层字段

| 字段                                                                               |    必填    | 具体能填的值                                                                                                                                                                     |
| ---------------------------------------------------------------------------------- | :---------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                                                                 |     ✅     | `disk.v1`                                                                                                                                                                      |
| `candidate_id`                                                                   |     ✅     | = 文件夹名；`{generator}_{YYYYMMDDHHMMSS}_{hash8}`                                                                                                                             |
| `campaign_id`                                                                    |     ✅     | 与所属`config.json` 完全一致                                                                                                                                                   |
| `formula`                                                                        |     ✅     | 单行 DSL；算子/字段符合 §4                                                                                                                                                      |
| `expression_type`                                                                |    建议    | `dsl` · `lqtp_dsl` · `python`（见 §4 各框架说明）                                                                                                                       |
| `generator_name` / `generator_version`                                         |     ✅     | 与 campaign config 一致                                                                                                                                                          |
| `mined_by`                                                                       |     ✅     | 与 campaign config 一致                                                                                                                                                          |
| `market` / `universe_id` / `domain_root` / `domain` / `frequency_bucket` |     ✅     | 与 campaign config 一致；`domain_root` 填数据域（如 `price_volume`），**不是** `market`；`frequency_bucket` **必须显式**写 `daily`（缺省 AFV 会用 `1d`） |
| `signal_structure` / `asset_class`                                             |    可选    | 可只在 campaign config 里写；manifest 缺失时 Gateway 从 campaign config 补                                                                                                       |
| `description`                                                                    |     ✅     | 非空，≤800 字；建议 LLM 根据 formula 生成                                                                                                                                       |
| `metrics`                                                                        |    建议    | §3.4                                                                                                                                                                            |
| `born_timestamp`                                                                 |    建议    | ISO 8601 UTC                                                                                                                                                                     |
| `config`                                                                         | ❌ 默认不写 | 见 §3.3                                                                                                                                                                         |
| `iteration_id` / `batch_id` / `depth` / `parent_local_hashes`              |    可选    | 迭代追溯                                                                                                                                                                         |
| `generation_stats`                                                               |    可选    | 单因子 LLM 消耗，见 §3.5；框架能拆则填，不能拆可只写 campaign 级`mining_run_stats`                                                                                            |

**Gateway 最低要求（AFV）：** `schema_version`、`candidate_id`、`campaign_id`、`formula`、`domain_root`、`frequency_bucket`（其余可由 campaign `config.json` 补）。

### 3.3 manifest 内嵌套 `config`

**不要写。** 公共字段只在 campaign `config.json` + manifest 顶层；若导出器仍写嵌套块，须与二者完全一致。

### 3.4 `metrics`

顶层键 **`train` · `valid` · `test`** 三段都要有（分别对应 `mining_config` 的 `train_period` / `valid_period` / `test_period`）。

每段可选键：`ic` · `icir` · `rank_ic` · `turnover`（number）。

### 3.5 `generation_stats`（可选）

单因子 LLM 消耗；字段同 §2.9。框架无法拆分时只填 campaign 级 `mining_run_stats` 即可。

---

## 4. 公式、字段、算子（按数据对齐）

> **字段四层命名**（逻辑表 / manifest 公式列 / parquet 物理列 / YAML 映射）：[`canonical_data_fields.md`](canonical_data_fields.md)（机器可读 JSON 同目录）。
> **多框架对接**：[`算子与导入教程.md`](算子与导入教程.md) · 代码入口 [`../api/mining_integration.py`](../api/mining_integration.py)。

### 4.1 美股 `dsl`（`US_MASSIVE_ALL`）

**方言：** AFV **`factor_engine` DSL**（蛇形小写函数名，如 `ts_mean` / `rank`）。
**不要**以 QuantaAlpha 内置算子表为准——该框架白名单尚未与 AFV 对齐，且命名/覆盖与投递标准不一致。

**主表字段（公式内用小写列名，对应 `StockDailyBar` / `PanelDaily`）：**

> 完整映射（含 Massive SIP 短列名 `o/c/v` ↔ 清洗层 `Open/Close/Volume`）见 [`canonical_data_fields.md`](canonical_data_fields.md) §2.1。

| 列名                                           | 说明                                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `open` `high` `low` `close` `volume` | OHLCV（原始价量；复权用`close * adj_factor` 等组合）                                |
| `adj_factor`                                 | 后复权因子（parquet 列`AdjFactor`）                                                 |
| `ret_price` `ret_total`                    | 来自`FactReturnsDaily` / `PanelDaily`                                             |
| `transactions`                               | 成交笔数（Massive`day_aggs_v1` 物理列；本地 `StockDailyBar` 需 ETL merge 后才有） |

**尚无物理列、常用公式组合代替（待 ETL 派生列登记）：**

| 概念              | 推荐写法（DSL）                                         |
| ----------------- | ------------------------------------------------------- |
| 昨收`pre_close` | `delay(close, 1)`                                     |
| 日收益            | `close / delay(close, 1) - 1` 或直接用 `ret_price`  |
| 成交额`amount`  | `close * volume`                                      |
| VWAP 近似         | `(high + low + close) / 3`（真 VWAP 需分钟聚合）      |
| 隔夜收益          | `(open - delay(close, 1)) / (delay(close, 1) + 1e-9)` |

**字段引用写法（挖掘侧标准）：** 公式字符串里 **直接写列名**，**不要**包 `col()`：

```text
rank(ts_mean(close, 20))
close - open
```

`col("close")` 仍合法，与裸写 `close` 等价；**新产出统一用裸字段名**。详见 [`算子与导入教程.md`](算子与导入教程.md) §3。

#### 公式写法规则（美股 PV 日频）

| 规则                      | 说明                                                                     | 示例                                              |
| ------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| **字段 = 裸标识符** | 小写蛇形名；字母/数字/下划线；**不能**以数字开头                   | `close` `volume` `ret_price` `adj_factor` |
| **算子 = 函数调用** | 蛇形小写；须在 Gateway 白名单内                                          | `ts_mean(x, 20)` `rank(x)` `delay(x, 5)`    |
| **窗口 `d`**      | 正整数，表示**bar 数**（日频即交易日根数）                         | `ts_mean(close, 20)` 里的 `20`                |
| **常数**            | 直接写数字                                                               | `1e-9` `0.01` `2`                           |
| **四则**            | `+ - * /` 或 `add`/`subtract`/…                                   | `(high - low) / (close + 1e-9)`                 |
| **比较**            | 单段比较，**不支持** `a < b < c`                                 | `close > open`                                  |
| **逻辑**            | 必须用`and_` `or_` `not_`，**不能**写 `and`/`or`/`not` | `if_else(close > open, close, open)`            |
| **字符串参数**      | 关键字用引号                                                             | `ts_macd(close, line="hist")`                   |
| **单行**            | `formula` 只能一行                                                     | ❌ 不要换行、不要分号                             |
| **字段来源**        | 只能引用本场`mining_scope.primary_tables` 里有的列                     | PV 场见上表；勿引用 forbidden 表                  |

**怎么区分字段和算子：** 名字后跟 `(` 且出现在白名单里 → **算子**；单独出现的标识符（如 `close`、`volume`）→ **字段**。

**除 OHLCV 外常用字段（仍须数据里有该列）：**

| 字段                        | 典型用途                      |
| --------------------------- | ----------------------------- |
| `ret_price` `ret_total` | 收益、动量、反转              |
| `adj_factor`              | 复权：`close * adj_factor`  |
| `transactions`            | 成交笔数（日 K 扩展列，若有） |

**禁止写法：**

| ❌ 不要写                         | 原因                                                               |
| --------------------------------- | ------------------------------------------------------------------ |
| `TS_MEAN` `$close` `#close` | QuantaAlpha 另一套语法                                             |
| `*_stub` `vec_avg`            | 不在白名单                                                         |
| `and` `or` `not`            | Python 关键字，用`and_` 等                                       |
| `StockDailyBar.Close`           | 美股 DSL 用**`close`**，不要表前缀（A 股 LQTP 另论 §4.2） |
| 未来函数                          | 如`delay(close, -1)` 等                                          |

#### 因子长什么样（示例，`formula` 单行字符串）

**价量动量 / 反转**

```text
rank(ts_mean(close, 20))
rank(close / delay(close, 1) - 1)
rank(ts_mean(close, 5) - ts_mean(close, 60))
zscore(close / delay(close, 20) - 1)
```

**价量 + 成交量**

```text
rank(volume / (ts_mean(volume, 20) + 1e-9))
rank(ts_delta(volume, 1))
rank(((high - low) / (close + 1e-9)) * volume)
```

**形态 / 区间**

```text
rank((open - delay(close, 1)) / (delay(close, 1) + 1e-9))
rank((close - low) / ((high - low) + 1e-9))
rank(high - low)
```

**波动 / 技术（多列算子要带齐 high/low/close 等）**

```text
rank(ts_std(close / delay(close, 1) - 1, 20))
rank(ts_rsi(close, 14))
rank(ts_macd(close, line="hist"))
rank(ts_atr(high, low, close, 14))
```

**截面变换**

```text
zscore(close / delay(close, 1) - 1)
winsorize(ret_price, lower=0.01, upper=0.99)
rank(protected_div(close - open, delay(close, 1)))
```

**条件 / 清洗**

```text
if_else(volume > 0, rank(close - open), 0)
rank(protected_div(close - open, delay(close, 1)))
trade_when(volume > ts_mean(volume, 20), rank(close - open), 0)
```

**manifest 片段示例：**

```json
{
  "formula": "rank(ts_mean(close, 20))",
  "expression_type": "dsl",
  "market": "us_stock",
  "frequency_bucket": "daily",
  "domain_root": "price_volume"
}
```

> 本地自检（在 `factor_engine` 根目录）：
>
> ```bash
> PYTHONPATH=. python scripts/validate_delivery_formula.py "rank(ts_mean(close, 20))"
> PYTHONPATH=. python ../../factor-pool-standard/scripts/check_manifest_fields.py path/to/manifest.json
> ```

**算子（权威来源 — 勿手写穷举）：**

| 资源                                                                                    | 用途                                                              |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `build_dsl_allowlist()`                                                               | Gateway / 本地校验用的**完整可解析名**（`col` + ~512 名） |
| [`../cleaned_operators/operators_catalog.md`](../cleaned_operators/operators_catalog.md) | 分类、别名、实现状态                                              |
| [`dsl_operators_reference.md`](dsl_operators_reference.md)                               | **白名单分类 + 待迁移算子清单**                             |
| [`dsl_allowlist.json`](dsl_allowlist.json)                                               | 机器可读白名单                                                    |
| [`operators_semantics.md`](operators_semantics.md)                                       | 参数语义、列依赖、DSL 限制                                        |
| [`算子与导入教程.md`](算子与导入教程.md)                                                 | 挖掘组 import / 本地校验                                          |

**代表算子（已在白名单，可投递）** — 完整列表以 [`dsl_allowlist.json`](dsl_allowlist.json) / `build_dsl_allowlist()` 为准；分类见 [`dsl_operators_reference.md`](dsl_operators_reference.md)：

| 类别     | 代表算子                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 算术     | `add` · `subtract` · `multiply` · `divide` · `abs` · `log` · `sign` · `sqrt` · `power` · `sin` · `cos` · `exp`                                     |
| 逻辑     | `if_else` · `and_` · `or_` · `not_`（**不能**写 `and`/`or`/`not`）                                                                                             |
| 截面     | `rank` · `zscore` · `normalize` · `quantile` · `scale` · `winsorize` · `neutralize`                                                                               |
| 时序     | `ts_mean` · `ts_std` / `ts_std_dev` · `ts_sum` · `ts_max` · `ts_min` · `delay` / `ts_delay` · `ts_delta` · `ts_corr` · `ts_rank` · `ts_decay_linear` |
| 分组     | `group_rank` · `group_neutralize` · `group_zscore` · `group_mean`                                                                                                          |
| 清洗     | `protected_div` · `protected_log` · `protected_sqrt` · `nan_to_num`                                                                                                        |
| 信号     | `trade_when` · `hump_decay`                                                                                                                                                      |
| 技术指标 | `SMA` · `EMA` · `ts_rsi` · `ts_macd` · `ts_atr` · `ts_adx` · `ts_obv` · `ts_cci` · …                                                                         |

**禁止投递（当前不在白名单）：**
`pasteurize` · `tail` · `bucket` · `hump` · `ts_step` · `ts_sma`/`ts_ema`（请用 **`SMA`/`EMA`**）· `ts_donchian` · `orthogonalize` · `vec_avg` · `vec_sum` · 全部 `*_stub` — 详见 [`dsl_operators_reference.md`](dsl_operators_reference.md) §4。

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python scripts/export_dsl_allowlist.py   # → docs/dsl_allowlist.json
```

**禁止：** 未来函数 · 白名单外算子 · 多行公式 · 引用 A 股表名 · QuantaAlpha 的 `TS_MEAN`/`$close` 等另一套语法

### 4.2 A 股 factor_engine `dsl`（`A_SHARE_*` universe）

**与美股相同：** 公式必须是 **factor_engine DSL**（`ts_mean` / `rank` / `col('close')` 或裸字段名 `close`，见 [`算子与导入教程.md`](算子与导入教程.md)）。`expression_type` 填 **`dsl`**（历史 manifest 若标 `lqtp_dsl`，语法校验同等对待，建议新投递改为 `dsl`）。

**价量主表 `StockDailyBar` 常用列：**

> A 股 parquet 物理列 ↔ manifest 公式别名见 [`canonical_data_fields.md`](canonical_data_fields.md) §3.1。**执行在 factor_engine**（本地 panel / AFV），不在 LQTP gRPC。

| 全限定名                         | 公式中写法（factor_engine） | 说明                              |
| -------------------------------- | --------------------------- | --------------------------------- |
| `StockDailyBar.Close`          | `close`                   | 复权收盘                          |
| `StockDailyBar.Open`           | `open`           | 复权开盘                          |
| `StockDailyBar.High` / `Low` | `high` / `low` |                                   |
| `StockDailyBar.Volume`         | `volume`         | 复权成交量                        |
| `StockDailyBar.Amount`         | `amount`         | 成交额                            |
| `StockDailyBar.Return`         | `ret`            | 日收益（bp×100，评估时注意量纲） |
| `StockDailyBar.PreClose`       | `pre_close`      |                                   |
| `StockDailyBar.Vwap`           | `vwap`           | 复权 VWAP（`Vwap * Factor`）    |
| `StockDailyBar.IsSuspend`      | `is_suspend`     | 停牌                              |

**算子（与美股同一白名单，校验见 §2.6）：**

| 类别   | 代表算子                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------ |
| 截面   | `rank` / `cs_rank` · `zscore` / `cs_zscore` · `cs_demean` · `winsorize` · `cs_resid` |
| 时序   | `ts_mean` · `ts_std` · `ts_corr` · `ts_cov` · `delay` · `decay_linear` · `ema`     |
| 财务   | `ttm` · `quarter` · `yoy` · `avg2` + `StockIncome.*` / `StockBalance.*`                 |
| 中性化 | `industry_neutralize` · `size_neutralize` · `neutralize`                                       |

**样本 filter（A 股常用，可在框架 yaml 或 campaign config 备注）：**

```text
volume > 0 AND not is_suspend AND close < high_limit AND close > low_limit
```

### 4.3 CogAlpha `python`（中间态）

- 允许在 workspace 内用 pandas 挖因子；
- **投递 `candidate_pool` 时**应转为 factor_engine **`dsl`**，或暂标 `expression_type: python` 并附 `code` 字段供翻译；
- **禁止**仅投递 python 字符串却标 `dsl`。

---

## 5. 一场 campaign 怎么开（Checklist）

### 5.1 开挖前

- [ ] 确定 `market` + `domain_root`（一种类型一场，不混挖）
- [ ] 选定 `universe_id`（§2.4）；本地 parquet 已同步（可选填 `data_source` §2.8）
- [ ] 填好 `mining_scope.primary_tables` / `forbidden_tables`
- [ ] 选定 `operator_policy`（§2.6）
- [ ] `campaign_id` 符合命名；创建目录

### 5.2 每个因子

- [ ] `candidate_id` 与文件夹名一致；`hash8` 算法正确
- [ ] `formula` 单行、算子/字段在 scope 内
- [ ] `market` / `domain_root` / `domain` / `universe_id` 与 campaign config 一致
- [ ] manifest 显式写 `frequency_bucket: daily`（不可省略）
- [ ] `metrics` 含 `train` / `valid` / `test` 三段
- [ ] `description` 非空
- [ ] **字段 / domain**：`check_manifest_fields.py` 通过（§5.4）
- [ ] **DSL（A 股 / 美股 `expression_type: dsl`）**：`validate_delivery_formula.py` 通过（§5.4）

### 5.3 投递后

- [ ] `mining_run_stats` 已写入 campaign `config.json`（**具体 API model id**、provider/base_url、Token 合计、起止时间与 `duration_seconds`）
- [ ] 未写入 tier / 因子值 parquet / `afv.json`
- [ ] 本地 `candidate_pool/{campaign_id}/` 目录完整（`config.json` + 每个因子的 `{candidate_id}/manifest.json`）
- [ ] 已同步到 COS `candidate_pool/candidate/{campaign_id}/...`（结构与本地一致，见 §1.2）
- [ ] 通知评估侧跑 AFV pipeline

### 5.4 投递前校验（CLI）

在拷贝到 `candidate_pool/` 或同步 COS **之前**执行：

```bash
# 1. 字段注册表 + domain_root + 跨域约束
python3 ~/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py \
  path/to/manifest.json

python3 ~/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py \
  ~/quant_projects/data/factor_pools/candidate_pool/{campaign_id}/ --recursive

# 2. factor_engine DSL 语法（A 股 / 美股，expression_type: dsl 或历史 lqtp_dsl）
cd ~/quant_projects/factor_engine
PYTHONPATH=. python scripts/validate_delivery_formula.py --manifest path/to/manifest.json
```

**分工：** `check_manifest_fields.py` 查列名是否在 canonical 注册、是否跨 domain、算子名是否误当列名；`validate_delivery_formula.py` 查 factor_engine 能否解析公式。**A 股与美股均经 factor_engine 执行**，不依赖 `platforms.lqtp`。

**canonical 注册表更新**（本地 parquet schema 变更后）：

```bash
cd ~/quant_projects
python factor_engine/scripts/build_canonical_fields.py
```
