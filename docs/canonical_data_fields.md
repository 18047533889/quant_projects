# 原始数据字段规范（Canonical Fields）

> **权威数据源（全量列清单 + 分域约束）**：[`canonical_data_fields.json`](canonical_data_fields.json)（**v3**）  
> **分域枚举与规则**：[`../../factor-pool-standard/enums/domain_roots.yaml`](../../factor-pool-standard/enums/domain_roots.yaml)  
> **同步副本**：[`../../factor-pool-standard/enums/canonical_data_fields.json`](../../factor-pool-standard/enums/canonical_data_fields.json)  
> **生成**：[`../scripts/build_canonical_fields.py`](../scripts/build_canonical_fields.py) · **校验**：[`../../factor-pool-standard/scripts/check_manifest_fields.py`](../../factor-pool-standard/scripts/check_manifest_fields.py)

---

## 1. 两层分类（必理解）

### 1.1 表级：`domain_root`（挖哪一类因子）

| `domain_root` | 挖什么 | A 股 operator_policy | 美股 operator_policy |
|---------------|--------|---------------------|---------------------|
| `price_volume` | 价量/收益/股本 | `lqtp_pv_daily` | `afv_us_pv_daily` |
| `fundamental` | 财报/估值/PIT | `lqtp_fs_event` | `afv_us_pv_daily` |
| `alternative` | 股东/融券/新闻 | `lqtp_pv_daily` | `afv_us_pv_daily` |
| `microstructure` | 分钟/逐笔/报价 | `lqtp_pv_daily` | `afv_us_pv_daily` |

**一场 campaign 只能有一个 `domain_root`**，对应 `mining_scope.primary_tables`。

### 1.2 列级：每列四个属性

| 属性 | 含义 |
|------|------|
| `role` | key / signal / filter / auxiliary / metadata |
| `formula_usage` | **primary_signal** · filter_only · auxiliary_only · **forbidden** |
| `signal_domain` | 该列作**主信号**时属于哪个 `domain_root` |
| `canonical` | manifest 公式里写的名字 |

**约束关系：**

```text
campaign.domain_root = price_volume
  → 公式主信号列必须 signal_domain = price_volume
  → 不得引用 signal_domain = fundamental 的列（如 pit_revenue）
  → filter 可用 is_suspend / high_limit 等 filter_only 列
  → metadata/key 列 forbidden，不得出现在 formula
```

### 1.3 宽表 PanelDaily（美股）

同一张表，**按列拆 domain**：

| 列前缀/列名 | signal_domain |
|-------------|---------------|
| `Open` `Close` `Volume` … | `price_volume` |
| `pit_*` `ltm_*` | `fundamental` |
| `news_*` | `alternative` |
| `batch_id` `join_log` … | forbidden（metadata） |

价量 campaign 可声明 `primary_tables: [PanelDaily]`，但公式**只能**用其中 `signal_domain=price_volume` 的列。

---

## 2. 覆盖范围

| 范围 | 表/数据集 | 列数（约） |
|------|-----------|-----------|
| 美股本地 | 7 | 81 |
| A 股本地 | 14 | 370 |
| Massive 外部 | 11 | 194 |

JSON 内 **`domain_scopes`** 段：每个 `domain_root` × `market` 下汇总：

- `primary_tables` — 允许的主信号表
- `forbidden_primary_tables` — 其他 domain 独占的表（宽表除外）
- `signal_columns_by_table` — **该 domain 下公式可引用的主信号列清单**
- `filter_columns_global` — 全局 filter 列
- `auxiliary_tables` — 仅中性化/分组

---

## 3. 美股本地表（7 张）

| 逻辑表 | 默认 domain | 列数 | 说明 |
|--------|------------|------|------|
| `StockDailyBar` | price_volume | 8 | OHLCV + `adj_factor`（PascalCase 物理列） |
| `FactReturnsDaily` | price_volume | 5 | `ret_price` / `ret_total` |
| `PanelDaily` | **按列** | 52 | 价量 + PIT + news 混合 |
| `Dim*`（4 张） | auxiliary | — | 日历/映射 |

### 3.1 美股价量：物理列 vs 公式组合

**`domain_scopes.price_volume.us_stock` 已登记的主信号列**（本地 `StockDailyBar`）：  
`open` `high` `low` `close` `volume` `adj_factor`。

**Massive `day_aggs_v1` 额外物理列**（外部表 `massive:us_stocks_sip/day_aggs_v1`）：  
`transactions` 等 — 见 JSON `tables_massive_external`。

**常见但本地尚无物理列** — manifest 中请用 DSL 组合（与 [`miner_delivery_spec.md`](miner_delivery_spec.md) §4.1 一致）：

| 概念 | 推荐写法 |
|------|----------|
| 昨收 | `delay(close, 1)` |
| 日收益 | `close / delay(close, 1) - 1` 或 `ret_price` |
| 成交额 | `close * volume` |
| VWAP 近似 | `(high + low + close) / 3` |
| 隔夜跳空 | `(open - delay(close, 1)) / (delay(close, 1) + 1e-9)` |

**A 股 `StockDailyBar`** 另有物理列：`amount` `pre_close` `vwap` `ret` `factor` 及涨跌停 filter 列 — 见 JSON `domain_scopes.price_volume.ashare`。

---

## 4. A 股本地表（14 张）

| 逻辑表 | domain | 列数 |
|--------|--------|------|
| `StockDailyBar` | price_volume | 16 |
| `StockCapitalDaily` | price_volume | 7 |
| `StockBalance` | fundamental | 111 |
| `StockIncome` | fundamental | 53 |
| `StockCashFlow` | fundamental | 59 |
| `StockIndicator` | fundamental | 36 |
| `StockValuationDaily` | fundamental | 19 |
| `StockDividend` | fundamental | 8 |
| `StockTopTenShareholder` | alternative | 18 |
| `StockTopTenFloatShareholder` | alternative | 16 |
| `StockList` / `StockStatus` / `StockIndustry` / `Calendar` | auxiliary | — |

---

## 5. 校验

```bash
# 重新生成（扫描 parquet + 写 domain_scopes）
python3 factor_engine/scripts/build_canonical_fields.py

# 校验 manifest：登记 + domain_root 分域
python3 factor-pool-standard/scripts/check_manifest_fields.py data/factor_pools/translated/ --recursive
```

---

## 6. 相关文档

- [`算子与导入教程.md`](算子与导入教程.md)
- [`miner_delivery_spec.md`](miner_delivery_spec.md) §4
- [`dsl_operators_reference.md`](dsl_operators_reference.md)
- [`massive_parquet_data_dictionary.md`](massive_parquet_data_dictionary.md)
