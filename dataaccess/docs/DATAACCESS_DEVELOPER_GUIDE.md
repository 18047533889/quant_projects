# DataAccess 开发指南（current-state）

> 面向在此仓库上**新增/扩展**数据能力的开发者。本文档讲「怎么做」，不讲历史。
> 包名 `data_access`；仓库根 `dataaccess/`。入口 `from data_access import get_store`。

---

## 1. 新增 dataset

### 1.1 在 `config/datasets.yaml` 登记（单一真源）

业务代码只写数据集名，路径/格式/分区/权限全在这里声明。Loader 是 **strict schema**：
未知 key / 非法枚举 / 未展开的 `${ENV}` 都会启动即失败。

静态数据集（路径一次定死）：

```yaml
my_daily_snapshot:
  kind: static
  access_mode: published          # published | namespaced | staging
  layout: plain                   # plain | hive
  format: parquet                 # parquet | csv | tsv | jsonl | arrow | feather
  root: ${MY_DATA_ROOT:-/data/mine}/snapshot
  glob: "**/*.parquet"
  time_column: TradeDate          # 业务约定时间列（可用 roles.event_time 替代）
  instrument_column: Symbol       # 业务约定标的列
  schema:                          # 可选：首访自检（列名 → 类型字符串）
    TradeDate: date
    Symbol: string
    Close: double
  storage:                         # 可选：local / s3 / cos / http / cli / clickhouse
    type: local
```

参数化数据集（路径含运行时参数，如因子湖按 `factor_id`）：

```yaml
my_factor_lake:
  kind: parametric
  access_mode: published
  root_template: ${LAKE_ROOT}/factors/{factor_id}
  glob_template: "*.parquet"
  params_schema: {factor_id: str}
  authorized_root: ${LAKE_ROOT}/factors      # 白名单安全边界（比静态前缀更窄）
```

### 1.2 必填治理字段

| 字段 | 取值 | 说明 |
|---|---|---|
| `kind` | `static` / `parametric` | 路径是否含运行时参数 |
| `access_mode` | `published` / `namespaced` / `staging` | published 只读；写必须走 staging→publish |
| `layout` | `plain` / `hive` | 目录结构类型 |
| `mutation_owner` | `dataaccess` / `external_versioned` / `external_mutable` / `immutable` | manifest 新鲜度语义根（缺省 `dataaccess`） |
| `schema_migrations` | 数组 | 跨 epoch schema 演进审批；每项 `approved` 必须 bool（未批准不构成放行依据） |
| `generation_pointer` | bool | generation 指针模型（读路径只解析 manifest 指向代） |
| `generation_required` | bool | 与 `mutation_owner` 匹配；`external_mutable` 禁 true |

`schema_migrations` 示例（R29-P0 #194 契约）：

```yaml
my_daily_snapshot:
  # ...
  schema_migrations:
    - from_fingerprint: a1b2c3d4e5f60718
      to_fingerprint: b2c3d4e5f60718a9
      kind: add_column              # add_column | dtype_change | unit_change
      field: NewField
      approved: true
      reviewer: "quant-baseline"
```

### 1.3 构建 manifest

注册后读路径只有在 `_manifest.parquet` 存在且新鲜时做文件级裁剪。显式构建：

```python
from data_access import get_store

store = get_store()
summary = store.build_dataset_manifest("my_daily_snapshot", include_row_groups=False)
# → {"dataset": ..., "files": N, "rows": N, "bytes": N, "format": ...}
```

### 1.4 校验

```bash
python3 -c "from data_access import get_store; s = get_store(); print(s.registry.names())"
python3 scripts/audit_dataaccess_source_contracts.py   # 治理字段一致性
```

---

## 2. 新增 semantic concept

字段语义单一事实源是 `config/semantic_fields.yaml`（`SemanticFieldCatalog` 消费）。
FactorEngine `FIELD_REGISTRY` 与 DataAccess schema/COS 契约之上的统一语义层。

### 2.1 在 `config/semantic_fields.yaml` 注册

```yaml
# 无量纲行情字段
my_indicator:
  dataset: my_daily_snapshot
  physical_name: MyIndicator
  dtype: double
  frequency: daily
  grain: instrument            # instrument | cross_section | snapshot ...
  time_role: event_time
  temporal_model: panel
  join_policy: exact
  mining_allowed: true

# 带单位归一化的比值字段（percent → ratio）
my_roe:
  dataset: my_daily_snapshot
  physical_name: MyRoe
  dtype: double
  frequency: daily
  grain: instrument
  source_unit: percent
  canonical_unit: ratio
  scale: 0.01                  # percent→ratio 乘子
  dimension: ratio             # ratio | money | money_per_share | price | shares | identifier | boolean | count
  currency: CNY                # 无量纲可省略；money 维度必须声明
  cross_market_comparable: true
  requires_fx: false
```

### 2.2 语义字段关键字段

- **身份**：`logical_name`（可显式覆盖；同一逻辑字段跨市场用唯一 YAML key +
  `logical_name: total_assets` 让 catalog 的 `_by_name` 同时挂 A股/美股候选）。
- **时间语义**：`time_role`（event_time/knowledge_time/effective_time）、
  `temporal_model`、`knowledge_time`、`effective_time`、`period_time`、
  `availability`（PIT 可见性）、`availability_latency`。
- **PIT/join**：`join_policy`（exact / pit_asof_backward / latest_period）、
  `revision_order`、`primary_key`、`duplicate_policy`、`pit_fidelity`、
  `time_representation`、`time_precision`。
- **单位**：`source_unit` → `canonical_unit` + `scale`；`dimension` + `currency`
  （money 必须有币种）；`cross_market_comparable` + `requires_fx`（不要同时 true）。
- **derived 字段**：不落盘，`dataset` 可省略但必须有 `derived_expression`，
  `derived_from` 声明依赖（coverage audit 对 derived 跳过 COLUMN_MISSING）。

### 2.3 关于 ConceptId

R30 `concepts` 层（`data_access.r30`）声明了 `ConceptId` 作为跨字段/跨市场概念的
身份抽象（字段 → `ConceptId` 归一化）。当前落地状态：`r30/_shared.py` 提供
`stable_digest`（确定性概念摘要）与 `SEMANTIC_SCHEMA_VERSION` 版本门；具体
`ConceptId` 子模块为声明的惰性扩展点。**新增语义字段的当前唯一入口**就是
`semantic_fields.yaml`；在字段上声明 `dimension`/`currency`/`unit`/`market` 即
为该字段建立可被概念层消歧的身份。

### 2.4 校验

```python
from data_access.read.semantic_catalog import get_semantic_catalog

catalog = get_semantic_catalog()
f = catalog.get("my_roe", market="ashare")   # 跨市场需显式 market / dataset
print(f.scale, f.dimension, f.cross_market_comparable)
```

```bash
python3 scripts/audit_semantic_concept_coverage.py   # 概念覆盖一致性
```

---

## 3. 新增 market / source adapter

### 3.1 新物理文件格式 → `FormatAdapter`

`read/formats.py` 的 `FormatAdapter` 抽象：每种格式一个 adapter，负责生成 DuckDB
的 FROM 子句，或声明需要 PyArrow 直读（arrow/feather）。

```python
from data_access.read.formats import FormatAdapter, DataFormat, _ADAPTERS

class MyDelimitedAdapter(FormatAdapter):
    data_format = DataFormat.CSV          # 或新枚举
    default_glob = "**/*.csv"

    def scan_options(self, *, hive_partitioning, union_by_name) -> str:
        opts = []
        # 把 self.spec 的选项渲染成 ", k=v" 后缀
        return (", " + ", ".join(opts)) if opts else ""

    def build_from_clause(self, path_param, *, hive_partitioning, union_by_name) -> str:
        return f"read_csv(?{self.scan_options(...)})"

# 注册进适配器表
_ADAPTERS["myfmt"] = MyDelimitedAdapter(spec=FormatSpec(type="myfmt"))
```

数据集声明 `format: myfmt` 即走该 adapter。`uses_duckdb=False` 的格式（arrow/feather）
由 `store.read` / `read_uri` 切到 PyArrow 引擎。

### 3.2 新存储后端 → `StorageSpec`

`core/storage.py` 把 registry 的 `storage:` 声明归一化成 `StorageSpec`，提供
`is_remote_storage` / `to_s3_uri` / `authorize_path` 分派。接新后端（S3/MinIO/R2/…）
只需加一个 backend + 分派，不新增 Store 特殊分支。路径鉴权分派：
local → `PathAuthorizer`；s3/cos → `authorize_s3_path`。

### 3.3 新市场

1. `config/datasets.yaml` 加市场数据集（命名建议 `us_*` / `ashare_*`，`semantic_catalog`
   按前缀推断 market）；
2. `config/semantic_fields.yaml` 加该市场的字段条目（market: `ashare`/`us`/`any`）；
3. `session_calendar` 注册该市场交易日历（`MarketCalendar`/`get_market_calendar`）。

### 3.4 R30 SourceAdapter

`data_access.r30.adapter` 声明 `SourceAdapter + SourceCapabilities + BackendCapabilities`
作为**跨数据源能力声明**的未来接口（惰性扩展点）；当前数据源接入仍走
`FormatAdapter`（物理格式）+ `StorageSpec`（存储后端）+ `Datasets.yaml`（数据集注册）。

---

## 4. 写 PIT contract

PIT（point-in-time）契约 = **数据可见时间**语义 + 读取时的可用性延迟。DataAccess
有三层 PIT 物化：

| 层 | 文件 | 用途 |
|---|---|---|
| 日频日历 | `read/session_calendar.py` | `MarketSession` / `MarketCalendar`；`available_from` 编译 |
| 财务 PIT 索引 | `read/pit_event_index.py` | `PITEventIndex`（filing_date 精确裁剪） |
| 字段级契约 | `config/semantic_fields.yaml` | `knowledge_time` / `effective_time` / `availability` / `join_policy` |

### 4.1 日频面板数据集

在 `semantic_fields.yaml` 声明 `time_role: event_time` + `temporal_model: panel`，
数据集声明 `time_column`。`session_calendar` 负责把「决策时间 → 可见 bar」换算
（时区感知，处理午休/节假日/盘前盘后）。

```yaml
close:
  dataset: ashare_stock_daily
  physical_name: Close
  time_role: event_time
  temporal_model: panel
  join_policy: exact
  frequency: daily
  grain: instrument
```

### 4.2 财务事件（knowledge_date）

财务字段声明数据可见时间列与生效时间列，`join_policy` 用 PIT：

```yaml
net_income:
  dataset: ashare_stock_income
  physical_name: NetProfit
  time_role: knowledge_time
  knowledge_time: PubDate              # 数据可见时间列（如公告日）
  effective_time: ReportPeriodEndDate  # 数据生效时间列（如会计期）
  period_time: ReportPeriodEndDate
  temporal_model: financial_event
  join_policy: latest_period           # 最新可见会计期 + 该期内最新修订
  availability: next_trading_day
  revision_order: [ReportPeriodEndDate, PubDate]
  pit_fidelity: knowledge_date_pit
```

### 4.3 美股 filing 精确裁剪（pit_event_index）

美股物理文件按 `{period_end}.parquet` 命名、真正的 PIT 是文件里的 `filing_date`，
路径无法按 filing_date 裁剪。构建二级索引：

```python
from data_access import get_store
from data_access.read.pit_event_index import build_pit_event_index

store = get_store()
build_pit_event_index(store, "us_stock_income")   # 落盘 _pit_event_index.parquet
```

索引只有 `complete == True` 且 snapshot 一致时才允许 authoritative prune；否则
fail-open（回退全量路径）。

---

## 5. 写 benchmark / test

### 5.1 单元测试

放在 `tests/unit/`（另有 `tests/concurrency/`、`tests/contract/`）。直接 import
稳定接口，不连真实数据源：

```python
# tests/unit/test_my_feature.py
from data_access.read.manifest import uuid4_hex

def test_uuid4_hex_format():
    assert len(uuid4_hex()) == 16
```

跑法（串行，仓库约定小内存）：

```bash
python3 -m pytest tests/unit/test_my_feature.py -q
python3 -m pytest tests/unit -q          # 全量
```

### 5.2 benchmark workload

放 `benchmarks/`（现有 `bench_read.py`）。新增 workload 直接调用
`store.read_*`，测量耗时/内存：

```python
# benchmarks/bench_read.py（追加一个 workload 函数）
def bench_my_read(store):
    t0 = time.perf_counter()
    df = store.read_frame("my_daily_snapshot", columns=["Close"],
                          time_range=("2024-01-01", "2024-12-31"))
    return {"elapsed_ms": (time.perf_counter() - t0) * 1000,
            "rows": len(df)}
```

```bash
python3 benchmarks/bench_read.py
```

### 5.3 静态审计（回归门）

```bash
python3 scripts/audit_dataaccess_source_contracts.py   # registry contract 一致性
python3 scripts/audit_semantic_concept_coverage.py     # 语义字段概念覆盖
python3 scripts/audit_dataaccess_perf_paths.py         # 热路径 pandas 物化检测
```

三个脚本 0 违规时 exit 0；`--json` 可输出机器可读结果。**它们只做静态检查，
不连接数据源**。
