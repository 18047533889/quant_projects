# `data_access` — 团队统一数据读写入口

> **一句话定位**：所有读写 parquet 的代码，都应该走这里。
>
> 维护人：量化基础平台组｜最后更新：2026-07-09｜当前版本：PR8+（读/写/publish/upsert/sql/stream/polars/CH/QueryBudget）

## 为什么有这个模块

在这之前，团队各处都在做同一件事：`pd.read_parquet(...)` → 拼 glob → 合并 DataFrame → 转 MultiIndex。
每个调用方实现略有不同，每次加新数据集都要到处改，性能也拉不开差距。
写的路径更乱：谁高兴写哪写哪，没人知道昨天晚上那批因子是谁算的。

**当前能力（PR1–PR8+）**：

1. **读入口收敛**：`read_arrow` / `read_frame` / `load_columns` / `read_arrow_stream` / `scan_polars`
2. **DuckDB 共享引擎**：跨线程 buffer pool / parquet footer cache
3. **数据集登记 YAML**：`config/datasets.yaml`（含 factor_lake 元数据列 schema）
4. **namespace 隔离**：namespaced / staging / published 三态
5. **写与发布**：`write_arrow` / `upsert` / `publish_from_staging` / `delete_rows`（支持 dry_run）
6. **有限 SQL**：`sql()` 只读 SELECT + 审计 + QueryBudget
7. **读审计**：生产模式（`QUANT_PRODUCTION_MODE=1`）默认记录 read；也可 `QUANT_AUDIT_READS=true`
8. **ClickHouse**：`clickhouse_panel` / `clickhouse_write` / 写后 `verify_factor_write`
9. **Schema 首访自检** + **instrument_filter 守卫**

**与 factor_engine 的配合**：读 parquet 一律 `data_source.type: data_access`；计算默认 `backend.type: auto`（DuckDB SQL 子树 + Polars/Pandas fallback）。纯 SQL 因子可用 `duckdb_sql`；ClickHouse panel 用 `clickhouse` + `clickhouse_sql`。详见 [`factor_engine/README.md`](../factor_engine/README.md)「底层栈与 backend 选择」。

PR4 及以前的分阶段说明见 `docs/data_access/10_架构设计.md`。

---

## 最常用的 3 个调用

```python
from data_access import get_store
store = get_store()

# 1) 读 DataFrame（末端场景：画图、落 CSV、快速查看）
df = store.read_frame(
    "us_stocks_sip_day_aggs",
    columns=["align_time", "ticker", "close"],
    time_range=("2024-01-01", "2024-12-31"),
    instrument_filter=["AAPL", "MSFT"],
)

# 2) 读 Arrow Table（pipeline 中段首选，零拷贝）
tbl = store.read_arrow("us_stocks_sip_day_aggs", columns=["align_time", "ticker", "close"])

# 3) 批量多列 → dict[name, MultiIndex Series]（因子引擎路径）
cols = store.load_columns(
    "us_stocks_sip_day_aggs",
    columns=["close", "volume", "open"],
    time_range=("2024-01-01", "2024-12-31"),
)
cols["close"]   # (timestamp, instrument) MultiIndex Series
```

### 参数化数据集（factor_lake 等）

```python
tbl = store.read_arrow(
    "factor_lake",
    factor_id="single_asset_mom_3_v1",    # params_schema 里要求的参数
    columns=["datetime", "asset", "value"],
    time_range=("2024-01-01", None),
)
```

---

## 写数据：`write_arrow`（PR2）

```python
import pyarrow as pa
from data_access import get_store

store = get_store()

# 1) 写 namespaced 数据集（你自己的回测结果）
tbl = pa.table({"timestamp": [...], "symbol": [...], "pnl": [...]})
store.write_arrow(
    "single_asset_backtest_runs",
    tbl,
    strategy_id="mom_3d",
    version="v1",
    mode="overwrite",            # 或 "append"
)

# 2) 写 staging（待发布的因子，带 hive 分区）
tbl = pa.table({"datetime": [...], "asset": [...], "value": [...], "year": [...]})
store.write_arrow(
    "factor_lake_staging",
    tbl,
    factor_id="my_new_factor_v1",
    mode="overwrite",
    partition_by=["year"],       # 生成 year=YYYY/part-xxxx.parquet
)
```

**规则**（看 `store.py::write_arrow` 的 docstring 是最新的）：

- `mode="overwrite"` 先清目标目录再写；`"append"` 加新文件不动旧的（uuid 前缀防撞名）
- 写 `published` 数据集 → `ValidationError`（必须走 `publish_from_staging`，PR3 实装）
- 没 `export QUANT_RUN_NAMESPACE` 不拦，但日志里 warning + 审计标 `namespace_explicit=false`
- 每次写都在 `workspace_data/logs/data_access_audit.jsonl` 追一行（成功、失败都记）

---

## 合并写入 `upsert`（PR3）

```python
# factor 增量算完以后合并进 staging；新键覆盖旧键，不重复算历史
store.upsert(
    "factor_lake_staging", new_tbl,
    factor_id="mom_3d",
    upsert_on=["datetime", "asset"],   # 同键 → 后写覆盖先写
    partition_by=["year"],             # 每年一个 data.parquet，只重写涉及的分区
)
```

**和 `write_arrow` 的区别**：
- `write_arrow(overwrite)` 清目录整写；`upsert` 保留旧数据、按键去重合并
- `write_arrow(append)` 单纯加文件、不处理重复；`upsert` 会 dedup
- `upsert` 带跨进程文件锁（父目录 `.upsert.<name>.lock`，5s 轮询），streaming 场景也能用

---

## 发布 `publish_from_staging`（PR3）

```python
# 在 namespace 自己的 staging 写完 + 校验后，晋升到全员只读的 published
store.publish_from_staging(
    "factor_lake_staging", "factor_lake",
    factor_id="mom_3d",          # 两边 params_schema 必须一致
)
# → {"target_path": "...", "archive_path": ".../_archive/...", "rows": ...}
```

流程：copy staging 到候选目录 → 锁 + 归档旧 published → rename 候选为 published → 读回校验。
任何阶段失败都会尽力回滚（归档的 revert 回来、候选目录清掉），锁保证同名 publish 不会并发打架。

---

## 临时分析 `sql()`（PR3）

```python
# 带 GROUP BY / JOIN / 多表聚合的临时查询；比 read_arrow 灵活但更严格
tbl = store.sql(
    \"\"\"
    SELECT f.asset, f.datetime, f.value, p.close
    FROM factor_lake f JOIN us_stocks_sip_day_aggs p
      ON p.align_time = f.datetime AND p.ticker = f.asset
    WHERE f.value > ? ORDER BY f.datetime
    \"\"\",
    read_datasets=["factor_lake", "us_stocks_sip_day_aggs"],
    read_params={"factor_lake": {"factor_id": "mom_3d"}},
    params=[0.5],
)
```

**规则**：
- 只能 SELECT；出现 `INSERT / UPDATE / DELETE / COPY / read_parquet / PRAGMA` 等关键字直接 raise
- 每个数据集按它的 registry 注册成 TEMP VIEW，查询结束自动清
- 用户 `?` 绑定参数走 DuckDB prepared statement；路径/谓词由 registry 控制，不能被 query 影响

为什么要这层：大部分 GROUP BY / window / 多表 JOIN 用 `read_arrow` 表达不了，
但又不能裸开 `read_parquet` —— 路径白名单、审计、配额限流都得经过这条收敛。

### 流式 `sql_stream()`（PR8+）

大结果集用 RecordBatch 迭代，避免一次性 materialize：

```python
from data_access import QueryBudget, get_store

store = get_store()
budget = QueryBudget(max_rows=500_000, require_columns=True)
with store.sql_stream(
    "SELECT align_time, ticker, close FROM us_stocks_sip_day_aggs WHERE close > ?",
    read_datasets=["us_stocks_sip_day_aggs"],
    params=[10.0],
    budget=budget,
) as stream:
    for batch in stream:
        process(batch)  # pyarrow RecordBatch
```

与 `sql()` 相同的安全约束；无 LIMIT 时自动按 `QueryBudget.max_rows` 包装子查询。

---

## 加新数据集：只改一个文件

编辑 [`config/datasets.yaml`](config/datasets.yaml)，追加一段：

```yaml
my_new_dataset:
  kind: static                  # static | parametric
  access_mode: published        # published | namespaced | staging
  layout: plain                 # plain | hive
  root: ${SOME_ENV:-/fallback/path}/my_new_dataset
  glob: "**/*.parquet"
  time_column: ts
  instrument_column: symbol
  hive_partitioning: false
  union_by_name: true
```

然后业务代码就可以 `store.read_arrow("my_new_dataset", ...)`，**无需改 Python**。

三种 `access_mode` 的语义：

| mode | 谁能读 | 谁能写（PR2 已实现） | 典型用途 |
|---|---|---|---|
| `published` | 全员只读 | **仅发布流程（PR3）**，`write_arrow` 直写会 raise | 清洗后的行情、已发布的 factor_lake |
| `namespaced` | 路径里含 `${RUN_NAMESPACE}`，天然隔离 | 当前 namespace 的 owner | 个人回测结果、临时实验 |
| `staging` | 发布候选人 | 发布候选人（待 `publish_from_staging` 晋升） | 待发布的产物（如 `factor_lake_staging`） |

---

## 模块文件速览

| 文件 | 作用 | 何时读它 |
|---|---|---|
| `store.py` | 对外 API 本体：`read_arrow / read_frame / load_columns / write_arrow / publish_from_staging` | 调用方看这个就够 |
| `engine.py` | DuckDB 连接单例 + PRAGMA | 调优线程/内存时 |
| `registry.py` | YAML → Dataset 对象，解析路径模板 | 加数据集字段、改 schema 时 |
| `predicate.py` | time_range / instrument_filter 编译成参数化 SQL | 想加新谓词时 |
| `adapters.py` | Arrow Table → `(timestamp, instrument)` MultiIndex Series | 因子引擎兼容层 |
| `paths.py` | `${VAR:-default}` 展开 + 路径白名单 | debug 路径不对时 |
| `namespace.py` | `QUANT_RUN_NAMESPACE` / `QUANT_OPERATOR` 解析 | 定位是谁在跑什么 |
| `audit.py` | JSONL 审计日志（PR2） | 想查谁什么时候写了啥 |
| `retry.py` | `@retry_io` — 只对 IO 错误重试 | 不轻易改，改了测试会挂 |
| `exceptions.py` | `ValidationError / DataError / EngineError` | 写 try/except 时 |

## PR2 新增的环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `QUANT_AUDIT_LOG` | `${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/logs/data_access_audit.jsonl` | 审计日志文件路径，测试里常指向 tmp |
| `QUANT_AUDIT_READS` | `false` | 设 `true` 时连 read 也记审计；默认只记 write（量太大） |
| `QUANT_OPERATOR` | 空 | 审计日志里的 operator 字段（允许 `@+.-_`，如 `shw@team`） |

---

## 延伸阅读

- [团队使用规范](../docs/data_access/01_团队使用规范.md) — **第一次用必读**
- [5 分钟快速上手](../docs/data_access/02_快速上手.md)
- [架构设计 / 取舍](../docs/data_access/10_架构设计.md)
- [`config/datasets.yaml`](config/datasets.yaml) — 数据集登记表本体
- [`factor_engine/README.md`](../factor_engine/README.md) — 因子引擎批量 YAML（`run_many_from_config` / `materialize_many_from_config`）与 `staging_clickhouse` 物化

## 发反馈 / 报 bug

- 线上问题：Slack `#quant-platform`
- 改动建议 / PR：联系基础平台组
