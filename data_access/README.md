# `data_access` — 团队统一数据读写入口

> **一句话定位**：所有读写 parquet 的代码，都应该走这里。
>
> **对外使用说明（随包分发）**：**[用户使用手册.md](用户使用手册.md)** — 只看这一份即可。
>
> 维护人：量化基础平台组｜最后更新：2026-07-12｜当前版本：PR8+（读/写/publish/upsert/sql/stream/polars/CH/QueryBudget/自选路径）

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

PR4 及以前的分阶段说明见本目录 `README.md` 历史与 `用户使用手册.md`。

---

## pip 安装

```bash
# 本地 editable（开发）
pip install -e ".[all]"

# 从 GitHub 只装远程客户端（团队机器，不必 clone 全仓库）
pip install "data-access[client] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=data_access"

# 读数 API 服务端
pip install "data-access[service] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=data_access"
export DATA_ACCESS_API_KEY=团队密钥   # 生产必设；QUANT_PRODUCTION_MODE=1 时无密钥会拒绝启动接口
data-access-server --host 0.0.0.0 --port 8765
```

| extra | 包含 |
|-------|------|
| 默认 | `get_store()` 本地读数 |
| `[client]` | `DataAccessClient` HTTP 客户端 |
| `[service]` | FastAPI 读数服务 + `data-access-server` 命令 |
| `[polars]` | Polars lazy scan |
| `[all]` | 以上全部 |

### COS 远程直读（不落地）

默认 ``mirror``：读前先同步到 ``ASHARE_PARQUET_ROOT`` 等本地目录。若机器磁盘小或不想缓存，可改为 DuckDB 直连 COS（S3 兼容 API）：

```bash
export DATA_ACCESS_COS_READ_MODE=remote   # 或 auto：本地已有则本地，否则 remote
export DATA_ACCESS_COS_REMOTE_BACKEND=auto  # auto|httpfs|cli（无 DuckDB httpfs 时用 cli）
# cli：按需用 clean-cos-ro 拉到独立 cache，不改 COS，也不写永久 mirror 根
# httpfs：需 INSTALL httpfs + DATA_ACCESS_COS_S3_ENDPOINT + COS_SECRET_ID/KEY（或 ~/.cos.yaml）
```

```python
from data_access import get_store
store = get_store()
df = store.read_frame(
    "ashare_stock_daily",
    columns=["TradeDate", "Symbol", "Close"],
    time_range=("2024-01-01", "2024-01-31"),
)
```

| 模式 | 行为 |
|------|------|
| `mirror` | 先 ``cos_mirror`` 拉到永久本地根，再读（默认） |
| `remote` | 按需从 COS 读：``httpfs`` 直连或 ``cli`` 拉到独立 cache |
| `auto` | 该 ``time_range`` 本地永久镜像齐全则用本地，否则 remote |

| `DATA_ACCESS_COS_REMOTE_BACKEND` | 行为 |
|------|------|
| `auto`（默认） | 有 httpfs+凭证用 httpfs，否则 ``clean-cos-ro`` cache |
| `httpfs` | DuckDB ``read_parquet('s3://...')`` |
| `cli` | ``clean-cos-ro`` 按需下载到 ``DATA_ACCESS_COS_CACHE_ROOT`` |

已登记 **43** 个 COS 数据集：A 股 lqtp（``clean_data/ashare/lqtp_data``）、美股 massive（``clean_data/us_stock/massive_data``）、美股 clean 衍生层（``universe_daily`` / ``adj_factor`` 等）。其他服务器只需配同一套 COS 前缀环境变量 + ``clean-cos-ro``（或 httpfs 凭证），**不依赖本机原有目录布局**。

真实小批量验收::

```bash
source env.sh
export DATA_ACCESS_COS_READ_MODE=remote
export DATA_ACCESS_COS_REMOTE_BACKEND=cli
export QUANT_RUN_NAMESPACE=cos_smoke
python3 scripts/smoke_cos_compute_write.py
```

### 读 COS → 运算 → 结果写到别处（不改源数据）

COS / published 源只读。运算结果写入 **staging / namespaced**（如 ``factor_lake_staging``），需要正式发布再 ``publish_from_staging``。

```bash
export DATA_ACCESS_COS_READ_MODE=remote
export DATA_ACCESS_COS_REMOTE_BACKEND=cli   # 或其他机器有 httpfs 时用 auto/httpfs
export QUANT_RUN_NAMESPACE=shw
```

```python
from data_access import get_store

store = get_store()

# 一条 API：远程读 + SQL 运算 + 落 staging（不碰 COS 原文件）
store.compute_and_write(
    """
    SELECT
      TradeDate AS datetime,
      Symbol AS asset,
      Close AS value,
      CAST(strftime(TradeDate, '%Y') AS INTEGER) AS year
    FROM {{ashare_stock_daily}}
    """,
    read_datasets=["ashare_stock_daily"],
    read_time_ranges={"ashare_stock_daily": ("2024-01-01", "2024-01-31")},
    write_dataset="factor_lake_staging",
    factor_id="close_raw_v1",
    mode="overwrite",
    partition_by=["year"],
)

# 可选：校验通过后晋升到 factor_lake（仍不改 COS 行情）
# store.publish_from_staging("factor_lake_staging", "factor_lake", factor_id="close_raw_v1")
```

更复杂的因子 DSL 运算走 ``factor_engine``：``data_source.type: data_access`` 读行情（同样支持 ``DATA_ACCESS_COS_READ_MODE=remote``），物化到 ``factor_lake_staging``。

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

### 自选读/写路径（可选）

默认路径来自 `datasets.yaml`。其他机器目录不同时可用：

```python
# 读：覆盖数据集根
store.read_frame("ashare_stock_daily", columns=["Close"], read_root="/data/my_ws/.../StockDailyBar")

# 写：最终目录 / 或只换根（仍拼 factor_id）
store.write_arrow("factor_lake_staging", tbl, factor_id="x", write_dir="/data/my_out/x")
store.write_arrow("factor_lake_staging", tbl, factor_id="x", write_root="/data/my_ws/staging/factors")
```

自定义路径需先：`export DATA_ACCESS_EXTRA_ALLOWED_ROOTS=/data/my_ws,/data/my_cache`  
也可用环境变量 `DATA_ACCESS_READ_ROOT_<数据集>` / `DATA_ACCESS_WRITE_ROOT_<数据集>`。

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
# 表名必须用 {{dataset}} 占位符，避免字符串字面量被误替换
tbl = store.sql(
    """
    SELECT f.asset, f.datetime, f.value, p.close
    FROM {{factor_lake}} f JOIN {{us_stocks_sip_day_aggs}} p
      ON p.align_time = f.datetime AND p.ticker = f.asset
    WHERE f.value > ? ORDER BY f.datetime
    """,
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
for batch in store.sql_stream(
    "SELECT align_time, ticker, close FROM {{us_stocks_sip_day_aggs}} WHERE close > ?",
    read_datasets=["us_stocks_sip_day_aggs"],
    view_columns={"us_stocks_sip_day_aggs": ["align_time", "ticker", "close"]},
    params=[10.0],
    query_budget=budget,
):
    process(batch)  # pyarrow RecordBatch
```

大表扫描也可用 ``read_auto_stream()``（按 footer 估算路由，始终 batch 返回）：

```python
for batch in store.read_auto_stream(
    "us_stocks_sip_day_aggs",
    columns=["align_time", "ticker", "close"],
):
    process(batch)
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
| `paths.py` | `${VAR:-default}` 展开 + 路径白名单 + 自选根 | debug 路径不对时 |
| `namespace.py` | `QUANT_RUN_NAMESPACE` / `QUANT_OPERATOR` 解析 | 定位是谁在跑什么 |
| `audit.py` | JSONL 审计日志（PR2） | 想查谁什么时候写了啥 |
| `retry.py` | `@retry_io` — 只对 IO 错误重试 | 不轻易改，改了测试会挂 |
| `exceptions.py` | `ValidationError / DataError / EngineError` | 写 try/except 时 |
| `cos_remote.py` / `cos_mirror.py` | COS 远程直读 / 本地镜像 | `DATA_ACCESS_COS_READ_MODE` |
| `query_budget.py` | 生产读配额 / columns 强制 | 排查「必须指定 columns」 |
| `service/` | HTTP 读数服务 + client | `data-access-server` / `DataAccessClient` |
| `用户使用手册.md` | 对外使用说明（随 wheel 分发） | 外部同事首选 |

## PR2 新增的环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `QUANT_AUDIT_LOG` | `${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/logs/data_access_audit.jsonl` | 审计日志文件路径，测试里常指向 tmp |
| `QUANT_AUDIT_READS` | `false` | 设 `true` 时连 read 也记审计；默认只记 write（量太大） |
| `QUANT_OPERATOR` | 空 | 审计日志里的 operator 字段（允许 `@+.-_`，如 `shw@team`） |

---

## 延伸阅读

- **[用户使用手册](用户使用手册.md)** — **外部使用者首选**（功能、数据、COS、自选路径、写发布、HTTP 服务）；随本包分发
- [`config/datasets.yaml`](config/datasets.yaml) — 数据集登记表本体
- [`factor_engine/README.md`](../factor_engine/README.md) — 因子引擎批量 YAML（`run_many_from_config` / `materialize_many_from_config`）与 `staging_clickhouse` 物化

## 发反馈 / 报 bug

- 线上问题：Slack `#quant-platform`
- 改动建议 / PR：联系基础平台组
