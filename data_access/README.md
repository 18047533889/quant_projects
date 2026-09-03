# data_access — 团队统一数据读写层（77 数据集 · PIT · 成本路由）

全团队唯一的量化数据读写入口。所有读/写 parquet、csv、因子值、COS 对象、ClickHouse
面板的代码都必须走这里，**禁止散落 `pd.read_parquet`**。读什么表、怎么复权、
什么口径、谁有权限，全部收敛到这一层统一治理。

**定位:** 企业级 A 股横截面日频多因子量化项目的**数据层** —— 因子计算
（factor_engine）、评估（quant_evaluator）、回测（vectorbt_qs）、组合优化
（riskfolio_qs）的所有行情/因子/风险模型读取都从这里走，是全平台"唯一读通道"。

**版本:** 0.10.2 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/data_access (私有)
**用户手册（中文，优先看）:** [`docs/用户使用手册.md`](docs/用户使用手册.md) ｜
架构: [`docs/DATAACCESS_ARCHITECTURE.md`](docs/DATAACCESS_ARCHITECTURE.md) ｜
开发者指南: [`docs/DATAACCESS_DEVELOPER_GUIDE.md`](docs/DATAACCESS_DEVELOPER_GUIDE.md) ｜
COS 语义与 PIT 契约: [`docs/COS语义与PIT契约.md`](docs/COS语义与PIT契约.md) ｜
运维手册: [`docs/DATAACCESS_OPERATIONS_RUNBOOK.md`](docs/DATAACCESS_OPERATIONS_RUNBOOK.md)

---

## 它是什么 / 不是什么

**做什么：** 数据集注册表（77 个）、统一读门面（55 个公共方法）、Filter AST 三编译器、
成本路由、PIT/asof 时点读、多数据集 join、受控写与发布、COS 三模式镜像、
ClickHouse 面板、查询预算治理、安全授权、HTTP 服务。

**不做什么：** 不算因子（factor_engine）、不做清洗加工后的衍生表落库以外的业务逻辑、
不做回测/评估。它是纯粹的 **I/O + 契约 + 治理层**。

## 安装与第一个读

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/data_access.git
cd data_access
pip install -e ".[all]"
```

```python
from data_access import get_store
store = get_store()

# ① 基础读（DataFrame）
df = store.read_frame("ashare_stock_daily",
                      columns=["TradeDate", "Symbol", "Close", "Vwap"],
                      time_range=("2024-01-01", "2024-03-31"))

# ② 统一读 → ReadHandle（to_arrow / to_pandas / to_lazy / stream）
handle = store.read("ashare_stock_daily", columns=[...], time_range=(...),
                    filters={"Symbol": ["600000.SH"]})

# ③ Arrow 直读（最快路径，回测层用这个，比 pandas 快 5-10×）
tbl = store.read_arrow("ashare_stock_daily", columns=[...], time_range=(...))

# ④ 一次读多因子（单查询 UNION ALL / PIVOT）
handle = store.read_factors(["mom_3d", "vol_20"], time_range=(...), layout="wide")

# ⑤ PIT 时点读（财务/事件类，防未来函数）
df = store.read_asof("ashare_stock_indicator", as_of="2024-06-30")
```

HTTP 服务：`data-access-server --host 0.0.0.0 --port 8765` + `DataAccessClient`。

## 门面方法全景（`DataAccessStore`，store.py）

| 类别 | 方法 |
|---|---|
| 读（基础） | `read` / `read_frame` / `read_uri` / `scan` / `scan_polars` / `read_arrow` / `read_arrow_stream` / `read_auto` / `read_auto_stream` / `read_result` / `prepare_read` / `execute_prepared_read` |
| 读（SQL/Join） | `read_joined`（exact / asof / **pit_asof** 三种 join 策略，DuckDB 内执行）/ `sql_relation` / `sql` / `sql_result` / `sql_stream` / `build_sql_snapshot` |
| PIT | `read_asof` / `pit_event_index` / `prune_pit_paths` / `manifest_version` / `touch_manifest_epoch` / `is_snapshot_stale` |
| 因子 | `read_factors`（wide/long）/ `get_factor_catalog` / `refresh_factor_catalog` / `compute_and_write`（读→算→写一步） |
| 聚合/分钟 | `aggregate_minute_to_daily` / `aggregate_minute_bundle` / `materialize_daily_aggregate` / `route_minute_storage` |
| 快照/契约 | `describe_dataset` / `resolve_fields` / `plan` / `contract_ir` / `contract_ir_fingerprint` / `coverage` / `metadata_plane` / `registry_fingerprint` / `calendar_snapshot_id` |
| 缓存 | `enable_result_cache` / `read_cached` / `read_cached_result` |
| 写 | `write_arrow` / `upsert` / `delete_rows` / `publish_from_staging` / `resolve_dataset_path` / `dataset_axis_columns` |
| 治理/安全 | `authorize_dataset` / `authorize_uri` / `security_principal` / `access_policy` / `get_calendar` / `set_calendar` / `lock_calendars` |
| 观测 | `dataset_read_stats` / `build_dataset_manifest` / `load_columns` |

`__init__.py` 显式导出 48 个符号（`get_store` / `ReadHandle` / `QueryBudget` /
`PITEventIndex` / `MarketCalendar` / `COS_DATASET_CONTRACTS` / `build_contract_ir` 等），
并**重密封** `data_access.store.get_store` —— 从任何 import 路径进来都绕不过 COS 契约。

## 数据集注册表（`config/datasets.yaml`，77 个）

- **A 股 LQTP 全家桶（25 个）**：`ashare_stock_daily` / `_adj`（复权）、
  `ashare_stock_minute` / `_adj`、`ashare_stock_valuation_daily`、
  `ashare_stock_capital_daily`、`ashare_stock_indicator` / `balance` / `income` /
  `cashflow`（财务四表）、`ashare_turnover_base_daily`、`ashare_stock_dividend`
  （分红送转）、`ashare_index_daily` / `ashare_index_constituent`、
  `ashare_etf_daily` / `ashare_etf_list`、`ashare_calendar`、`ashare_stock_list`、
  `ashare_stock_industry` / `ashare_stock_status`、`ashare_universe_daily`、
  `ashare_stock_topten_shareholder`（十大股东）/ `_topten_float_shareholder`
- **美股（30 个）**：`us_stock_daily`、财务四表、`us_adj_factor` / `_clamped`、
  `us_security_master` / `_daily_snap`、`us_ticker_map` / `us_ticker_alias`、
  `us_fact_news`、SIP 分钟/tick 聚合（`us_stocks_sip_day_aggs` /
  `us_stocks_sip_minute_aggs` / `us_stocks_sip_quotes` / `us_stocks_sip_trades`）、
  short_interest / short_volume 等
- **因子湖（8 个）**：`factor_lake` / `factor_lake_wide` / `factor_matrix` /
  `factor_lake_staging` / `factor_values_stream` / `single_asset_backtest_runs`
- **流式/原始**：`streaming_bars` / `streaming_ticks_raw` / `massive_ticks` /
  `daily_market_summary` / `stocks_floats`

逻辑字段走 **语义字段目录**（`config/semantic_fields.yaml`）：
`close` / `open` / `high` / `low` / `vwap` / `return_bp` / `adj_factor` 等，
物理列名解耦（换数据源不改下游代码）；语义契约指纹 `semantic_contract_fingerprint()`。

## 读链路核心机制

```
read/read_auto → DataRequest 校验（engine/result/join/snapshot 白名单枚举）
  → contract_ir 契约 IR + 必填过滤强制（如必须给时间范围）
  → physical_plan（PlanNode）→ partition_planner 分区裁剪 + manifest min/max
  → scan_cost 成本估算 → read_auto_router 成本路由（本地 parquet / COS / DuckDB）
  → 引擎执行（auto/duckdb/polars/pyarrow）→ ReadHandle（auto/arrow/pandas/polars/lazy/stream）
```

- **Filter AST**：谓词抽象语法树 + DuckDB/Polars/PyArrow 三编译器（`predicate_ast.py`），
  同一过滤条件在哪个引擎都能下推
- **成本路由**：`read_auto` 按预估扫描成本（行数/字节/列数/文件数/远端延迟/选择性）
  自动选本地直读、DuckDB 查询还是远端扫描
- **查询预算**：`QueryBudget` deadline 到期主动取消（`DeadlineExceeded`）；
  `ManagedBatchReader` 批次托管；`runtime/resource_governor.py` 资源门控
- **结果缓存**：`query_cache.py` 读写身份指纹（data_read_identity / execution_identity），
  相同读请求命中缓存；缓存写入走 `core/atomic.py` 原子落盘
- **快照保真**：`snapshot/`（manifest / resolver / verifier / fidelity）——
  读到的数据对应哪个文件清单版本可追溯，manifest 变了 `is_snapshot_stale` 会报
- **流完整性**：`streaming_integrity.py` 流式读的行数/校验核对

## PIT（point-in-time，防未来函数）

- `PITContract` / `PITEventIndex` / `pit_event_index` —— 事件（财报/分红/公告）建立
  asof 索引，`read_asof` / `read_cos_events_asof` 只暴露**当时已知**的信息
- join 三策略：`exact`（严格对齐）/ `asof`（向后看齐）/ `pit_asof`
  （时点安全对齐，财务数据用它）
- 快照令牌：`manifest_version` / `is_snapshot_stale` —— 因子重算前先验快照没换

## 写与发布（受控写入）

`write/` 子包全链路：`generation.py`（新版本数据文件生成）→
`atomic_generation.py` + `generation_atomicity.py`（**原子落盘**，半写文件不可见）→
`publish.py` + `publish_manifest.py`（发布 = 换 manifest 指针，旧版本可回溯）→
`upsert.py`（幂等更新）→ `mutation_lock.py`（**变更锁**，写冲突直接拒绝）→
`authorization_boundary.py` / `metadata_security.py` / `object_storage_boundary.py`
（写权限/元数据/对象存储三道边界）。

配套 `compute_and_write`：读数据 → 本地计算 → 结果写到另一个数据集，
中间不落临时文件、不碰源数据。

## COS（对象存储，读 COS 一律走 quant-admin 的 admin-cos）

- 三模式：**mirror**（本地为主 COS 兜底）/ **remote**（COS 为主）/ **auto**（按成本路由）
- `cos://` URI 直接可读；底层 DuckDB httpfs + Secret Manager 取凭证
- **跨 domain（跨账号）bucket**（如 `qs-cold`）：本机静态密钥 IAM 不可见 →
  httpfs/boto3 直连 `NoSuchBucket`/404。snapshot 的 LIST/HEAD 元数据钩子
  （`store._cos_list_objects` / `_cos_head_object`）boto3 失败后自动落回
  **COS CLI 网关**（`cos_cli_ls` / `cos_cli_head`，解析 coscli `ls` 表格），
  回填真实 `content_length`/etag → governor 按真实 scan-bytes 准入（此前
  未知大小被按保守上界拒绝，表现为「remote 读永远被 gate 拒」）。数据面在
  跨账号 bucket 上仍走 **cli backend**（`DATA_ACCESS_COS_REMOTE_BACKEND=cli`
  或 auto 探测失败自动选 cli，经 clean-cos-ro materialize 到 cache 后本地读）
  ——httpfs 直读需要对该 bucket 有 IAM 权限的凭证（部署注入）。
- 顶层运行时模块：`cos_contract.py` / `cos_contract_ashare.py` / `cos_contract_us.py`
  （A 股/美股表契约）、`cos_panel_runtime.py` / `cos_factor_runtime.py` /
  `cos_event_runtime.py` / `cos_registry_runtime.py` / `cos_storage_runtime.py`
  （面板/因子/事件/注册表/存储五类运行时）、`cos_runtime.py` 总装
- `require_cos_contract(dataset)` —— 没有契约的表禁止从 COS 读（fail-closed）

## ClickHouse 面板

`clickhouse/panel.py` 面板表读（因子宽表落 CH 后查询）、`clickhouse/write.py`
`ensure_factor_table`（建表）/ `verify_factor_write`（写后校验行数与校验和）。

## 治理 / 安全 / 服务

- **安全**（`security/`）：principal/authorizer 模型、`governed_frame.py`
  （敏感列出域脱敏）、`redaction.py`、`run_mode.py`（研究/生产模式权限差异）、
  `credentials.py`
- **质量**（`quality/`）：`data-access-quality` CLI + 数据契约校验
- **服务**（`service/`）：FastAPI 服务端（`data-access-server`）+ `DataAccessClient`
  HTTP 客户端，含 HTTP 资源管理
- **导出**（`export/`）：serializers / importers / converters / versioning
- **遥测**（`telemetry/`）：opt-in 指标
- **构建身份**：`_build_info.py` 打包时冻结（build_sha / dirty 状态），
  安装产物不查环境变量——出问题能精确对到构建版本

## 目录

```
store.py          # 对外门面 DataAccessStore（~9600 行，55 公共方法），get_store()
core/             # DuckDB 引擎、原子 IO、审计、构建身份、重试
registry/         # 数据集注册：yaml 加载、schema/layout 校验、路径解析、边界
read/             # 读契约/AST/物理计划/成本路由/缓存/PIT/快照（45 个模块，最大子包）
write/            # 原子生成、发布、upsert、变更锁、三道安全边界
cos/ + 顶层 cos_*.py   # COS 三模式 + A股/美股契约 + 五类运行时
clickhouse/       # 面板读写
snapshot/         # 快照 manifest/解析/校验/保真
security/         # principal/授权/脱敏/run_mode
runtime/          # 读流水线、缓存管理、deadline、资源门控、启动门
service/          # FastAPI 服务 + HTTP 客户端
quality/          # data-access-quality CLI + 数据契约
export/           # 序列化/导入/转换/版本
config/           # datasets.yaml（77 表）、semantic_fields.yaml（语义字段）
telemetry/ ops/ benchmarks/ scripts/ deploy/  # 可观测/运维/基准/脚本/部署
tests/            # 168 个测试文件，1808 个测试函数（contract/security/r32/r39 分层）
```

## 口径（红线）

- **复权**：复权因子唯一来源 `adj_factor` / `adjusted_price_backward`；
  因子计算用后复权价
- **收益**：`return_bp` 单位是 bp（×1/10000）；下游标签一律 **vwap-to-vwap** 后复权
  （全平台硬性口径）
- **PIT**：历史快照读走 PIT/asof，禁止未来函数；好得离谱的回测先查数据口径
- **写**：所有写必须走受控写链路（原子生成 + 发布指针），禁止直接写数据目录

## 依赖与接口（谁 import 谁）

- **被谁调用**：`factor_engine`（build_data_source 的 data_access 后端）、
  `vectorbt_qs`（`mvp/data/adapter.py`，read_arrow 快路径 + COS fallback）、
  `riskfolio_qs`（benchmark/tradable/Return 取数）、`quant_evaluator`、
  `lightgbm_qs`（set_data_root 数据根）、`quant_platform`
- **依赖**：pyarrow / duckdb / polars / pandas / fastapi（按 extras 可选）
- **COS 凭证**：读 COS 走 quant-admin 的 admin-cos，勿用 research-cos

## 相关仓库

- **factor_engine** — 因子计算（数据源后端对接本库）
- **vectorbt_qs / riskfolio_qs / lightgbm_qs** — 回测与组合链（消费方）
- **quant_platform** — 平台服务（HTTP/DTO 对接）
