# FactorEngine R21：最终收官轮——服务安全、任务治理、DQ、可观测、CI、发布与生产就绪独立审计整改提示词

> **独立文档**：不要与 R17/R18/R19/R20 合并。  
> **前提**：R17–R20 已提出的问题视为其它 AI 正在并行整改，本轮只处理前面没有系统覆盖的生产入口、服务治理、质量、可观测、发布和运行收官问题。  
> **GitHub 最新可见基线**：`b947c690a119c6fe42fe51b7fd2f9b5f0c746807`。服务器本地如更晚，以服务器真实 HEAD/dirty tree 为准。  
> **要求**：一次性修改、测试、重生 evidence/artifacts，并给出“可收官/不可收官”的机器结论；不要分优先级让我确认，不要修一部分就停。

---

## 0. R21 的收官目标

R17–R20 解决“引擎内部是否正确”；R21 解决“正确的引擎能否被安全、稳定、可恢复地长期运行”。

收官标准必须覆盖：

```text
production endpoint 不可降级
认证/授权不可绕过
HTTP 不能任意访问本地路径/内网数据源
公式/SourceRef/请求有资源预算
任务有排队、限流、超时、取消、重启恢复
幂等有请求绑定
DQ 能识别 schema/coverage/freshness drift
依赖/构建可复现
CI 有真实 HEAD 证据
性能有 SLO 和 regression gate
上线有 canary / rollback / runbook
```

---

# 一、Production HTTP 执行策略

### R21-001
当前 `/production/compute` 会先把 payload `run_mode` 写成 production，但 `config_path` 执行路径直接调用 `FactorEngine.run_from_config(config_path)`，没有把 endpoint 的 production policy 作为不可降低的 floor。

**整改**：引入 `EndpointExecutionPolicy.PRODUCTION`，production endpoint 调 config 时必须强制 production；config 中 research/关闭 PIT/DQ 等不得降低入口策略。

### R21-002
配置与 production endpoint 冲突时直接：
`PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT`，不能按 research 跑。

### R21-003
`/production/materialize` 同样必须把 production policy传到底层 `materialize_from_config`；当前 payload 的 `run_mode` 不能只停留在 service 层。

### R21-004
production endpoint 到 config loader、engine、source、PIT、DQ、write target、publish、catalog 全链使用同一不可降级 policy object。

### R21-005
增加 regression：
- production endpoint + research config -> reject；
- production endpoint + pit_enforce=false -> reject；
- production endpoint + direct-local write -> reject；
- production endpoint + source production=false -> reject。

---

# 二、Validate 与 Execute 不能使用不同上下文

### R21-006
`validate_spec()` 会读 `surface/dialect/dialect_version`，但 inline `_run_compute()` 当前 `parse_factor(formula, name=name)` 使用默认值。

### R21-007
同样检查 `freq/universe/market/calendar/decision_time_policy` 是否验证与执行完全一致。

### R21-008
新增 immutable `ValidatedFactorRequest`，验证后只执行该对象，不从 raw dict重新解析。

### R21-009
其 digest 至少包含：
`canonical_formula/surface/dialect/dialect_version/frequency/market/universe/calendar/decision_time_policy/production_policy/catalog generations/complexity budget`。

### R21-010
执行前：
`validated_request_digest == execution_request_digest`。

---

# 三、Service 认证/授权

### R21-011
当前 `_require_service_api_key()` 判断“production 是否必须 API key”主要看 `QUANT_PRODUCTION_MODE`，而不是 endpoint 的 production policy。

### R21-012
production compute/materialize route必须无条件 authenticated，不能因 `QUANT_PRODUCTION_MODE` 未设而 anonymous。

### R21-013
`FACTOR_ENGINE_SERVICE_ALLOW_OPEN` 只能影响明确 research/open route，不能影响 production route。

### R21-014
`X-Request-Identity` 不能作为可信 principal；任何持有共享 API key的人都能伪造该 header。

### R21-015
principal必须来自 API-key→principal mapping、JWT、mTLS、反向代理认证身份之一。

### R21-016
job 增加 owner principal / tenant / project。

### R21-017
`GET /jobs/{id}` 和 `/artifacts` 不仅认证，还要授权；默认 owner-only。

### R21-018
若系统明确单租户，就明确 single-principal contract，不要做“可填 identity 但实际共享 key”的伪多租户。

---

# 四、HTTP 数据源权限：当前存在路径访问和 SSRF 风险

### R21-019
inline HTTP 请求的 `data_source` 会直接进入 `build_data_source()`。

### R21-020
ClickHouse source允许 payload 指定 `host/port/database/username/password/table`。

**整改**：production HTTP只允许 `approved_source_profile_id`，不得由请求传 raw host/credentials。

### R21-021
增加 ClickHouse host/database/table allowlist 和 network egress policy。

### R21-022
研究服务若允许自定义 host，也必须单独权限，不与 production 共用。

### R21-023
Parquet source的 `root` 经过 `resolve_path()` 时允许 arbitrary absolute path。

### R21-024
production HTTP必须限制到 approved data roots / registered DataAccess datasets。

### R21-025
绝对路径不在 approved root -> reject。

### R21-026
production service优先只开放 DataAccess dataset ID，不开放任意 filesystem root。

### R21-027
config 文件中的 source 同样要经过 approved-source policy，不因 config_path 已授权就放行任意内部 host/path。

---

# 五、Config path hardening

### R21-028
当前 `_authorized_config_path()` 先 `resolve()` 再 `is_symlink()`；如果目标本来是symlink，resolve 后通常已看不到原symlink身份。

### R21-029
若政策是“任何symlink禁止”，对原始 path 每个component做 `lstat()`，或用 no-follow/openat。

### R21-030
防 TOCTOU：授权检查和最终读取尽量基于同一 fd/verified inode。

### R21-031
config root不可由普通 request覆盖。

---

# 六、HTTP 请求必须 typed

### R21-032
停止直接用 raw dict + `bool(payload.get("sync"))`。

### R21-033
例如 `"sync":"false"` 在 Python 中是 truthy，当前可能误触发同步执行。

### R21-034
所有 request/response 用 Pydantic model，`extra="forbid"`。

### R21-035
枚举字段严格：
`run_mode/backend/source_profile/surface/dialect/job_type/write_target`。

### R21-036
设置长度/cardinality：
formula、factor name、idempotency key、metadata、instrument_filter、source refs。

---

# 七、DSL / SourceRef 资源型 DoS

### R21-037
ComplexityBudget 当前在 `ast.parse()` 后 visitor阶段生效；超大文本已经先完成 parser 分配。

### R21-038
parse 前增加 `max_formula_bytes/max_formula_chars`。

### R21-039
LQTP normalization后再次检查长度，防 normalization expansion。

### R21-040
增加 `max_string_literal_bytes`；一个数MB字符串目前仍只是一个 AST Constant。

### R21-041
增加 `max_identifier_length/max_keyword_length`。

### R21-042
SourceRef base64/json decode增加 token bytes、decoded bytes、param count、key/value length限制。

### R21-043
所有 size check在 `json.loads/base64 decode/ast.parse` 前尽可能早执行。

---

# 八、AST 大小不等于计算成本

### R21-044
建立 `FactorCostEstimate`，在任务入队前估算：
rows/cells、history、window、operator complexity、intermediate bytes、source count、SQL count、materialization bytes。

### R21-045
production service设置：
`max_cost_per_job/max_rows/max_cells/max_history/max_sources/max_expected_memory`。

### R21-046
high-cost算子即使 AST很短也必须触发 cost gate。

### R21-047
cost estimate写入 job manifest 和 lineage。

---

# 九、任务队列必须 bounded

### R21-048
ThreadPoolExecutor只限制 running workers，不等于有界业务队列。

### R21-049
建立 bounded queue；队列满返回 429/503 + Retry-After。

### R21-050
限制 global / per-principal：
running、queued、hourly cost、daily cost。

### R21-051
暴露 queue depth / wait time metrics。

---

# 十、禁止 heavy sync 阻塞 HTTP

### R21-052
当前 async endpoint允许 payload `sync=true` 后直接同步执行 FactorEngine。

### R21-053
production compute/materialize全部入job system；用户不能强制 heavy sync。

### R21-054
如保留sync，只用于validate/tiny smoke并有严格deadline。

---

# 十一、Job deadline / cancel / heartbeat

### R21-055
JobStatus扩展：
`submitted/queued/running/cancelling/cancelled/timed_out/rejected/interrupted/succeeded/failed`。

### R21-056
JobRecord增加：
deadline_at、timeout_seconds、cancel_requested_at、attempt、worker_id、heartbeat_at、phase。

### R21-057
增加 cancel endpoint。

### R21-058
cancel必须传播到 DuckDB/ClickHouse/Polars/materializer，而不是只改状态。

### R21-059
running job定期 heartbeat；无heartbeat超阈值标 stuck/interrupted。

### R21-060
真实phase：
VALIDATING/COMPILING/PREFLIGHT/READING/EXECUTING/DQ/MATERIALIZING/PUBLISHING/FINALIZING。

---

# 十二、ClickHouse 资源治理不能绕过 QueryBudget

### R21-061
DuckDB路径有 QueryBudget；ClickHouse执行当前直接 `execute_query(config, sql)`。

### R21-062
ClickHouse统一接入：
max_execution_time、max_result_rows/bytes、max_memory_usage、max_threads、max_read_rows/bytes、query_id、cancel token。

### R21-063
切 backend不能同时绕过资源治理。

### R21-064
QueryBudget env非法值 production启动直接失败，不能静默退默认。

---

# 十三、Idempotency 必须绑定请求

### R21-065
当前 idempotency index本质是 `key -> run_id`。

### R21-066
正确key scope：
`principal + job_type/endpoint + idempotency_key`。

### R21-067
持久化 `canonical_request_hash`。

### R21-068
同key不同 request hash -> `409 IDEMPOTENCY_KEY_CONFLICT`。

### R21-069
failed job重试语义明确：logical_job_id和attempt_id分离。

---

# 十四、Job manifest必须可恢复

### R21-070
当前 manifest直接 `write_text`，改为 tmp+fsync+replace+checksum+schema version。

### R21-071
坏manifest不能 silent continue；quarantine + alert + corruption metric。

### R21-072
当前 to_public manifest不保存完整 request，restart无法真正resume submitted/running job。

### R21-073
选择：
- 持久化安全的 canonical request并可resume；
- 或restart时把所有旧 nonterminal job标 INTERRUPTED。
不能留下幽灵running。

### R21-074
startup reconciliation重建 idempotency 时处理 interrupted/stale running。

### R21-075
manifest增加 request_digest / execution_policy_digest / attempt / worker_id。

---

# 十五、Process-local JobStore 不足以多worker

### R21-076
当前 STORE是模块内存dict；多uvicorn worker/多pod会各有一套job/idempotency。

### R21-077
production改 durable queue/store（SQLite事务队列/Postgres/Redis等），或强制single process。

### R21-078
如果暂时single-process，启动时检测workers>1直接拒绝。

### R21-079
多process idempotency并发测试。

---

# 十六、Graceful shutdown

### R21-080
FastAPI lifespan：
停止收新job、处理queued、请求running取消/宽限、flush manifests、关闭executor/catalog/source。

### R21-081
SIGTERM/K8s termination测试，不能留下半物化/幽灵running。

### R21-082
production禁止 `--reload`。

---

# 十七、Liveness / Readiness 分离

### R21-083
`/livez` 只证明进程活。

### R21-084
`/readyz` 验证：
registry generation、field/provider/evidence artifact一致、resource plan、catalog schema、disk capacity、queue admission。

### R21-085
readyz不做昂贵全量查询。

### R21-086
R17–R20 blocker非零时 production ready=false。

---

# 十八、错误输出与secret redaction

### R21-087
public job error只返回 stable error_code + sanitized message + error_id。

### R21-088
full traceback只内部日志/受控artifact。

### R21-089
统一redact password/token/DSN/API key/Authorization/internal secrets。

### R21-090
异常信息包含 SQL/path/host 时按安全策略清洗。

---

# 十九、Output DQ 当前明确问题

### R21-091
duplicate `(timestamp,instrument)` 检查不能只在 unique timestamp>=2 时执行；单日重复也必须失败。

### R21-092
coverage使用finite，但 `min_instruments_per_day` 当前用notna；统一finite policy。

### R21-093
preserve_invalid_rows时Inf不能被当有效instrument。

### R21-094
DQ报告保存 sample policy和role contract。

---

# 二十、DQ 必须 role-aware

### R21-095
不要一套 DQThresholds 打所有operator/factor。

### R21-096
建立 Alpha/Condition/Event/State/GroupState/GlobalState/Intermediate DQ contract。

### R21-097
Condition：输出只能 `{0,1,NaN}` 或声明的三值逻辑。

### R21-098
Rank/Pct：范围 `[0,1]`。

### R21-099
Category：合法 code domain。

### R21-100
GlobalState：允许同截面广播；Alpha不应意外全截面constant。

### R21-101
DQ不做IC/收益筛选，只做结构正确性。



# 二十一、DQ 要看时间分布，不只全样本平均

### R21-102
增加：
daily coverage、latest-day coverage、rolling coverage、coverage quantiles、coverage change。

### R21-103
识别“历史99天正常、最新一天全空”。

### R21-104
识别 all-constant / unexpected all-zero / variance collapse / cardinality collapse。

### R21-105
输出 DQ保存按日期/股票池的 coverage profile。

---

# 二十二、Input DQ 必须来自 Field/Provider contract

### R21-106
fundamental/event/news/snapshot字段天然稀疏，不能全部用统一 `min_non_null_ratio=0.01`。

### R21-107
阈值来自 FieldSpec / provider expected coverage + requested universe/window。

### R21-108
Input DQ增加 dtype、unit、timestamp timezone、instrument dtype、duplicate key、schema version、partition schema一致性。

### R21-109
source freshness：
latest source timestamp、expected trading session、lag、freshness SLA。

### R21-110
multi-source factor检查共同 date/instrument overlap和snapshot coherence。

---

# 二十三、核对 DataAccess sidecar stats 的 null/non-null 含义

### R21-111
当前 `adjust_input_dq_thresholds_from_stats()` 读取名字为 `column_null_ratio` 的属性，但逻辑/注释把数值当“非空率”使用。

### R21-112
去 DataAccess真实 DatasetStats schema核实。

### R21-113
若字段确实是 null ratio，则必须先 `1-null_ratio`。

### R21-114
禁止模糊名字，typed schema分别保存 `null_ratio/non_null_ratio/finite_ratio`。

### R21-115
增加极端fixture：
0% null / 90% null / 100% null。

---

# 二十四、数据质量异常必须分“数据坏”还是“因子本身稀疏”

### R21-116
DQ错误码区分：
SOURCE_EMPTY、SOURCE_STALE、SOURCE_SCHEMA_DRIFT、EXPECTED_SPARSE、FACTOR_DEGENERATE、OUTPUT_DOMAIN_VIOLATION。

### R21-117
Event factor稀疏不能误判为数据源坏。

### R21-118
Alpha factor突然仅1只股票有效必须阻断。

---

# 二十五、Production 依赖必须可复现

### R21-119
pyproject 当前核心/optional dependency多为 `>=` 下限；生产部署不能直接用无限上界解析。

### R21-120
保留library兼容range，但production deployment生成exact constraints/lock。

### R21-121
lock至少覆盖 Python、numpy、pandas、pyarrow、scipy、numba、polars、duckdb、clickhouse client、data_access。

### R21-122
build artifact记录 lock digest。

### R21-123
任何依赖升级先跑 R19/R20 numerical/execution matrix再进入production。

---

# 二十六、Execution provenance记录真实依赖版本

### R21-124
每次run记录全部关键库版本。

### R21-125
依赖版本属于 RunExecutionProvenance，不一定属于 FactorSemanticIdentity。

### R21-126
当版本变化造成语义变化时，通过 semantic policy/hash体现，而不是靠猜。

---

# 二十七、版本号只允许一个authority

### R21-127
pyproject和service app当前都出现 `0.3.1`，删除硬编码复制。

### R21-128
统一由 package metadata / build metadata导出。

### R21-129
HTTP `/readyz`、lineage、artifact manifest输出相同 build version/hash。

---

# 二十八、运行时不能修改 sys.path 来寻找 DataAccess

### R21-130
SQL executor当前 `_ensure_data_access()` 会把 quant_projects root插到 `sys.path`。

### R21-131
这会破坏“安装环境实际使用哪个 data_access”的可复现性。

### R21-132
production删除runtime sys.path injection。

### R21-133
DataAccess作为正常package dependency加载，启动时验证 imported path/version/build hash。

---

# 二十九、Wheel 必须在脱离 monorepo 的环境可运行

### R21-134
构建wheel后在clean venv + 临时目录运行 smoke。

### R21-135
不得依赖 editable install、repo parent、monorepo `.env`、source checkout。

### R21-136
安装wheel后至少测试 parser/compiler/pandas smoke和可选extras安装。

### R21-137
evidence/docs package data与wheel中的代码generation一致。

---

# 三十、Python 顶层 namespace 冲突审计

### R21-138
当前安装包暴露 `api/backend/runtime/storage/fields/...` 等通用顶级包名。

### R21-139
做 clean environment namespace collision audit。

### R21-140
长期分发建议迁移到 `factor_engine.*` namespace；短期不迁则生产env严格隔离。

---

# 三十一、`.env` 自动加载不能成为production authority

### R21-141
生产服务默认禁止自动读取monorepo `.env`，除非明确dev profile。

### R21-142
production secrets来自secret manager/deployment environment。

### R21-143
启动输出 non-secret config digest。

---

# 三十二、非法 run mode 必须 fail closed

### R21-144
若 `FACTOR_ENGINE_RUN_MODE` 明确存在但值非法，例如 `prodution`，当前不能退回research。

### R21-145
环境变量存在但非法 -> startup hard fail。

### R21-146
`QUANT_PRODUCTION_MODE` 与 `FACTOR_ENGINE_RUN_MODE` 冲突 -> startup fail。

### R21-147
production endpoint不依赖ambient run mode判断自身安全等级。

---

# 三十三、Feature flags 统一 typed policy

### R21-148
FASTPATH/POLARS_EXPR/QUERY_BUDGET/ALLOW_OPEN等不能分散在任意模块临时读env。

### R21-149
启动期构造 `RuntimeFeaturePolicy`。

### R21-150
每个run保存 feature-policy digest。

### R21-151
未知/非法flag值production直接失败。

---

# 三十四、Formula / API 版本迁移

### R21-152
persisted source_expr必须有 `formula_schema_version`。

### R21-153
operator rename、param rename、dialect upgrade建立 migration registry。

### R21-154
定义 deprecate_since / remove_after / replacement。

### R21-155
旧factor不能永久靠legacy alias存活。

### R21-156
新增 `migrate_formula(from_version,to_version)`。

---

# 三十五、Canonical textual serializer

### R21-157
从 typed Expr/IR生成唯一canonical formula representation。

### R21-158
`parse -> canonical serialize -> parse` identity必须一致。

### R21-159
用于 catalog diff、migration、human review、reproducibility。

### R21-160
用户原始source_expr与canonical_expr都保存，二者角色不同。

---

# 三十六、Public API compatibility matrix

### R21-161
测试 parse_factor / Factor / FactorEngine.run / run_many / run_incremental / materialize / YAML / HTTP request。

### R21-162
破坏性变化必须版本升级并给migration。

### R21-163
旧生产factor corpus定期重放。

---

# 三十七、统一 error taxonomy

### R21-164
稳定错误族：
VALIDATION、POLICY、PIT、SOURCE、SCHEMA、DQ、RESOURCE、TIMEOUT、CANCEL、CACHE、MATERIALIZE、BACKEND、INTERNAL。

### R21-165
HTTP映射明确 400/401/403/409/422/429/503/500。

### R21-166
错误码进入metrics，不依赖解析message文本。

---

# 三十八、Retry policy按错误类型

### R21-167
validation/PIT/schema/identity/DQ/policy violation永不retry。

### R21-168
网络/瞬时DB故障 bounded retry + jitter。

### R21-169
每个retry有 attempt id。

### R21-170
materialization retry必须幂等，不重复publish/watermark。

---

# 三十九、Job 运行日志与审计

### R21-171
每个job绑定：
request_id、run_id、execution_id、principal、factor identity、snapshot、attempt。

### R21-172
所有 backend/fallback/DQ/materialization日志可按 run_id关联。

### R21-173
结构化JSON logging，不靠自由文本grep。

### R21-174
日志不得输出完整未脱敏config/secrets。

---

# 四十、核心 metrics

### R21-175
至少：
job_queue_depth、job_wait_seconds、run_latency、compile_latency、source_read_latency、rows/bytes read、peak RSS、cache hit/miss、backend routes、fallback count、DQ failures、materialize bytes、SQL queries、cancel/timeout。

### R21-176
按 mode/backend/factor role/market聚合，控制标签基数。

### R21-177
factor name不要无控制作为高基数Prometheus label。

---

# 四十一、Tracing

### R21-178
支持 OpenTelemetry或等价span：
request→validate→compile→source read→backend execute→DQ→materialize/publish。

### R21-179
SQL query_id与trace/span关联。

### R21-180
慢factor能定位到底是source、operator、serialization还是materialization。

---

# 四十二、Production SLO/性能收官

### R21-181
不要只“跑得动”，建立性能基线：
compile p50/p95、single factor、100 factor batch、1000 factor batch、incremental、cold/warm cache。

### R21-182
代表性 A股 daily panel / minute-to-daily / fundamental mix分别benchmark。

### R21-183
指标：
latency、peak RSS、query count、bytes scanned、stack/unstack count、fallbacks。

### R21-184
性能gate基于相对baseline，不写死跨机器毫秒数。

### R21-185
默认 regression阈值可配置，例如 p95/peak RSS 恶化>15%需要显式批准。

---

# 四十三、SQL batch 当前“chunk alias”不等于真正降低物化峰值

### R21-186
检查 batch SQL long/wide结果是否先完整 materialize到 Arrow/Polars，再chunk select。

### R21-187
如果是，chunk只降低下游view宽度，不降低SQL结果本身峰值内存。

### R21-188
真正大batch需要列分批查询、streaming、partitioned materialization或结果sink。

---

# 四十四、Backpressure 不只在HTTP队列

### R21-189
source read、SQL materialization、CSE shared panels、final results之间都要有背压。

### R21-190
禁止“上游无限生产、下游慢慢落盘”。

### R21-191
sink/materialize路径优先流式，不保留所有factor结果。

---

# 四十五、Capacity test

### R21-192
逐步提高：
concurrent jobs、factor count、stock count、history length、minute data、source count。

### R21-193
找到资源拐点并形成 `CapacityEnvelope`。

### R21-194
超过容量在admission阶段拒绝，不等OOM。

---

# 四十六、CI：当前 HEAD 必须有实际证据

### R21-195
不能因为GitHub没有红status就认为绿。

### R21-196
最终release commit必须有明确 CI generation / test artifact。

### R21-197
CI保存 test summary、coverage、semantic audit artifacts、dependency versions。

---

# 四十七、CI 测试矩阵

### R21-198
至少：
unit、integration、backend parity、market/PIT、operator audit、service security、materialization/incremental、concurrency、package-wheel。

### R21-199
支持的 Python / pandas / numpy / Polars / DuckDB版本矩阵。

### R21-200
production锁定版本必须是强制必跑组合。

---

# 四十八、Property-based tests

### R21-201
对 DSL、params、SourceRef、Plan serializer、CSE/optimizer、DQ加入 Hypothesis或等价生成式测试。

### R21-202
性质包括：
roundtrip、prefix invariance、permutation invariance、chunk invariance、cache cold/warm、serial/parallel。

---

# 四十九、Fuzz tests

### R21-203
fuzz DSL parser：
随机AST、恶意大输入、Unicode、边界字面量、深度、奇怪字符串。

### R21-204
fuzz SourceRef decoder：
bad base64、truncated JSON、huge payload、duplicate semantic fields。

### R21-205
fuzz SQL emitter：
identifier、string literal、date、instrument codes，确保无注入和quoting错误。

### R21-206
fuzz service JSON：
类型错、超长、extra fields、sync字符串、巨大instrument lists。

---

# 五十、Mutation testing

### R21-207
对最重要的 semantic gates做 mutation test：
把 `>`改`>=`、删PIT检查、删semantic hash、改ddof、关fail-closed。

### R21-208
测试必须能杀死这些mutation，否则“绿测试”覆盖不足。

---

# 五十一、Static typing / lint / dead-code

### R21-209
对 production core开启 pyright/mypy或等价严格检查。

### R21-210
尤其 service request、PlanNode semantic contracts、identity、DQ reports不能大量Any逃逸。

### R21-211
ruff/flake等静态规则加入 CI。

### R21-212
dead imports/dead compatibility shims定期清理。

---

# 五十二、安全扫描

### R21-213
CI加入 dependency vulnerability scan（pip-audit等）。

### R21-214
secret scan。

### R21-215
Bandit/semgrep类规则重点：
subprocess、eval/exec、SQL字符串、path、pickle、yaml unsafe loader、requests network。

### R21-216
生产镜像最小权限、只读filesystem区域、非root用户。

---

# 五十三、Config schema版本与未知字段

### R21-217
YAML/JSON run config有 schema_version。

### R21-218
production unknown fields全部reject。

### R21-219
config migration显式，不默认忽略旧字段。

### R21-220
配置canonical serialization/hash可复现。

---

# 五十四、No hidden research escape hatch

### R21-221
全库扫描参数/flag：
research、unsafe、allow_unknown、skip_validation、allow_future、fallback、warn_only、force。

### R21-222
production入口必须证明这些escape hatch全部被锁死或有明确权限。

### R21-223
不能通过generic `/jobs/compute` 构造一个“看似production但绕过production endpoint validation”的请求。

---

# 五十五、Service endpoint contract收敛

### R21-224
明确：
research compute、production compute、production materialize、validate、operators、job status/cancel。

### R21-225
generic `/jobs/compute` 若保留，不能成为绕过production专用route的后门。

### R21-226
根据 run_mode自动走同一个 validation/admission policy，不允许少一道门。

---

# 五十六、Materialize privilege应高于 compute

### R21-227
生产落盘/发布是写权限，不应只凭普通compute API key。

### R21-228
RBAC区分 READ/COMPUTE/MATERIALIZE/PUBLISH/ADMIN。

### R21-229
publish动作记录principal和审批/operation id。

---

# 五十七、Factor lake/Catalog backup 与恢复

### R21-230
定义 catalog backup策略。

### R21-231
factor partitions/catalog不一致时有 reconcile工具。

### R21-232
定期演练：
catalog丢失、单partition损坏、checkpoint损坏、manifest损坏。

### R21-233
恢复后 identity/watermark仍一致。

---

# 五十八、Schema migration必须可回滚

### R21-234
Catalog/factor lake/job manifest schema migration不做不可逆原地修改而无backup。

### R21-235
migration有 from/to version、dry-run、rollback或forward recovery。

---

# 五十九、Canary / shadow validation

### R21-236
R17–R21全部实现后，不直接宣布收官。

### R21-237
用固定真实snapshot跑“旧稳定路径 vs新路径”shadow comparison。

### R21-238
比较：
values、NaN mask、universe、lineage、backend path、runtime/memory。

### R21-239
已解释的预期差异形成 migration report；未解释差异 blocker。

---

# 六十、真实因子 corpus

### R21-240
构建 production golden corpus，不只是toy operator。

### R21-241
至少覆盖：
量价、技术、基本面、事件、group neutralize、minute→daily、SourceRef、多source、stateful/high-cost。

### R21-242
包含真实 AlphaProbe/AlphaMiner能生成的表达式形态。

---

# 六十一、Regression corpus长期冻结

### R21-243
保存 formula + execution scope + data snapshot + expected semantic identity + expected output digest。

### R21-244
每次重大改动自动重放。

### R21-245
数据本身不能漂；使用immutable test snapshot。

---

# 六十二、上线 rollback

### R21-246
新版本上线必须可以快速切回上一已认证engine generation。

### R21-247
factor lake写入按generation隔离，避免新版本先污染旧版本可见数据。

### R21-248
rollback不依赖手工删文件。

---

# 六十三、Feature rollout

### R21-249
新backend/new optimizer/new cache按feature policy/canary开启。

### R21-250
不要直接100%全量切换。

### R21-251
canary metrics异常自动停止扩大，不自动修改factor定义。

---

# 六十四、Production runbook

### R21-252
至少写清：
PIT失败、DQ失败、source stale、schema drift、OOM、cache corruption、SQL outage、materialization partial failure、job stuck、artifact generation mismatch。

### R21-253
每类给：
symptom、error code、first checks、safe remediation、禁止操作、rollback。

---

# 六十五、Ownership / on-call 信息

### R21-254
每个生产模块有owner：
data contract、operator semantics、runtime、storage、service、deployment。

### R21-255
审计artifact标 owning module/team，避免问题无人处理。

---

# 六十六、Documentation必须自动生成

### R21-256
operator docs、field docs、service schemas、config docs尽量从metadata生成。

### R21-257
禁止手工文档与真实ParamSpec/roles/fields长期漂移。

### R21-258
docs generation diff进入CI。

---

# 六十七、Example代码也要测试

### R21-259
README/docs中的示例作为 doctest/smoke。

### R21-260
防止公开API已改但文档仍教旧入口。

---

# 六十八、生产默认值必须审计

### R21-261
扫描所有 `default=` / env fallback。

### R21-262
对production关键项：
run mode、PIT、DQ、source policy、write target、fallback、auth、resource budget——默认必须fail-safe。

### R21-263
“未配置”与“明确关闭”区分。

---

# 六十九、启动期 ProductionPreflight

### R21-264
服务进入ready前一次性检查：
config、auth、artifact generation、dependency versions、resource limits、data roots/hosts、catalog schema、disk space。

### R21-265
任何安全/语义关键UNKNOWN -> not ready。

### R21-266
不要等第一笔实盘job才发现配置错。

---

# 七十、运行期 invariant monitor

### R21-267
每个job结束验证：
no unauthorized fallback、no identity drift、no resource-accounting negative、no unresolved source、no partial publish、DQ status known。

### R21-268
invariant failure把job标 INTERNAL_CONTRACT_VIOLATION。

---

# 七十一、内存/文件/线程泄漏 soak test

### R21-269
连续执行数百/数千短job，监测：
RSS baseline、fd count、thread count、temporary files、DuckDB connections、cache bytes。

### R21-270
job完成后资源回到稳定区间。

### R21-271
失败/cancel路径同样测试。

---

# 七十二、多并发 soak

### R21-272
重复 concurrent submit/cancel/materialize/query。

### R21-273
验证无deadlock、manifest corruption、idempotency race、catalog lock starvation。

---

# 七十三、故障注入

### R21-274
在关键阶段注入：
DB断连、disk full、permission denied、OOM guard、timeout、process kill、bad cache、catalog busy。

### R21-275
每个故障必须产生可恢复状态，不出现“成功status但数据没写完”。

---

# 七十四、Clock/timezone 测试

### R21-276
service/job timestamps固定UTC。

### R21-277
deadline用monotonic clock计算duration，wall clock只记录展示时间。

### R21-278
系统时钟回拨不能让timeout失效。

---

# 七十五、审计数据保留

### R21-279
job manifest/log/lineage有retention policy。

### R21-280
删除策略不能先删审计证据后留factor数据。

### R21-281
敏感metadata按合规要求脱敏/最小化。

---

# 七十六、生产写入必须有审批边界

### R21-282
如果materialize/publish影响共享因子湖，至少支持：
dry-run、validate-only、staging、explicit publish。

### R21-283
research用户不能直接publish production namespace。

---

# 七十七、没有“自动修复数据”式危险恢复

### R21-284
DQ/schema/PIT失败默认阻断。

### R21-285
系统不能在production偷偷fill/clip/drop直到通过。

### R21-286
任何 repair必须显式policy并进入lineage。

---

# 七十八、最终 artifacts

### R21-287
生成：
`R21_SERVICE_SECURITY_AUDIT.{md,json,csv}`

### R21-288
`R21_JOB_LIFECYCLE_AUDIT.{md,json}`

### R21-289
`R21_DQ_CONTRACT_AUDIT.{md,json,csv}`

### R21-290
`R21_REPRODUCIBLE_BUILD_AUDIT.{md,json}`

### R21-291
`R21_OBSERVABILITY_SLO_AUDIT.{md,json}`

### R21-292
`R21_CI_TEST_MATRIX.{md,json,csv}`

### R21-293
`R21_RELEASE_READINESS.{md,json}`

### R21-294
`R21_FAILURE_INJECTION_REPORT.{md,json}`

### R21-295
`R21_PRODUCTION_CLOSURE_SCORECARD.{md,json}`

---

# 七十九、R21 Release Blockers

### R21-296
至少定义：
S01_PRODUCTION_ENDPOINT_DOWNGRADE
S02_PRODUCTION_AUTH_BYPASS
S03_JOB_AUTHORIZATION_MISSING
S04_UNAPPROVED_REMOTE_SOURCE
S05_UNAPPROVED_LOCAL_PATH
S06_UNBOUNDED_REQUEST
S07_UNBOUNDED_JOB_QUEUE
S08_NO_DEADLINE_CANCEL
S09_CLICKHOUSE_BUDGET_BYPASS
S10_IDEMPOTENCY_REQUEST_MISMATCH
S11_JOB_RECOVERY_BROKEN
S12_MULTIPROCESS_JOBSTORE_UNSAFE
S13_GRACEFUL_SHUTDOWN_MISSING
S14_READINESS_FALSE_POSITIVE
S15_DQ_CONTRACT_BUG
S16_SCHEMA_FRESHNESS_UNKNOWN
S17_UNREPRODUCIBLE_DEPENDENCIES
S18_RUNTIME_SYSPATH_INJECTION
S19_INVALID_RUNMODE_FAIL_OPEN
S20_PUBLIC_API_MIGRATION_MISSING
S21_RETRY_SIDE_EFFECT_RISK
S22_OBSERVABILITY_INCOMPLETE
S23_PERFORMANCE_CAPACITY_UNKNOWN
S24_CI_EVIDENCE_MISSING
S25_SECURITY_FUZZ_MISSING
S26_RELEASE_ROLLBACK_MISSING
S27_RECOVERY_NOT_TESTED
S28_RESOURCE_LEAK
S29_ARTIFACT/BUILD_GENERATION_MISMATCH
S30_ZERO_BLOCKER_GATE_NOT_MET

---

# 八十、最终“可收官”判定

### R21-297
不能以“R21文档修完”为收官。

### R21-298
必须满足 R17–R21 所有 release blocker = 0。

### R21-299
最新实际服务器HEAD必须完成全套测试，不允许拿旧commit evidence。

### R21-300
fresh process / clean environment重跑全套关键audit。

### R21-301
连续至少3次完整production-readiness run结果一致且无flaky retry。

### R21-302
真实固定snapshot golden corpus全部通过。

### R21-303
serial/parallel、CSE on/off、backend matrix、cache cold/warm、incremental/full、materialized reload全部在R20矩阵收口。

### R21-304
service security/admission/cancel/restart/failure injection在R21收口。

### R21-305
built wheel clean-install smoke通过。

### R21-306
production locked dependency environment通过。

### R21-307
canary/shadow comparison无未解释差异。

### R21-308
rollback演练通过。

### R21-309
zero open blocker + zero UNKNOWN critical contract。

---

# 八十一、整改 AI 最终一次性执行顺序

1. 读取真实server HEAD/dirty/source hash；
2. 核对R17–R20当前完成情况，只把未完成项作为dependency，不在R21重复改写；
3. 修production endpoint policy override；
4. 修service auth/authz；
5. 收紧source profile/path/network权限；
6. 建typed request；
7. 建pre-parse size/cost budget；
8. 建bounded queue/deadline/cancel/heartbeat；
9. 修idempotency和durable job recovery；
10. 修graceful shutdown/readiness；
11. 修DQ具体bug并升级role-aware/freshness/schema contracts；
12. 收敛deployment config/env；
13. 建exact dependency lock和wheel clean install；
14. 去runtime sys.path injection；
15. 建error taxonomy/retry policy；
16. 建metrics/tracing/SLO/capacity；
17. 建property/fuzz/mutation/security CI；
18. 建formula/config migration和canonical serialization；
19. 建backup/recovery/fault-injection；
20. 建canary/rollback/runbook；
21. 生成R21全部artifacts；
22. 与R17–R20最终blockers合并成一个**只用于验收的 closure scorecard**（注意：不要把五份整改文档本身合并）；
23. 只有全部critical blocker=0才输出 `PRODUCTION_CLOSURE_READY=true`。

---

# 八十二、最后的原则

FactorEngine真正收官，不是“operator多、测试多、代码大”，而是：

```text
正确性可证明
安全门不可绕过
任何输入有边界
任何任务可治理
任何失败可恢复
任何结果可追溯
任何版本可复现
任何上线可回滚
```

只要 production endpoint还能降级为research、HTTP还能指定任意服务器路径/内网主机、任务不能取消/重启恢复、DQ和schema drift仍是UNKNOWN、最新HEAD没有CI证据，就不能宣布收官。


---

# 八十三、当前离收官还有多远：工程判断框架

> 这不是按代码行数估算，而按 production closure domains 估算。整改 AI 应在服务器真实 HEAD 上重新打分。

建议分 6 个域，各 0–5 分：

```text
A. Data / PIT / Market contract
B. Operator / Math / Parameter correctness
C. Compiler / Backend / Cache / Incremental equivalence
D. Mining direct-use / grammar / deletion cleanup
E. Service / Security / Job / DQ / Observability
F. CI / Reproducibility / Release / Recovery
```

评分：

```text
0 = 基本未做
1 = 有框架
2 = 主路径可用但大量 UNKNOWN
3 = 大多数主路径完成，仍有 blocker
4 = blocker基本清零，只差真实canary/soak
5 = production closure evidence完整
```

### R21-310
生成 `R21_PRODUCTION_CLOSURE_SCORECARD` 时必须逐域列：
score、evidence、open blockers、unknowns、next closure action。

### R21-311
不要因为某域测试数量多就给高分；必须看 blocker 和 UNKNOWN。

### R21-312
只要任一域 <4，不允许宣布最终收官。

### R21-313
全部域>=4且critical blocker=0，才进入canary/soak。

### R21-314
全部域达到5，才可标：
`FACTOR_ENGINE_PRODUCTION_CLOSED=true`。

### R21-315
当前 GitHub visible main 已经显示 R17/R18 部分代码在继续落地，但 R19/R20 文档刚同步到 main，不能仅凭文档存在给 B/C 高分；必须以实际代码和运行 evidence重新评分。
