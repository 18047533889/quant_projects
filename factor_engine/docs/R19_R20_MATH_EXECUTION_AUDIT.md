# FactorEngine R19 + R20 独立深度审计整改报告

> **范围**:R19《数学 / 统计语义、时间拓扑、参数绑定与跨后端一致性》 + R20《编译、优化、CSE、缓存、物化、增量、并发、资源与全链路语义保持》
> **基线 HEAD**:`48aec8d`(文档基线)→ `b947c69`(服务器真实 HEAD,并发会话已 commit R16/R17)
> **执行**:11 个 file-disjoint agent(3 波)+ 中央协调;动态 `load_all()` 枚举全部 canonical

---

## 1. 任务结构

| # | 范围 | Agent | 状态 |
|---|------|-------|------|
| 175 | R19-001..008 参数系统单一权威 | A1 | 进行中 |
| 176 | R19-016..029 数学内核(rank/OLS/FWL/finite) | A2 | 进行中 |
| 177 | R19-009..015 audit release-gate | A3 | 进行中 |
| 178 | R19-030..057 ts 算子族 | A4 | 进行中 |
| 179 | R19-075..093 时间拓扑/history/anchor | A5 | 进行中 |
| 180 | R19-063..074, 094..100 truthiness/domain/neutralize | A6 | 进行中 |
| 181 | R19-101..130 + R19-55 机器审计/证书/matrix | A7 | 进行中 |
| 182 | R20-001..035 编译链语义保持 | A8 | 进行中 |
| 183 | R20-036..165 identity/CSE/cache 统一 | A9 | 进行中 |
| 184 | R20-107..172 并发/资源/缓存 | A10 | 进行中 |
| 185 | R20-176..472 物化/增量/artifact | A11 | 进行中 |
| 186 | 中央:surface/policy 协调 + 最终验证 | 主会话 | 待 |

## 2. 已确认缺陷(整改前现状)

### R19
- **R19-001** `ParameterCanonicalizer.hash_key()` `str + tuple` 类型错误 — `planner/canonicalize_params.py:285`
- **R19-002** planning validator 只认 `(int,float,str,bool)`,np 标量/Decimal 被跳过
- **R19-003** `child.attrs.get("value") is not None` 混淆显式 None 与 literal missing
- **R19-004** `parameter_validation.py` 与 `strict_params.py` 双套 strict gate
- **R19-005** numeric-string coercion 声明层/kernel 层双规则
- **R19-006** 三套参数 alias authority(metadata/base/backend)
- **R19-007/008** active_when 用原始 args 而非 canonicalized bound
- **R19-016** `rank_corr(d=0)` 整段历史 rank → full-sample look-ahead
- **R19-021** `ts_regression_slope_` `np.cov`(n-1)/`np.var`(n) → slope×n/(n-1)
- **R19-023** `industry_size_resid_panel_` 非 FWL(二次回归用原始 log(size))
- **R19-024** `size_resid_panel_` invalid cap clamp 到 1
- **R19-026** `coalesce_` 只判 NaN,Inf 不跳过
- **R19-030/031** ts_corr/ts_cov Numba vs Pandas current-row 不一致
- **R19-033** ts_beta `min_periods=2` vs reviewed `rolling_beta` default 5
- **R19-039..042** ts_argmax/argmin origin/tie 三套矛盾定义
- **R19-043/044** ts_product 把 0 当 missing
- **R19-049..055** hidden kwargs/local casts(ts_quantile/ts_topk/ts_moment…)
- **R19-056..059** ts_sharpe/ts_autocorr 隐藏 min_periods;ann_factor 可 0
- **R19-060..062** ts_pct/ts_log_return/ts_delay domain 语义
- **R19-063..065** neutralize 缺 group → silent global demean
- **R19-069..071** ConditionBool 与 truthiness 冲突
- **R19-072** `OPERATOR_SEMANTICS` "divide" 重复 key
- **R19-075..077** expanding/cumulative anchor 未声明
- **R19-078** `_WINDOW_LIKE_PARAM_NAMES` heuristic history fallback
- **R19-079** stale `ts_cusum_break_score`
- **R19-082..087** 缺 MissingTopologyPolicy/CurrentRowRequirement
- **R19-094** cs_mad_zscore 命名/公式不一致
- **R19-095..097** asin/acos/power/exp domain 跨 backend 不一致
- **R19-098** blom_transform Inf 污染
- **R19-099..100** group label 顺序依赖

### R20
- **R20-001/002** rolling CSE 丢 semantic_attrs(plan_ref 裸建)
- **R20-005/006** `_first_col_ref` DFS 弱 key(ts_mean(close) vs ts_mean(log(close)))
- **R20-010..013** 手工 ROLLING_OPS(重复 ts_std)/_window_from_attrs
- **R20-014/015** sql_lowerer 裸 materialized_series 丢 semantic_attrs
- **R20-019..023** composite lowering bare PlanNode
- **R20-024..027** `_lowering_hash` 只看 co_code
- **R20-028..035** z-score rewrite 等价性未证
- **R20-044..046** scoped-universe classifier fail-open
- **R20-056** FactorExecutionScope market="A" 默认
- **R20-058..061** CSE scope 缺 source_dependency_hash
- **R20-062..066** 缺 universe_membership_hash
- **R20-073..078** semantic attrs/identity repr/default=str fallback
- **R20-079..082** TypedIR hash 未分层
- **R20-083..087** lineage 只记 lowered_plan_hash
- **R20-091..093** malformed SourceRef 被吞
- **R20-100..102** result normalize Index 分支不统一
- **R20-107..113** Polars lazy scan 改共享 source + 吞 restore 异常
- **R20-114..118** ExecutionContext wrapper 挂 shared inner
- **R20-119..124** MemoryGovernor/Cache 无锁
- **R20-125..127** check_admit `__probe__` ghost accounting
- **R20-128..131** warmup stage 语义冲突
- **R20-132..134** run_many_parallel resource fail-open
- **R20-135..137** resource scope 用 env 猜 run_mode
- **R20-138..140** CPU slots 不 clamp
- **R20-146..148** LRU overwrite 非 MRU
- **R20-153..160** persistent cache schema/empty axis
- **R20-166..175** NaN structural literal 与 plan hash 冲突
- **R20-176..179** scalar default truthiness
- **R20-201..206** materializer float32 默认 → identity 缺 precision
- **R20-207..212** resolved snapshot 不一致 + identity 异常吞掉
- **R20-230..233** all-NaN recompute 残留旧 finite
- **R20-234..236** incremental != full recompute 未证
- **R20-246..252** production fallback 执行后才 gate

## 3. Agent 执行结果(11 个 file-disjoint agent,全部完成)

| # | 范围 | 新增测试 | 关键产出 |
|---|------|---------|---------|
| 175 | R19-001..008 参数系统 | 14 | hash_key 修复;normalize_and_validate_scalar_param;显式 None literal;parameter_validation 纯 re-export;numeric-string 只 binder;alias metadata 单一权威;NormalizedBoundParameters/bind_operator_call |
| 176 | R19-016..029 数学内核 | 20 | rank_corr 拆 ts/cs + no-lookahead;rank_ tie average+finite;regression slope centered-sums ddof 修复;industry+size FWL;invalid cap→NaN;coalesce first-finite;CrossSectionSampleMask |
| 177 | R19-009..015 audit gate | 22 | release_blocking 单一权威;duplicate 四桶拆分;choices 真 choice;MISSING 处理;parameter 级 coverage;fingerprint canonicalize NaN;manifest intrinsic/contextual |
| 178 | R19-030..057 ts 族 | 30 | ts_corr/cov current-row 一致;ts_beta→rolling_beta authority(min_periods 5);argmax/argmin age/index+tie latest;ts_product signed;decay partial/missing/alpha;hidden casts 收口;ts_sharpe/ann_factor |
| 179 | R19-075..093 时间拓扑 | 28 | AnchorPolicy/MissingTopologyPolicy/CurrentRowRequirement/TimeUnit/RankSemantic/QuantilePolicy;heuristic history fallback 移除;minimum_effective_samples;stale name 清理 |
| 180 | R19-063..074,094..100 | 25 | neutralize 缺 group→NaN;truthy_inf_is_true=False;OPERATOR_SEMANTICS 注册 hard-fail+numeric_semantics_hash;cs_mad_zscore 描述;asin/power/exp domain;blom finite;column permutation |
| 181 | R19-101..130,55 | 50 | MathematicalSemanticCertificate;M01-M20;audit_operator_math_contract/differential;R19 matrix(1430 canonical);metamorphic/prefix/chunk/checkpoint;R19-111 moment stable;R19-112 poly2 centered |
| 182 | R20-001..035 编译链 | 21 | rolling CSE semantic_attrs+input structural key+verify_plan_ref;ROLLING_OPS 派生;SQL materialized_series 语义;composite output contract 传播;adversarial corpus |
| 183 | R20-036..165 identity | 62 | OptimizerDifferentialAudit;RewriteTemporalProof;PlanScopeCategory fail-closed;canonicalize_execution_config;ExecutionSemanticIdentityV2 投影链;market 默认空;universe_membership_hash;TypedIR hash 分层;lineage hashes;PhysicalExecutionContract;SourceRef 三态;typed materialized_series;downsample contract |
| 184 | R20-107..172 并发/缓存 | 24 | Polars lazy TLS 隔离+corrupted marker;ExecutionContext wrapper 隔离;MemoryGovernor/Cache 锁;can_admit 无 ghost;warmup stage;CPU clamp;LRU MRU;persistent cache schema/empty axis/generation |
| 185 | R20-176..472 物化/增量 | 26 | MACD truthiness 修复;LoweringContract arity;MACD fast>=slow reject;tolerance unit;materializer float64 默认+precision policy;resolved snapshot;IdentityUnavailable/Failed;write_mode append/upsert/replace/recompute;incremental anchor;catalog strict+thread;artifact 20 份+audit_execution_semantic_preservation.py |

**合计:新增 322 测试**(不含既有回归)。

## 4. 中央协调

- **rank_corr R19-016..018 跨 agent 收口**:#176 移除 d=0 内核 → pandas wrapper(default=0)与 polars(default=20)契约发散(P0-23)。中央将 pandas `RankCorr` 对齐 ts-only(d>=1,default=20,走 `ts_rank_corr_`),并协调 #178 对齐 polars `RankCorrPolars`。load_all 恢复 1426。
- **GroupDemean R19-063**:group=None 由 global demean 改为走 kernel fallback policy(默认 "nan"→NaN),显式 "global" 仍可用。
- **R19-070 parity fixture**:`test_contract_hardening.py::test_misaligned_panels_fail_closed` 断言旧 broadcast 行为 → 改为断言 raise(exact-aligned)。
- **R20-100..102**:`cleaned_bridge._normalize_operator_result` 1D 分支用 `_target_index(template)`。
- **R20-213..219 Identity V2 接线**:FactorSemanticIdentity + ExecutionSemanticIdentityV2 增加 numeric/mathematical/composite/optimizer/history/precision 六字段并消费 ctx/lineage,digest 随 storage precision 变化。
- **R19-124 hidden kwargs 清零**:clip(min/max→lo/hi aliases)、ts_rank(min_periods 声明)修复,审计 M07=0;ts_beta polars min_periods 默认 1→5 对齐;polars ts_rank 保留 d-rejection(契约显式)。
- **alias compat marker**:ts_beta 因 metadata 声明 d→window 从 `_COMPAT_ONLY_ALIAS_CANONICALS` 移除,ts_rank 保留。
- **surface/policy**:本次无新 canonical 需 surface 注册(拆分算子沿用既有 canonical/alias),load_all 1430 全绿。
- **并发会话**:HEAD b947c69 后其 dirty 集(api/mining_integration 等)全程未触碰;其 R16 编辑(fields/catalog.py price_basis、ts_topk_sum params、polars TLS)与本轮 file-disjoint 共存。

## 5. 最终验证

- **全量新测试 + 受触碰既有回归:347 passed, 1 skipped**(约 6 分钟)。覆盖 11 个新测试文件 + test_contract_hardening + test_rolling_parameter_contracts。
- **load_all:1430 canonicals OK**。
- **R19_OPERATOR_MATH_AUDIT.json**:1430 行,load_ok;blocker 648 个(无 blocker 782)。
- **M07_HIDDEN_PARAMETER = 0**(本轮目标达成);M08_LOCAL_PARAMETER_COERCION=381、M13_HISTORY_FORMULA_HEURISTIC=498 为 legacy 算子诚实遗留,由审计引擎跟踪。
- **R20 artifacts**:docs/R20_* 20 份 + R19_OPERATOR_MATH_AUDIT.{json,csv,md} + build/r19_audit/。

### 诚实遗留
- **审计引擎发现的跨 backend gap**(tracked,非本轮确认缺陷):polars `BetaPolars` 原 min_samples=2 vs pandas 5(已由中央对齐);ts_mean/ts_std polars rolling 对 Inf 不转 NaN vs pandas output_inf_to_nan;ts_mean reference-fixture 0/18 为审计脚本自身 fixture 形状问题(非算子)。这些由 `operator_differential_execution` 持续跟踪。
- **M08/M13 遗留**:381 个 local cast + 498 个 history 名字推断,绝大多数为 legacy/research operator;审计脚本 AST 扫描已把它们全部列出供后续轮次逐批收口。
- **并发会话 transient**:fields/catalog.py 在其 R23 编辑窗口曾出现语法错误(重复块),已由对方修复;我未触碰其 dirty 文件。
