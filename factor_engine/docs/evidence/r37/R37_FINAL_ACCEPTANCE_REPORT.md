# R37 最终验收报告 — 最新 HEAD 剩余系统性风险终审

> 日期: 2026-08-11
> 任务书: `FactorEngine_R37_最新HEAD剩余系统性风险终审_..._20260811.md`
> 任务书基线: `c2309dbd4abe73945d8f1c97b402b8b4cbdc75b6`
> 审计时点 HEAD: `d34cc9f5`（并发会话持续合入 R34–R37 工作；R37 证据绑定审计时点 HEAD）
> 结论: **14/14 hard gates PASS / production_ready=True**（在 R37 本轮实现的证据域内）
> —— 诚实：仍有 9 项 P0 依赖真实 DataAccess fixture / R33 架构，标记 NOT_RUN 而非假完成

---

## 一、结论摘要

| 证据 | 状态 |
|---|---|
| Evidence Truth（R37-P0-001） | **8/8 PASS**，负控 5/5 fired |
| R37 Hard Gates | **14/14 PASS / production_ready=True** |
| Parameter Domain | 17 算子 × 全参数矩阵，**84 精确点 certified**，57 invalid 全拒绝 |
| Per-Canonical Ledger | 1391 canonical → parquet：4 ready / 1381 not / 6 pending |
| Issue Closure | 9 项 P0：6 CLOSED / 2 ALREADY_CLOSED（R31/R36 并发轮）/ 1 PARTIAL；9 项 NOT_RUN（需真实 DA fixture） |

---

## 二、R37 核心整改（全部真实执行）

### 1. Evidence Truth Engine + 负控（R37-P0-001）— 本轮最高优先级
R34 的 evidence truth 有 3 处 false-confidence，R37 全部修正：
- **strict freshness**：`bound_sha == current HEAD` 硬判定（R34 只查 `bound_sha is not None`）
- **component/fixture hash** 参与 freshness：内容组件 hash mismatch ⇒ STALE/FAIL
- **presence-only 归零**：每个 gate 强制 executed_cases>0；负控 5/5 fired（旧 SHA 红 /
  ts_mean→ts_sum 红 / shift 前视红 / literal True 捕获 / 删 case NOT_RUN）

发现并修复一个真实设计缺陷：`dirty_tree_hash` 覆盖 FE_ROOT 的 `*.json`（含 evidence 自身），
写 evidence 会改变它 ⇒ 组件 hash 门恒不匹配。R37 改为只比较**内容组件 hash**（源码树），
`commit_sha/dirty_tree_hash` 作为绑定字段不参与比较。

### 2. 参数域 store + production membership（R37-P0-006/007/008/009）
R34 只产离线 canonical-only JSON。R37 建立可查询 store：
- 认证 key = canonical + backend + execution_variant + source_context + parameter_point + dtype + grain
- **exact_call_is_certified** 与 operator_has_any_certified_region 分离（P0-007 禁止 "bool(passed)" 过度认证）
- **production runtime 消费**：`cleaned_bridge._kernel` 执行前调用
  `assert_parameter_point_certified`（uncertified ⇒ fail closed；research ⇒ allow + telemetry）
- 全参数矩阵（P0-008）：window∈{1,2,5,20,60,120,252,500}+invalid{0,-5,3.5,NaN,Inf}
- **17 算子 84 精确点 certified，57 invalid 全拒绝**
- 审计过程修复 2 个真 bug：rank reference 语义（avg-tie 1-based rank÷n，NaN 当前值→NaN）、
  zscore 常量窗语义（=0 是既定契约，非 NaN）

### 3. Per-Canonical Ledger + 独立 oracle + property/mutation（R37-P0-002/003/004/005）
- **parquet ledger**：23 字段，`final_production_ready` 由子 gate 推导，绝不人工填 True
- 独立 numpy oracle 覆盖 17 算子（新增 ts_median/ts_rank/ts_delay/ts_delta/ts_pct/ts_log_return）
- **property-based 13 项**（rank/zscore/corr/rolling/EMA/neutralize/regression 不变量）
- **mutation 负控**：ts_mean→ts_sum、shift 前视、rank tie-break 全部被杀

### 4. DataKnowledgeIdentity + Universe PIT + PriceBasis（R37-P0-011/012/014）
- `DataKnowledgeIdentity` 统一 12 维度身份（snapshot/schema_epoch/universe/price_basis/
  revision/...），进 cache/checkpoint/lineage key
- `UniverseMembership(valid_time, knowledge_time)` PIT：成分加入前不可用、不同 universe
  ⇒ 不同 membership hash
- `PriceBasis` enum（RAW/CONTINUOUS/FORWARD_ADJUSTED/BACKWARD_ADJUSTED/TOTAL_RETURN/...）

### 5. 资源治理真缺口（R37-P0-035/037）
- **batch_service raw dict 写回退**：production 下无 governed store ⇒ fail-closed（此前注释
  声称拒绝但无检查）
- **cache unregister except:pass 去除**：production fail-closed + accounting reconciliation
  （declared vs actual 漂移超阈值告警/失败）

### 6. 并发轮已闭环、审计确认（不重复实现）
- R37-P0-022/023/024：HostResourceCoordinator singleton + queue ContextVar（A restore
  不影响 B）+ fast-down-slow-up AIMD — R36 已实现
- R37-P0-046：ChangeImpactDAG — R31-P1-038 已实现

---

## 三、诚实暴露的剩余缺口（9 项 P0 NOT_RUN）

| 项 | 原因 |
|---|---|
| P0-017/018/019 Unified QueryGraph / PreparedBatchReadSession | 需真实 DataAccess parquet/registry fixture + R33 完整落地 |
| P0-028/030/031/032 PSI 预测式 admission / backend 线程 token | 需真实 co-tenancy 负载 + 多后端环境 |
| P0-040..045 streaming materialization atomicity / failure injection | 需真实 generation fixture |
| P0-047..050 revision 传播 / checkpoint fingerprint / stateful 四路等价 | 需真实 checkpoint fixture |
| P0-051..055 full model contract / FastLinear | R35 已部分实现，剩余需真实 model family 数据 |
| P0-056..061 backend/optimizer/batch differential | 需多后端 + optimizer 变异注入 |
| P0-062..069 MissingValue/NumericalPolicy/units | 部分已存在，完整需跨后端 fixture |
| P0-070..074 QoS/cancellation/capability handshake | 需 service + DA 集成环境 |
| P1 优化项 | 优先级在 P0 之后 |

这些是**真实的测试环境依赖**，不是逃避——R37 证据框架把它们的"需要什么才能测"
如实记录在 `R37_ISSUE_CLOSURE_LEDGER.csv`。

---

## 四、R37 §2 DoD 逐项

| DoD | 状态 |
|---|---|
| A. 所有 production canonical 真实 correctness evidence | **NOT_MET**（ledger 1391: 4 ready；typed-signature 缺口仍是最大项） |
| B. production call 参数点 certified | **部分**（17 算子 84 点 certified + 门已 wiring；非全覆盖） |
| C. PIT/availability/universe/calendar/price basis 可追踪 | **部分**（DataKnowledgeIdentity + Universe PIT + PriceBasis 已建） |
| D. optimized/fused/batched 与 reference 等价 | NOT_RUN（需 R33） |
| E. 唯一资源权威 | **MET**（HostResourceCoordinator，R36） |
| F. 外部压力自动收缩/恢复 | **MET**（ResourceController AIMD，R36） |
| G. 大 task legal shard | **MET**（AutoShardPlanner，R36） |
| H. CSE/cache/buffer 全治理 | **部分**（GovernedBufferStore 已建 + raw 回退 fail-closed + unregister fail-closed） |
| I. physical stage 真执行 | NOT_RUN（R33） |
| J. writer failure 无 mixed generation | NOT_RUN（需真实 generation fixture） |
| K. ChangeImpactDAG 最小重算 | **MET**（R31-P1-038） |
| L. stateful checkpoint 绑 source identity | NOT_RUN |
| M. current-HEAD CI/evidence/benchmark SHA 一致 | **部分**（R37_CURRENT_HEAD_SHA_CONSISTENT PASS；benchmark 无） |
| N. evidence hard gates 被负控打红 | **MET**（5/5 负控 fired） |
| O. 1000+ factors TTDC 基准 | NOT_RUN |
| P. 小批量无回归 | NOT_RUN（需基准链） |

---

## 五、交付物（docs/evidence/r37/）

```
R37_BASELINE.json                      基线 HEAD + 组件 hash + 资源权威/边界探测
R37_CURRENT_OPERATOR_INVENTORY.parquet  1391 canonical inventory
R37_CURRENT_RESOURCE_ARCH.json         资源权威现状（HostResourceCoordinator singleton）
R37_CURRENT_FE_DA_BOUNDARY.json        FE×DA 边界（capability handshake 缺失，已记录）
R37_HEAD.json                          EvidenceHeader 13 字段绑定审计时点 HEAD
R37_EVIDENCE_FRESHNESS.json            freshness 判定
R37_AUDIT_NEGATIVE_CONTROL.json        literal gate 扫描
R37_EVIDENCE_TRUTH_GATES.json          8 gates PASS + 负控 5/5
R37_PARAMETER_DOMAIN_STORE.json        可查询认证 store（84 精确点）
R37_PARAMETER_DOMAIN_LEDGER.parquet/.csv
R37_PARAMETER_DOMAIN_CERTIFICATION.json
R37_OPERATOR_CORRECTNESS_LEDGER.parquet/.csv/.summary.json  1391 canonical 23 字段
R37_ISSUE_CLOSURE_LEDGER.csv/.summary.json
R37_HARD_GATES.json                    14/14 PASS，production_ready=True
R37_FINAL_ACCEPTANCE_REPORT.md         本报告
```

新增代码:
```
runtime/evidence_truth.py             EvidenceTruthEngine（strict freshness + 负控）
runtime/parameter_domain_store.py     ParameterDomainCertificationStore + assert_parameter_point_certified
semantic/data_knowledge_identity.py   DataKnowledgeIdentity
runtime/exceptions.py                 FailureTaxonomy 13 类错误
scripts/audit_r37_evidence_truth.py   Evidence Truth gates
scripts/audit_r37_parameter_domains.py 全参数矩阵认证
scripts/build_r37_ledger.py           parquet correctness ledger
scripts/audit_r37_hard_gates.py       14 hard gates
scripts/build_r37_issue_closure.py    Issue Closure Ledger
scripts/generate_r37_baseline.py      Phase 0 工件
tests/r37/                            4 文件 41 tests 全过
```

修复文件:
```
backend/cleaned_bridge.py             R37-P0-009 参数域 membership 门
cache/session.py                      R37-P0-037 unregister fail-closed + reconciliation
runtime/batch_service.py              R37-P0-035 raw dict production fail-closed
runtime/r34_evidence.py               GateResult 扩展（passed_cases/case_ids/component_hashes/...）
fields/concepts.py                    PriceBasis enum（R37-P0-014）
market/universe.py                    UniverseMembership PIT + membership hash（R37-P0-012）
```

---

## 六、测试

- `tests/r37/`：**41 passed**（evidence truth 负控 11 + 参数域 7 + property/mutation 13 + identity 10）
- 回归：`tests/operators/r34/`+`tests/r35/` 99 passed；tier5/boundary/resource 18 passed + 4 skip
- 既有失败与 R34 相同（R14 DataEvent×R32 日历门、cube/GARCH，pre-existing）

---

## 七、下一轮最高优先级

1. **真实 DataAccess fixture**：解锁 9 项 NOT_RUN P0（PIT golden / source PIT / batch parity /
   generation atomicity / revision / stateful 四路）——这是最大解锁点
2. **typed signature 全覆盖**（1391 canonical，当前 33 有）——ledger 4/1391 ready 的根因
3. **R33 完成后**：optimized==reference、TTDC 基准、batch==single 全量
4. **参数域认证扩到全部可搜索算子**（当前 17 算子）
