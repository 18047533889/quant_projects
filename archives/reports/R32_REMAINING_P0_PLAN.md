# DataAccess R32 剩余 P0 项实施计划

**日期**: 2026-08-12  
**完成状态**: 60/112 P0 完成（54%）  
**剩余**: 52 P0 待执行

---

## 已完成的 6 个簇（60 项）

1. ✅ **身份版本打包** (P0-107..111, 5 项)
2. ✅ **资源租约 deadline** (P0-001..012, 12 项)
3. ✅ **Snapshot/Manifest/凭证** (P0-025..050, 26 项)
4. ✅ **启动门服务流** (P0-013..024 + P0-092..098, 19 项)
5. ✅ **会话缓存读计划** (P0-051..067, 17 项)
6. ✅ **写入治理** (P0-099..106, 6/8 项)

---

## 剩余 P0 项分簇计划（52 项）

### 第七簇：DQ Coverage 与变更影响 (P0-068..086, 19 项) 🔴 HIGH PRIORITY

**范围**: Data quality checker、Coverage grain、ExperimentSnapshot identity、Change impact

**关键缺口**:
1. **P0-068..072**: DQ checker 异常返回 PASS（fail-open）
2. **P0-073..077**: Coverage 自然日 vs 交易 session grain 错配
3. **P0-078..082**: ExperimentSnapshot 单一 ID 混合 source/execution/security
4. **P0-083..086**: FE change_impact 只取 changed_columns[0]，多列丢失

**估计工作量**: 中等（需审计 DQ checker 全路径 + ExperimentSnapshot 重构）

**依赖**: 无（可立即开始）

**产出**:
- `dataaccess/quality/dq_checker.py` 修改（fail-closed）
- `dataaccess/snapshot/experiment_snapshot.py` 重构（分离三类身份）
- `dataaccess/coverage/grain.py` 新增（session-based grain）
- 测试: `tests/unit/test_r32_dq_coverage_2026_08.py`

---

### 第八簇：FE×DA 绑定层 (P0-087..091, 5 项) 🔴 HIGH PRIORITY

**范围**: FactorSourcePlan typed binding、依赖提取 fail-closed、双 planner 权威统一

**关键缺口**:
1. **P0-087**: `FactorSourcePlan` 未存 typed `ColumnSourceBinding`
2. **P0-088**: leaf_concepts / source_datasets 分离元组需合并
3. **P0-089**: 依赖提取失败返回 None（fail-open）
4. **P0-090**: FactorBatchPlan vs ReadWavePlanner 双权威
5. **P0-091**: Planner estimate vs actual backend counter 边界

**估计工作量**: 中等（跨 FE-DA 边界，需协调）

**依赖**: 无（可立即开始，但需 FactorEngine 协同）

**产出**:
- `dataaccess/read/source_binding.py` 新增（typed ColumnSourceBinding）
- `dataaccess/read/dependency_extract.py` 修改（fail-closed）
- `dataaccess/read/planner.py` 修改（deprecate FactorBatchPlan）
- 测试: `tests/unit/test_r32_fe_da_binding_2026_08.py`

---

### 第九簇：元数据写授权审计 (P0-105 补完, 1 项) 🟡 MEDIUM PRIORITY

**范围**: Audit all metadata write call sites and add authorization gates

**关键缺口**:
- Action 常量已定义（`ACTION_METADATA_WRITE`）
- 实际调用点未审计：manifest write、DQ write、coverage write

**估计工作量**: 小（审计 + 插入 authorize_dataset 调用）

**依赖**: 无

**产出**:
- `dataaccess/write/publish_manifest.py` 修改（add auth gate）
- `dataaccess/quality/dq_store.py` 修改（add auth gate）
- `dataaccess/coverage/coverage_store.py` 修改（add auth gate）
- 测试: 扩展 `tests/security/test_r32_p0_099_106.py`

---

### 第十簇：HTTP 与流控制 (P0-112..120, 散项 9 项) 🟡 MEDIUM PRIORITY

**范围**: HTTP 缓冲、stream 断流释放、exact-once semantics

**关键缺口**:
1. HTTP response 缓冲未限制（OOM 风险）
2. Stream 断开后资源未释放
3. HTTP exact-once 语义未证明

**估计工作量**: 小到中等

**依赖**: P0-092..098 已完成（HTTP budget 字段保留）

**产出**:
- `dataaccess/service/app.py` 修改（buffering limit）
- `dataaccess/service/stream.py` 修改（disconnect hook）
- 测试: `tests/integration/test_r32_http_stream_2026_08.py`

---

### 第十一簇：Symlink 与 TOCTOU (P0-121..128, 散项 8 项) 🟢 LOW PRIORITY

**范围**: Symlink TOCTOU、production lock、price basis fail-open

**关键缺口**:
1. Symlink 解析存在 TOCTOU 窗口
2. Production lock 未 fail-closed
3. Price basis fallback 逻辑 fail-open

**估计工作量**: 小

**依赖**: 无

**产出**:
- `dataaccess/store/path_resolver.py` 修改（symlink follow=False）
- `dataaccess/security/production_lock.py` 修改（fail-closed）
- `dataaccess/market/price_basis.py` 修改（fail-closed）
- 测试: `tests/security/test_r32_symlink_toctou_2026_08.py`

---

### 第十二簇：其他散项 (剩余 10 项) 🟢 LOW PRIORITY

**范围**: 未分类的其他 P0 项

**估计工作量**: 小

**依赖**: 需逐项评估

---

## 推荐执行顺序

### Phase 1（立即开始）: High Priority 簇

1. **第七簇**: DQ Coverage 与变更影响 (P0-068..086) — 19 项
2. **第八簇**: FE×DA 绑定层 (P0-087..091) — 5 项

**预计时间**: 2-3 个工作会话  
**预计产出**: 24 项 P0 完成，84/112 P0 (75%)

### Phase 2（后续）: Medium Priority 簇

3. **第九簇**: 元数据写授权审计 (P0-105) — 1 项
4. **第十簇**: HTTP 与流控制 (P0-112..120) — 9 项

**预计时间**: 1 个工作会话  
**预计产出**: 10 项 P0 完成，94/112 P0 (84%)

### Phase 3（收尾）: Low Priority 簇

5. **第十一簇**: Symlink 与 TOCTOU (P0-121..128) — 8 项
6. **第十二簇**: 其他散项 — 10 项

**预计时间**: 1-2 个工作会话  
**预计产出**: 18 项 P0 完成，112/112 P0 (100%)

---

## 并行策略

**可并行**:
- 第七簇（DQ Coverage）与第八簇（FE binding）无依赖，可分配不同 subagent 同时执行
- 第九簇（metadata auth）可与第十簇（HTTP stream）并行

**必须串行**:
- 第十二簇散项需先评估依赖再执行

---

## 风险评估

### 高风险

1. **FE×DA 协同**: 第八簇跨 FE-DA 边界，需 FactorEngine 同步修改
2. **ExperimentSnapshot 重构**: 第七簇涉及核心身份契约，影响面广

### 中风险

1. **DQ checker 全路径审计**: 可能发现更多 fail-open 点
2. **HTTP exact-once**: 需证明语义，可能无法完全保证

### 低风险

1. 其他簇均为局部修改，影响可控

---

## 成功标准

每个簇完成后：
1. ✅ 所有该簇 P0 项 CLOSED
2. ✅ 新增测试全绿
3. ✅ 全量回归不退化
4. ✅ 文档更新（CANONICAL_MIGRATION_LEDGER / FAIL_OPEN_AUDIT）

---

**下一步**: 启动第七簇（DQ Coverage）和第八簇（FE binding）并行实施
