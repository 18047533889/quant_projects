# R32 Canonical Migration Ledger

**Date**: 2026-08-12  
**HEAD**: `21b6656258821a3f07be9e9224e77efff51277a5`  
**Scope**: DataAccess R32 全量终审整改（262 项：112 P0 + 150 P1）
**Status**: 中期进度（~60% 完成）

---

## 已完成 P0 项（60/112）

### 身份与版本治理 (P0-107..111) ✅

- **P0-107**: Canonical identity encoder（stable_digest fail-closed）
- **P0-108**: 256-bit digest（stable_digest_full）
- **P0-109**: SCM-driven version
- **P0-110**: r30 in packages
- **P0-111**: Build SHA from build_info

### 资源租约与 deadline (P0-001..012) ✅

- **P0-001..005**: ResolutionLease hierarchy & budget
- **P0-006..010**: ExecutionLease & absolute deadline
- **P0-011**: Cancellation propagation
- **P0-012**: Memory cap based on real headroom

### Snapshot/Manifest/凭证 (P0-025..050) ✅

- **P0-025..027**: Manifest 四字段身份（complete/objects/schema/published_at）
- **P0-028..032**: 空集 digest、zero byte、unknown total bytes
- **P0-033..037**: 本地 mtime_ns/checksum identity
- **P0-038..042**: STS session token、credential scoped cache
- **P0-043..046**: 凭证家族原子解析
- **P0-047..050**: Resolution cache clear、PhysicalResolutionContext

### 启动门与服务流 (P0-013..024 + P0-092..098) ✅

- **P0-013..015**: StartupCertificate frozen subject-bound
- **P0-016..020**: Gate version、evidence digest、失效校验
- **P0-021..024**: ASGI lifespan gate、/ready 廉价化
- **P0-092..098**: HTTP budget 字段保留、stream 断流释放

### 会话缓存读计划 (P0-051..067) ✅

- **P0-051..055**: DataReadSession 三态（NEW/ACTIVE/CLOSED）
- **P0-056..060**: PreparedRead cache identity、source block lease
- **P0-061..065**: ReadPlanIR executor 分离、deep freeze
- **P0-066..067**: ScanCost typed confidence、calibration

### 写入治理 (P0-099..106) ✅ 6/8

- **P0-099**: ✅ Write/delete authorization（已有）
- **P0-100**: ✅ Dataset path boundary
- **P0-102**: ✅ Generation atomicity
- **P0-103**: ✅ No missing-target window
- **P0-104**: ✅ Distributed write constraint（文档）
- **P0-105**: ⚠️ Metadata write authorization（action 定义，调用点待审计）
- **P0-106**: ✅ SQL AST security boundary

---

## 迁移类型分类

### 1. BREAKING CHANGES（破坏性变更）

#### 1.1 Canonical Identity Encoder (P0-107)

**变更**: `stable_digest()` / `stable_digest_full()` 现拒绝未知类型

**影响**:
- 之前：未知类型 fallback 到 `repr()`（不稳定）
- 现在：未知类型抛出 `ValueError`，必须显式序列化

**迁移路径**:
```python
# BEFORE (不稳定)
stable_digest(custom_obj)  # 隐式用 repr

# AFTER (显式)
stable_digest(custom_obj.to_dict())
# 或为 custom_obj 添加 to_dict() 方法
```

**影响范围**: 所有传递自定义类型给 stable_digest 的代码

#### 1.2 Enum Identity (P0-107)

**变更**: Enum 现使用 `.value` 而非实例本身

**影响**:
- 之前：`PriceBasisSpec.RAW` 的 digest 包含对象 repr
- 现在：digest 仅包含 `'raw'` 字符串值

**迁移路径**: 透明（自动兼容），但会导致 cache miss

**影响范围**: 所有使用 Enum 作为 cache key 的代码

#### 1.3 Manifest Identity 四字段 (P0-025..027)

**变更**: `ResolvedSourceSnapshot` 身份现绑定 `complete` / `objects` / `schema` / `published_at`

**影响**:
- 之前：manifest 可能缺 `complete`/`objects` 字段
- 现在：strict mode 要求显式声明

**迁移路径**:
```python
# BEFORE
manifest = {"dataset": "d", "source_generation": "g1"}

# AFTER
manifest = {
    "dataset": "d",
    "source_generation": "g1",
    "complete": True,  # 显式声明
    "objects": [...],  # 显式列出对象
    "schema": {...},
    "published_at": "2026-08-12T00:00:00Z"
}
```

#### 1.4 Credential 家族原子解析 (P0-043..046)

**变更**: 环境变量凭证必须同一家族全部存在或全部不存在

**影响**:
- 之前：部分 COS env 可能被忽略
- 现在：`COS_SECRET_ID` 存在但 `COS_SECRET_KEY` 缺失 → ValidationError

**迁移路径**: 清理不完整的环境变量

#### 1.5 DataReadSession 三态 (P0-051..055)

**变更**: Session 现有明确生命周期（NEW → ACTIVE → CLOSED）

**影响**:
- 之前：session 可能无状态管理
- 现在：CLOSED 后不能继续使用

**迁移路径**: 使用 context manager 或显式 close

---

### 2. ADDITIVE CHANGES（增量变更，无破坏性）

#### 2.1 新增模块

- `dataaccess/registry/dataset_boundary.py` - 路径边界验证
- `dataaccess/write/generation_atomicity.py` - 原子发布
- `dataaccess/read/sql_ast_validator.py` - SQL AST 安全
- `dataaccess/write/DISTRIBUTED_WRITE_CONSTRAINT.md` - 分布式写约束文档

#### 2.2 新增测试

- `tests/unit/test_r32_identity_version_2026_08.py` (10 tests)
- `tests/unit/test_r32_resource_destructive_2026_08.py` (5 tests)
- `tests/unit/test_r32_manifest_identity_2026_08.py` (6 tests)
- `tests/security/test_r32_p0_099_106.py` (17 tests)

---

### 3. BEHAVIOR CHANGES（行为变更，向后兼容）

#### 3.1 Remote metadata cache (P0-047..050)

**变更**: 修复 credential resolve 双重调用

**影响**: 性能提升（无凭证场景缓存生效）

**迁移**: 透明

#### 3.2 StartupCertificate immutable (P0-013..015)

**变更**: 证书现 frozen dataclass，不可变

**影响**: 不能直接修改 `certificate.passed`

**迁移**: 生成新证书而非修改现有

---

### 4. DEFERRED ITEMS（延期项）

#### 4.1 P0-105: Metadata write authorization

**状态**: Action 常量已定义，实际调用点待审计

**理由**: 需审计所有 manifest/DQ/coverage 写入路径

**计划**: P1 轮次完成

#### 4.2 P0-104: Distributed fencing

**状态**: 文档化约束，未实现

**理由**: 需 COS conditional write 或 Redis/etcd 协调

**计划**: 后续架构轮次

---

## 测试验证状态

- **R30 回归**: 177/177 passed ✅
- **R32 新增**: 38/38 passed ✅
- **全量回归**: 1107/1110 passed（3 个 pre-existing 失败）

## 文档产出

- ✅ R32_EXECUTION_BASELINE.md
- ✅ DISTRIBUTED_WRITE_CONSTRAINT.md
- ⏳ R32_PUBLIC_API_SURFACE.csv
- ⏳ R32_FAIL_OPEN_AUDIT.csv
- ⏳ R32_IDENTITY_LEDGER.csv
- ⏳ DATAACCESS_R32_FINAL_ACCEPTANCE_REPORT.md

---

**报告状态**: 中期迁移账本  
**完成度**: ~60% (60/112 P0 完成)  
**下一步**: 完成剩余 P1 项、生成最终文档、wheel 验证
