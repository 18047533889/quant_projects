# platform_web — 量化研究平台 Web 前端

Quant Research Platform 前端（QRP-P9 / task #43）：React 18 + TypeScript (strict) + Vite 5。
**20 个页面**镜像平台信息架构；仪表盘**只渲染后端 API 返回的数据**，前端不做任何计算。

**定位:** 企业级 A 股横截面日频多因子量化项目的**展示层** —— 只读后端 API，零领域逻辑。

**仓库:** https://github.com/HKUST-QUANT-SOCIETY/platform_web (private)
**后端合同来源:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform（`app/contracts/*`）

> 本环境**没有 node/npm**——类型检查与构建在 GitHub Actions CI 里跑
> （`.github/workflows/platform-web.yml`，位于 monorepo 根）。

## 技术栈

| 关注点 | 库 |
|---|---|
| UI | React 18.3 |
| 语言 | TypeScript 5.6（strict，tsconfig 全开严格项） |
| 构建 | Vite 5.4 |
| 数据请求/缓存 | TanStack Query 5.59（`useQuery` hooks） |
| 表格 | TanStack Table 8.20 |
| 路由 | React Router 6.26（`NavLink` 侧栏） |
| 图表 | ECharts 5.5（已声明，暂未使用） |
| 测试 | Vitest 2.1（node 环境，只测 `src/lib` 纯 helper） |

engines `node>=18`；37 个 TS/TSX 文件（约 3056 行，不含 vite.config.ts）。

## 页面（20，导航为 spec §25 的 1 级 6 组）

**Overview / Research / Models / Portfolio / Production / Platform**

| 路由 | 页面 | 一句话功能 |
|---|---|---|
| /overview | Dashboard | 系统运营态势：10 KPI + 9 段流水线漏斗 + 8 项运营信号（spec §26） |
| /factors | Factors | 因子目录，列镜像平台 factor-summary 响应（§27） |
| /campaigns | Campaigns | 因子生成活动列表 |
| /treatments | Treatments | 处理搜索/治疗结果，TreatmentRef 由 factor_assets 携带（§8.4） |
| /clusters | Clusters | 全局聚类运行 ClusterSetVersion + 分层聚类模型（§14/§34/§35） |
| /libraries | Libraries | FactorLibraryVersion 治理资产集，晋升/回滚（§15/§16） |
| /feature-sets | Feature Sets | 带类型、有序成员的版本化特征集（§17/§18） |
| /models | Models | 模型注册表，只读/占位直至正式注册表接入（§2） |
| /backtests | Backtests | 正式执行回测（合同 + 未来 adapter，§41） |
| /optimizers | Optimizers | 处理搜索优化器（§35） |
| /live-health | Live Health | 生产因子资产运行健康（§9 HealthState） |
| /streaming | Streaming | 增量更新：signal-available 时钟 vs label-matured 时钟（§40） |
| /alerts | Alerts | 平台健康告警 |
| /jobs | Jobs | JobSpec/JobRecord/JobResult（§9/§42/§43），需 `job:read` |
| /data-snapshots | Data Snapshots | DataAccess 数据快照（§38） |
| /artifacts | Artifacts | 已发布制品注册表，content_hash 不可变引用（§7.2） |
| /standards | Standards | 平台标准/策略注册表，需 `standards:read` |
| /exports | Exports | 报告导出（§48） |
| /audit | Audit Log | 访问决策与治理动作审计追踪（§23/§24.4），需 `audit:read` |
| /users | Users & Roles | 主体/团队/角色/安全分类（§24.1），需 `user:manage` |

## API 层如何镜像 contracts

- **`src/api/types.ts`**：逐节（§4.1–§4.14 注释标注）镜像 `quant_platform/app/contracts/*` DTO：
  ArtifactRef / JobSpec / JobRecord / EventEnvelope / FactorCandidateManifest /
  FactorAsset（前端专用聚合）/ ClusterSetVersion / ClusterVersion / FactorLibraryVersion /
  LibraryMembership / FeatureSetVersion / FeatureMemberRef / 身份 refs /
  AdmissionRequest+Verdict / Backtest+Model DTO / TimingContract / ModelVersion /
  SnapshotManifest / WorkflowSpec+Run / HumanPrincipal+WorkloadPrincipal /
  DashboardSummary 三件套（Kpis/Funnel/Signals）。全部 thin mirror，无逻辑。
- **`src/api/enums.ts`**：枚举/常量镜像 —— LifecycleState(8) / QRPPipelineStage(11) /
  HealthState(6) / JobStatus(10) / EvidenceStatus(4) / ClusterConversionType(6) /
  LibraryStatus(4) / SecurityClassification(5) / Role(5) / Team(5) / ArtifactType(15) /
  EventType(19) / FeatureSetDiffCategory(8)。
- **`src/api/client.ts`**：`request<T>()` 包装 fetch，`API_BASE_URL = VITE_PLATFORM_API_BASE_URL ?? '/api/v1'`；
  解析规范错误信封（§50：code/message/request_id/retryable）；403→`PlatformApiError`
  （code=PERMISSION_DENIED）；`platformApi` 对象约 **30 个端点**。
- **`src/hooks/index.ts`**：20 个 useQuery hooks，queryKey 派生自资源路径，staleTime 30s，
  retry 只尊重 API retryable 标记。

## 约束

- 纯前端脚手架：**无领域逻辑、无 Python**，绝不在前端重算指标。
- `src/api/types.ts` 镜像 `quant_platform/app/contracts/*`（类型即合同）。
- **"无数据"绝不渲染成 0**：`src/lib` formatters（formatDate/formatBytes/formatCount）
  null/undefined → `—`；`EvidenceCell` 无状态 → `—`（KPI 值为 0 是合法数字，只有缺失值显示 `—`）。
- **六态渲染**（spec §49）：loading/empty/permission-denied/partial-evidence/error/stale，
  由 `classifyState` 纯函数 + `StatePanel` 组件实现，`ResourceListPage` 统一驱动；
  `LABEL_NOT_MATURE`/`NOT_COMPUTED` → partial-evidence 态（"Values are not zero-filled until labels mature"）。

## 脚本与构建

```sh
npm run dev        # vite dev server
npm run typecheck  # tsc --noEmit
npm test           # vitest run（纯 helper：classifyState 六态/empty/partial 文案/formatters）
npm run build      # tsc --noEmit && vite build → dist/
```

**CI**（`.github/workflows/platform-web.yml`，monorepo 根）：setup-node 20 + `npm ci` +
typecheck + test + build + upload dist/ artifact；push main / PR / workflow_dispatch 触发。
需要本地生成 `package-lock.json`（`npm install`），让 setup-node 缓存 key 生效。

## 相关仓库

- **quant_platform** — 后端合同/API（types 的唯一真相源）
