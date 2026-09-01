# platform_web — 量化研究平台 Web 前端

Quant Research Platform 前端（QRP-P9 / task #43）：React 18 + TypeScript (strict) + Vite 5。
20 个页面镜像平台信息架构；仪表盘**只渲染后端 API 返回的数据**，前端不做任何计算。

**仓库:** https://github.com/HKUST-QUANT-SOCIETY/platform_web (private)
**后端合同来源:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform（`app/contracts/*`）

> 本环境**没有 node/npm**——类型检查与构建在 GitHub Actions CI 里跑
> （`.github/workflows/platform-web.yml`）。

## 技术栈

| 关注点 | 库 |
|---|---|
| UI | React 18 |
| 语言 | TypeScript（strict） |
| 构建 | Vite 5 |
| 数据请求/缓存 | TanStack Query |
| 表格 | TanStack Table |
| 路由 | React Router 6 |
| 图表 | ECharts（已声明，暂未使用） |
| 测试 | Vitest |

## 页面（20）

Overview（总览）、Factors、Campaigns、Treatments、Clusters、Libraries、FeatureSets、
Models、Backtests、Optimizers、LiveHealth、Streaming、Alerts、Jobs、DataSnapshots、
Artifacts、Standards、Exports、Audit、UsersRoles。

## 脚本

```sh
npm run dev        # vite dev server
npm run typecheck  # tsc --noEmit
npm test           # vitest run（纯 helper）
npm run build      # tsc --noEmit && vite build → dist/
```

## 构建（CI）

`platform-web.yml`：setup-node + `npm ci` → `typecheck` + `test` + `build`，上传 `dist/`。
需要本地生成 `package-lock.json`（`npm install`），让 setup-node 缓存 key 生效。

## 约束

- 纯前端脚手架：**无领域逻辑、无 Python**，绝不在前端重算指标。
- `src/api/types.ts` 镜像 `quant_platform/app/contracts/*`（类型即合同）。
- "无数据"绝不渲染成 0：缺失值显示 `—`；标签未成熟的证据显示
  `LABEL_NOT_MATURE` 部分证据状态（spec §40/§49）。

## 相关仓库

- **quant_platform** — 后端合同/API（types 的唯一真相源）
