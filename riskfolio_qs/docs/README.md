# riskfolio_qs 文档索引

## 用户指南

| 文档 | 说明 |
|---|---|
| [`CLI使用指南.md`](CLI使用指南.md) | 回测 CLI 安装、配置、data_access 契约、持仓串联、输出验收与故障排查 |

## docs/design/ —— 设计规范（v0.2 现行）

| 文档 | 说明 |
|---|---|
| `riskfolio_qs_v0.2_交付形态与使用形态总纲.md` | 产品形态、使用入口、配置驱动总览 |
| `riskfolio_qs_v0.2_纯净数学定义.md` | 统一符号、收益分解、风险/成本/暴露的数学公式 |
| `riskfolio_qs_v0.2_数据链路初版.md` | 数据准备、字段契约、pivot 规则、输入校验 |
| `riskfolio_qs_v0.2_优化器映射规范.md` | optimizer_name → 目标/约束/依赖/降级 映射 |
| `riskfolio_qs_v0.2_冻结参数规范.md` | 参数分层、默认值冻结、白名单覆盖、校验规则 |
| `riskfolio_qs_v0.2_优化器与参数总表.md` | 单页总览：映射 + 参数 + 路由 + 审计 |
| `riskfolio_qs_v0.2_实施路线图_阶段拆解.md` | 阶段拆解、里程碑、交付节奏 |
| `上游信号平滑与换手门控方案_v0.1.md` | 信号平滑器设计：topN 换手率 + EMA 触发 |

**代码实现对应位置**：`src/riskfolio_qs/`，注释中标注了关联文档。

**配置资产**：`src/riskfolio_qs/configs/optimizer/mapping.v0.2.0.yaml`
和 `parameters.v0.2.0.yaml`，作为包资源随 wheel 分发。

---

## docs/archive/ —— 早期方案（v0.1 已归档）

| 文档 | 说明 |
|---|---|
| `组合优化器方案_v0.1.md` | 优化器初始方案 |
| `组合优化器实现方案_v0.1.md` | 实现方案初版（含 vendor 设计） |
| `组合优化与回测接口协议_v0.1.md` | 回测接口初版 |

---

## docs/reference/ —— 外部参考

| 文档 | 说明 |
|---|---|
| `Riskfolio-Lib九类接口数学与调用说明.md` | Riskfolio-Lib 九类接口的数学公式与 Python 调用路径 |

> 注意：v0.2 已移除 Riskfolio-Lib 依赖，直接使用 cvxpy。此文档保留作为数学参考。
