# V3–V8 整改直接集成 main

用户明确要求直接修改 server-c 主目录并推送 main。本轮未新建分支、worktree 或整库副本。

## 集成位置

- 正式工作目录：`/home/sunhaiwei/quant_projects`，分支 `main`。
- 集成前 main：`91b6539c0df3bbfdd96262828375fd71860fd567`。
- 将此前整改目录相对 `2db45f4e446309003381a51a5e03071d31859502` 的改动三方合并到最新 main，而不是覆盖最新 main。
- 合并范围包含 QE、FA、FO、FP、quant_platform、modeling、jobs，以及关联 FE、DataAccess、VectorBT 代码和测试。
- 缓存、行情数据、大型历史归档不进入提交；原目录和必要备份保留。

## 冲突及集成修复

- 保留 main 新增的 FE 拟合作用域、失败证据、Butterworth 采样频率及全量重放语义。
- 合并 CPU/GPU 回撤实现：保留缺失值、并列值、年化参数和精确高水位；净值为零后保持零，低于 −100% 的收益在无负资本合同时明确拒绝。
- Calmar 保留 main 的 CAGR 默认口径；算术年化仍可显式选择。旧测试改为分别断言两种口径，并未放宽数值断言。
- 保留等总敞口持仓记账和不依赖未来收益的成员选择。
- 保留 main 的构建元数据；验收任务的运行目录改为从当前源码位置推导。
- 发现并修复 SQLite 去重索引初始化竞争：有界重试、失败连接清理；并发测试有超时和失败传播，不再无限等待。该并发测试另连续运行 20 次通过。

## 主目录冻结回归

全部测试从正式主目录运行，分别启动进程；验证前后源码哈希一致。

| 测试组 | 通过 | 失败 | 跳过或预期失败 |
|---|---:|---:|---:|
| QE | 1230 | 0 | 2 |
| FA 主测试 | 1480 | 0 | 0 |
| FA 独立并发测试 | 1 | 0 | 0 |
| FO | 885 | 0 | 0 |
| FP | 377 | 0 | 1 |
| 平台 | 464 | 0 | 21 |
| 建模与任务 | 466 | 0 | 0 |
| VectorBT | 102 | 0 | 0 |
| FE / DataAccess / 历史验收账本接口 | 186 | 0 | 0 |
| 合计 | 5191 | 0 | 24 |

源码哈希清单 SHA256：`2e22619f0cda71e7178498dea6c5fd3e85fdda0a8fde4db12fb5f6f830ec168e`。

执行摘要和源码哈希清单位于 `evidence/main_remediation_20260909/`。原始日志、首轮旧 Calmar 测试失败和第二轮并发等待的记录保留在服务器 `/home/sunhaiwei/quant-main-integration.Fwo5WJ/`。未将中断或跳过的测试当作通过；上表只统计最后完整成功回归。

## 实际运行环境

修复了主环境仍导入旧 FO/FP wheel 的问题。当前虚拟环境中的 QE、FA、FO、FP 和平台均验证为从正式主目录导入。安装过程没有下载依赖或复制虚拟环境。

嵌套布局的 FO/FP 使用 setuptools 的 editable compatibility 模式；在其他开发环境复现时，可从仓库根目录安装：

```sh
python -m pip install --no-deps --no-build-isolation -e ./quant_evaluator -e ./quant_platform -e ./factor_assets
python -m pip install --no-deps --no-build-isolation --config-settings editable_mode=compat -e ./factor_optimizer -e ./factor_preprocess
```

## 边界

本次是代码集成和主目录验证，不是生产部署、数据库迁移或交易策略切换。原 V8 仍未闭合的 T70、T116、T117、T119、T120、T123、T165，不会因合并到 main 而自动成为通过；真实数据校准、完整下游失效迁移、最终组合压力回放、十万规模实测、断段 HMM 和异步能力仍按原限定范围处理。
