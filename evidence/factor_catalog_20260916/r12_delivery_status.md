# R12 实施与验证交接（2026-09-16）

## 正式产物

- server-c 主工作树：/home/sunhaiwei/quant_projects。
- 最新表：evidence/factor_catalog_20260916/factor_catalog_review_r12e.csv.gz
- SHA256：d9668cd615ea28c9008159a6cd84bc99c9e1d29e38e04d459b3c4f0d2abb75d5
- 相邻 manifest 记录源表、编译证据哈希和适用范围。
- 原 Mac CSV 未改；未创建分支、仓库副本，未 commit/push/部署/发布生产因子。

## 本轮修改

1. 依据解释、正式算子契约进行精确形状 DSL 修订，保留原式与 SEMANTIC_REDESIGN 记录。r8.1→r12d 独立逐行核对：3774 条不同因子公式修订；不意味着全部已通过。
2. 财务累计值转单季值、平均资产、明确财务期间映射；无明确来源或指数身份的输入仍拒绝臆造。
3. fin_net_borrowing_cashflow 优化只允许四输入代数展开；五输入保留期间依赖及原生执行，未放宽依赖守恒检查。
4. 六个正式涨跌停事件算子签名和 IR 输出修为 Bool/EventBool，保留 0/1/NaN；连续价格不能冒充事件。
5. StockDailyBarAdj.ret 补 RETURN 元数据，显式 SourceRef 保留 ReturnDecimal。
6. 校验工具拒绝重复 ID、公式错配、截断输入、哈希不符；编译证据不提升执行状态。发布前重新检查输入哈希。
7. 新增 watchdog 崩溃信号证据与限定输入测试支持，不把未知 native crash 改成通过。

## 实测证据

- 本轮真实小样本 128 条：127 条有限值、1 条全非有限值，没有新增运行失败。
- 范围：8 标的，2025-01-01 至 2026-04-30，pandas；不是全历史/所有后端测试。
- 45 条净借款公式重新编译：45/45 通过，无数据读取；只合并 compile 字段。
- 主代理独立回归：
  - r12-root-final-regressions.log：68 passed，2 warnings。
  - r12-root-lowering-final.log：41 passed，2 warnings。
  - r12-root-event-final.log：29 passed，2 warnings。
  - 套件有重叠，不应相加称为独立测试数。warnings 包括 PhysicalImplementationSpec 缺失；日志还保留已有 contract divergence 提示。
- 主代理逐行验证 r12d→r12e：仅45行、4个编译证据字段变化；所有公式、身份、静态/执行证据不变；三个输入输出哈希匹配。
- 净借款 pandas/polars 活动实现五输入数值 fixture 已覆盖正常值、零分母、NaN。不能代替真实财务数据端到端测试。

## 因子口径汇总

114132 行中有 113893 个非空唯一因子 ID；空 ID 行不计以下数量。

| 检查 | 数量 |
| --- | ---: |
| 编译通过 | 106487 |
| 编译失败 | 4300 |
| 编译未跑 | 3106 |
| 静态解析且字段绑定 | 104987 |
| 字段未解析 | 7454 |
| 解析失败 | 1452 |
| 已有匹配公式的有限值执行证据 | 7827 |
| 已执行但全非有限值 | 186 |
| 未执行 | 105699 |
| 执行失败 | 26 |
| 执行记录为编译失败 | 121 |
| 准备失败 | 14 |
| 原生崩溃 | 20 |

这些检查维度不同，不可相加。历史匹配公式执行证据不是当前代码的重新认证。

## 仍然未解决，禁止宣称全量完成

- 原28条缺字段因子中15条已绑定且编译，13条仍需信息：3条 Top10PledgeRatio 缺聚合分母，3条 IndexWeight 缺指数身份/面板定义，7条 IndexReturn 缺基准身份。
- 8条 index_entry_exit_event(IndexSymbol) 仍编译失败：字符串标识不是数值成员状态，且 StateSigned 不能被冒充 EventBool。调换参数不解决输入语义。未放宽类型约束。
- 本轮唯一全空因子 source_row=60447：确认枢轴价格经复合表达式进入 MMD；全空原因尚未逐层实证，不能填零或标有效。
- 20条 native crash 仍未定位；105699条未执行。不能称十一万因子全部可落值。
- GARCH 旧测试已按正式别名规范化修正，并新增五个状态算子非 SQL 断言；生产路径未改。主代理全文件39项通过，日志 r12-root-stateful-alias.log。
- 扩大测试的 spectral/Supertrend/PSAR 旧断言已按正式契约修正；研究算子显式使用 compat_research，不放宽 Agent 认证。主代理完整文件14 passed、2 warnings，日志 r12-root-deep-dive-r3.log。先前两轮失败日志保留。
- 全局跨 wave CSE、十万因子完整性能验证、默认有界落盘与中断终态仍未完成全面验收。GPU/所有后端也未全量认证。
- DataAccess 本轮主要通过真实受限读取参与验证，不能据此宣称其所有缺陷解决。

技能采用先复现、最小修复、独立复验的流程；因此新表明确区分编译、执行和数据缺口，没有为了通过率补造数据。
磁盘复查：754 GiB 可用。只保留压缩明细、日志及测试证据，未存大面板。
