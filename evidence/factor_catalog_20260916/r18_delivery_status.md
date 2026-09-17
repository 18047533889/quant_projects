# R18 主代码修复与交付账本（2026-09-16）

正式树：server-c /home/sunhaiwei/quant_projects。所有修复直接进入现有工作树，
保留其他改动。未新建分支/worktree、未 commit/push、未部署或发布生产因子。

## 已完成的真实代码修复

- 数据源依赖、列收集与源绑定改为按节点身份去重的迭代遍历，拒绝循环。
  覆盖 5,000 层深表达式、60 层共享菱形，避免递归溢出和重复扫描。
- SourceRef v2 的市场、提供方、数据集、版本等语义身份进入依赖摘要，
  保留旧 v1 清单形状；这不是跨波次多源快照复用的实现。
- RAW 日线字段从权威字段目录绑定到 ashare_stock_daily，复权字段保持
  ashare_stock_daily_adj；不能把原始价格默认为复权锚点数据。
- 二级数据源允许列表包含同市场权威字段目录的数据集，仍拒绝不明数据集和跨市场引用。
- 规划器的列表/元组控制参数测量加入循环与引用数量限制，避免无限展开。
- 通用编译检查阻止面板表达式占用已声明的标量参数槽；保留合法常量和可证明的
  标量表达式，不把错误窗口参数放行到运行期。
- 精确展开 17 条 TRIX 与 17 条 ADXR；14 条非法调用未猜测补参数。
  ADXR 以当前注册 ADX 的完整窗口规则为准，不使用旧辅助实现的提前出值规则。
- 撤回三条旧 Gaussian 方法的无依据假设，原式和变更记录保留。
  只有明确肯定的方法声明才可触发该类迁移；普通“高斯”字样不够。

## 根代理直接核验的测试

| 证据日志 | 结果 | 看门狗耗时 | 进程族峰值 RSS |
| --- | --- | --- | --- |
| r17-root-engine-final-integration.log | 85 passed | 58.20 秒 | 865,632,256 B |
| r18-root-source-dependency-r2.log | 122 passed | 7.23 秒 | 355,545,088 B |
| r18-root-source-engine-integration.log | 98 passed | 52.54 秒 | 869,588,992 B |
| r18-root-recipes-integration.log | 23 passed | 57.58 秒 | 462,856,192 B |
| r18-root-static-recipes-integration-r2.log | 70 passed | 57.35 秒 | 799,252,480 B |

测试集有重叠，不能将这些数目相加为独立测试数。
static-recipes-integration 首轮因根代理提供了错误测试文件路径而未收集测试，
修正路径后的 r2 才是有效证据。

## 小样本实际批量落值

数据窗口 2025-01-01 至 2026-04-30，8 只股票；均为研究测试，
auto、batch-only，不退化成逐因子重试，不发布生产因子。

| 批次 | 真实结果 | 看门狗耗时 | 峰值 RSS |
| --- | --- | --- | --- |
| 原 CCI 剩余 5 条 | 5 有有限值 | 72.53 秒 | 972,214,272 B |
| 外层技术公式 40 条 | 40 有有限值 | 93.25 秒 | 1,470,414,848 B |
| TRIX/ADXR 34 条 | 34 有有限值；87,040 个值中 80,736 有限 | 79.29 秒 | 969,818,112 B |
| 原混合源 40 条 | 40 批次失败：源估算缺失 | 57.35 秒 | 751,190,016 B |
| 仅剔除 Hawkes 的 37 条 | 37 批次失败：其他公式也将 ret 传入 window | 91.32 秒 | 1,450,385,408 B |
| 完整预检后的 22 条 | 20 有有限值，2 全非有限；无运行异常 | 97.56 秒 | 1,431,805,952 B |

另有两条旧 NATIVE_CRASH 因子在同源小批中执行成功。这只证明当前两条可执行，
旧的 20 条崩溃标记来自同一次进程 SIGSEGV，不能据此认定 20 个算子分别存在崩溃缺陷。

路径账本区分 planned_backend 和 actual_backend；本轮成功的技术公式实际为
pandas_numpy。identity 传递记 actual_bytes=0，并另记真实 payload_bytes，
不再将预测转换量冒充测量量。耗时包含不同阶段，不能据此比较各算子快慢。

Quant Evaluator 的独立 CUDA smoke 与 2 项 CPU/GPU Rank IC 对照已通过，
包含并列值与缺失值。这不是 FactorEngine 全部后端或所有 GPU 指标的认证，
也不代表异步双缓冲传输已经实现。

## 默认入口的准确范围

在已配置 engine 和写入回调的前提下，调用形状为：

    result = engine.run_many(factors, result_policy="sink", sink=configured_sink)

不需要通过 enable_cse 或 worker 数启用基本 DAG/CSE 和自动调度。
sink 模式的大列表会转为有界波次；默认 return 模式会保留全部结果，两者不同。
这里不虚构未配置的生产 profile、COS bucket 或写入接口，也没有执行生产写入。

尚未完成：全目录 11 万异构因子的一次 auto 实测、多源 SourceRef 跨波次共享值、
全部后端的数值等价及性能认证。字段缺失、参数含义不明、样本支持不足与真正代码
缺陷应分别记录；不能把“编译通过”当作数学语义或生产可用性证明。

## 版本与传输

R17c 已交付 Mac 的 Documents，保留原表。随后发现的三条旧 Gaussian 假设
正在纳入 R18c 更正，不应把旧表的这三条有效值继续视为已确认公式的证据。
R18a 只包含 34 条公式修复，R18b 合并其真实落值及其他闭合测试结果。
最终 R18c 的完整校验、数量和交付路径见下方更新。

## R18c 最终核验与 Mac 完整原格式交付

- 服务器审计 CSV SHA256：ccf6a0effe8a09471491b94d3845767309bb4ebc380bf5cd3c016b680144c7c3。
- R18b→R18c 恰好 43 行变化：3 条 Gaussian 公式更正、40 条编译/执行证据更新。
  114,089 行逐列完全相同。构建看门狗 60.96 秒，634,736,640 B 峰值。
- 有非空 ID 的编译记录：109,681 COMPILED，4,212 COMPILE_FAILED。
- 全行执行状态：8,151 EXECUTED，188 EXECUTED_ALL_NONFINITE，105,595 NOT_RUN
  （含 239 条无 ID 行），140 COMPILE_FAILED，26 EXECUTION_FAILED，
  18 NATIVE_CRASH，14 PREPARE_FAILED，0 BATCH_ABORTED。
- 这些是不同时间与范围的证据记录，不是全目录当前代码认证。各行保留核验范围。
- Gaussian 与证据合并规则的根代理回归：58 passed，58.64 秒，451,887,104 B 峰值，
  r18-root-gaussian-reconcile-integration.log。

两条 event_mark_autocorr 全非有限结果已有数据级说明：默认 252 行窗口、lag=1，
至少需要 5 个有效事件。8 只股票中的跌停事件最大只有 3 个，炸板最大只有 4 个。
事件对应 return marks 均有效且非恒定，支持不足而不是字段丢失或算子崩溃。
6 事件正向控制产生有效值，改变未来 mark 不影响之前输出。
证据 diagnostics/r18-mark-autocorr-diagnosis-r3.log，72.28 秒、751,751,168 B。
诊断 r1/r2 是探针输出归一化错误，不计为算子缺陷。

Mac 最终文件：
/Users/shw/Documents/新因子统一整合_DSL更新版_R18c_20260916.csv
大小 836,286,093 bytes（约 798 MiB），114,132 行。
SHA256：1b553c81bba0b7db54011cf6f3502b0bb70d8a39eaef825f53c905eec5163a2c。
保留原始 21 列结构及方向、经济解释、分类、来源、完整原始 JSON 等内容。
主列“公式或构造”使用 current_formula，“所需字段或输入”和“输入表”在已有
明确绑定时更新；对应原值保留在 original_formula/original_fields/original_tables。
新增编译、落值、绑定和变更记录等审计列，不改写旧来源叙述。
独立重新读取输出后，114,132 行全部可以逐字段还原成未修改的原始 CSV。
相对最初原表累计有 21,535 行公式变化；不是本轮新增 21,535 条，也不等于
21,535 条均已通过实际落值。
完整原格式交付核验清单：r18c_mac_delivery.json。

原文件 SHA256：3ed7406a30634eedddf102a56a3217d2f7bf2379c5615b5c45b05b3f8fdc55eb，
生成前后未变。Mac 传输 gzip 与中间输出临时文件已清理；正式源文件和审计证据保留。
交付后 Mac 约 101 GiB 可用，server-c 约 749 GiB 可用。没有复制仓库或数据集。

18 条错误事件公式是否按正式默认窗口重新定义，已向用户明确询问，尚未授权时
保持编译失败。未执行无依据的窗口/方法替换、未补零冒充有效因子。
