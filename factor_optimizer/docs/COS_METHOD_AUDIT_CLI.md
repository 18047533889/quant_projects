# COS 逐方法自动审查与下跌冲击因子复验

后续更新：该因子的谱系识别已补齐，见 [真实接线对照](SHOCK_LINEAGE_20260922.md)。
下文 lineage_unknown 为本轮历史结果；修复后基础方案因训练指标缺失被拒绝，尚非经济准入完成。

## 一条命令运行

在 server-c /home/sunhaiwei/quant_projects 使用已有 DataAccess 配置：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python factor_optimizer/examples/cos_batch_audit.py \
  --manifest cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/d591b622354c2e44072064e8a8586aa5b3958cacbbafa8c882ec40acf6a286ed/landing_manifest.json \
  --factors 1 --assets 256 --max-factor-mib 64 --audit-methods --optimize
```

新增 --audit-methods 将既有 method_audit.audit_methods 接到同一个 COS 入口。
不指定则不增加逐方法审查耗时。--optimize 独立控制是否运行自动选择，
可同时使用，也可仅做 TRAIN 诊断与逐方法执行检查。

输出：
- methods：逐方法参数、状态、失败/缺输入原因、前缀与资产重排不变性、
  有效覆盖率、RankIC 变化及含成本的训练指标；
- method_status_counts：executed / requires_additional_inputs_or_control / failed；
- method_audit_partition：TRAIN only；
- automatic：仅 --optimize 输出的 TRAIN 选择与 VALIDATION 确认结果。

任何案例状态为 failed 时，先完整输出 JSON，再以退出码 1 结束，
避免自动化调用把“进程能运行但某个方法已坏”认作成功。
缺外部输入或控制操作不伪装成 executed，也不自动产生运行失败退出码；
使用者必须核对这类记录，退出码 0 不代表所有方法完成真实验收。
命令参数或读取错误仍按原入口拒绝，不放宽预算、路径和来源验证。

## 新真实样本

通过 DataAccess 读取 downside_shock_accumulation_asym，约 50.33 MB，
内容 SHA256 为 66d3a91e2eaead81528d3e30b8e8f1afea208352475155277f8c5de87e523b9d。
这是下跌冲击频次与波动不对称的复合因子，区别于此前估值变化/成交量样本。

先仅抽查清单元数据：最新 12 个清单多为重复快照；另在按时间排列的
清单上均匀抽取 8 个，发现此预算内样本。不是遍历全部因子，不按收益选样。
只下载选中清单中符合既有 64 MiB 单对象、128 MiB 整批上限的对象。
样本使用 500 日 × 256 股票，2024-08-02..2026-08-25，
267 个实际 TRAIN 日选资产，5080 只股票满足 TRAIN 覆盖要求，
最终固定 256 只。不读取 TEST 指标来决定样本或候选。

独立回放：65 案例，57 executed、8 requires_additional_inputs_or_control、0 failed。
默认 104 个自动候选中 10 个通过训练评分准入，但没有可靠改善，
最终保留 RAW；预算设为 90 时明确保留 RAW，输出逐元素等于输入。
这不证明因子没有价值，只表示本研究网格与当前准入规则没有找到可靠改进。

## 边界

三个中性化方法仍缺可信历史暴露，三种 DSL 重编译方法仍未接真实重算；
ABANDON 和 drop 为控制/行选择。复合表达式谱系不完整不能冒充基础处理已完成。
本轮没有声称提速、生产准入、TEST 通过或未来收益保证。
实际 CLI 已以 --audit-methods --optimize 跑完，退出码 0，57 个执行成功、8 个额外输入/控制项。
它明确输出 baseline_diagnostics.omissions=[lineage_unknown]，基础操作为空；
因此尚不能视为完成用户要求的基础预处理链。需要继续核验这条复合 DSL 的算子
语义与处理谱系，再决定是否可安全补齐，不应直接把未知谱系当作“未处理”。
当前训练诊断 issues 为空，不能把保留 RAW 解释成已发现并修复具体经济问题。

接口测试先观察到 3 个预期失败（缺少新参数），接线后完整优化库与预处理库
1845 passed、1 xfailed、16 warnings，100.72 秒。HP 预期失败保持原生产禁用限制。
一组单独测试的 SSH 观察连接中断；随后确认其进程已退出，未根据超时重复启动。
最终以上述完整回归的退出码 0 和完整日志为准。

真实证据见 [JSON](shock_cos_method_audit_20260922.json)：为减小体积，省去
逐方法重复 RAW 指标和逐候选训练指标字典，保留方法候选指标、参数、状态、原因和最终诊断。
