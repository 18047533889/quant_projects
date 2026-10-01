# 四因子 COS 候选审计，2026-10-01

本轮审计读取 4 个既有 COS 因子，使用 500 个交易日、256 只股票。自动优化接受一个倒 U 型变换，其余三个因子保留 RAW。逐方法审计包含 340 个案例：308 executed、32 requires_additional_inputs_or_control、0 failed。你可以在 [JSON 记录](cos_candidate_audit_20261001_v2.json) 中检查参数、来源身份和候选拒绝原因。

## 数据与选择边界

日期范围为 2024-08-02 至 2026-08-25。资产选择只使用 TRAIN 覆盖率，要求至少 90%；本次没有隔离因子。TRAIN 有 267 个有效决策日。标签使用复权 VWAP：

$$r_{t,i}=\frac{\mathrm{AdjVwap}_{t+2,i}}{\mathrm{AdjVwap}_{t+1,i}}-1.$$

决策时点为 $t$，执行延迟一日。审计不评分 TEST，也不发布生产因子。组合诊断采用 `joint.v1`、`cost_rate=0.001`、`empty_leg_policy=signal_cash`；成本作用于全名义换手。上游 PIT、中性化暴露和可投资性认证仍需独立完成。

本次从内容哈希为 `00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66` 的 landing manifest 选择记录。4 个压缩因子对象共下载 50,250,363 字节。单对象准入上限为 32 MiB，批次对象上限为 128 MiB；这些数字不是进程峰值 RSS。

## 自动优化结果

| 因子 | 最终方案 | TRAIN gain | VALIDATION 下界 |
|---|---|---:|---:|
| weekly_23629be243c54739 | RAW | 0.826392 | -1.014938 |
| weekly_4cbd7ca6dccc61dc | INVERTED_U_REPAIR | 0.196956 | 0.117909 |
| weekly_52e1e67ab375742b | RAW | 1.124779 | -0.608681 |
| weekly_5c866fa073382822 | RAW | 0.453495 | 无可用下界 |

优化器在 TRAIN 选择一个方案，然后冻结身份做 VALIDATION 确认。表中的 TRAIN gain 属于训练赢家，RAW 行的最终结果仍为原始因子；你不能把训练提升当作验证收益。最后一行未得到可用置信下界，因此保留 RAW。

接受方案的参数为 `center=0.5, power=1.0, asymmetry=False, orientation=1`，计划身份为 `66c4b482beca199150cf47772cf79ab8f83ac544833e2809a9dd4df4ad13eef4`。其验证期指标如下：

| 指标 | RAW | 候选 |
|---|---:|---:|
| Rank IC | -0.021546 | 0.013763 |
| Rank ICIR | -0.335685 | 0.212187 |
| 成本后 Sharpe | -0.177365 | 0.877073 |
| 最大回撤 | 0.095506 | 0.077173 |
| 全名义平均换手 | 0.041163 | 0.036343 |
| 最差时序区块 Sharpe | -3.345268 | -1.169952 |

本次验证样本支持该方案的联合改善；它没有证明未来收益或其他股票池中的改善。

## 候选口径与未完成项

共享静态目录有 46 个方案，逐方法审计目录有 85 个案例，覆盖 20 个修复族。逐因子的实际自适应候选记录数为 58、58、58、60。审计目录覆盖的执行分支比优化器搜索目录更广，你应分别读取 `optimizer_static`、`optimizer_adaptive_actual_by_factor` 和 `executable_audit_cases`，不能把它们相加解释为优化器预算。

4 个因子的预算要求分别为 104、103、104、106，上限均为 128；`candidate_budget.evaluated` 分别为 68、18、83、33。这个字段计入满足评分准入的候选；其他候选仍在记录中提供拒绝原因。32 个逐方法案例需要额外输入或控制，不计为已执行方法。当前结果不能证明这些案例已完成，也不能证明不存在其他缺陷。

本轮修复了 manifest 绑定的一个实测缺陷：清单含 10 个有限浮点元数据字段，旧校验器仅接受整数等类型。新校验器用类型标签和 `float.hex()` 绑定浮点值，保留整数/浮点、正零/负零差异，并拒绝 NaN、无穷和未支持的对象类型。测试还验证修改浮点元数据后，复用快照必须拒绝读取。

## 复现

在 server-c 的 `/home/sunhaiwei/quant_projects` 执行，使用新的输出路径：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_SKIP_COS_MIRROR=1 \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python factor_optimizer/examples/cos_batch_audit.py \
  --factors 4 --assets 256 --max-factor-mib 32 --coverage-policy isolate \
  --audit-methods --optimize --output-json NEW_UNIQUE_REPORT.json
```

JSON SHA-256：`3904c9a7f78b53ebbd2137c4aebfc897b3d9117b39b4de733da0ca868e7a95bb`。本次入口没有记录源码前后指纹；共享工作树在审计前后观察到不同 HEAD，因此本报告不提供固定源码的性能认证，也不报告提速。最新代码测试结果为 optimizer 1645 passed、QE 4654 passed / 26 skipped；这些回归结果和真实因子审计提供不同范围的证据。
