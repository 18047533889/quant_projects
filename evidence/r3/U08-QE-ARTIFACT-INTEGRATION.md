# U08 QE 来源传递与合成读回证据

范围：server-c 正式工作树，真实 FE parquet writer/reader 与 QE CPU evaluator；不是批准行情、默认 run_many、GPU 或吞吐验收。

发现并修复：

- `FactorValueRef` / `LabelBundleRef` 的嵌套 FrozenMapping 仅被浅层转 dict，`json.dumps(ref.to_dict())` 报 TypeError。现在递归分离容器，保留内存中深冻结，不把未知对象 stringify。
- QE 单指标及 metric-instance 汇总丢弃请求的 factor_value_ref。现在引用进入配置身份与结果元数据；声明 factor_ids 与实际批次不一致时拒绝。保留 UNVERIFIED 不代表给上游数值颁发证书。

回归使用 8 个时间点 × 40 个证券的固定种子合成值，经 FE 真 parquet 写出、hash 验证、完整 Series 读回，再交给 QE CPU 算 `rank_ic_series`。标签显式给定为合成值的两倍，完整时间/证券轴对齐，独立数学结果每期为 1，容差 1e-12。测试普通入口与 metric-instance 入口，并验证来源用途、不可变性及错误 factor axis 拒绝。

这里的 broker 是有界合成测试替身；输入不是生产数据，没有证明默认执行、80% RSS、真实 COS、GPU 或十万因子。没有将 artifact 成功写盘当成算子全域认证。

执行证据：

- `.venv/bin/python -m pytest quant_evaluator/tests/test_r3_fe_reference_metadata.py quant_evaluator/tests/test_dlib_qe_002_004.py quant_evaluator/tests/test_v3_immutable_inputs.py quant_evaluator/tests/test_v5_metric_instances.py -q --tb=short --show-capture=no`：26 passed，日志 `U08-FE-artifact-QE-cpu-final3.log`。
- `.venv/bin/python -m pytest quant_evaluator/tests -q --tb=short --show-capture=no`：1255 passed / 2 skipped / 10 warnings，144.44s，日志 `U08-QE-full.log`。
- 两个 skip 是既有 build/lib 中不存在 zscore/standardize 与 dropna/row-drop 内核的文档契约测试；未把 skip 算通过。

早期新增测试自身的导入、支持样本数和时间类型问题已纠正，原日志保留。最终测试没有放宽值、掩码或轴检查。AU29 仍为 PARTIAL：批准真实数据的默认落值→QE 闭环尚未运行。
