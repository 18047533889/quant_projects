# 本轮回归结果（2026-09-21）

## 已验证范围

- Optimizer + Preprocess 全部测试：1555 passed，1 xfailed。预期失败是 HP filter 非前缀不变；它仍只允许离线研究。
- Quant Evaluator 全部测试：1270 passed，2 skipped。跳过的是 build/lib 中不存在 zscore/standardize 及 dropna kernel 的 metamorphic 检查。
- 原有修复族、分层路由、条件搜索、形态及 U/倒 U 端到端定向集：102 passed。
- 新增/修正测试覆盖：真实候选执行、SMA 手算、五种平滑时序因果、KAMA 参数域、倒 U 中心冻结、分层成本、IIR 强度方向、公开多暴露中性化及最小样本数、CPU/CUDA 的小截面拒绝。
- 2026-09-18 文档交付中记录的两项失败已处理：中性化修复实际适配器；CUDA 测试纠正样本量不足，并保留拒绝小样本的反例。

## 未通过的平台根目录测试收集

另执行了平台根目录默认 pytest：1 skipped，14 个收集错误，未进入全套运行。这些发生在本轮优化库/预处理库之外的旧测试，涉及缺失文件路径、导入环境及测试依赖；本轮没有改写其他任务正在处理的 Factor Engine 实现，不能宣称整个平台全绿。

错误文件：

- tests/backend/test_polars_native_batch2_causality.py
- tests/operators/test_cs_polars_native_batch1.py
- tests/operators/test_r11_round2_closure_audit.py
- tests/operators/test_r11_round3_closure_audit_ext.py
- tests/operators/test_ts_advanced_batch1_polars_native.py
- tests/operators/test_ts_nth_value_polars_causality.py
- tests/r44/test_r44_strict_remote_audit.py
- tests/test_ashare_feature_pipeline.py
- tests/test_sql_statistical_nonfinite.py
- tests/test_status_industry_features.py
- tests/test_status_minute_features.py
- tests/test_status_partition_runner.py
- tests/test_status_publish_validator.py
- tests/test_ts_batch1_rank_if_direct.py

三个库测试直接混在同一进程还会因两份 test_review_regressions.py 的模块名冲突而收集失败。本轮按 Optimizer+Preprocess、Evaluator 两个独立进程运行，未删除缓存或跳过失败用例掩盖此问题。

## 验证不代表的内容

没有执行真实因子批量重算、部署或生产入库；没有复现用户尚未提供的外部测试日志。新增平滑执行桥接只用于研究，不自动迁移旧业务脚本；不承诺每个合法候选都提高收益。
