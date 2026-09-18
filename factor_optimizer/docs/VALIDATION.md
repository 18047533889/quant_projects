# 文档交付验证（2026-09-18）

本次仅补充文档、接口索引生成器与文档一致性测试，不修改优化或预处理算法。验证来自 server-c 正式工作树；其他任务同时修改 Factor Engine，因此结果代表该时点的集成状态，不是生产认证。

两个库的既有测试合跑（不含新增文档测试）：1527 passed、2 failed、1 xfailed。

新增文档测试合跑：5 passed。两个 API 索引与源码一致；5 个可运行示例实际执行成功；所有新增手册和索引的本地链接均可解析。业务接线示意不算独立可运行示例。

未通过项：

- factor_optimizer/tests/search/test_qe_adapter_v3.py::test_real_public_adapter_cuda_batch_path：CUDA Rank IC 的 observation_count 为 0，测试期望正数。
- factor_preprocess/tests/test_fe_operator_parity.py::test_ols_neutralize_fe_backed_matches_fp_kernel：FE 中性化结果全 NaN，与 FP 参考结果不一致。

已知预期失败：HP filter 非 prefix-invariant，已按 OFFLINE_ONLY 禁止生产使用。

以上两项失败未在文档任务中改动跨库实现，也不能被文档测试通过替代。部署前必须单独解决并重新验证。
