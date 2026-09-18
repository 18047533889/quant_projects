# 文档交付验证（2026-09-18）

本次仅补充文档、接口索引生成器与文档一致性测试，不修改算法。两个库既有测试合跑：1527 passed、2 failed、1 xfailed；测试时其他任务正在修改 Factor Engine。

与本库直接相关的失败为 tests/test_fe_operator_parity.py::test_ols_neutralize_fe_backed_matches_fp_kernel：FE 中性化输出全 NaN，与 FP 参考内核不一致。另一个失败为优化库 CUDA QE adapter 的 Rank IC observation_count 为 0。

HP filter 的 prefix-invariance 测试为已知预期失败；其全样本解依赖未来，已标记 OFFLINE_ONLY。

两个库新增文档测试合跑 5 passed；检查 API 与源码一致及变换目录完整性，兼容 40 项基础注册与 3 项可选小波注册。手册中的 4 个预处理示例和 1 个优化器示例实际执行成功，文档源码链接检查通过。

这不是生产验收通过的声明；两个非预期失败未在文档任务中修改跨库实现，部署前必须解决并重新验证。
