# 批量 source 性能资格：运行环境观测

`scripts.benchmark_real_cos_source_batch.run_backend` 可传入
`context_observer`。缺省为 `None`，不改变既有调用与报告字段。

回调收到关键字参数：`phase`、实际 `source`、`labels`、`metrics`、
`requested_tile_size`、`policy`。`phase` 分别是 `before` 和 `after`；
返回结果保留在报告的 `context_before` 与 `context_after`。

顺序为：创建 source → before 观测 → 开始计时 → 整个评估 API →
结束计时 → after 观测 → 校验执行回执 → 关闭 source。
计时不包括 source 创建、观测与关闭。观测或评估发生异常仍关闭 source；
评估失败不生成成功报告，也不会调用 after 回调。

回调必须只观察，不能读取因子 tile、改变 source 或校准后端。
传递的是 API 请求上限，而不是预算裁剪后的实际计算宽度；例如请求 16、
source 预算 5、GPU 4，这三个数不能合并。正式资格上下文应使用
`capture_factor_tile_source`、API 的语义请求 fingerprint 和
`capture_source_route_profile_context` 从实际对象重新捕获。

这只是观测接口，不自动证明最快、正确或 COS 身份。正式 ABBA 资格还必须
验证四次运行上下文一致、相反运行顺序均有相同严格赢家、完整执行范围、
零 OOM、每个后端独立的实际 tile schedule，并通过独立数值参考检查。
CPU/GPU 彼此一致不能替代独立参考。预热必须在首次上下文观测之前显式完成。
