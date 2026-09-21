# 冷启动排查：否决后端分类缓存（2026-09-22）

## 结论

后端分类的源码缓存**未证明改善冷启动或真实优化耗时**，因此没有发布缓存代码。
试验修改已通过逐行补丁撤回，polars_backend_kind.py 与本轮开始前完全一致；
仅删除本轮新增的缓存测试文件，没有删除其他 AI 的改动或正式数据。

真正需要继续处理的是 cleaned_operators/registry.py 中
_freeze_value_unbounded 对类调用 inspect.getsource 的重复成本。
带 profiler 的新一轮冷启动共 66.915 秒；getsource 的 2,081 次类源码读取
累计 20.435 秒。这个调用关系证据纠正了此前只看累计耗时表的定位偏差。
下一步只考虑复用未变的源码文本，不能缓存整个可变语义指纹或绕过认证。

## 对照证据

详见 cold_start_rejected_cache_20260922.json。

- 第一版按函数对象缓存：完成三次试验后停止，命中率低；不伪装成完整六次。
- 第二版按代码对象及文件版本缓存：四个独立进程按 old/new/new/old 交错。
- 冷启动中位：47.4417 秒 → 48.3226 秒，没有收益。
- 4,191 个已注册实现的分类与 execution_kind 摘要全部一致。
- DataAccess 读取同一 SHA256 绑定真实因子、500 日×256 股票、103 候选、
  99 次自助抽样；两个独立进程完整优化：58.8130 秒 → 58.8992 秒。
- 真实输出数组、候选记录、选择身份及验证下界摘要完全一致；TEST 未评分。
- 这些是当前共享工作树的观测，包含其他 AI 未提交代码；
  不将绝对耗时描述为干净已发布版本的保证，不声称统计显著差异。

## 测试发现与边界

试验缓存的 6 个专项测试通过，其他相关后端契约 26 项通过。
FO+FP 全套 1813 passed、1 xfailed，HP 非因果性仍是预期失败。
平台根测试仍有 14 个收集错误，模块清单见 METHOD_AUDIT_20260922.md。

test_operator_capability.py 有以下 7 项失败。
用原始、无缓存源码读取路径独立重跑后，仍是同样 7 项失败：
不是缓存引入，但也没有据此宣称这些能力正常。

- test_ts_mean_production_capabilities
- test_bfill_is_removed_from_runtime
- test_if_else_polars_long_tier
- test_get_best_backend_respects_production_safe
- test_capability_matrix_covers_implemented
- test_production_fast_path_whitelist
- test_get_best_backend_matches_registry_preferred

没有放宽生产准入以消除失败。registry.py 已存在其他 AI 的 134 行未提交新增，
本轮没有覆盖、暂存或发布这部分代码。

