# 优化暴露读取修复与边界（2026-09-22）

## 修复内容

行业与日估值注册项缺少文件名日期分区声明，导致读取一天时仍把数千个文件纳入预算预检。现在注册 TradeDate 的 daily 文件名模式 `{date}.parquet`，由现有规划器裁剪；没有扩大预算或绕过 DataAccess。

COS 行业运行时覆盖把正确的 UpdateTime 类型 timestamptz 改成 timestamp。现在保留时区类型，严格 schema 检查仍启用。

构建元信息脚本另外发现路径歧义：工作树存在名为 HEAD 的文件时，git show 失败。加入版本与路径分隔符，不删除该文件，不改变构建门禁。

## 测试与真实数据

三项新增集成测试使用实际注册表、COS runtime、DuckDB 及微型 Parquet：
两种数据各读取一天，最多扫描一个文件；行业 UpdateTime 必须通过严格检查。
修改前 3 项失败；仅增加分区后 2 项失败；修正时区后全部通过。

真实数据比较见 [完整证据](exposure_read_real_20260922.json)：
2024-08-02 与 2026-08-24，行业分别为 5107/5211 行、6 列，
估值分别为 5107/5211 行、19 列。自动日期裁剪与显式单文件范围的
全部列逐值完全一致，均启用 strict schema，max_scan_files=1。
另外核对 2026-08-22/23 周末行业快照，记录日期与文件名一致。

证据中的耗时仅为单次读取，不是重复预热 A/B，不据此声称吞吐提升。
构建信息与新读取用例合计 10 passed（0.73 秒）。
DataAccess 全套复验为 2317 passed、1 failed、11 warnings（66.40 秒）。
唯一失败为 tests/test_check_allowlist.py::test_real_repo_passes：其他审查工作在
evidence/、work/ 下存在绕过 DataAccess 的直读脚本；本轮未擅改或批量豁免。

## 未解决边界

读取正确不等于历史时点可用性成立。历史 UpdateTime 常为后续回填时间，
不能伪造 available_time=TradeDate，也不能把这些暴露直接注入历史自动选优。
本轮没有评分 TEST、部署服务或发布生产因子。

优化库与预处理库回归为 1768 passed、1 xfailed（134.18 秒）；
HP 滤波的非因果性仍明确隔离为 OFFLINE_ONLY。
总仓库 pytest 仍有 14 项收集错误，详见优化库 METHOD_AUDIT_20260922.md；
不能将相关库通过描述为全项目通过。
