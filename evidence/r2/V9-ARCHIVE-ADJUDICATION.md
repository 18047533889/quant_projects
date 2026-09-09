# 清理前代码草稿逐项对账

范围：两个已校验的小源码备份共 315 项（同路径的不同草稿分别计数），不是整库复制。
程序账本 `V9-ARCHIVE-RECONCILIATION.json` 保留 276 项主干历史 blob、2 项逐字相同、37 项需语义复核的原始分类，不将历史版本机械覆盖回当前源码。

## 20 项 M37 旧草稿差异

逐项比较草稿与正式源码的完整差异；以下为主干后续修复，不能倒退：

| 文件（factor_engine/ 下） | 主干保留的后续实现 |
|---|---|
| backend/operator_semantic_version.py | BDS、事件、拟合、频域等修复后的语义版本和缓存身份 |
| runtime/resume_validation.py | M11 索引、摘要、范围、归属和缺失证据校验；此次再补安全 pending 子集 |
| runtime/persistent_run_state.py | M11 有界失败证据及按 ordinal 链接的持久状态 |
| runtime/bounded_pipeline.py | M11 输送/预算/实际失败 slot 归属；此次补回 pending 和恢复兼容性 |
| cleaned_operators/event_response.py | 全历史稳定 episode 锚点与局部未知事件覆盖 |
| cleaned_operators/ts_model/dynamic_regression.py | FitResult/失败原因/窗口上下文凭据 |
| cleaned_operators/registry.py | relational_specs 先校验再发布及空后端回填 |
| cleaned_operators/advanced_expectile.py | 归一化预测/score 收敛和安全单位还原 |
| cleaned_operators/extreme_tail.py | 统一稀疏 pinball 拟合，取代独立稠密 LP |
| cleaned_operators/cross_section/peer_ops.py | 极端数值 Decimal 路径由平方复杂度降至线性 |
| cleaned_operators/group_ext.py | 带原单位校验的稀疏全局 LAD，取代启发式局部拟合 |
| cleaned_operators/polars_native/ts_advanced_batch5.py | 实际 Hill 内核取代分位比占位，明确 CPU delegate |
| cleaned_operators/candle_state_space.py | 历史预处理、相对协方差几何与不可识别 nullspace 拒绝 |
| cleaned_operators/research_spectral.py | 训练块归一化、不惩罚截距、双响应 solve、BDS common-center 方差 |
| tests/operators/test_v9_hsic_solve_domain.py | 新截距合同的独立 KKT 参考，保留旧模型参考并明确历史身份 |
| tests/operators/test_v9_expectile_location.py | 新增真实 Polars 绑定数学等价性 |
| tests/operators/test_v9_m32_hill_domain.py | 实际赢家后端、极端单位、fresh-process 及委托分类测试 |
| tests/runtime/test_v9_contract_identity.py | 新增已修正算子的身份覆盖 |
| tests/runtime/test_v6_bounded_pipeline.py | 实际故障注入、准确 slot 终态及证据链；此次补回草稿中 db.close |
| tests/runtime/test_v8_slot_failure_ownership.py | 实际失败 slot 的 attempt、终态、原因及证据归属断言 |

## 17 项小草稿差异

- `test_pending_positive.py` 与旧 `test_v9_pending_resume_safe_subset.py`：早期合成 checkpoint 测试，整合入正式 `factor_engine/tests/runtime/test_v9_pending_resume_safe_subset.py` 并补拒绝路径和默认 spawn/direct 的三路 admission 覆盖，不另增根目录重复测试。
- 两项 `.pytest_cache/README.md`：生成缓存，不是代码。
- 四项 package `__init__.py`：旧 overlay 的 `extend_path` 脚手架。主干保留实际包入口、兼容别名和 source-window 合同安装，不合入 overlay 脚手架。
- `test_fit_failure_evidence.py`：主干额外保留编码上界、重复组、递增 sequence 校验。
- `test_v6_bounded_pipeline.py`、`test_v8_slot_failure_ownership.py`：按上述较新语义保留；已补回前者独有的显式连接关闭。
- 五个 M36 `.patch`：v1–v4 为已被审查驳回/修订的中间草稿；v5 已实装。对 v5 在主干执行反向 dry-run 成功，未执行反向写入。
- `tests/backend/test_m36_resource_admission.py`：已整合并改名为 `factor_engine/tests/backend/test_v9_m36_resource_admission.py`，SHA256 同为 `5b9fae3431bb26d50ef9e026e404ecbd737c9e27f141e6f0d83d4fa7d7ae01b3`，不是遗漏。

## 边界

此账本证明上述草稿的去向，不是 1756 算子无 bug、全 GPU 原生、100k 真实数据性能认证。新增恢复逻辑和回归结果以本轮最终整合证据为准，中间失败结果保留。
GitHub 上其他 34 个七八月历史分支没有本任务的迁移证据；未将它们盲目合并或删除。三个 backup 分支的提交已证明为 main 祖先，删除分支引用不丢失提交。
