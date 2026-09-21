# server-c 上传边界和操作说明

## 唯一映射

正式工作树是 /home/sunhaiwei/quant_projects，直接在 main 修改。
个人 18047533889/quant_projects/main 接收整个已提交快照。
HKUST-QUANT-SOCIETY 下每个独立仓库只接收该快照的同名目录：

factor_engine、data_access、vectorbt_qs、riskfolio_qs、quant_evaluator、
quant_platform、modeling、factor_preprocess、factor_optimizer、factor_assets、
alphaprobe、platform_web、lightgbm_qs。

例如 HKUST factor_engine 的根 tree 必须与个人快照的 factor_engine 子目录 tree 完全一致。
不能把总仓库 HEAD 推给 factor_engine。远端名称和提交消息都不能证明内容正确。
独立库依赖其他库应声明依赖，不复制其他库源代码进去。

## 操作

先查看 git status，取得当前工作的提交/上传授权，只提交自己验证过的相关文件。
运行 ./push_both.sh --dry-run，检查每个库的添加、修改、删除列表。
确认后运行 ./push_both.sh。只上传一个库用 --only factor_engine（或 fe）。

默认拒绝脏工作树。多人同时工作时，若明确只发布已提交 main，
使用 --allow-dirty-committed（可配合 --only）；其他人的未提交文件保留且不会上传。
脚本不会自动提交。预检和上传共用互斥锁，禁止重复后台运行。

个人仓库普通直推 main，不创建分支/PR，不强推；远端历史分叉时停止。
独立库以远端 main 为父提交创建同名子树快照，再普通直推 main；
不 checkout、不创建本地分支、不更改镜像缓存的工作树、不复制数据。
利用已有镜像缓存和 Git 对象；缓存缺失时停止，不能自动复制全库。

## 防误传

上传前按 GitHub 主机、组织、仓库名校验目标；按 tree 对象校验实际上传内容。
独立库顶层包含其他受管理库时拒绝上传。
上传后重新获取 main，核验提交和 tree；网络故障不能报成功。
logs/push_both_result.json 保存每个镜像的 sourceTree、remoteParent、mirrorCommit、
verifiedRemoteTree 和状态。失败后先检查回执；部分仓库可能已成功，重试会跳过相同内容。

server-c 总仓库及现有 13 个镜像缓存均配置 core.hooksPath 为
/home/sunhaiwei/quant_projects/scripts/hooks。
该目录中的 pre-push 同步检查所有更新：只允许 main，拒绝删除，
不信任 origin/hkust 等别名，不后台启动同步。
旧 .git/hooks/pre-push 保留但不再生效；其中旧的后台同步逻辑不能恢复使用。

这是本机防误操作保护，不是不可绕过的安全边界。有文件系统权限的程序仍能禁用 hook，
其他机器也不会自动继承本机 Git 配置。所有 AI 禁止 --no-verify 或覆盖 hooksPath。
需要更强保护应另外配置 GitHub 权限/规则；本次未变更组织安全策略。

## 修复误传的独立库

先确认远端独有改动：预检的删除列表不代表那些代码应从服务器删除。
只有用户授权修复的同步才能把镜像恢复到同名目录。普通新提交保留旧历史，可追溯恢复，
不会清除以前错误上传内容在 Git 历史中的副本，也不是敏感信息清除措施。
禁止对服务器平级的其他库执行删除来“修复”远端。

## 本轮验证（2026-09-22）

上传工具专项测试：18 passed。测试使用新建的极小 Git 仓库，
没有复制正式仓库/数据。覆盖整库误传、错库、非法目标、多个 push URL、
非 main、删除 main、嵌套兄弟库、pre-push 实际入口、并发锁和原有同步测试。

根目录 .venv/bin/python -m pytest -q：1 skipped、2 warnings、
14 collection errors（69.55 秒），并非全平台通过。错误模块：

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

本轮原始记录在 /tmp/push_scope_root_pytest_20260922.log。
这些错误不在上传工具修复范围，未改动其他 AI 的算子或任务代码。
