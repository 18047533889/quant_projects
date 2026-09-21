# FactorEngine / DataAccess：WorkBuddy 完整交接

交接日期：2026-09-19，Asia/Hong_Kong。用户要求本次停止扩展修复，等当前子代理收尾后交接。三个 GPT-5.6-sol 子代理均已结束。本文件供没有聊天上下文的接手 AI 使用。

## 0. 先读结论

你要继续在 **server-c 的 `/home/sunhaiwei/quant_projects` 正式工作树**修 FactorEngine、DataAccess 和约 11 万条因子的 DSL。用户最终要一次提交因子集合，通过 `run_many(auto)` 自动构建共享 DAG、复用中间值、按有效剩余内存的 80% 调度、分波读取/计算/写出，并给每个因子一个可解释的有限终态。

**尚未完成全量验收。不要告诉用户“全部算子可用”“11 万因子都已落值”或“所有后端/GPU 全通”。**

交接时状态：

| 项目 | 已核对的状态 |
| --- | --- |
| 正式代码分支 | server-c 主目录的 `main`，没有为本轮创建分支/worktree |
| 最新已提交、已 push 的代码 | `ad216adfd9992a45b2e946f1b09769f7eb186629` |
| 推送目标 | `origin/main`，已核对对应 `18047533889/quant_projects` |
| 未提交源代码 | `factor_engine/cleaned_operators/layer_composite_fixes.py`，ADX 候选修复，尚有 engine 级失败，见第 5 节 |
| 未提交测试修改 | `factor_engine/tests/operators/r28/test_all_canonicals_execute.py`，合法样本/参数修正，尚未全量复测 |
| 未跟踪新测试 | `factor_engine/tests/operators/test_r56_adx_physical_contract.py` |
| 普通启动注册数 / pytest 数 | 已诊断为 1789 / 1826；差额 37 个不是 alias 统计噪声 |
| 34 个公共算子加载修复 | 尚未解决；候选方案会改坏既有契约，已撤回源码，保留小补丁证据 |
| 最新 CSV | 仍是 R20 版本，路径见第 3 节。本轮没有生成新 CSV |
| 11 万因子执行 | 全量小样本执行尚未完成；旧清单中大量 NOT_RUN |
| 磁盘 | 最后检查 `/dev/vda2` 约 2.0T，总使用 1.3T，空闲 666G；接手时重新检查 |

**先看未提交修改，禁止 `git reset --hard`、`git checkout --`、`git clean` 或覆盖整文件来“整理”工作树。** 其他 AI 留下的未跟踪文件也不等于垃圾。

## 1. 用户要求与授权边界

### 1.1 工作位置、协作和 Git

- 用户说“本地”指 server-c 正式目录，不是 Mac 副本，也不是直接改 GitHub。
- 代码直接改 `/home/sunhaiwei/quant_projects`，FactorEngine 在 `factor_engine/`，DataAccess 在 `data_access/`。注意目录有下划线，不叫 `dataaccess/`。
- 不创建分支、worktree、整库副本或独立 staging 代码树。多个代理按文件划分责任，共用正式树。
- 保留用户和其他 AI 的现有改动。读 diff 后局部修改；不要整文件覆盖别人的进展。
- 子代理只能用 **GPT-5.6-sol**。不要给子代理选 GPT-6。用户特别强调额度。
- 用户于 2026-09-18 明确允许以后直接 commit 和 push。此前“未经授权不提交”的历史约束已被这条授权更新。根 `AGENTS.md` 已记录。
- 提交只包含你核对、测试过的相关文件。不要 `git add .`。不强推。
- **commit/push 授权不包含部署、发布生产因子或把研究结果晋升生产。** 测试只做隔离的小样本研究计算。
- 用户曾要求合并旧分支中有用改动、清理无用分支、检查上传脚本并同步 HKUST 下约 13 个仓库。历史是否全部完成，本轮没有重新审计，不能替历史任务签收。

### 1.2 磁盘和数据

- 用户曾因误复制的 `/tmp/v9-m37-stage` 占用 333 GB 强烈反对大临时目录。不得复制仓库、`.git`、虚拟环境或数据集来测试。
- 操作前检查磁盘、内存。只保留小且有界的日志、补丁、样本；用完清理自己确认无用的临时文件。
- 有独有代码或失败诊断价值的小文件保留并说明，不能为了“干净”删除。
- 大型持久产物通过项目 **DataAccess** 写入已有配置和授权的 **COS**。不能猜 bucket、凭据、路径或绕过授权。
- 不删除正式数据；不按目录名字猜“无用”。这次没有重新检查历史 333 GB 目录是否仍存在。
- 用户要求清理 FactorEngine 内无关 HTML/MD、重复代码和乱命名目录。应先查引用、打包用途、Git 差异；保留库的有效文档和本次必要测试证据。

### 1.3 产品目标

- 一次提交全部十万级因子，自动选择合适后端和执行路径，持续落值。
- 默认启用已有有效性能能力，例如 DAG、CSE、共享读取、并发调度；用户不想每次手动选择 worker、内存、开关。
- 同一次调用中复用相同中间式，例如 `ts_min(close, 10)`。DAG 太大时拆成有界子 DAG / waves，但不能在每波中无意义地重算相同输入和共享节点。
- 统一使用**有效剩余内存的 80%**，包括 CPU/cgroup 约束以及读、算、写、缓存、结果队列的统一预算。不是机器总内存的 80%，也不是每个 worker 各拿 80%。
- 零余量时不得偷偷保底分配大块内存。需要有限等待、明确失败/取消、预算回收和可恢复状态。
- `auto` 选择最快的**可用且正确**路径。用户也希望保留明确可选参数，但默认无需性能调参。
- 多引擎后端与 Quant Evaluator 的 GPGPU/GPU 路径都在需求内。没有真实 GPU 测试就不能说已支持或已加速。
- 全算子范围历史上称 1700/1750/1756 个 Agent available 算子。数字随注册面和政策变化，必须按当前 catalog 和正常启动口径重新核对。
- 修语法、参数、后端、数值、缺失、坐标、逻辑、未来函数问题。确属未来泄漏或不适合挖因子的算子从生产挖掘面隔离/移除，保留明确研究用途和迁移说明；不要为凑数量伪装成生产可用。
- 没有后端的算子可以补实现，但 CPU delegate 只能标为 CPU delegate；不能把标签改成 native/GPU 就当完成。

### 1.4 因子表与 DSL

- 因子表要使用当前数据源口径，尤其复权价格 `close` 等。当前 v2 默认配置要求 `ashare`、`hfq`；具体字段仍以 DataAccess 的正式绑定为准。
- 结合原公式、解释、方向、数据域修 DSL。未知算子可换成已有算子，但必须保持可解释语义；若重设计而非等价替换，应留原式、理由和变更记录。
- 用户批准：**99 条旧日内涨跌停公式按当前正式算子定义重写，同时保留原式和变更说明。**
- 字段、来源表、指数身份、参数、类型、单位需要明确。真没有数据就记录数据阻塞，不伪造字段或用无关 `close` 替代来凑编译通过。
- 用户希望因子 DSL 全部正确，然后构造巨大 DAG；每个因子先用很少的数据简单落值一次，定位表达式还是引擎问题。
- 每次交付修订因子表，都要另存新版 CSV，并给用户 **Mac 本地文件**，不是只留 server-c 路径。不要覆盖唯一原表。
- 用户另提过一份不泄露公式/私有信息的对外因子概况：数量、方向、量价/基本面/混合/分钟/Level2 分类，新表或新文档，不修改原表。该交付的历史完成情况和路径本轮未核实，需要时查已有产物，不应把 DSL/内部数据发给外部人。

## 2. 连接、环境和安全命令

### 2.1 连接 server-c

在用户 Mac 上使用现有 SSH 配置：

```sh
ssh qs-server-c
cd /home/sunhaiwei/quant_projects
pwd
git branch --show-current
git status --short --untracked-files=no
git diff --stat
df -h .
free -h
```

已知登录用户 `sunhaiwei`，服务器 shell hostname 历史显示 `qs-compute-gpu-hk-01`，Ubuntu 24.04。不要把登录 banner 的内网 IP 当成新的连接凭据。用现有 SSH alias，不创建密码、密钥或新主机配置。

解释器：`/home/sunhaiwei/quant_projects/.venv/bin/python`。在正式根运行，通常设 `PYTHONPATH=.`。

本次 Mac 工具工作目录是 `/Users/shw/Documents/Codex/2026-09-06/qin`。它不是正式代码库。现有 SSH ControlPath `/tmp/fe-r20-ssh.OJGbj8/socket` 只是加速连接的临时 socket；失效时直接用普通 SSH。

### 2.2 凭据与 push

**不要打印 `git remote -v` 或原始 remote URL。现有 remote URL 可能内嵌 token。** 不把 `.env`、COS 凭据、SSH 配置原文写进日志/交接文件。

上传脚本：

- `/home/sunhaiwei/quant_projects/push_both.sh`
- `/home/sunhaiwei/quant_projects/scripts/push_both.py`

本轮只核对并推送了个人主仓库 origin/main，未声称同步了 HKUST 下全部仓库。现有安全推送方式：

```sh
.venv/bin/python -c 'from scripts.push_both import ROOT,git,verify_push_remote; verify_push_remote(ROOT,"origin","18047533889","quant_projects"); print(git(ROOT,"push","origin","HEAD:main")); print(git(ROOT,"ls-remote","origin","refs/heads/main"))'
```

先检查脚本仍符合此接口和授权目标，再执行。不要盲跑会推多个仓库的脚本。

### 2.3 编辑方式

有服务器原生编辑工具就直接局部 patch 正式文件。本次环境的 `apply_patch` 只能操作 Mac，所以采用：在 Mac 创建**小型 unified diff**，scp 到服务器小临时文件，服务器 `git apply --check` 后 `git apply`，最后删除自己的传输补丁。没有建源码副本或 worktree。

每个代理使用独立补丁名。不要共用传输文件，不要直接用旧补丁覆盖已变化的源文件。

## 3. 因子表、原始需求文件与代码导航

### 3.1 因子表位置与当前证据

| 用途 | 路径 |
| --- | --- |
| 用户原表，Mac/iCloud | `/Users/shw/Library/Mobile Documents/com~apple~CloudDocs/quantsociety/新因子统一整合.csv`，最后核对约 598 MB |
| 已给用户的 R20 CSV | `/Users/shw/Documents/新因子统一整合_DSL更新版_R20_20260917.csv`，最后核对约 876 MB |
| server-c R20 压缩 CSV | `/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/factor_catalog_review_r20_final.csv.gz`，约 12 MB |
| R20 manifest | 同目录 `factor_catalog_review_r20_final.csv.gz.manifest.json`，约 11 MB，含大量 code_hashes，不要整份输出 |
| 全部历史因子修订/执行证据 | `/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/` |

交接时选择性读取 R20 manifest 顶层得到：

| 字段 | 历史记录 |
| --- | ---: |
| CSV 总行数 | 114132 |
| 非空因子 ID 编译通过数 | 113893 |
| R20 当轮修改行数 | 12369 |
| EXECUTED | 8191 |
| EXECUTED_ALL_NONFINITE | 188 |
| NOT_RUN | 105707 |
| EXECUTION_FAILED | 12 |
| NATIVE_CRASH | 18 |
| BATCH_ABORTED | 16 |

这些状态来自 **2026-09-17 的 R20 证据**，不是现在代码的重跑结果。执行状态计数覆盖 CSV 全行，不能直接与非空 ID 的编译计数混用。`EXECUTED` 标签也需要复核值和数据语义。聊天中更早的“109681 编译通过 / 8151 有效值 / 4212 编译问题”已经是旧阶段数字。

R20 manifest 的 `output_sha256` 为 `80044269dc29b209dfae5136488bde50adcfc283dd8ee2d90147746ce89463d5`。核对生成脚本对该字段的定义后再比较压缩/解压文件，别把不同对象的哈希混比。

R20 manifest 自述范围：113893 条因子 DSL 的严格字段绑定和引擎编译，**没有承诺全量执行认证**。后续 R21-R56 修过源码和语义版本，所以你必须用当前树重新编译、执行，不能沿用这个数字宣布验收完成。

处理 CSV 要流式/分块、按 ID 建审计记录。避免一次读取 876 MB CSV 再复制数份。新 CSV 包含稳定 ID、原 DSL、新 DSL、变更原因、编译状态、执行状态、失败原因、代码/数据快照身份；保持原解释和分类。

### 3.2 用户历次审计材料

下列文件均在用户 Mac 的 `/Users/shw/Documents/`，本次已核对文件名存在。按任务选择章节读取，不把 ZIP 全量 CSV/JSONL 一次加载进上下文：

1. `FactorEngine_全链路审计与整改方案_v2_f9e7237.md`
2. `FactorEngine_DataAccess_全面复审与整改方案_v3_2db45f4.md`
3. `FactorEngine_v6_默认80内存_Auto持续落值_AI交接包_6e783a57.zip`
4. `FactorEngine_DataAccess_v7_模型算子复审_证据与整改包_2e2feb9.zip`
5. `FactorEngine_DataAccess_v8_复审证据与整改包_118cd21.zip`
6. `FactorEngine_v9_模型与全算子补充审计_证据与整改包_三次增补版_118cd21.zip`
7. `FactorEngine_DataAccess_主代码合并审计与统一整改.md`
8. `FactorEngine_DataAccess_主干合并补充审计_新增任务.md`
9. `FactorEngine_DataAccess_整改后复审与剩余问题_e94ac507.md`

用户对 v6 指定“先读 `00_START_HERE.md`，再按工作包实施”，重点是先修零余量预算，再接通资源、执行和错误状态机。先看 ZIP 目录、单独读入口，不把数据和整个仓库解压复制。旧报告中的“完成”必须由当前源码和当前测试证据支持。

### 3.3 代码导航

以下路径相对 `/home/sunhaiwei/quant_projects/`：

| 范围 | 优先入口 |
| --- | --- |
| 全算子加载/注册/准入 | `factor_engine/cleaned_operators/__init__.py`、`registry.py`、`operator_surface.py`、`semantic_certification.py` |
| 物理后端与证据 | `factor_engine/backend/polars_backend_kind.py`、`backend/contracts/`、`backend/evidence_delta.py` |
| 语义版本/缓存身份 | `factor_engine/backend/operator_semantic_version.py`、`factor_engine/runtime/factor_identity.py` |
| 传统 run_many 入口 | `factor_engine/runtime/engine.py`，`run_many`、`run_many_iter`、`run_many_stream` |
| 批执行/跨波复用 | `factor_engine/runtime/batch_service.py`、`streaming_batch_service.py` |
| 默认持久化 v2 入口 | `factor_engine/runtime/default_engine.py` 中 `DurableFactorEngine.run_many` |
| 默认策略 | `factor_engine/runtime/default_execution_policy.py`，`DefaultExecutionPolicy` |
| 有界落值状态机 | `factor_engine/runtime/bounded_pipeline.py`、`finite_manifest.py` |
| 资源池/预算 | `factor_engine/runtime/resource_broker.py`、`resource_governor.py` |
| DAG/后端选择 | `factor_engine/planner/`，先查 `physical_lowerer.py` 和 batch global optimizer |
| DataAccess 桥 | `factor_engine/storage/sources/data_access_source.py`、`data_access_source_helpers.py` |
| DataAccess 正式包 | `data_access/`，测试 `data_access/tests/`；不是其 `build/lib/` 产物 |
| Quant Evaluator | `quant_evaluator/`，GPU 需求需在该真实调用路径验收 |
| DSL 解析 | `factor_engine/api/dsl_parser.py` |
| 历史 DSL 修订规则 | `factor_engine/tools/catalog_*recipes.py`、`catalog_r20_*proposals.py` 等 |
| 全算子严格测试 | `factor_engine/tests/operators/r28/test_all_canonicals_execute.py` |
| 公用样本/参数构建 | `factor_engine/scripts/audit_all_factor_production.py` |
| 批量测试脚本 | `factor_engine/scripts/run_r28_execute_batches.py`，使用前核对参数和统计口径 |

## 4. 已提交并 push 的近期成果

下面测试数是各自测试命令的结果，套件有重叠，**不能相加成算子覆盖数**。除特别注明外，本轮主代理检查过日志；更早阶段以对应提交和 evidence 为准。

| 提交 | 改动和验证 |
| --- | --- |
| `ad216adf` | 跳空回补：纠正方向、双向计数与分母、缺失窗口、零缺口；语义版本 2。新增 16 项手算/后端/多股/因果/单位缩放测试。联合 **97 passed**。 |
| `94910535` | EVT threshold stability：至少 3 个 Hill 阈值；严格正值不再用绝对 EPS 拒绝；log 差防比值溢出；语义版本 2。联合 **92 passed**。 |
| `61271040` | run_many 缓存按数据快照隔离；跨 wave 快照变化停止后续写出；权限/快照错误不吞；联合 **150 passed**。同时 turnover/CPT 保留时间/身份坐标、严格窗口域、语义版本更新；联合 **259 passed**。 |
| `d8fd4eb8` | McGinley 遇缺失/Inf 后可在完整有限窗口恢复、巨大 seed 均值防溢出；递归因果/公开后端/身份联合 **117 passed**。另做 DataAccess/cache/manifest **53 passed**。 |
| `3f532f3c` | 日内 jump finite support、session grid/时区/coverage、holder 多面板正式契约、share ratio 参数绑定、HHI/entropy 域、计数溢出。联合 **255 passed**。 |
| `84307e26` | 257 个合成因子，6 条观测，32 因子一波共 9 波；共享 `ts_mean` 实际调用一次，逐项 sink 核对，lease 释放。完整套件 **16 passed**。 |
| `69f90dce` | membership 只统计已观察到的成员转换。 |
| `6eb5a2eb` | Hill 参数域与时间/股票轴修复。 |
| `3408f7a9` | 日内 profile 网格与 rolling pair history 修复。 |

主要总结文件在正式根 `evidence/`：

- `R56_GAP_FILL_REPAIR_20260919.md`
- `R56_EVT_THRESHOLD_REPAIR_20260919.md`
- `R55_CACHE_AND_CHIP_REPAIR_20260918.md`
- `R54_RECURSIVE_AND_DATAACCESS_REGRESSION_20260918.md`
- `R53_INTRADAY_HOLDER_REPAIRS_20260918.md`
- `R50_BOUNDED_DAG_REUSE_REGRESSION_20260918.md`
- 更早 `R32` 至 `R49` 的 repair/status 文档以及 `*_SOURCE_HASHES_*.sha256`，按问题读取。

关键原始日志：

- `evidence/r56-gap-before-root.log`：修前 12 failed / 2 passed；`r56-gap-final-root.log`：97 passed。
- `evidence/r56-evt-before-root.log`：修前 5 failed / 4 passed；`r56-evt-final-root.log`：92 passed。
- `evidence/r55-cache-final-root.log`：150 passed。
- `evidence/r55-chip-final-root.log`：259 passed。
- `evidence/r54-recursive-public-identity-root.log`：117 passed。
- `evidence/r54-dataaccess-cache-root.log`：53 passed。
- `evidence/r53-combined-root.log`：255 passed。
- `evidence/r50-257-shared-factor-root.log`：16 passed。

缓存修复细节：caller cache 保持原对象/原 namespace，通过 `with_scope` 共享 backing 而不是复制、clear 或改 caller。源 snapshot 不明时不复用；共享 cache 中不同 dataset 即使 generation 相同也隔离。默认无 cache 路径同样不能吞权限或 ApprovedSnapshotMismatch。第二 wave 之前 snapshot 变化时，禁止第二 wave sink。失败释放资源 lease。

Turnover 合法样本使用 decimal turnover `.2`、window `20` 和变化正价格，`exp(-4) < .10` 才有足够已观察筹码质量。没有放宽旧质量门槛。窗口 2..4 原来永远无法估计，正式最小窗口改为 5。12 个 turnover 语义版本 3，CPT 版本 2。实现仍是每列 NumPy CPU kernel，不是 GPU。

日内 segment realized vol、部分 limit、holder 等采用正式 CPU delegate 保持契约。不要把“有 Polars 包装”误称为原生 Polars/GPU 加速。

## 5. 接手时的未提交内容与紧急问题

### 5.1 ADX：部分修复留在正式树，尚不能签收

文件：

- 修改：`factor_engine/cleaned_operators/layer_composite_fixes.py`
- 新测试：`factor_engine/tests/operators/test_r56_adx_physical_contract.py`
- 源码 SHA256：`2cca9cfae48d84977213707ed9e91b98efa1a9bc18ce453f2ed654edc384c993`，接手时若不一致先查别的改动。

子代理 GPT-5.6-sol 已做：

- 将 high/low/close 非有限值转 null；TR 要求当前 h/l/c 与前 close 有效。
- Polars EWM 的 null 输出按 pandas 路径恢复，修 NaN/Inf 后差异。
- 为最终 registered ADX winner 加诚实 `PhysicalImplementationSpec`：`POLARS_NATIVE_EXPR`，CPU、stateful、requires_sorted、materializes_full_panel；`supports_lazy=False`、`supports_streaming=False`。
- 最终 catalog backend_meta 同步上述实际拓扑。authoring tokens 经 `complete_physical_specs()` 绑定当前代码的 64hex digests；不要把 authoring token 自身当测试认证。

证据：

- `evidence/r56-adx-physical-contract-final.log`：**4 passed**。独立 Wilder reference、warmup、zero denom、多股、时间轴、缺失恢复、prefix、直接 registry 双后端和物理说明。
- `evidence/r56-adx-existing-regressions.log`：`test_recursive_kernel_parity.py::test_adx_matches_pandas` 通过；`test_polars_phase2_parity.py::test_ts_adx_parity` **失败**，engine Polars 与 pandas 全段不同。
- 更早失败日志：`r56-adx-physical-contract.log`、`r56-adx-physical-contract-after.log`。

必须继续做：

1. 追踪 engine 级测试实际选中的 ADX 实现，不要只测最终 registry 对象。它可能仍走别的老实现/转换路径，原因尚未证明。
2. 当前 ADX semantic version **还没有从 1 bump 到 2**。数值改动在 dirty tree，接手后应先补版本和 identity 回归，防止复用旧缓存。主代理为等待交接未再启动这一轮修改。
3. 正常 bootstrap 仍有两条 `unknown ... lacks explicit PhysicalImplementationSpec` warning，来源是最终 replacement 前的 ADX reconcile。别用全局 suppress 隐藏。
4. ADX capability 目前是 `IMPLEMENTED`，**不是 `PRODUCTION_SAFE`**；production auto 仍不准入，证据集合没有 ADX。若要准入，需要真实认证链，不可改字符串冒充。
5. 修完 engine 级路径后重跑 new/legacy/prefix/identity，再精确提交。当前 ADX 文件未提交、未 push。

### 5.2 37 个仅在 pytest 加载的算子：真实加载缺口，候选修复已撤回

普通 `load_all()` 为 1789，pytest 因 `factor_engine/tests/conftest.py` 的 `_LATE_SURFACE_MODULES` 在 building 阶段预热 17 个模块而出现 1826。37 个差额在普通启动下 `get(name, backend="any") is None`，alias 也不存在。`resolve()` 原样返回名字不等于注册成功。

其中 34 个公共名称需要安全接通注册；23 个原状态为 production/implemented/extended，11 个 experimental 仍应受准入限制：

```text
event_window_return_asof
financial_snapshot_lag
fiscal_capital_stock
fiscal_delta
fiscal_logit_score
fiscal_rolling_slope
fundamental_staleness_days
intra_covariance_manifold_shift
intra_critical_transition_score
intra_dmd_koopman_features
intra_functional_motif_score
intra_kalman_latent_price
intra_market_profile_corr_ex_self
intra_matrix_profile_session_features
intra_price_peak_ridge_valley_state
intra_smart_money_fcm_score
intra_state_space_volume_components
intra_visibility_graph_features
intra_volume_peak_ridge_valley_state
intraday_value_at_extreme_state
laborforce_efficiency
panel_day_night_beta_gap
pastor_stambaugh_beta
price_delay_score
report_asof
same_calendar_day_mean
same_calendar_month_return
ts_mcginley_dynamic
ts_nlms_filter
ts_one_euro_filter
ts_returns
ts_rls_filter
ts_vidya
years_since_date
```

另 3 个 `research_only` 不应混入 production registry：`ts_deviation_from_mean`、`ts_jump_bipower`、`ts_lag1_autocorr`。需要时使用正式 ResearchToolRegistry 的研究入口。

34 项来自这些模块，均相对 `factor_engine.cleaned_operators`：

- `panel_batch1`，`time_semantic_gap`
- `fundamental.fiscal_logit_score_op`，`common.fiscal_operators`
- `common.polars_ts_basic`，`cross_section.panel_batch1`
- `intraday.state_space`，`intraday.intra_state_space`，`intraday.topology_manifold`
- `technical.adaptive_filters`，`fundamental.fiscal_batch3`

**不要重复已经失败的修法：**

- 直接搬 pytest 预热清单到正式启动，会改变 20 个已有 winner/契约：`fiscal_perpetual_inventory` 以及 `ts_argmax_age/argmin_age/coverage_ratio/days_since_high/days_since_low/delay/delta/max/max_if/mean/min/min_if/new_high/new_low/quantile/rank_if/std/sum/valid_count`。
- 子代理尝试“导入模块前深拷贝 registry，导入后仅保留 allowlist 新名字，再恢复原 registry”。新增集合、research 隔离、experimental 不晋升和 no-thaw 测试通过，但仍有约 99 个原有契约变化。
- 已确认真实反例：`cs_hartigan_dip.param_specs['min_cross'].param_role` 从 `SUPPORT_POLICY` 变成 `None`，不是 repr 地址噪声。聚合模块导入会改 registry 之外的全局规格/后续治理来源。
- 最后候选测试 1 passed / 1 failed，119.63 秒。前轮 1/1，126.69 秒；初轮 2 failed，232.55 秒。这是子代理回报，完整日志未作为本轮已提交证据验收。
- 子代理已完整撤回 `cleaned_operators/__init__.py` 候选修改，并移除新 `test_r56_missing_public_bootstrap.py`。交接时这两个路径没有本任务 diff。

四个小补丁保留在 server `evidence/`：`.r56-bootstrap-sol.patch`、`.r56-bootstrap-fix2.patch`、`.r56-bootstrap-test-fix.patch`、`.r56-bootstrap-test-fix3.patch`。**只作失败诊断，不应直接重施。** Mac `.../qin/outputs/` 也有小补丁/测试片段，正式树才是权威。

建议正确方向：拆出无导入副作用的注册函数/专用模块，只注册缺失项，避免触发整批旧模块治理。使用 fresh subprocess，绕开 pytest conftest 的预热；对新增集合、既有 backend winner、instance metadata、参数域/角色、alias、状态、物理说明和无 thaw 做完整断言。普通 baseline 与 candidate 的比较需要稳定结构序列化，不能删除真实差异来“通过”。

### 5.3 全算子合法样本测试：保留修改，仍有未解决项

唯一修改文件：`factor_engine/tests/operators/r28/test_all_canonicals_execute.py`。

交接 SHA256：`f60a1fb05ac94d84dd9cd4ac15d69807401c7866cac44d54373c916f7710c254`。改动约 544 加行 / 71 删行；未提交，因为最终全量复测还没做。后续变更需重新冻结 hash。

约束保持：

- `_CONTRACT_REJECTED` 仍为空，未通过异常豁免凑数。
- 空值、全 NaN、Inf、非数值、空输出都不能算成功。
- 不能按算子名字前缀猜 daily/minute grain；以正式契约/已验证输出拓扑为准。
- 当前这个套件每 canonical 优先测 pandas_numpy，否则选一个实现，**不是全后端认证**。它也不替代时间轴原生验证和数值 oracle。

旧严格基线：

- lower 0..899：609 finite-pass / 291 fail / 0 skip。291 中 175 为日内到日频的 fixture shape 误断言、108 为 allNaN、8 为 runtime fixture 问题。
- upper 900..1788：808 pass / 81 fail / 0 skip。先前已修 8 参数/域 + 13 财政样本，形成 remaining60。
- 以上是在旧冻结 fixture/旧列表上的统计，后面的局部修复不能直接算成当前全量通过率。
- 旧 lower manifest：`evidence/r28_lower_0_899/manifest.txt`。
- 旧 upper 日志：`evidence/r44-strict-exact-batch0900-0999.log` 等到 1700 段。
- 真实 pytest 1826 名单曾保存 `/tmp/r51-collected-canonicals.txt`。临时文件可能消失，接手应重新 collect 并保存 manifest。

本轮 fixture 小包：

| 小包 | 结果/证据 |
| --- | --- |
| pattern4、micro recipe3、relation2、session1、state6、directional-change3、envelope1 | 20 passed，`evidence/r55-upper20-pass.log` / `r55-upper20-hashed.log` |
| report/revision3 + turnover11 | 14 passed，`evidence/r55-report3-turnover11-pass.log` |
| remaining26 前20 第一轮修复 | 9 passed / 11 allNaN，`evidence/r55-upper-rem-b1-iter1.log` |
| EVT fixture 最终绑定 | 1 passed，`evidence/r55-evt-binding-pass.log` |
| Huber5 + state_age_percentile | 6 passed，`evidence/r55-huber5-state1.log` |

旧 upper remaining26 中，还有以下 **3 项 fixture 未修好**：

```text
ts_expected_shortfall_asymmetry
ts_extrema_divergence_strength
ts_first_passage_conditional_time
```

`ts_gap_fill_ratio` 的真实源代码错误由主代理修复并提交了 `ad216adf`，独立 public 双后端测试通过，但 **R28 对应 case 尚未用最终 fixture 重跑**。

remaining26 后 6 项本轮未完成针对性修复/验收：

```text
ts_state_exit_hazard
ts_state_residual_life
ts_support_fit_r2
ts_threshold_cycle_asymmetry
ts_threshold_cycle_period
ts_transition_intensity
```

此外 lower108 allNaN 尚未完整收敛，差额37的整个小包也没有最终全组验收。不要把上面 upper 小包完成误当全库完成。

避免重复踩坑：

- `_apply_overrides()` 后面的 broad defaults 会覆盖前面设置。审查 **最终** `_build_call(canonical, op, panels)` 的 args/kwargs，而非只看你刚写的赋值。
- EVT generic `side='lower'` 与正 lognormal 样本相冲突；最终绑定为 `window=10, k_min=2, k_max=4, side='upper'` 才验证通过。官方独立样本在 `test_r56_evt_threshold_contract.py`。
- Huber 正式参数是 `y, x1, x2, x3, x4`，不是 `x/y`；每个设计列要非共线，不能把同一 panel 塞进四个特征。
- state age 旧 sample 收到 `min_completed_runs=20`，220 行每 7 行切换、按状态分组后支持不足。修合法样本/最小已完成 run 数，不放宽正式统计门槛。
- micro recipe 原来每交易日只有一条记录，session 内 pct_change 全 NaN。合法输入是 4 个明确 session × 60 个分钟 bar，正且变化的 close/volume，minute 输出保持 minute 形状。
- report/revision 需要真实 filing 事件和 vintage：参考 `tests/runtime/test_r55_report_revision_contracts.py`。revision_delta 用重复 period 的后续 vintage，差额例如 5/-10/15。
- lower 的财务样本要 quarter end、连续 period、非零 signed cashflows；CS 需足够横截面和非共线特征，Hartigan 至少 100；relation 需多个 peer；日内需真实 SessionCalendar/slot；holder share 必须 bounded/nonnegative。

## 6. run_many / DAG / 内存 / GPU：现状与仍需验收

有两个入口，交付默认例子时必须区分：

1. `factor_engine/runtime/engine.py::run_many`：已默认全批 DAG/CSE/自适应并发；但其 Python 签名仍是 `result_policy='return'`，会累积返回结果。`sink` 模式且大批次/显式 wave 参数才走有界 streaming；`wave_size` / `sink_queue_bytes` 必须配 `result_policy='sink'`。不能把十万因子直接用默认 return 全驻留当最佳方案。
2. `factor_engine/runtime/default_engine.py::DurableFactorEngine.run_many`：调用 `execute_run_many_durable`，使用 `DefaultExecutionPolicy` 默认 `backend='auto'`、`result_mode='artifact'`、`memory_fraction=.80`、有限 retry、逐因子 report-and-continue、integrity abort、`automatic_production_publish=False`。它要求已有批准 business profile，不猜 dataset/universe/date/artifact root。

v2 policy 当前允许 backend 选项为 `auto/pandas/pandas_numpy/polars_long/duckdb_sql`；这不等于 GPU 已接入。用户要 Quant Evaluator GPGPU，接手者应查硬件/驱动和实际 evaluator 入口，不替其宣布完成。

257 合成因子的 9-wave 实验只证明那组有预算的共享节点能跨波复用一次，不能外推成 11 万因子的大 DAG 已安全执行。还需验收：

- 真实表达式集合的共享图构建内存、规划时间、分区边界、lookback/warmup。
- shared node 何时保留、何时驱逐，cache 预算不足时允许的重算，跨 snapshot 禁止重用。
- 统一读/算/写预算，队列回压，少数昂贵因子不阻塞整个队列。
- 零余量、cgroup 更小、外部进程抢内存、sink 慢/失败、timeout、cancel、resume、OOM 后有限重计划。
- 后端转换次数、转换字节数、实际选路理由和性能账本。CPU delegate 不应重复转换大 panel。
- 自动选择只在语义、输入类型、单位、时间/股票轴、PIT 和证据门禁允许范围内比较性能。

用户希望“最快”，验收应比较同一数据/代码/因子集合的 wall time、峰值内存、CPU、GPU、I/O 和转换开销。pytest 总耗时含导入，不是算子吞吐。采样 RSS watchdog 也不是硬内存上限。

## 7. 11 万因子测试：接手后按这个顺序执行

### 阶段 A：冻结输入与代码身份

1. 记录 Git HEAD、dirty diff 的文件/hash、Python/依赖版本、正式 registry manifest。
2. 读取 CSV header 与稳定 ID，查空 ID、重复 ID、空 DSL、源表/单位/复权/频率。按 chunk 处理。
3. 记录 CSV hash、输入 snapshot token/content digest、calendar/timezone、universe、start/end、adjustment。
4. 以当前正式代码重新编译全部非空 ID。记录字段绑定、canonical、参数域、禁用/research 状态、实际后端和失败原因。不能把 pytest 预热后的可见面当真实用户面。

### 阶段 B：合法小样本实际求值

1. 从已授权 DataAccess 数据读取**很少股票和很短但满足 warmup 的历史**。普通日频、财报、持有人、指数、分钟、Level2 需要各自合法输入，不可一张假 close panel 通吃。
2. 通过正式 `run_many` / durable 入口提交因子集合，让它形成共享 DAG 和有界 waves；不要退化成外层逐个 `run()`。
3. 每因子至少一次真实值路径；逐条检查数值/finite mask、日期/股票顺序、输出 grain、类型、单位、warmup。
4. 全 NaN 单独记录：统计支持不足/本次没有事件/真缺字段/算子 bug/参数错误。不能把全 NaN 自动当成功，也不能把合法无事件强行填零。
5. 失败隔离：区分 DSL 修复、source binding、算子逻辑、backend、native crash、整批 abort。发生错误后继续处理不受影响的因子，并保留每项终态。
6. 旧 R20 的 12 failed、18 native crash、16 batch aborted、188 all-nonfinite 和 105707 NOT_RUN 都要在新 manifest 中逐条重新分类。不要丢弃难例。

### 阶段 C：算子与后端正确性

- 相同输入的 pandas/reference、Polars、DuckDB 以及真实支持的 GPU 路径数值/NaN mask/坐标一致。
- 独立公式或手算 oracle。两个后端一起算错也会 parity 通过，gap-fill 就是已发现反例。
- prefix causality、禁止未来索引、严格过去训练、PIT/as-of、财务修订 vintage 可见时点。
- 测 NaN/Inf/常数/零除/极端量级、排序、多股、不齐时间、空输入、参数边界/非法参数、long/wide 与时区。
- 对不适合生产挖因子的 diagnostic/research 算子保留真实分类；不要为了编译全通过绕过门禁。
- 参数/算法变更同时更新 semantic version、缓存身份及变更账本。

### 阶段 D：十万级批入口与资源压力

- 先小数据×全表达式，确认规划、去重和终态完整，再逐步增加数据量。
- 检查根因子数、成功/失败/取消/阻塞总数能对账，不能 silently drop。
- 用共享子表达式计数证明 CSE 生效；跨 wave 缓存和数据 snapshot 绑定。
- 验证结果按完成即写出，不保留十万结果 panel；sink 失败和重试须幂等。
- 测小预算/零预算/慢写/取消/resume、内存抢占、OOM 重计划次数，确认最终停止而非无限循环。
- 大产物走已授权 DataAccess COS，不在 `/tmp` 或正式盘堆数据。

### 阶段 E：最终交付

- 新版 CSV 给 Mac 本地文件，保留原式、新式、修改说明与完整状态；另留 server 压缩产物和 hash。
- 交付当前代码下的全量编译/执行 manifest，明确尚缺真实数据的项。
- 交付默认调用示例，明确选传统 sink 还是 v2 durable，以及业务范围配置；不让用户再猜性能开关。
- 交付性能与转换账本、后端可用范围、尚未认证/未测 GPU 项。
- 完成代码提交与授权远端 push；没有部署/生产发布授权时不要发布。

## 8. 推荐接手顺序、测试命令与证据纪律

建议按文件分配最多几个 GPT-5.6-sol 子代理：

1. 一个负责 ADX engine 路径和语义版本闭环，独占相关源文件和测试。
2. 一个负责缺失 34 项的无副作用 bootstrap，必须 fresh-process 验收既有契约不变。
3. 一个负责 R28 fixture 全量收敛和明确失败清单，独占该共享大测试文件。
4. 主代理负责跨组件验收、DataAccess/资源状态机、11 万 DSL 编译/执行清单与用户 CSV 交付。

如果模型槽位不足，按上述顺序推进。不要再建新任务工作树或把源码复制给代理。

有界测试模板，在正式根运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=. \
.venv/bin/python evidence/r3/test_watchdog.py \
  --log evidence/workbuddy_unique_test.log \
  --max-rss-mib 1536 --timeout 240 \
  -- .venv/bin/python -m pytest -q --tb=short \
  factor_engine/tests/operators/test_r56_gap_fill_semantics.py
```

日志名必须唯一。watchdog 的 source_hashes 默认可能为空，所以另记录源码和 fixture hash。不要覆盖失败日志，不要为了补 watchdog JSON 重跑整个已完成大套件。

R28 小包示意：

```sh
R28_EXECUTE_SUBSET=ts_evt_threshold_stability,ts_gap_fill_ratio \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=. \
.venv/bin/python evidence/r3/test_watchdog.py \
  --log evidence/workbuddy_r28_unique.log --max-rss-mib 1536 --timeout 240 \
  -- .venv/bin/python -m pytest -q --tb=short \
  factor_engine/tests/operators/r28/test_all_canonicals_execute.py
```

先确认指定名字确实被收集，0 selected 不算通过。调合法 fixture 时每包不超过约 20 项；冻结后再分批全量跑。保留 ordered canonical list 和 hash，避免 1789/1826 不同列表的下标错位。

不得使用 `ulimit -v` 作为这个环境的测试内存限额，此前造成启动 segfault。历史 ulimit/bootstrap/SSH 中断日志不能统计成算子源码错误。

Git 检查仅排除原始日志尾随空格，不改写失败证据：

```sh
git diff --check
git diff --cached --name-only
git diff --cached --check -- . ':(exclude)evidence/*.log'
```

先 review 再按精确路径 stage。提交 ADX 前先完成当前失败和 semantic version；提交 R28 fixture 前冻结并完成相应复测。其他 untracked evidence、COS ACL 脚本、`.orig`、旧补丁要查归属，不能全加或全删。

## 9. 仍然不能做出的结论

- 不能说 1750/1789/1826 个算子已逐个跨后端认证。
- 不能说 113893 条因子已全部有有效值，旧 manifest 大部分仍 NOT_RUN。
- 不能说默认 auto 在所有输入上最快，尚无十万级真实性能账本。
- 不能说 Polars 包装就是原生表达式，更不能说它等于 GPU。
- 历史 source-bound paired 证据覆盖是 514/1756；普通单测通过数没有自动增加这个认证数，接手应查当前证据集合。
- 不声称历史分支已全部合并删除、HKUST 多仓库全已 push、依赖锁定已全部一致或磁盘已清干净，本轮未做这些全量复核。
- 本交接没有新生成 CSV、没有跑完整 11 万真实因子、没有部署和发布生产因子。

用户要求持续修到完成。现在暂停扩展工作是因为用户明确要交给 WorkBuddy；接手者应从上面未完成项继续，不需要重新从 R1 审计开始。
