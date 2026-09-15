# R3 整改状态与调用链门禁（2026-09-12）

状态：PARTIAL。不是“全部算子已可执行/已认证”的交付声明。

正式代码仅修改 server-c `/home/sunhaiwei/quant_projects` 的 main 工作树，基线为 `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`。未创建分支、worktree、源码副本，未提交、推送、部署或发布因子。其他任务的 QE/FP/FA/FO/quant_platform 改动保留，见 `evidence/r2/v2_recheck_20260912/SUMMARY.md`，不能当成本轮 FE 实测结果。

## 实际改动与门禁

|环节|处理|保留边界|
|默认 profile / get_engine|新增不可变 ExecutionPurpose，默认 research_compute；资源策略独立|数据源和 ExecutionContext 仍为 production 严格输入模式；批准的显式 production_compute 不降级|
|发现与公式编写|同一 runtime 目录的分页、完整研究解析入口与可用性台账|旧生产/daily 接口保留；发现、可绑定、可计算、可发布不是同一状态|
|compile / PIT|证书准入按用途判断；明确的结构时间契约不因 experimental 标签消失|未来依赖、错误 source_expr、缺失来源及未知时间契约继续拒绝；派生字段递归传用途、不可变批准 execution scope 与原资源 broker|
|bridge|实际获选后端精确绑定；完整参数/实现身份仍检查；精确负证据拒绝|未认证不等于已知错误；自动重规划正确替代绑定尚未实现|
|物理规划与执行|execution_ready 与 production_ready 分开；未知成本保守估计|没有修改统计窗口、ddof 或精度来省资源；未知 native 身份不升级|
|资源准入|SQL 滚动 LIST 工作区进入物理估算；ROOT 的真实 TaskResourceContract 计入可达区域、转换和输出|这是估算，不是 RSS 实测上限；原生分配器、未知成本和原子不可行计划仍需规模验收|
|多源读取|依赖批准集、child 范围与 broker 继承；财务 prepared read 精确 digest 核对|不拿当前快照替代批准值；缺分钟/财务依赖明确 DATA_SOURCE_MISSING|
|durable writer / reader|manifest、receipt、恢复绑定用途和 run identity；有界读取及真实 broker 租约随物理 owner 生命周期释放|研究结果保持 UNVERIFIED；不登记或发布生产因子；QE 尚无本轮真实默认端到端证据|
|逐项编译失败|typed envelope 保留原因；有效同批项继续|预算、超时、错误作用域仍沿用有限状态机；没有把完整 AU33–36 当作全部通过|
|兼容性|修 frozen ParamSpec 的 capability 读取与旧物理入口；补新字段/资源契约测试夹具|不绕过生产门禁来迁就旧测试；Pandas 2/3 分开记录|

R01/R02 修复真实 ts_kurt 的完整有限窗口与稳定中心矩计算，未跳过原失败 node 或放宽容差。R03/R06 补执行依赖身份和两条参考函数；R04/R05 补 lazy 输出物理所有权及加权入口共同支持。各子包文件记录具体反例和范围。

## 默认调用

管理员先提供已有的批准 profile（不是让 Agent 选择性能参数）。脚本主入口内使用：

```python
from factor_engine import get_engine

if __name__ == "__main__":
    with get_engine() as engine:
        receipt = engine.run_many(all_factors)
```

完整研究公式示例见 `factor_engine/examples/run_many_v2_example.py`。当前 SSH 环境未设置 `FACTOR_ENGINE_V2_PROFILE`，真实调用预检返回 `DEPLOYMENT_CONFIGURATION_REQUIRED`，证据为 `R08-default-profile-preflight.log`。不得由此编造日期、股票池、批准快照或输出目录。

## AU01–AU40 逐项状态

PASS_LIMITED 只表示列明的有限测试，不表示该编号的所有真实数据要求均已关闭。PARTIAL 包含代码/单元或合成证据，但缺要求中的闭环；NOT_RUN 表示没有本轮验收证据。真实数据默认端到端整体仍未完成。

|编号|状态|已确认与缺口|
|AU01|PARTIAL|runtime 台账记录 1756 canonical、287 alias、248 recipe、4057 binding；不是 1756 数值认证|
|AU02|PARTIAL|研究发现/解析与真实模型 compile 已放开标签限制；默认真实落值待验|
|AU03|PASS_LIMITED|分页、总量和 canonical 直达测试；完整长目录不作为一次工具返回|
|AU04|PARTIAL|未认证真实模型 compile、候选数学和合成 artifact 分别验证；不是同一次真实默认调用|
|AU05|PARTIAL|研究 CPU 候选可规划；默认 pandas-only 真实落值未验|
|AU06|PARTIAL|参数绑定与证书准入分离；未覆盖所有联合参数域|
|AU07|PARTIAL|研究不把缺/过期生产证书当已知错误；过期证据真实工作流未验|
|AU08|PARTIAL|精确坏绑定可拒绝；自动选择其他正确 CPU 绑定尚未实现|
|AU09|NOT_RUN|缺批准 profile，未执行真实默认 durable 调用|
|AU10|PASS_LIMITED|显式生产和旧物理接口的认证拒绝保留；生产兼容回归通过|
|AU11|PARTIAL|严格 source/context/profile/PIT 负控通过；真实批准输入未读|
|AU12|PARTIAL|worker 配置、用途校验和协议回归；非真实 DataAccess spawn 全链路|
|AU13|PARTIAL|发现/DSL 同源；新增 HTTP /factor-engine/default/compute 与 CLI default-compute 均委托同一 durable facade，保留旧入口；真实多入口数值闭环未验|
|AU14|PARTIAL|旧字典 API 与生产入口保留；CLI 新旧兼容8项通过，HTTP身份和回执追加修复/测试另记；非真实业务迁移验收|
|AU15|PASS_LIMITED|注册错误显性化及 frozen capability 修复，未吞成完整空菜单|
|AU16|PARTIAL|批准依赖与 prepared read 接线/负控通过；日线+财务真实端到端未验|
|AU17|PARTIAL|缺批准依赖保留 DATA_SOURCE_MISSING；真实分钟配置尚未验|
|AU18|PARTIAL|发现不等同输出频率白名单；真实分钟到日频计算未验|
|AU19|PARTIAL|保留 HFQ 语义 catalog 校验；本地与 COS 同一批准数据对照未跑|
|AU20|PARTIAL|区域执行与转换规划保持；真实混合后端性能/转换账本未跑|
|AU21|PARTIAL|保守未知成本与真实任务预算接线通过；真实原子可行性未验|
|AU22|PARTIAL|R01/R02 统计定义/单位变化有限回归；高成本模型全域未验|
|AU23|NOT_RUN|本轮未完成高成本模型公平调度真实负载验收|
|AU24|PARTIAL|候选条件/状态发现与数学 lane；默认单独预览/落值角色未全接通|
|AU25|PARTIAL|嵌套未来依赖仍拒绝；完整输出角色/可知时间传播未全验|
|AU26|NOT_RUN|独立 label/diagnostic 用途及批准分区尚未实现，不放开未来数据门禁替代|
|AU27|NOT_RUN|本轮未完成真实模型训练成熟/预测/诊断全场景验收|
|AU28|PARTIAL|真实 parquet 序列化、hash/轴/值读回合成契约通过；缺真实默认因子运行|
|AU29|PARTIAL|合成 FE parquet/hash/完整读回→QE真实CPU单指标与metric-instance已通过，研究来源进入引用/配置身份/结果元数据；批准行情默认闭环与GPU仍未验|
|AU30|PASS_LIMITED|研究 artifact 不发布、用途不可升级；不是生产发布授权|
|AU31|PASS_LIMITED|manifest/receipt 不洗白用途证据的负控通过|
|AU32|PASS_LIMITED|用途/身份恢复隔离及父子协议负控通过，非真实并发部署认证|
|AU33|PASS_LIMITED|HTTP/CLI共用typed公式适配器；真实Pandas合成混合批2项verified parquet精确读回、2项REJECTED且无artifact；全拒绝/超限重复名/篡改负控通过；批准真实数据混合批未跑|
|AU34|PARTIAL|typed 原因与持久 attempts 接线；全部错误改写/跨重规划去重未验|
|AU35|PARTIAL|预算接线及原有有界处理保留；真实零余量/原子超限阶梯未跑|
|AU36|PARTIAL|source digest mismatch 释放与作用域原因测试；真实 writer 不确定态未全验|
|AU37|PARTIAL|最终runtime台账1756 canonical、367 research_callable；修复面板参数误判，未知标量契约仍阻断；无批准业务context，不标全可执行|
|AU38|PASS_LIMITED|4 个未认证真实 pandas 候选有独立数学 lane；其余后端/参数 NOT_RUN|
|AU39|NOT_RUN|缺真实默认入口端到端日志；不能据 parse/compile 宣称完成|
|AU40|PARTIAL|epoch5冻结源码及98个改动文件前后摘要一致；双环境完整后端回归无失败，skip保留；未真实小数据→100k阶梯，不能称最快或零bug|

## 主要证据索引

- `R03-R06-IDENTITY-REFERENCES-20260912.md`：身份/参考函数。
- `R04-R05-OWNERSHIP-SUPPORT-20260912.md`：缓存与加权支持。
- `R07-BACKEND-PARITY-CANDIDATE-LEDGER-20260912.md`：初次完整回归与逐 skip 原因，运行中源码曾变化。
- `U05-DEPENDENCY-SOURCE-REVIEW-20260912.md`：批准多源/资源释放与完整 LQTP 复测。
- `U08-U09-DURABLE-PURPOSE-20260912.md`：artifact/receipt/租约/恢复证据。
- `U10-TYPED-COMPILE-REASONS-20260912.md`：逐项原因与连续处理。
- `U03-U06-production-compat-final3.log`：80 passed；早期失败日志保留，不覆盖历史。
- `U06-physical-admission-root.log`：25 passed，物理工作区进入任务准入。
- `U11-runtime-availability.json`：逐 canonical/binding 可用性，不是数值 PASS。

## 必须继续的工作

自动避开精确已知坏绑定需要规划器拿到完整调用/来源/dtype/实现身份，不能猜 float64 或只按名字隔离；有限重规划尚未实现。HTTP/CLI混合非法DSL现在进入原生manifest逐项拒绝，而非服务端拼接回执；结构错误及重复名请求校验仍保留。标签/离线诊断完整工作流仍有代码与验收缺口，需要批准的定义、价格/日历口径、成熟时间与输出分区契约，不能放松默认PIT替代。QE合成artifact消费已通过，批准真实数据默认闭环仍未验。真实批准profile缺失另行阻断真实小样本、GPGPU、十万因子吞吐/转换/80%预算验收。以上均不能靠更改默认字符串或认证标签关闭。

## 本轮集成补充

- CLI `python -m factor_engine.run_pipeline default-compute factors.json` 仅接受公式清单，调用同一默认 facade，部分失败返回非零退出码；说明见正式 example。
- HTTP 默认入口绑定实际资源策略、用途、catalog、批准配置和 timeout；同名 principal 的 tenant/project 不同不能互相读取/取消任务。显式 ADMIN 保持原全局语义。
- ABORTED 回执经批准根目录、大小上限、run/policy/purpose/来源身份校验后保留；显式手动 retry 是新任务及新 deadline，保留请求/策略/attempt lineage，不能冒称原运行 resume。
- Pandas 3 时间轴回归共169个节点，已在两个环境分别全部通过；不同来源的测试输入不再一侧保留timestamp、另一侧降为date，没有放宽dtype比较。
- `ts_corr` 使用窗口内稳定中心矩；两点为精确符号公式；Numba旧原始矩旁路不再参与。Polars使用原生list-of-struct表达式，避免旧展开方案导致的计划膨胀。新增工作区进入ROOT资源准入，同region并存工作区累加；仍是估算，不是硬RSS限额。
- 中间epoch3运行时源码变化且有失败，不作为冻结全绿证据。epoch4保留历史冻结结果；追加ts_skew、目录及逐项DSL修复后的最终结果以epoch5为准，见 `R3-FINAL-REVIEW-20260912.md` 与 `R3-FINAL-VERIFICATION-20260912.json`。
- 合成读回/QE的26项聚焦回归、QE全套1255 passed/2 skipped见 `U08-QE-ARTIFACT-INTEGRATION.md`；HTTP全套84 passed及双环境定向29 passed见 `R3-service-default-durable-evidence-20260912.md`。

最后版本仍为PARTIAL：自动负证据重规划、标签/离线诊断完整契约、全部算子/参数覆盖，以及批准真实数据的默认落值/100k/GPU验收没有因此自动关闭。
