# 本轮整改与最终冻结证据

状态仍为 PARTIAL，不是“全部1756算子正确/默认最快/绝不OOM”。全部代码直接修改 server-c `/home/sunhaiwei/quant_projects` 的既有main工作树，保留其他人的改动；未创建分支、提交、推送、部署或发布因子。HEAD仍为 `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`。

## 已落地的代码

- ts_kurt完整有限窗口与稳定中心矩、ts_corr稳定窗口内中心化和精确两点公式；真实三后端回归。SQL结果轴不得通过模板无声增删键。
- 实现依赖身份、lazy缓存输出所有权、日内共同正权重支持及两条参考函数窄修。
- 默认研究用途与严格数据治理分离，贯穿scope/父子进程/依赖源/产物/回执；生产资格和发布授权仍独立。
- 物理区域工作区、共享转换和ROOT任务预算接线；同区域额外工作区累加。估算额度不冒称实际RSS硬限额。
- HTTP/CLI共用 `iter_parsed_factor_definitions`，语法错误和未知算子生成原生manifest REJECTED记录，保留ordinal、原公式摘要、定义字节数和错误码；不伪造Factor或服务端拼接回执。seal/resume同步校验拒绝元数据；结构错误仍是请求错误。
- QE嵌套不可变引用序列化修复，研究来源进入配置身份和结果；合成parquet精确读回并进入真实CPU评估，保持UNVERIFIED。
- 目录使用权威OperatorSpec面板参数推断：ts_mean/ts_corr不再误报；未知标量契约没有被硬填为已知。
- 独立候选数学复测发现ts_skew对称窗口偏度误差，已用窗口内中心化/尺度归一化修正，覆盖极端有限输入。Pandas实现为O(N×window)的rolling.apply，存在逐窗口Python回调开销，不能声称比旧内核快；没有用剪裁、降低精度或放宽容差制造通过。

## 冻结与测试

epoch5 FE Python源码摘要：`bcd3a2676020101923082c25a22066d0b5ed4de16fbaed8af88cf157721461f1`。完整回归前后相同；记录的98个非evidence/非服务临时产物的改动文件内容也完全相同。清单包括保留的其他AI改动，不是作者归属声明。

|测试范围|结果|证据|
|---|---|---|
|项目Python3.12.3/Pandas2.3.3完整backend_parity|1027 passed /340 skipped /35 warnings，319.88s|R07-epoch5-project.log/xml|
|系统Python3.12.3/Pandas3.0.5完整backend_parity|1019 passed /348 skipped /630 warnings，319.06s|R07-epoch5-pandas3.log/xml|
|冻结后服务全套+CLI+manifest/receipt|153 passed /29 warnings，56.48s|R3-epoch5-service-cli-manifest.log|
|四个未认证候选实际Pandas绑定与独立数学参考|两环境各ts_skew/ts_kurt/ts_quantile/is_nan PASS_LIMITED，生产拒绝负控保留|R07-epoch5-*-ledger.json|
|目录契约定向|两个真实环境各22 passed|U11-targeted-environments.md|
|ts_skew稳定性定向|两个真实环境各7 passed|ts-skew-stability-final-*.log|
|QE先前子任务全套|1255 passed /2 skipped /10 warnings；不是epoch5重跑|U08-QE-ARTIFACT-INTEGRATION.md|

各集合可能重叠，不相加成唯一测试或算子覆盖率。逐节点skip及不能精确映射到canonical/backend的项目保留在账本，未推测映射。候选lane仅w=5/q=.5固定夹具及4个Pandas绑定，不是所有参数或后端认证。历史ts_skew失败、早期回归失败和源码漂移证据均保留。

完整后端回归采样进程族峰值分别为783544320和663179264字节；服务联合测试为1361854464字节。三次returncode均0，未触发测试保护。watchdog每0.1秒采样，**不是内核强制内存限制，也不是80%生产资源验收**；原测试工具保存在 `test_watchdog.py`，完整命令见对应watchdog JSON。

## 可用性、默认调用及剩余边界

最终 `U11-runtime-availability-final.json`：1756 canonical、287 alias、248公开recipe、4057 runtime binding、367 research_callable。未装载数值证据时production_callable为0/NOT_RUN，不代表已证明所有生产实现不可用或无已知错误。该目录不是当前批准数据上下文的实际可执行总数；例如cs_huber_resid的add_intercept标量契约仍未知。

已配置管理员批准profile后，默认调用仍是 `with get_engine() as engine: receipt = engine.run_many(all_factors)`（置于main guard）；调用者无需选择后端、并发数或内存比例。CLI为 `python -m factor_engine.run_pipeline default-compute factors.json`，文件只含 `{"factors":[{"name":"mean20","formula":"ts_mean(close,20)"}]}`。此处是接口说明，不是已运行批准真实行情的声明。

未关闭项：精确坏绑定的自动替代/有限重规划；标签与离线诊断完整工作流和业务时间契约；其余算子/参数/recipe的完整认证及最低依赖矩阵。缺少已有批准profile另行阻断真实DataAccess→默认落值→读回→QE/GPU、真实小样本→100k和性能/转换/80%进程族RSS验收。不能把所有代码缺口都归因于缺profile，也不能自行编造日期、证券池、快照或COS目标。

逐AU状态见 `R3-STATUS-AND-GATE-MAP-20260912.md`；机器可核验的冻结摘要与测试结果见 `R3-FINAL-VERIFICATION-20260912.json`。
