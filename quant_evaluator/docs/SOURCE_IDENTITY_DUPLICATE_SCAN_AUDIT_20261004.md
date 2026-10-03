# 严格身份重复扫描：成本与安全反例（2026-10-04）

## 实际成本

qualified context 的 `_capture_live_context_state` 先完整读取 QE Python 源码，再调用 `capture_source_dependency_identity` 完整读取 QE、DataAccess、FO、FP。依赖闭包里的 QE 与前面的 QE 同 root，因此存在重复 I/O。

独立 Luna 在正式 server-c 源树测量三次顺序 CPU 子项，每次都新建严格 identity scanner。QE：753文件、8,135,691字节；闭包：1,580文件、18,509,882字节。观测窗口三个摘要一致、full_content_checked=true、drifted=false。QE扫描耗时为0.230013、0.039177、0.032196秒；依赖扫描为0.316298、0.103402、0.082433秒；合计为0.546311、0.142579、0.114629秒。

这不是完整 context 或端到端性能测试：没有运行 COS/CUDA/runtime 捕获，也不能用冷热差异推断每次收益。文件数是当时工作树观测，不是固定生产配置。

## 不能直接去重的反例

最初考虑在同一次 capture 内交接带 root/scope 的严格 QE receipt，省去闭包里的第二次 QE 读取，运行前后分别重新捕获。但额外审计发现，这会减少现有的时间观察窗口。

独立有界 fixture（一个3字节 Python 文件，临时目录自动清理）先读版本A，两次扫描之间变成B，再恢复A进行 post 两次扫描。实测：pre两次摘要不同且各自scan成功；post两次摘要相同；原dual context的pre/post不同，而去重后的single context相同。因此不能宣称单次交接保留原来全部漂移检测能力。

修复前没有显式比较最先 QE digest 与闭包 QE component digest，而是将它们一起组成 executable identity。两次扫描间变化会形成混合时点身份，后续 context 比较可以观察到；但单次 capture 本身不直接拒绝这类不一致。

## 当前决定与后续

不删除第二次全字节扫描，不使用 mtime/stat 缓存替代内容验证，也没有将拟议去重方案用于 qualification。

本轮已增加两个 QE digest 的显式一致性校验：组件绑定必须完整、按声明顺序且无重复，摘要为合法 SHA-256；独立 QE 与闭包 QE 不一致时 fail closed。新增反例先在旧实现上出现 `DID NOT RAISE SourceQualificationError`，修复后相关身份、依赖与资格独立回归 72 passed（1 条既有 fork 警告）。真实工作树的正向探测见 `benchmarks/source_cross_digest_live_probe_20261004.json`。

此检查不是跨扫描原子快照，也不证明已加载字节码与磁盘内容一致；完全发生在观察间隔内的 A→B→A 仍可能不可见。真实 COS 基准期间仍保持 QE/DA/FO/FP Python 源码冻结，源码改变后重新生成性能资格。
