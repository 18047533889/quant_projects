"""
data_access.exceptions —— 统一错误类型

职责：
    把 data_access 能抛出的错误分成三类，便于上游写重试/告警策略。
非职责：
    不负责格式化错误信息（由调用方或 logging 处理）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations


class DataAccessError(Exception):
    """data_access 所有错误的基类，方便上层 except 时一把抓。"""


class ValidationError(DataAccessError):
    """配置/参数/路径白名单不通过时抛出。

    典型场景：
        - 数据集名在 datasets.yaml 里不存在
        - 参数化数据集缺少必填参数（例如 factor_lake 没传 factor_id）
        - 写路径落到了白名单以外
        - 试图绕过 publish() 直接写 published_root
    """


class DataError(DataAccessError):
    """数据本身的问题：文件不存在、schema 对不上、空文件、空结果。

    典型场景：
        - 指定 time_range 过滤后行数为 0（上游可决定是否算错）
        - parquet 文件存在但列名不匹配请求列
        - 目录存在但没有任何 parquet 文件
    """


class EngineError(DataAccessError):
    """DuckDB/Arrow/pyarrow 内部错误，一般是底层异常的包装。

    典型场景：
        - DuckDB 解析 SQL 失败（通常是 data_access 代码的 bug，不是用户错）
        - 读 parquet 时 pyarrow 抛 OSError / ArrowInvalid
        - 发布流程里 os.replace 失败
    """


class DeadlineExceeded(DataAccessError):
    """查询超过 max_elapsed_ms 被取消。

    由 watchdog 在截止时间后调用 ``conn.interrupt()`` 触发；表示查询被主动
    终止（而不是查完才发现超时）。
    """


class ResourceBudgetExceeded(DataAccessError):
    """资源预算超限（#P0-8：deadline 连接池并发上限等待超时等）。

    不同于 DeadlineExceeded（查询本身超时），这里是**准入失败**——池容量满、
    等待超时仍未获得连接。
    """


class EngineClosedError(EngineError):
    """在 ``DuckDBEngine.close()`` 之后仍尝试 acquire/execute。

    #P0-10：关闭后的 engine 必须 fail-fast，不能静默复用已关闭连接。
    """


class AmbiguousSemanticFieldError(ValidationError):
    """跨市场逻辑字段无法在无上下文下消歧（#54 fail-closed）。

    典型场景：``resolve_one("market_cap")`` 没传 market/dataset，而
    ``market_cap`` 在 A股/美股都登记了——生产模式直接抛错，禁止 YAML 顺序
    决定市场。
    """


class AmbiguousFieldError(ValidationError):
    """字段回退到 registry 全局查找时命中多个候选（#33 fail ambiguous）。

    典型场景：``resolve_fields("Close")`` 未传 dataset，registry 里几十张表都
    有 Close 物理列——R39 #69 起 research/production 一律抛错，禁止「取 registry
    第一个」。
    """


class UnsupportedFeatureError(ValidationError):
    """接口已声明但当前执行链真正不支持的降级/变换功能（R39 #67/#68）。

    不同于 generic ValidationError：这是 typed 的「功能不支持」错误——调用方
    可以精确 catch 并决定回退策略，而不是把「不支持的 feature」和「参数非法」
    混为一谈。
    """


class MatrixUnavailable(DataAccessError):
    """factor_matrix 物化层不可用（未登记 / 无数据）。

    只表示「矩阵不存在」，不表示版本/schema/参数错误——fallback 逻辑只捕这个。
    """


class MatrixCoverageMiss(DataAccessError):
    """factor_matrix 物化层存在但覆盖不到请求的 (universe, frequency, 因子)。"""


class SnapshotBuildError(DataAccessError):
    pass


class AuditWriteError(DataAccessError):
    """#P1-final closure 22：权威写入/发布的审计写失败。

    业务已成功但审计静默失败 = 合规证据缺失。``audit.record(durable=True)``
    在写审计日志失败时抛本错误，调用方据此知道「发布了但审计没落盘」。
    """
    """read_joined / sql 的多数据集 snapshot 构建不完整（#14 fail-closed）。

    任意参与数据集没有成功构建 snapshot → 抛此错，禁止「查询成功但 lineage /
    cache / replay 不可靠」。
    """


class AuthorizationError(DataAccessError):
    """逻辑授权层拒绝（R24 P0-S2）。

    表示「这个身份没有权限做这件事」——**不是 backend 错误，禁止 fallback 到
    更高身份**（不得读 ~/.cos.yaml 换 credential、不得换 profile、不得换一个
    更高权限的 bucket）。外部错误信息已经脱敏（P1-S6）。
    """


class AccessDeniedError(AuthorizationError):
    """显式 403 / Forbidden / PermissionDenied / AccessDenied。

    与 ``NotFound`` / ``NoRows`` 严格区分（P1-S9）：universe 解析 / calendar /
    metadata probe / coverage / remote fallback / mirror 遇到本错误必须原样
    传播，绝不能 ``except Exception: return None`` 或降级成空 universe。
    """


class TemporalContractError(DataAccessError):
    """时间语义契约违例（R24 P0-PIT 系列 / P1-S9）。

    与「数据不存在」区分：请求违反 knowledge/effective/period 时钟、availability、
    date_label vs instant、timeframe 等时间契约时抛出。
    """


class CalendarUnavailableError(ValidationError, TemporalContractError):
    """R25 P0-005 / §78：calendar-required availability 需要真实日历，但日历不可用。

    与旧裸 ``ValidationError`` 区分：PIT/availability 在「无法证明可见时点」时
    fail-closed，供调用方按 §113（Calendar unavailable → PIT-required production
    reject）处理。同时继承 ``ValidationError`` 保证 R24 既有契约（match=日历不可用）
    仍成立。
    """


class AvailabilityLatencyError(TemporalContractError):
    """非零 availability latency 无法应用到已编译的可见时点。

    延迟应用失败时必须保留底层异常为 ``__cause__``，禁止返回未延迟值，因为后者
    会把尚不可见的数据提前暴露给 PIT 查询。
    """


class SemanticCatalogUnavailableError(DataAccessError):
    """权威 semantic catalog 无法导入、bootstrap 或读取其存储。

    这与 catalog 成功加载但数据集没有 availability 声明严格不同；只有后者才可
    按显式解析结果使用 ``same_day``。
    """


class FilterContractError(ValidationError):
    """R25 P0-004 / §78：过滤契约违例（required filter 缺失 / exactly-one 违反）。

    与普通 ValidationError 区分：request 违反 dataset 级过滤契约（如 US finance
    timeframe 缺失/多值）时抛。
    """


class SourceSnapshotUnavailable(DataAccessError):
    """R25 P0-009/012 / §78：source snapshot 无法解析 / 无法证明。

    production remote 遇到 unresolved wildcard / 无法证明 source identity 时
    fail-closed，绝不把「目录非空」当「完整」。
    """


class SourceSnapshotChanged(DataAccessError):
    """R25 P0-012/013 / §78：source snapshot 在执行期间变化（ETag/version 变化）。

    remote 读前 HEAD 与执行时不一致 / auto hybrid 混 source epoch 时抛。
    """


class PhysicalPartitionResolutionError(DataAccessError):
    """R25 P0-001/016 / §78：物理分区解析失败（partition_clock 无法映射）。

    不能用 request time_range 展开 period/event 文件名且无 source index 时抛。
    """


class MissingRequiredPartition(DataAccessError):
    """R25 P0-015 / §78：dense 数据集缺失必需 partition（missing=ERROR）。

    不能统一「404 → debug 跳过」——交易行情缺失交易日文件必须 fail-closed。
    """


class SchemaContractError(ValidationError):
    """R25 P0-033 / §78：schema evolution 契约违例（跨 epoch 字段缺失/dtype 变化）。

    不允许 ``union_by_name=True`` 当「什么 schema 都能拼」的万能开关。
    """


class ContractCompilationError(ValidationError):
    """R26-P0-011：RuntimeDatasetContract 编译失败。

    production / automated_research 下 contract 编译异常 = 不可用 ≠ fallback。
    绝对禁止「except Exception → 旧逻辑继续」。仅对明确无 contract 的合法
    dataset 允许走声明好的 legacy-compatible contract。
    """


class PITUnavailable(DataAccessError):
    """R26-P0-013：PIT availability 无法证明（calendar mapping 缺失 / 右边界未知）。

    production SQL/Python/Polars 一律拒绝，绝不 COALESCE 到 knowledge date
    same-day 放行。
    """


class SemanticFieldNotRegisteredError(ValidationError):
    """R22 语义字段 fail-closed：production/strict 下逻辑名未登记即拒绝。

    与「登记了但声明不完整」严格区分（semantic_catalog 加载时已强制完整声明）。
    ``store.resolve_fields`` 对不在 catalog、也不在显式 dataset 物理 schema 的
    逻辑名，在 production 抛出本错误——禁止物理列名静默回退（掩盖未登记
    字段的前视/单位/跨市场语义）；research/legacy 允许回退保持兼容。
    """


class SemanticUnitError(ValidationError):
    """R25 §36/37 / §78：单位/币种/flow semantics 不兼容。

    跨市场单位/货币/定义不可比、requires_fx 无 FX PIT 时抛。
    """


class ResourceAdmissionError(DataAccessError):
    """R25 §26/27 / §78：资源准入拒绝（object/bytes/remote 预算、全局 governor）。

    不能「执行完才知道扫了多少」——入场前 object count / bytes 超限即拒绝。
    """


class CacheSecurityError(DataAccessError):
    """R25 §28 / §78：缓存安全违例（principal scope 不匹配 / 权限过宽）。

    cache root 权限不对 / 高权限缓存被低权限复用拒绝。
    """


class UnknownProvenanceError(DataAccessError):
    """R25 §50 / §78：裸 DataFrame/无 provenance 帧进入 production。

    FactorEngine production 接裸 dataframe 时抛 UnknownProvenanceError。
    """


class CapabilityUnavailableError(DataAccessError):
    """R39 P0 #43：后端能力不可用（组合读共享 pool 数据库不可用）。

    与「数据库损坏 / schema 错误 / pool 状态错误」严格区分：只有能力（如
    DuckDB 共享 pool 文件不可用）明确缺失时才抛本错误，调用方可以回退
    （临时 parquet）；其余底层错误必须原样传播，禁止 ``except Exception``
    一把抓当能力缺失回退。
    """


class SourceResolutionError(DataAccessError):
    """R39 P0 #44：物理读取范围解析失败（不是「数据集为空」）。

    与 ``EmptyPhysicalScope``（合法空数据集）严格区分：解析异常（glob 失败 /
    远程不可达 / 授权拒绝）抛本错误；``[]`` 只能表示「合法空」，绝不静默
    同时代表「解析失败」。
    """


def propagate_authorization(exc: BaseException) -> None:
    """R24 P1-S9 §25：AuthorizationError 必须原样传播。

    调用方在 ``except Exception:`` 宽捕获里应先行调用本函数——若当前异常是
    ``AuthorizationError``（含 ``AccessDeniedError``）就原样重抛，绝不能被
    ``return None`` / 空 universe / 降级数据 吞掉。
    """
    if isinstance(exc, AuthorizationError):
        raise exc


class FinancialRevisionAmbiguityError(DataAccessError):
    """财务 duplicate/revision 冲突无法确定性解决（R24 P1-PIT7）。

    同一 semantic key 多行且值冲突、又没有可信 revision availability 时
    production fail-closed——绝不依赖 parquet 扫描顺序。
    """


class CommittedButAuditFailed(AuditWriteError):
    """#P0 收官（0.9.5）：**数据已提交、仅审计确认失败**。

    与普通 ``AuditWriteError`` 的区别：body 本身**已经成功提交**（candidate →
    target rename、post-verify 都完成，``ok=True``），只有最后 durable audit 落盘
    失败。外层的 ``_dataset_mutation`` 必须把这种情形当 **COMMITTED** 处理——
    仍然 rebuild manifest（数据确实发布了，manifest 必须追平），再单独向上报告
    审计失败。真正的 mutation failure（body 中途抛错）仍是 ABORTED，绝不 rebuild。
    """
