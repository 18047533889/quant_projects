"""因子落盘系统自定义异常。"""


class FactorHashMismatchError(Exception):
    """因子 AST Hash 与注册记录不一致时抛出。

    参数:
        无


    触发场景：某人修改了因子公式代码但沿用了旧的 ``factor_id``，
        此时 Materializer 拒绝覆盖落盘，强制研究员新建版本号。
    """


class FactorSemanticIdentityMismatchError(Exception):
    """因子语义身份（factor_version）与注册记录不一致时抛出（R11 #6）。

    AST 相同但执行语义变化（data_source/universe/market/pit/backend/dialect/
    算子或字段 catalog 变化）时，production 下拒绝沿用同一个 ``factor_id``，
    强制新版本或显式全量重建——避免「前半段来自 source A、后半段来自 source B
    但 catalog 认为是同一个 factor」。
    """


class UnreconstructableDataSource(Exception):
    """数据源的 ``execution_spec()`` 无法还原为可重建的 canonical 配置（#收官轮 P1）。

    production 下 Composite 任一 child 缺失 / 无法序列化 source contract 时抛出，
    绝不伪造 ``{"type": "data_access"}`` 这类缺 ``dataset`` 的残缺 spec——那会把
    「无法恢复 source contract」伪装成「有 source config」，事件增量 rebuild 时会
    拿到一个无法工作的假配置。research 下返回 ``None``（不发明 config）。
    """


class FactorNotFoundError(Exception):
    """因子未在 Catalog 注册时抛出。

    参数:
        无
    """


class MaterializePartitionError(Exception):
    """分区物化部分失败时抛出。
    
    参数:
        无
    """


class ClickHouseWriteError(Exception):
    """ClickHouse 写入失败时抛出。
    
    参数:
        message: 见函数签名
        summary: 见函数签名（可选）
    
    
    ``summary`` 携带 ``materialize()`` 返回的部分 summary，便于运维定位。
    """

    def __init__(self, message: str, *, summary: dict | None = None):
        """初始化实例。
        
        参数:
            message: 见函数签名
            summary: 见函数签名（可选）
        
        返回:
            无
        """
        super().__init__(message)
        self.summary = summary or {}


class MaterializedButCatalogCommitFailed(Exception):
    """因子数据已落盘但依赖 / full-definition 写入 catalog 失败（#收官轮 P0）。

    物理数据 + 水位线已提交，但依赖边 / full definition 未持久化——事件增量
    重建链路（``DataEvent`` → dependency lookup → reconstruct 完整因子 →
    incremental recompute）无法还原它。production 下抛此异常而非 warning 吞掉：
    该因子处于 **IN_DOUBT**，调用方必须对账（重试 catalog 写入 / 标记
    NEED_RECONCILE / 回滚水位线），绝不能把这次运行当作完整 PUBLISHED success。

    ``materialization`` 携带已落盘的 ``materialize()`` 摘要供对账。
    """

    def __init__(self, message: str, *, materialization: dict | None = None):
        """初始化实例。

        参数:
            message: 见函数签名
            materialization: 见函数签名（可选）

        返回:
            无
        """
        super().__init__(message)
        self.materialization = materialization or {}


class DualWriteError(Exception):
    """多目标物化部分成功时抛出。

    参数:
        message: 见函数签名
        summary: 见函数签名（可选）
        cause: 见函数签名（可选）
    """

    def __init__(self, message: str, *, summary: dict | None = None, cause: Exception | None = None):
        """初始化实例。

        参数:
            message: 见函数签名
            summary: 见函数签名（可选）
            cause: 见函数签名（可选）

        返回:
            无
        """
        super().__init__(message)
        self.summary = summary or {}
        self.cause = cause


class CatalogCorruptionError(Exception):
    """catalog JSON 字段损坏/不可解析时抛出（NEW-P0-45）。

    production 下 ``_parse_json_field`` 遇到非法 JSON（如 ``"{broken"``）或
    checksum 不匹配时，抛本异常而非返回 ``{}``——否则「catalog 数据损坏」会被
    伪装成「config 缺失」，事件增量 rebuild 拿到假空配置。research 下保留容错
    返回 ``{}`` 的历史行为。
    """


class CatalogSerializationError(Exception):
    """catalog 全量定义序列化遇到不支持的对象时抛出（NEW-P0-44）。

    production 下 full definition / data-source config 必须走 typed JSON schema：
    未知类型（callable / 自定义类 / enum-like 配置）直接报错，绝不
    ``json.dumps(..., default=str)`` 字符串化——字符串化后的对象无法还原重建。
    """


class FactorRetiredError(Exception):
    """deleted-factor 复用被拒时抛出（NEW-P0-56/57）。

    ``delete_factor`` 只删 catalog 元数据、不删物理分区文件。复用同一
    ``factor_id`` 而不显式 ``rebuild=True`` 时，新世代数据会与旧世代分区
    （2020-2025 旧因子 / 2026 新因子）混在同一个目录下。抛本异常强制调用方
    升级 factor_id 或显式声明全量重建。
    """


class ProductionModeResolutionError(RuntimeError):
    """R32-P0-014: production authority 解析意外异常时抛出（fail-open 关闭）。

    catalog 等 production 决策点解析运行模式时，意外异常不得默认 ``False`` 静默
    回落成 research —— 那会把「production authority 解析失败」伪装成「非生产」，
    绕过 production 硬门（identity / precision / typed-JSON / direct-local 禁写）。
    """


class CatalogMigrationError(RuntimeError):
    """R32-P0-012: catalog schema migration 失败 / 校验和不匹配时抛出。

    schema 版本化独占迁移：from_version → to_version、migration checksum、
    destructive migration 前 backup、migration report。任何一步失败都必须在
    迁移前状态停止（事务回滚），绝不半迁移。
    """
