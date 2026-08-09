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
