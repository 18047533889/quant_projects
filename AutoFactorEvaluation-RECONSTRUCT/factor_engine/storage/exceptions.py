"""因子落盘系统自定义异常。"""


class FactorHashMismatchError(Exception):
    """因子 AST Hash 与注册记录不一致时抛出。
    
    参数:
        无
    
    
    触发场景：某人修改了因子公式代码但沿用了旧的 ``factor_id``，
        此时 Materializer 拒绝覆盖落盘，强制研究员新建版本号。
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
