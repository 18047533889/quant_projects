"""因子落盘系统自定义异常。"""


class FactorHashMismatchError(Exception):
    """因子 AST Hash 不匹配：公式已变更，请升级版本号（另起 factor_id）。

    触发场景：某人修改了因子公式代码但沿用了旧的 ``factor_id``，
    此时 Materializer 拒绝覆盖落盘，强制研究员新建版本号。
    """


class FactorNotFoundError(Exception):
    """因子未在 Catalog 中注册，无法执行读取或水位线查询。"""


class MaterializePartitionError(Exception):
    """分区落盘部分失败：已成功分区已写入，失败分区记录在 checkpoint。"""


class ClickHouseWriteError(Exception):
    """ClickHouse 写入失败：主存储（staging/local）可能已成功，需人工对账或补偿。

    ``summary`` 携带 ``materialize()`` 返回的部分 summary，便于运维定位。
    """

    def __init__(self, message: str, *, summary: dict | None = None):
        super().__init__(message)
        self.summary = summary or {}


class DualWriteError(Exception):
    """多目标物化部分成功：例如 staging 已 upsert 但 ClickHouse 失败。"""

    def __init__(self, message: str, *, summary: dict | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.summary = summary or {}
        self.cause = cause
