"""因子落盘系统自定义异常。"""


class FactorHashMismatchError(Exception):
    """因子 AST Hash 不匹配：公式已变更，请升级版本号（另起 factor_id）。

    触发场景：某人修改了因子公式代码但沿用了旧的 ``factor_id``，
    此时 Materializer 拒绝覆盖落盘，强制研究员新建版本号。
    """


class FactorNotFoundError(Exception):
    """因子未在 Catalog 中注册，无法执行读取或水位线查询。"""
