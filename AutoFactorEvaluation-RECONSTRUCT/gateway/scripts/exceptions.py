"""
自定义异常类
"""

class GatewayError(Exception):
    """网关基础异常"""
    pass


class SchemaValidationError(GatewayError):
    """Schema 校验失败"""
    pass


class OperatorNotAllowedError(GatewayError):
    """算子不在白名单中"""
    pass


class DataTypeError(GatewayError):
    """数据类型错误"""
    pass


class FutureFunctionError(GatewayError):
    """检测到未来函数"""
    pass


class ComplexityError(GatewayError):
    """复杂度预算超出"""
    pass


class ConfigError(GatewayError):
    """配置错误"""
    pass