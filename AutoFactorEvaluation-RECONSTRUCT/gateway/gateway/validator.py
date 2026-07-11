"""兼容旧 gateway.validator 导入。"""
try:
    from gateway.scripts.validator import *  # noqa: F401,F403
except ImportError:
    from scripts.validator import *  # type: ignore # noqa: F401,F403
