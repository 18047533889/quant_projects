"""兼容旧 gateway.complexity 导入。"""
try:
    from gateway.scripts.complexity import *  # noqa: F401,F403
except ImportError:
    from scripts.complexity import *  # type: ignore # noqa: F401,F403
