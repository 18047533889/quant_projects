"""兼容旧 gateway.config 导入。"""
try:
    from gateway.scripts.config import *  # noqa: F401,F403
except ImportError:
    from scripts.config import *  # type: ignore # noqa: F401,F403
