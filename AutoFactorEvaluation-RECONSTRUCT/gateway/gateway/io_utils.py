"""兼容旧 gateway.io_utils 导入。"""
try:
    from gateway.scripts.io_utils import *  # noqa: F401,F403
except ImportError:
    from scripts.io_utils import *  # type: ignore # noqa: F401,F403
