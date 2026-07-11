"""兼容旧 gateway.future_scanner 导入。"""
try:
    from gateway.scripts.future_scanner import *  # noqa: F401,F403
except ImportError:
    from scripts.future_scanner import *  # type: ignore # noqa: F401,F403
