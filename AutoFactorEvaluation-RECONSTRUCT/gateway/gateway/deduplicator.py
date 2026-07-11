"""兼容旧 gateway.deduplicator 导入。"""
try:
    from gateway.scripts.deduplicator import *  # noqa: F401,F403
except ImportError:
    from scripts.deduplicator import *  # type: ignore # noqa: F401,F403
