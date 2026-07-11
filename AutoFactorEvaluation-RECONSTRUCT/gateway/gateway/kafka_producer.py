"""兼容旧 gateway.kafka_producer 导入。"""
try:
    from gateway.scripts.kafka_producer import *  # noqa: F401,F403
except ImportError:
    from scripts.kafka_producer import *  # type: ignore # noqa: F401,F403
