"""data_access HTTP 读数服务（服务端 + 轻量客户端）。"""

from .client import DataAccessClient
from .models import ReadRequest, ReadResponseMeta

__all__ = ["DataAccessClient", "ReadRequest", "ReadResponseMeta"]
