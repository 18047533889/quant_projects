"""
data_access.exceptions —— 统一错误类型

职责：
    把 data_access 能抛出的错误分成三类，便于上游写重试/告警策略。
非职责：
    不负责格式化错误信息（由调用方或 logging 处理）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations


class DataAccessError(Exception):
    """data_access 所有错误的基类，方便上层 except 时一把抓。"""


class ValidationError(DataAccessError):
    """配置/参数/路径白名单不通过时抛出。

    典型场景：
        - 数据集名在 datasets.yaml 里不存在
        - 参数化数据集缺少必填参数（例如 factor_lake 没传 factor_id）
        - 写路径落到了白名单以外
        - 试图绕过 publish() 直接写 published_root
    """


class DataError(DataAccessError):
    """数据本身的问题：文件不存在、schema 对不上、空文件、空结果。

    典型场景：
        - 指定 time_range 过滤后行数为 0（上游可决定是否算错）
        - parquet 文件存在但列名不匹配请求列
        - 目录存在但没有任何 parquet 文件
    """


class EngineError(DataAccessError):
    """DuckDB/Arrow/pyarrow 内部错误，一般是底层异常的包装。

    典型场景：
        - DuckDB 解析 SQL 失败（通常是 data_access 代码的 bug，不是用户错）
        - 读 parquet 时 pyarrow 抛 OSError / ArrowInvalid
        - 发布流程里 os.replace 失败
    """


class DeadlineExceeded(DataAccessError):
    """查询超过 max_elapsed_ms 被取消。

    由 watchdog 在截止时间后调用 ``conn.interrupt()`` 触发；表示查询被主动
    终止（而不是查完才发现超时）。
    """


class AmbiguousSemanticFieldError(ValidationError):
    """跨市场逻辑字段无法在无上下文下消歧（#54 fail-closed）。

    典型场景：``resolve_one("market_cap")`` 没传 market/dataset，而
    ``market_cap`` 在 A股/美股都登记了——生产模式直接抛错，禁止 YAML 顺序
    决定市场。
    """


class AmbiguousFieldError(ValidationError):
    """字段回退到 registry 全局查找时命中多个候选（#33 fail ambiguous）。

    典型场景：``resolve_fields("Close")`` 未传 dataset，registry 里几十张表都
    有 Close 物理列——生产模式抛错，禁止「取 registry 第一个」。
    """


class MatrixUnavailable(DataAccessError):
    """factor_matrix 物化层不可用（未登记 / 无数据）。

    只表示「矩阵不存在」，不表示版本/schema/参数错误——fallback 逻辑只捕这个。
    """


class MatrixCoverageMiss(DataAccessError):
    """factor_matrix 物化层存在但覆盖不到请求的 (universe, frequency, 因子)。"""


class SnapshotBuildError(DataAccessError):
    """read_joined / sql 的多数据集 snapshot 构建不完整（#14 fail-closed）。

    任意参与数据集没有成功构建 snapshot → 抛此错，禁止「查询成功但 lineage /
    cache / replay 不可靠」。
    """
