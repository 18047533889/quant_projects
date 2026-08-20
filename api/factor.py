"""因子容器：名称 + ``Expr`` 树 + 业务元数据（频率、股票池、描述）。

``Factor`` 本身不参与计算；经 ``FactorEngine.compile`` 变为 ``PlanNode`` 后由 backend 执行。
``freq`` / ``universe`` 写入 YAML 与物化元数据，不改变 IR 类型推导。
"""

from dataclasses import dataclass

from expr.base import Expr


@dataclass(frozen=True)
class FactorExecutionScopeHint:
    """因子执行作用域提示（P1-01 / P1-33）。

    携带跨执行路径完整的执行作用域字段（market / universe_id / frequency /
    calendar_id / decision_time_policy / source_scope_hash），供
    ``runtime.engine._scope_from_factor`` 推断 ``FactorExecutionScope``。
    字段缺省 ``None``，保持向后兼容：未显式声明的字段不覆盖因子级属性。

    .. note:: 这是**执行作用域提示**，不是因子语义身份（runtime
       ``FactorSemanticIdentity`` 才是单一定义身份，见
       ``runtime/factor_identity.py``）。P1-33：本类原名为
       ``FactorSemanticIdentity``，与 runtime 身份重名造成歧义；保留
       ``FactorSemanticIdentity = FactorExecutionScopeHint`` 模块级别名以兼容
       既有导入。

    Attributes
    ----------
    market : str | None
        市场标识（``"A"`` / ``"US"`` …）；``None`` 表示未声明，绝不默认 ``"A"``。
    universe_id : str | None
        股票池标识（``"ALL"`` / ``"CSI300"`` / ``"US_NASDAQ100"`` …）。
    frequency : str | None
        业务语义频率（如 ``"1d"``）。
    calendar_id : str | None
        交易日历标识（如 ``"SSE"`` / ``"NYSE"``）。
    decision_time_policy : str | None
        决策时间策略。
    source_scope_hash : str | None
        数据源作用域哈希；为空时由引擎按真实数据源配置现算（P1-04）。
    """

    market: str | None = None
    universe_id: str | None = None
    frequency: str | None = None
    calendar_id: str | None = None
    decision_time_policy: str | None = None
    source_scope_hash: str | None = None


# R32-P1-044: 兼容别名退役。历史导入 ``from api.factor import FactorSemanticIdentity``
# 拿到的仍是执行作用域提示（供 ``Factor.semantic_identity`` 注解 / 引擎 scope
# 推断）。真正的因子语义身份在 ``runtime.factor_identity``。本模块不再直接绑定
# 该重名（会误导开发者把 scope hint 当 identity）；通过 ``__getattr__`` 抛
# ``DeprecationWarning`` 兼容旧导入，进入 deprecation 周期。
def __getattr__(name: str):
    """R32-P1-044: ``FactorSemanticIdentity`` 兼容名走 deprecation shim。"""
    if name == "FactorSemanticIdentity":
        import warnings

        warnings.warn(
            "api.factor.FactorSemanticIdentity is deprecated (R32-P1-044): this "
            "name is the execution-scope HINT, not the semantic identity. Use "
            "FactorExecutionScopeHint, or "
            "runtime.factor_identity.FactorSemanticIdentity for the real "
            "semantic identity.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FactorExecutionScopeHint
    raise AttributeError(name)


@dataclass(frozen=True)
class Factor:
    """可编译、可执行的一条因子；核心负载是 ``expr: Expr``。

    经 ``FactorEngine.compile`` 编译为 ``PlanNode`` 后由 backend 执行；
    ``freq`` / ``universe`` 仅写入 YAML 与物化元数据，不改变 IR 类型推导。

    Attributes
    ----------
    name : str
        因子唯一标识（YAML key / 物化列名）。
    expr : Expr
        由 ``col``、``CleanedCall`` 等节点组成的表达式树。
    freq : str
        业务语义频率（默认 ``"1d"``），供配置与文档。
    universe : str | None
        股票池或标签名；执行时以数据源 universe 为准。
    description : str | None
        人类可读说明（与 ``source_expr`` 不同）。
    source_expr : str | None
        原始 DSL 字符串，用于 lineage / 审计。
    """

    name: str
    expr: Expr
    freq: str = "1d"  # 业务语义频率，写入 YAML/物化元数据，不参与 IR 类型推导
    universe: str | None = None  # 股票池/标签，供配置与文档；执行时以数据源为准
    description: str | None = None
    source_expr: str | None = None  # 原始 DSL 字符串（lineage 用，勿与 description 混用）
    # R11 #5: parser-surface 元数据随 Factor 携带，供 full definition 持久化，
    # 使事件增量重建引擎时能还原原始 surface/dialect（而不是退化成默认值）。
    surface: str = "daily"
    dialect: str = "native"
    dialect_version: str | None = None
    # P1-01: 完整执行作用域提示；None 时引擎回退到因子级属性推断。默认 None
    # 保持向后兼容（parse_factor / 既有构造调用点不受影响）。
    semantic_identity: FactorExecutionScopeHint | None = None

    def __post_init__(self) -> None:
        """R32-P0-043: Direct Python API 的 ``Factor.name`` 走同一 domain validator。

        HTTP validator 已拒绝超长/穿越；Direct Python API 直接构造 ``Factor``
        可绕过 service —— 必须下沉到 domain 层校验（``security.factor_id``）。
        非法名抛 ``ValueError``（冻结 dataclass 不赋值，仅校验）。
        """
        from security.factor_id import FactorIdError, validate_factor_id

        try:
            validate_factor_id(self.name)
        except FactorIdError as exc:
            raise ValueError(str(exc)) from exc
