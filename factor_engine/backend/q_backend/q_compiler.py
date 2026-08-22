"""q/K Expression Compiler from Canonical IR.

将 FactorEngine Canonical IR 编译为 q 代码（文档 §25）。

Hard Gates (文档 §85):
- Q_BACKEND_CANONICAL_IR_ONLY: q 只能是 Canonical IR 的编译目标
- Q_BACKEND_ZERO_SEMANTIC_AUTHORITY: q 不得成为语义权威
- Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO: 禁止 Python→q op1→Python→q op2

遵循文档 §25: Region 级边界，不是 operator 级边界。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Literal

from backend.q_backend.q_capability import QBackendCapability
from backend.q_backend.q_physical_implementation_registry import (
    QPhysicalImplementationRegistry,
    get_installed_q_physical_implementation_registry,
    install_q_physical_implementation_registry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QRegionPlan:
    """q 执行 Region 的编译计划。

    文档 §10: 从 PlanRoute 升级为真正 PhysicalRegionPlan。
    """
    region_id: str
    node_ids: tuple[str, ...]
    q_code: str
    input_tables: tuple[str, ...]
    output_table: str
    requires_sort: bool = False
    estimated_rows: int = 0


class QLoweringKind(str, Enum):
    """Typed q expression shape used by compiler dispatch."""

    INFIX = "infix"
    PREFIX = "prefix"
    LAMBDA = "lambda"
    ROLLING = "rolling"
    AGGREGATE = "aggregate"
    SPECIALIZED = "specialized"


@dataclass(frozen=True)
class QLowering:
    source: str
    kind: QLoweringKind

    @property
    def evidence_source(self) -> str:
        """Bind certification to both q source and dispatch semantics."""
        return f"{self.kind.value}:{self.source}"


class QCompiler:
    """Canonical IR → q 代码编译器。

    遵循文档 §2 架构原则：
    - DSL/IR/OperatorSemanticRegistry/PIT/Unit/Factor identity 是 canonical authority
    - q 只负责 compile_to_q(region) 和 execute_q_region()
    """

    def __init__(self, capability: QBackendCapability | None = None):
        # Build the executable map before binding the instance-owned registry.
        # This keeps direct compiler instances and the global compiler on the
        # same lowering authority without sharing mutable registry state.
        self._operator_map = self._build_operator_map()
        if capability is None:
            registry = QPhysicalImplementationRegistry(
                lowerings=self.executable_lowerings(),
                declared_targets=QPhysicalImplementationRegistry._DECLARED_TARGETS,
            )
            self.capability = QBackendCapability(registry)
        else:
            self.capability = capability

    def executable_lowerings(self) -> dict[str, str]:
        return {
            canonical: lowering.evidence_source
            for canonical, lowering in self._operator_map.items()
        }

    def declared_targets(self) -> frozenset[str]:
        return self.capability.registry.declared_targets()

    def _build_operator_map(self) -> dict[str, QLowering]:
        """Build the typed canonical operator to q lowering map."""
        sources = {
            # Arithmetic
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "%",
            "negate": "neg",
            "abs": "abs",
            "power": "xexp",
            "sqrt": "sqrt",
            "log": "log",
            "exp": "exp",
            "log1p": "{log 1+x}",
            "expm1": "{(exp x)-1}",
            "sign": "signum",
            "floor": "floor",
            "ceil": "ceiling",
            "round": "{`long$x+0.5}",

            # Comparison
            "greater": ">",
            "less": "<",
            "greater_equal": ">=",
            "less_equal": "<=",
            "equal": "=",
            "not_equal": "<>",

            # Lag/Delta
            "lag": "prev",
            "delta": "deltas",
            "ts_diff": "deltas",
            "pct_change": "{(deltas x)%prev x}",
            "ts_returns": "{(deltas x)%prev x}",

            # Rolling (窗口函数)
            "ts_mean": "mavg",
            "ts_sum": "msum",
            "ts_std": "mdev",
            "ts_min": "mmin",
            "ts_max": "mmax",
            "ts_count": "mcount",

            # Cumulative operations
            "ts_cumsum": "sums",
            "ts_cumprod": "prds",
            "ts_cummax": "maxs",
            "ts_cummin": "mins",

            # Aggregations
            "mean": "avg",
            "sum": "sum",
            "std": "dev",
            "min": "min",
            "max": "max",
            "median": "med",
            "var": "{dev[x] xexp 2}",
            "product": "{(*/)x}",
            "first": "first",
            "last": "last",
            "count_nonzero": "{sum x<>0}",

            # Rank
            "rank": "rank",
            "cs_rank": "{(iasc iasc x)%count x}",

            # Cross-sectional operations
            "cs_zscore": "{(x-avg x)%dev x}",
            "cs_demean": "{x-avg x}",
            "cs_normalize": "{x%sum abs x}",
            "cs_mean": "avg",
            "cs_std": "dev",
            "cs_median": "med",
            "cs_var": "{dev[x] xexp 2}",

            # Conditional/Fill
            "where": "?",
            "fillna": "^",
            "ffill": "fills",
            "clip": "{(x&y)|z}",  # clip between z and y

            # Correlation
            "ts_corr": "cor",
            "ts_cov": "cov",

            # Simple indicators
            "ema": "ema",
            "sma": "mavg",
            "wma": "{wavg[til count x;x]}",
        }
        by_kind = {
            QLoweringKind.INFIX: {
                "add", "subtract", "multiply", "divide", "power", "greater",
                "less", "greater_equal", "less_equal", "equal", "not_equal",
            },
            QLoweringKind.PREFIX: {
                "negate", "abs", "sqrt", "log", "exp", "sign", "floor",
                "ceil", "delta", "ts_diff", "ts_cumsum", "ts_cumprod",
                "ts_cummax", "ts_cummin", "ffill", "first", "last",
            },
            QLoweringKind.LAMBDA: {
                "log1p", "expm1", "round", "pct_change", "ts_returns",
                "var", "product", "count_nonzero", "cs_zscore", "cs_demean",
                "cs_normalize", "cs_var",
            },
            QLoweringKind.ROLLING: {
                "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_count",
            },
            QLoweringKind.AGGREGATE: {
                "mean", "sum", "std", "min", "max", "median", "cs_mean",
                "cs_std", "cs_median",
            },
            QLoweringKind.SPECIALIZED: {
                "lag", "rank", "cs_rank", "where", "fillna", "clip",
                "ts_corr", "ts_cov", "ema", "sma", "wma",
            },
        }
        classified = [name for names in by_kind.values() for name in names]
        if len(classified) != len(set(classified)) or set(classified) != set(sources):
            raise RuntimeError("q lowering kinds must classify every operator exactly once")
        kinds = {
            name: kind
            for kind, names in by_kind.items()
            for name in names
        }
        return {
            name: QLowering(source=source, kind=kinds[name])
            for name, source in sources.items()
        }

    def has_lowering(self, op_name: str) -> bool:
        """Return whether this compiler has an executable lowering."""
        return op_name in self._operator_map

    def is_production_certified(self, op_name: str) -> bool:
        """Admit only operators in the registry's globally consistent production set."""
        return op_name in self.capability.registry.get_production_ready()

    def can_compile_operator(self, op_name: str) -> bool:
        """Return whether an executable lowering exists; production admission is separate."""
        return self.has_lowering(op_name)

    def compile_operator(
        self,
        op_name: str,
        inputs: list[str],
        params: dict[str, Any] | None = None,
    ) -> str:
        """编译单个算子为 q 表达式。

        参数:
            op_name: 算子名称
            inputs: 输入变量名
            params: 算子参数

        返回:
            q 代码表达式

        抛出:
            ValueError: 不支持的算子
        """
        if not self.can_compile_operator(op_name):
            raise ValueError(f"Operator {op_name} not supported in q backend")

        lowering = self._operator_map.get(op_name)
        if lowering is None:
            raise ValueError(f"No q mapping for operator {op_name}")
        q_func = lowering.source

        params = params or {}

        def required_param(name: str) -> Any:
            if name not in params:
                raise ValueError(f"{op_name} requires canonical parameter '{name}'")
            return params[name]

        def positive_int(name: str) -> int:
            value = required_param(name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{op_name} parameter '{name}' must be a positive integer")
            if not math.isfinite(float(value)) or int(value) != value or value <= 0:
                raise ValueError(f"{op_name} parameter '{name}' must be a positive integer")
            return int(value)

        def finite_number(name: str) -> Real:
            value = required_param(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{op_name} parameter '{name}' must be a finite number")
            return value

        # Preserve the public rank lowering contract before generic function
        # and lambda dispatch can change the emitted expression shape.
        if op_name == "cs_rank":
            if len(inputs) != 1:
                raise ValueError(f"cs_rank requires 1 input, got {len(inputs)}")
            return f"({q_func})[{inputs[0]}]"

        if op_name == "rank":
            if len(inputs) != 1:
                raise ValueError(f"rank requires 1 input, got {len(inputs)}")
            return f"rank {inputs[0]}"

        # Lag must run before generic q-function/lambda handling.
        if op_name == "lag":
            periods = positive_int("periods")
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"prev {inputs[0]}" if periods == 1 else f"{periods} prev\\{inputs[0]}"

        # Windowed special lowerings must run before generic lambda handling.
        if op_name in {"ts_corr", "ts_cov"}:
            window = positive_int("window")
            if len(inputs) != 2:
                raise ValueError(f"{op_name} requires 2 inputs, got {len(inputs)}")
            return f"{window} {q_func}[{inputs[0]};{inputs[1]}]"

        if op_name == "ema":
            span = positive_int("span")
            alpha = 2.0 / (span + 1)
            if len(inputs) != 1:
                raise ValueError(f"ema requires 1 input, got {len(inputs)}")
            return f"ema[{alpha};{inputs[0]}]"

        if op_name == "wma":
            window = positive_int("window")
            if len(inputs) != 1:
                raise ValueError(f"wma requires 1 input, got {len(inputs)}")
            return f"{{wavg[til {window};-{window}#{inputs[0]}]}}each {window}_mavg {inputs[0]}"

        if op_name == "clip":
            lower = finite_number("lower")
            upper = finite_number("upper")
            if len(inputs) != 1:
                raise ValueError(f"clip requires 1 input, got {len(inputs)}")
            return f"({inputs[0]}|{lower})&{upper}"

        if op_name == "where":
            if len(inputs) != 3:
                raise ValueError(f"where requires 3 inputs (cond, true_val, false_val), got {len(inputs)}")
            return f"?[{inputs[0]};{inputs[1]};{inputs[2]}]"

        if op_name == "fillna":
            fill_value = finite_number("fill_value")
            if len(inputs) != 1:
                raise ValueError(f"fillna requires 1 input, got {len(inputs)}")
            return f"{inputs[0]}^{fill_value}"

        if lowering.kind is QLoweringKind.INFIX:
            if len(inputs) != 2:
                raise ValueError(f"{op_name} requires 2 inputs, got {len(inputs)}")
            return f"{inputs[0]} {q_func} {inputs[1]}"

        if lowering.kind is QLoweringKind.PREFIX:
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{q_func} {inputs[0]}"

        if lowering.kind is QLoweringKind.LAMBDA:
            if len(inputs) == 1:
                return f"({q_func})[{inputs[0]}]"
            elif len(inputs) == 2:
                return f"({q_func})[{inputs[0]};{inputs[1]}]"
            elif len(inputs) == 3:
                return f"({q_func})[{inputs[0]};{inputs[1]};{inputs[2]}]"
            else:
                raise ValueError(f"{op_name} with lambda requires 1-3 inputs, got {len(inputs)}")

        if lowering.kind is QLoweringKind.ROLLING:
            window = positive_int("window")
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{window} {q_func} {inputs[0]}"

        if op_name == "sma":
            window = positive_int("window")
            if len(inputs) != 1:
                raise ValueError(f"sma requires 1 input, got {len(inputs)}")
            return f"{window} mavg {inputs[0]}"

        if lowering.kind is QLoweringKind.AGGREGATE:
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{q_func} {inputs[0]}"

        raise ValueError(f"Compilation not implemented for {op_name}")

    def compile_region(
        self,
        region_id: str,
        nodes: list[dict[str, Any]],
        *,
        input_tables: list[str],
        output_name: str,
        mode: Literal["production", "research"] = "production",
    ) -> QRegionPlan:
        """编译整个 Region 为 q 代码。

        遵循文档 §25: Region 级边界。

        参数:
            region_id: Region 标识
            nodes: 节点列表（按拓扑顺序）
            input_tables: 输入表名
            output_name: 输出表名

        返回:
            QRegionPlan
        """
        if not nodes:
            raise ValueError("Empty region cannot be compiled")

        is_valid, unsupported = self.validate_region(nodes, mode=mode)
        if not is_valid:
            raise ValueError(
                f"Cannot admit q physical region in {mode} mode: "
                f"unsupported operators {unsupported}"
            )

        # 生成 q 代码
        q_statements = []
        q_statements.append(f"/ Region {region_id}")
        q_statements.append("")

        # 逐节点编译
        for node in nodes:
            node_id = node["node_id"]
            op_name = node["op"]
            inputs = node.get("inputs", [])
            attrs = node.get("attrs", {})

            try:
                q_expr = self.compile_operator(op_name, inputs, attrs)
                q_statements.append(f"{node_id}: {q_expr};")
            except Exception as e:
                logger.error(f"Failed to compile node {node_id}: {e}")
                raise

        # 输出赋值
        final_node = nodes[-1]["node_id"]
        q_statements.append(f"{output_name}: {final_node}")

        q_code = "\n".join(q_statements)

        # 检查是否需要排序
        requires_sort = any(
            self.capability.requires_global_sort(n["op"])
            for n in nodes
        )

        # 估算行数
        estimated_rows = nodes[0].get("estimated_rows", 0)

        return QRegionPlan(
            region_id=region_id,
            node_ids=tuple(n["node_id"] for n in nodes),
            q_code=q_code,
            input_tables=tuple(input_tables),
            output_table=output_name,
            requires_sort=requires_sort,
            estimated_rows=estimated_rows,
        )

    def validate_region(
        self,
        nodes: list[dict[str, Any]],
        *,
        mode: Literal["production", "research"] = "production",
    ) -> tuple[bool, list[str]]:
        """验证 Region 中所有节点是否可编译。

        参数:
            nodes: 节点列表

        返回:
            (is_valid, unsupported_ops)
        """
        unsupported = []
        for node in nodes:
            op_name = node["op"]
            admitted = (
                self.is_production_certified(op_name)
                if mode == "production"
                else self.can_compile_operator(op_name)
            )
            if not admitted:
                unsupported.append(op_name)

        return len(unsupported) == 0, unsupported


# Global singleton
_COMPILER: QCompiler | None = None


def get_q_compiler() -> QCompiler:
    """Return the global compiler bound to the installed registry authority."""
    global _COMPILER
    installed = get_installed_q_physical_implementation_registry()
    if _COMPILER is None:
        _COMPILER = QCompiler(
            QBackendCapability(installed) if installed is not None else None
        )
        install_q_physical_implementation_registry(_COMPILER.capability.registry)
    elif installed is not None and _COMPILER.capability.registry is not installed:
        raise RuntimeError(
            "q physical implementation registry was replaced after compiler "
            "bootstrap; restart with the intended registry installed first"
        )
    return _COMPILER
