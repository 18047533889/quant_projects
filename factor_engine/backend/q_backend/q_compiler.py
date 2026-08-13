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
from dataclasses import dataclass
from typing import Any

from backend.q_backend.q_capability import QBackendCapability, get_q_capability

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


class QCompiler:
    """Canonical IR → q 代码编译器。

    遵循文档 §2 架构原则：
    - DSL/IR/OperatorSemanticRegistry/PIT/Unit/Factor identity 是 canonical authority
    - q 只负责 compile_to_q(region) 和 execute_q_region()
    """

    def __init__(self, capability: QBackendCapability | None = None):
        self.capability = capability or get_q_capability()
        self._operator_map = self._build_operator_map()

    def _build_operator_map(self) -> dict[str, str]:
        """构建算子名称 → q 函数映射。"""
        return {
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
            "ts_median": "{(w#0Nf),med (w-1)#x}",  # rolling median
            "ts_product": "{(*/)x}",
            "ts_var": "{dev[x] xexp 2}",

            # Cumulative operations
            "ts_cumsum": "sums",
            "ts_cumprod": "prds",
            "ts_cummax": "maxs",
            "ts_cummin": "mins",

            # Time series statistical moments
            "ts_skew": "{(avg((x-avg x)xexp 3))%(dev x)xexp 3}",
            "ts_kurt": "{(avg((x-avg x)xexp 4))%(dev x)xexp 4}",

            # Time series position/extrema
            "ts_argmax": "{x?max x}",
            "ts_argmin": "{x?min x}",
            "ts_days_since_high": "{((count x)-1)-x?max x}",
            "ts_days_since_low": "{((count x)-1)-x?min x}",

            # Time series rank/zscore
            "ts_rank": "{(iasc iasc x)%count x}",
            "ts_zscore": "{(x-avg x)%dev x}",
            "ts_demean": "{x-avg x}",
            "ts_normalize": "{x%sum abs x}",

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
            "ts_beta": "{cov[x;y]%dev[y] xexp 2}",

            # Simple indicators
            "ema": "ema",
            "sma": "mavg",
            "wma": "{wavg[til count x;x]}",
        }

    def can_compile_operator(self, op_name: str) -> bool:
        """算子是否可以编译为 q。

        参数:
            op_name: 算子名称

        返回:
            是否支持
        """
        return self.capability.supports_native(op_name)

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

        q_func = self._operator_map.get(op_name)
        if q_func is None:
            raise ValueError(f"No q mapping for operator {op_name}")

        params = params or {}

        # 简单二元算子
        if q_func in {"+", "-", "*", "%", ">", "<", ">=", "<=", "=", "<>"}:
            if len(inputs) != 2:
                raise ValueError(f"{op_name} requires 2 inputs, got {len(inputs)}")
            return f"{inputs[0]} {q_func} {inputs[1]}"

        # 一元算子 (简单函数)
        if q_func in {"neg", "abs", "sqrt", "log", "exp", "signum", "floor", "ceiling",
                      "sums", "prds", "maxs", "mins", "fills", "first", "last"}:
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{q_func} {inputs[0]}"

        # 一元算子 (lambda 表达式)
        if q_func.startswith("{") and q_func.endswith("}"):
            if len(inputs) == 1:
                return f"({q_func})[{inputs[0]}]"
            elif len(inputs) == 2:
                return f"({q_func})[{inputs[0]};{inputs[1]}]"
            elif len(inputs) == 3:
                return f"({q_func})[{inputs[0]};{inputs[1]};{inputs[2]}]"
            else:
                raise ValueError(f"{op_name} with lambda requires 1-3 inputs, got {len(inputs)}")

        # 滚动窗口算子
        if op_name.startswith("ts_") and q_func in {"mavg", "msum", "mdev", "mmin", "mmax", "mcount"}:
            window = params.get("window", 20)
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{window} {q_func} {inputs[0]}"

        # Lag
        if op_name == "lag":
            periods = params.get("periods", 1)
            if len(inputs) != 1:
                raise ValueError(f"lag requires 1 input, got {len(inputs)}")
            # q prev with repeat
            if periods == 1:
                return f"prev {inputs[0]}"
            else:
                return f"{periods} prev\\{inputs[0]}"

        # 聚合算子
        if q_func in {"avg", "sum", "dev", "min", "max", "med"}:
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"{q_func} {inputs[0]}"

        # Rank
        if op_name in {"rank", "cs_rank"}:
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            return f"({q_func})[{inputs[0]}]"

        # Correlation/Covariance/Beta
        if op_name in {"ts_corr", "ts_cov"}:
            window = params.get("window", 20)
            if len(inputs) != 2:
                raise ValueError(f"{op_name} requires 2 inputs, got {len(inputs)}")
            return f"{window} {q_func}[{inputs[0]};{inputs[1]}]"

        if op_name == "ts_beta":
            window = params.get("window", 20)
            if len(inputs) != 2:
                raise ValueError(f"ts_beta requires 2 inputs, got {len(inputs)}")
            return f"({q_func})[{window}#{inputs[0]};{window}#{inputs[1]}]"

        # EMA (exponential moving average)
        if op_name == "ema":
            span = params.get("span", 20)
            alpha = 2.0 / (span + 1)
            if len(inputs) != 1:
                raise ValueError(f"ema requires 1 input, got {len(inputs)}")
            return f"ema[{alpha};{inputs[0]}]"

        # WMA (weighted moving average)
        if op_name == "wma":
            window = params.get("window", 20)
            if len(inputs) != 1:
                raise ValueError(f"wma requires 1 input, got {len(inputs)}")
            return f"{{wavg[til {window};-{window}#{inputs[0]}]}}each {window}_mavg {inputs[0]}"

        # Clip
        if op_name in {"clip", "cs_clip"}:
            lower = params.get("lower", -999999)
            upper = params.get("upper", 999999)
            if len(inputs) != 1:
                raise ValueError(f"clip requires 1 input, got {len(inputs)}")
            return f"({inputs[0]}|{lower})&{upper}"

        # Where (conditional)
        if op_name == "where":
            if len(inputs) != 3:
                raise ValueError(f"where requires 3 inputs (cond, true_val, false_val), got {len(inputs)}")
            return f"?[{inputs[0]};{inputs[1]};{inputs[2]}]"

        # Fill operations
        if op_name == "fillna":
            fill_value = params.get("fill_value", 0)
            if len(inputs) != 1:
                raise ValueError(f"fillna requires 1 input, got {len(inputs)}")
            return f"{inputs[0]}^{fill_value}"

        # Quantile
        if op_name in {"ts_quantile", "cs_quantile"}:
            q_val = params.get("q", 0.5)
            if len(inputs) != 1:
                raise ValueError(f"{op_name} requires 1 input, got {len(inputs)}")
            window = params.get("window", 20) if op_name.startswith("ts_") else None
            if window:
                return f"{{({q_val}) mquantile[-{window}#x]}} each {inputs[0]}"
            else:
                return f"{q_val} quantile {inputs[0]}"

        # Percentile rank
        if op_name == "cs_percentile_rank":
            if len(inputs) != 1:
                raise ValueError(f"cs_percentile_rank requires 1 input, got {len(inputs)}")
            return f"({{(iasc iasc x)%count x}})[{inputs[0]}]"

        raise ValueError(f"Compilation not implemented for {op_name}")

    def compile_region(
        self,
        region_id: str,
        nodes: list[dict[str, Any]],
        *,
        input_tables: list[str],
        output_name: str,
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

        # 生成 q 代码
        q_statements = []
        q_statements.append(f"/ Region {region_id}")
        q_statements.append("")

        # 逐节点编译
        for node in nodes:
            node_id = node["id"]
            op_name = node["operator"]
            inputs = node.get("inputs", [])
            params = node.get("params", {})

            try:
                q_expr = self.compile_operator(op_name, inputs, params)
                q_statements.append(f"{node_id}: {q_expr};")
            except Exception as e:
                logger.error(f"Failed to compile node {node_id}: {e}")
                raise

        # 输出赋值
        final_node = nodes[-1]["id"]
        q_statements.append(f"{output_name}: {final_node}")

        q_code = "\n".join(q_statements)

        # 检查是否需要排序
        requires_sort = any(
            self.capability.requires_global_sort(n["operator"])
            for n in nodes
        )

        # 估算行数
        estimated_rows = nodes[0].get("estimated_rows", 0)

        return QRegionPlan(
            region_id=region_id,
            node_ids=tuple(n["id"] for n in nodes),
            q_code=q_code,
            input_tables=tuple(input_tables),
            output_table=output_name,
            requires_sort=requires_sort,
            estimated_rows=estimated_rows,
        )

    def validate_region(
        self,
        nodes: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """验证 Region 中所有节点是否可编译。

        参数:
            nodes: 节点列表

        返回:
            (is_valid, unsupported_ops)
        """
        unsupported = []
        for node in nodes:
            op_name = node["operator"]
            if not self.can_compile_operator(op_name):
                unsupported.append(op_name)

        return len(unsupported) == 0, unsupported


# Global singleton
_COMPILER: QCompiler | None = None


def get_q_compiler() -> QCompiler:
    """获取全局 q 编译器。"""
    global _COMPILER
    if _COMPILER is None:
        _COMPILER = QCompiler()
    return _COMPILER
