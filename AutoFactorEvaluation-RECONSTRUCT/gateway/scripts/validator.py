"""
静态校验器模块

负责:
- Schema 校验：检查 candidate.json 必需字段
- 算子白名单校验：检查表达式中的算子是否在白名单中
- 字段兼容性校验：检查算子是否用于兼容的字段类型
- 数据类型校验：检查算子参数类型是否正确
"""

import re
from typing import Tuple, Set, List, Optional, Any, Dict
from enum import Enum

from .config import GatewayConfig
from .exceptions import (
    SchemaValidationError,
    OperatorNotAllowedError,
    DataTypeError
)


class ValidationSeverity(Enum):
    """校验严重程度"""
    ERROR = "error"
    WARNING = "warning"


class ValidationResult:
    """校验结果"""

    def __init__(self, is_valid: bool, error_message: str = "", warnings: List[str] = None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.warnings = warnings or []

    def __repr__(self):
        return f"ValidationResult(is_valid={self.is_valid}, error='{self.error_message[:50]}...')"


class SchemaValidator:
    """
    Schema 校验器

    检查 candidate.json 的必需字段和格式
    """

    # 必需字段列表（符合 Disk Contracts）
    REQUIRED_FIELDS = [
        "schema_version",
        "candidate_id",
        "Expr",
        "Config",
        "BornTimestamp",
        "BasicInfo"
    ]

    # 支持的 schema 版本
    SUPPORTED_SCHEMA_VERSIONS = ["disk.v1"]

    # candidate_id 格式正则（示例：cand_YYYYMMDD_xxxxxxxx）
    CANDIDATE_ID_PATTERN = re.compile(r'^cand_\d{8}_[a-f0-9]{8}$')

    def __init__(self, config: GatewayConfig):
        """
        初始化 Schema 校验器

        Args:
            config: 网关配置对象
        """
        self.config = config

    def validate(self, candidate_dict: Dict[str, Any]) -> ValidationResult:
        """
        执行完整的 Schema 校验

        Args:
            candidate_dict: 候选因子的原始字典

        Returns:
            ValidationResult 对象
        """
        # 1. 检查必需字段
        missing_fields = self._check_required_fields(candidate_dict)
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                error_message=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # 2. 检查 schema_version
        version_result = self._validate_schema_version(candidate_dict["schema_version"])
        if not version_result.is_valid:
            return version_result

        # 3. 检查 candidate_id 格式
        id_result = self._validate_candidate_id(candidate_dict["candidate_id"])
        if not id_result.is_valid:
            return id_result

        # 4. 检查 BornTimestamp 格式
        timestamp_result = self._validate_timestamp(candidate_dict["BornTimestamp"])
        if not timestamp_result.is_valid:
            return timestamp_result

        # 5. 检查 Config 格式
        config_result = self._validate_config(candidate_dict["Config"])
        if not config_result.is_valid:
            return config_result

        return ValidationResult(is_valid=True)

    def _check_required_fields(self, candidate_dict: Dict[str, Any]) -> List[str]:
        """检查必需字段是否存在"""
        missing = []
        for field in self.REQUIRED_FIELDS:
            if field not in candidate_dict:
                missing.append(field)
        return missing

    def _validate_schema_version(self, version: str) -> ValidationResult:
        """检查 schema_version 是否支持"""
        if version not in self.SUPPORTED_SCHEMA_VERSIONS:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported schema_version: '{version}'. "
                              f"Supported: {', '.join(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return ValidationResult(is_valid=True)

    def _validate_candidate_id(self, candidate_id: str) -> ValidationResult:
        """检查 candidate_id 格式"""
        if not candidate_id:
            return ValidationResult(is_valid=False, error_message="candidate_id is empty")

        # 格式检查（可选，如果不匹配只警告不报错）
        if not self.CANDIDATE_ID_PATTERN.match(candidate_id):
            # 不强制要求格式，只记录警告
            return ValidationResult(
                is_valid=True,
                warnings=[f"candidate_id '{candidate_id}' does not match recommended pattern"]
            )

        return ValidationResult(is_valid=True)

    def _validate_timestamp(self, timestamp: str) -> ValidationResult:
        """检查时间戳格式（ISO 8601）"""
        if not timestamp:
            return ValidationResult(is_valid=False, error_message="BornTimestamp is empty")

        # 尝试解析 ISO 8601 格式
        try:
            from datetime import datetime
            # 尝试多种常见格式
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ]:
                try:
                    datetime.strptime(timestamp, fmt)
                    return ValidationResult(is_valid=True)
                except ValueError:
                    continue

            # 如果都不匹配，可能是其他格式，只警告
            return ValidationResult(
                is_valid=True,
                warnings=[f"BornTimestamp '{timestamp}' may not be ISO 8601 format"]
            )
        except Exception:
            return ValidationResult(
                is_valid=True,
                warnings=[f"BornTimestamp '{timestamp}' format could not be validated"]
            )

    def _validate_config(self, config: Any) -> ValidationResult:
        """检查 Config 字段格式"""
        if config is None:
            return ValidationResult(is_valid=False, error_message="Config is None")

        if not isinstance(config, dict):
            return ValidationResult(
                is_valid=False,
                error_message=f"Config must be a dictionary, got {type(config).__name__}"
            )

        return ValidationResult(is_valid=True)


class FieldClassifier:
    """字段类型分类器。

    匹配优先级:
    1. match_patterns: 对字段名做精确/后缀匹配（区分大小写敏感模式与忽略大小写模式）
    2. table_patterns: 对全路径做表名前缀匹配
    """

    def __init__(self, field_type_rules: dict):
        self._match_rules: dict[str, list] = {}   # type_name → [regex patterns for field name]
        self._table_rules: dict[str, list] = {}   # type_name → [regex patterns for table prefix]

        for type_name, rules in field_type_rules.items():
            self._match_rules[type_name] = [
                re.compile(p, re.IGNORECASE) for p in rules.get("match_patterns", [])
            ]
            self._table_rules[type_name] = [
                re.compile(p, re.IGNORECASE) for p in rules.get("table_patterns", [])
            ]

    def classify(self, field_name: str) -> str:
        """识别字段引用属于哪个类型。

        策略:
        - 若 field_name 包含 '.'（如 StockDailyBar.Close），拆分为 (table, field)
        - 优先 match_patterns 匹配纯字段名
        - 若未命中，尝试 table_patterns 匹配表名
        - 纯字段名（无 '.'）只走 match_patterns
        """
        table_part = ""
        field_part = field_name

        if "." in field_name:
            parts = field_name.split(".", 1)
            table_part = parts[0].strip()
            field_part = parts[1].strip()

        # 1. 字段名精确匹配（match_patterns）
        for type_name, patterns in self._match_rules.items():
            for pat in patterns:
                if pat.fullmatch(field_part):
                    return type_name

        # 2. 表名前缀匹配（table_patterns），需有 table 部分
        if table_part:
            qualified = f"{table_part}.{field_part}"
            for type_name, patterns in self._table_rules.items():
                for pat in patterns:
                    if pat.search(qualified):
                        return type_name

        return "other"


class OperatorValidator:
    """
    算子校验器

    检查:
    - 算子是否在白名单中
    - 算子是否应用于兼容的字段类型（如收益率算子不能用于价格字段）
    支持 AST 解析（如果有 factor_engine）或正则匹配（简化版）
    """

    # 常见算子的正则模式（简化版，用于无 AST 解析器的情况）
    OPERATOR_PATTERN = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(')

    # 函数调用提取：匹配 func_name(arg1, arg2, ...)
    FUNC_CALL_PATTERN = re.compile(
        r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)',
    )

    def __init__(self, config: GatewayConfig, use_ast_parser: bool = False):
        """
        初始化算子校验器

        Args:
            config: 网关配置对象
            use_ast_parser: 是否使用 AST 解析器（需要 factor_engine）
        """
        self.config = config
        self.whitelist = config.operator_whitelist
        self.use_ast_parser = use_ast_parser
        self.field_classifier = FieldClassifier(config.field_type_rules)

        # 如果启用 AST 解析器，尝试导入
        self.ast_parser = None
        if use_ast_parser:
            try:
                pass
            except ImportError:
                self.use_ast_parser = False

    def validate(self, expr: str) -> ValidationResult:
        """
        校验表达式中的算子及字段兼容性

        Args:
            expr: 因子表达式字符串

        Returns:
            ValidationResult 对象
        """
        if not expr or not expr.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Expression is empty"
            )

        # 1. 提取表达式中的所有算子
        operators = self._extract_operators(expr)
        warnings = []

        if not operators:
            return ValidationResult(
                is_valid=True,
                warnings=["No operators found in expression"]
            )

        # 2. 检查每个算子是否在白名单中
        invalid_operators = []
        for op in operators:
            if not self.config.is_operator_allowed(op):
                invalid_operators.append(op)

        if invalid_operators:
            return ValidationResult(
                is_valid=False,
                error_message=f"Operators not in whitelist: {', '.join(invalid_operators)}"
            )

        # 3. 检查算子-字段兼容性
        violations = self._check_field_compatibility(expr, operators)
        if violations:
            return ValidationResult(
                is_valid=False,
                error_message="Field-operator compatibility violations:\n" + "\n".join(violations)
            )

        return ValidationResult(is_valid=True, warnings=warnings)

    def _extract_operators(self, expr: str) -> Set[str]:
        """从表达式中提取算子名，统一转为大写。"""
        matches = self.OPERATOR_PATTERN.findall(expr)
        return {m.upper() for m in matches}

    def _check_field_compatibility(self, expr: str, operators: set) -> list[str]:
        """检查算子字段兼容性。

        对每个有限制的算子，检查其第一个参数引用的字段是否匹配允许的类型。
        """
        violations = []
        # 构建算子→限制信息的快速映射
        op_restrictions: dict[str, dict] = {}
        for group in self.config.operator_restrictions.values():
            for op_name in group.get("operators", []):
                op_restrictions[op_name] = group

        for match in self.FUNC_CALL_PATTERN.finditer(expr):
            op_name = match.group(1).upper()
            args_str = match.group(2)

            # 跳过无限制的算子
            restriction = op_restrictions.get(op_name)
            if restriction is None:
                continue

            allowed_types = set(restriction.get("allowed_types", []))
            group_desc = restriction.get("description", "")

            # 零参函数（如 REAL_TURNOVER_RATE）无需检查字段
            if restriction.get("operators") and op_name in restriction.get("operators", []):
                # 检查是否 zero_arg 类别
                if not allowed_types and op_name == "REAL_TURNOVER_RATE":
                    continue

            # 提取第一个参数中的字段引用
            first_arg = args_str.strip().split(",")[0].strip() if args_str else ""
            if not first_arg:
                continue

            # 从参数中提取字段名（去掉可能的包裹函数如 rank(x) 中的 x）
            field_ref = self._extract_field_ref(first_arg)
            if field_ref is None:
                continue

            # 分类字段类型
            field_type = self.field_classifier.classify(field_ref)

            # 检查兼容性
            if field_type not in allowed_types:
                violations.append(
                    f"  {op_name}({field_ref}) 的字段 '{field_ref}' 类型为 '{field_type}'，"
                    f"但该算子{group_desc}，仅允许: {sorted(allowed_types)}"
                )

        return violations

    @staticmethod
    def _extract_field_ref(arg: str) -> str | None:
        """从算子参数中提取字段引用。

        处理多种参数形式:
        - "close"                        → "close"
        - "StockDailyBar.Close"          → "StockDailyBar.Close"
        - "rank(close)"                  → "close"
        - "ts_mean(close, 5)"            → "close"
        - "col('close')"                 → "close"
        """
        arg = arg.strip()
        if not arg:
            return None

        # 去掉 col('...') 包裹
        col_match = re.match(r"col\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", arg)
        if col_match:
            return col_match.group(1)

        # 如果是字段别名（小写字母/点/下划线），直接返回
        if re.match(r'^[a-zA-Z_][\w.]*$', arg):
            return arg

        # 如果是函数调用如 rank(x)，取最内层参数
        inner = re.match(r'[A-Z_]+\((.+)\)', arg, re.IGNORECASE)
        if inner:
            inner_arg = inner.group(1).split(",")[0].strip()
            if re.match(r'^[a-zA-Z_][\w.]*$', inner_arg):
                return inner_arg

        # 数值常量或复杂表达式 → 无法确定字段类型
        return None

    def validate_with_ast(self, expr: str) -> ValidationResult:
        """
        使用 AST 解析器进行更精确的校验（如果可用）

        这是一个高级接口，需要 factor_engine 提供 AST 解析能力
        """
        if not self.use_ast_parser or self.ast_parser is None:
            # 降级到正则匹配
            return self.validate(expr)

        try:
            # 解析 AST
            ast = self.ast_parser(expr)

            # 遍历 AST 收集算子
            operators = self._collect_operators_from_ast(ast)

            # 检查白名单
            invalid = [op for op in operators if not self.config.is_operator_allowed(op)]

            if invalid:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Operators not in whitelist: {', '.join(invalid)}"
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"AST parsing failed: {str(e)}"
            )

    def _collect_operators_from_ast(self, ast_node) -> Set[str]:
        """
        从 AST 节点递归收集算子

        这是一个示例实现，实际需要根据 factor_engine 的 AST 结构调整
        """
        operators = set()

        # 假设 AST 节点有 type 和 children 属性
        # 实际实现需要根据具体的 AST 格式调整
        if hasattr(ast_node, 'type'):
            if ast_node.type == 'function_call':
                operators.add(ast_node.name)

        if hasattr(ast_node, 'children'):
            for child in ast_node.children:
                operators.update(self._collect_operators_from_ast(child))

        return operators


class DataTypeValidator:
    """
    数据类型校验器

    检查算子参数的类型是否正确
    """

    def __init__(self, config: GatewayConfig):
        self.config = config

    def validate(self, expr: str) -> ValidationResult:
        """
        校验数据类型

        这是一个简化实现，完整实现需要解析参数
        """
        warnings = []

        # 检查 MA 窗口期
        ma_pattern = re.compile(r'MA\s*\([^,]+,\s*(\d+)\s*\)', re.IGNORECASE)
        for match in ma_pattern.finditer(expr):
            window = int(match.group(1))
            if window < 1:
                warnings.append(f"MA window must be >= 1, got {window}")
            elif window > 252:
                warnings.append(f"MA window {window} is very large (max recommended: 252)")

        # 检查 STD 窗口期
        std_pattern = re.compile(r'STD\s*\([^,]+,\s*(\d+)\s*\)', re.IGNORECASE)
        for match in std_pattern.finditer(expr):
            window = int(match.group(1))
            if window < 2:
                warnings.append(f"STD window must be >= 2, got {window}")

        return ValidationResult(is_valid=True, warnings=warnings)


class StaticValidator:
    """
    静态校验器 - 整合所有校验

    这是网关调用的主入口
    """

    def __init__(self, config: GatewayConfig, use_ast_parser: bool = False):
        """
        初始化静态校验器

        Args:
            config: 网关配置对象
            use_ast_parser: 是否使用 AST 解析器
        """
        self.config = config
        self.schema_validator = SchemaValidator(config)
        self.operator_validator = OperatorValidator(config, use_ast_parser)
        self.data_type_validator = DataTypeValidator(config)

    def validate_all(self, candidate_dict: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """
        执行所有校验

        Args:
            candidate_dict: 候选因子字典

        Returns:
            (is_valid, error_message, warnings)
        """
        warnings = []

        # 1. Schema 校验
        schema_result = self.schema_validator.validate(candidate_dict)
        if not schema_result.is_valid:
            return False, schema_result.error_message, warnings
        warnings.extend(schema_result.warnings)

        # 2. 算子校验
        expr = candidate_dict.get("Expr", "")
        operator_result = self.operator_validator.validate(expr)
        if not operator_result.is_valid:
            return False, operator_result.error_message, warnings
        warnings.extend(operator_result.warnings)

        # 3. 数据类型校验（只警告，不阻止）
        data_type_result = self.data_type_validator.validate(expr)
        warnings.extend(data_type_result.warnings)

        return True, "", warnings

    def validate_schema_only(self, candidate_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        仅执行 Schema 校验

        Args:
            candidate_dict: 候选因子字典

        Returns:
            (is_valid, error_message)
        """
        result = self.schema_validator.validate(candidate_dict)
        return result.is_valid, result.error_message

    def validate_operators_only(self, expr: str) -> Tuple[bool, str]:
        """
        仅执行算子校验

        Args:
            expr: 因子表达式

        Returns:
            (is_valid, error_message)
        """
        result = self.operator_validator.validate(expr)
        return result.is_valid, result.error_message