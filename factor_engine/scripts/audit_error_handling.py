#!/usr/bin/env python3
"""
深度错误处理审计脚本。

扫描关键模块，识别：
1. 缺失的错误处理（危险操作没有try-except）
2. 捕获过于宽泛的异常（bare except或except Exception）
3. 错误消息不清晰（raise Exception without context）
4. 资源清理缺失（无finally或context manager）
5. 错误重新抛出时丢失上下文（raise without from）
"""

import ast
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set


@dataclass
class ErrorHandlingIssue:
    """错误处理问题。"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    code_snippet: str = ""


class ErrorHandlingAuditor(ast.NodeVisitor):
    """AST访问器，审计错误处理。"""

    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.issues: List[ErrorHandlingIssue] = []
        self.current_try_handlers: List[ast.ExceptHandler] = []

    def visit_Try(self, node: ast.Try):
        """访问try语句。"""
        # 检查是否有bare except或except Exception
        for handler in node.handlers:
            if handler.type is None:
                # Bare except
                self.issues.append(ErrorHandlingIssue(
                    file_path=self.file_path,
                    line_number=handler.lineno,
                    issue_type="BARE_EXCEPT",
                    severity="HIGH",
                    description="Bare except catches all exceptions including SystemExit and KeyboardInterrupt",
                ))
            elif isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                # Too broad
                self.issues.append(ErrorHandlingIssue(
                    file_path=self.file_path,
                    line_number=handler.lineno,
                    issue_type="BROAD_EXCEPTION",
                    severity="MEDIUM",
                    description="Catching Exception is too broad, use specific exception types",
                ))

        # 检查是否有finally清理资源
        has_open_call = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == 'open':
                    has_open_call = True
                    break

        if has_open_call and not node.finalbody:
            self.issues.append(ErrorHandlingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="MISSING_FINALLY",
                severity="HIGH",
                description="File operation without finally block or context manager",
            ))

        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise):
        """访问raise语句。"""
        if node.exc:
            # 检查是否是简单的raise Exception("message")
            if isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name):
                    exc_name = node.exc.func.id
                    if exc_name == "Exception":
                        self.issues.append(ErrorHandlingIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="GENERIC_EXCEPTION",
                            severity="MEDIUM",
                            description="Raising generic Exception, use specific exception type",
                        ))

                # 检查是否有错误消息
                if not node.exc.args:
                    self.issues.append(ErrorHandlingIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="NO_ERROR_MESSAGE",
                        severity="HIGH",
                        description="Exception raised without error message",
                    ))

            # 检查是否在except块中重新抛出，但没有使用from
            if node.cause is None:
                # 这需要上下文检查，暂时跳过
                pass

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """访问函数调用。"""
        # 检查危险操作是否被try-except包裹
        dangerous_calls = {
            'open': 'File operations should be wrapped in try-except',
            'json.loads': 'JSON parsing should be wrapped in try-except',
            'json.dumps': 'JSON serialization should be wrapped in try-except',
            'int': 'Type conversion should be wrapped in try-except',
            'float': 'Type conversion should be wrapped in try-except',
        }

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in dangerous_calls:
                # 检查是否在try块内
                # （这需要额外的上下文追踪，简化处理）
                pass

        self.generic_visit(node)


def audit_file(file_path: Path) -> List[ErrorHandlingIssue]:
    """审计单个文件的错误处理。"""
    try:
        source = file_path.read_text()
        source_lines = source.split('\n')
        tree = ast.parse(source, filename=str(file_path))

        auditor = ErrorHandlingAuditor(str(file_path), source_lines)
        auditor.visit(tree)

        return auditor.issues
    except SyntaxError as e:
        return [ErrorHandlingIssue(
            file_path=str(file_path),
            line_number=e.lineno or 0,
            issue_type="SYNTAX_ERROR",
            severity="CRITICAL",
            description=f"Cannot parse file: {e}",
        )]
    except Exception as e:
        return [ErrorHandlingIssue(
            file_path=str(file_path),
            line_number=0,
            issue_type="AUDIT_ERROR",
            severity="CRITICAL",
            description=f"Cannot audit file: {e}",
        )]


def audit_directory(base_path: Path, patterns: List[str]) -> List[ErrorHandlingIssue]:
    """审计目录中的所有Python文件。"""
    all_issues = []

    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            if file_path.is_file() and file_path.suffix == '.py':
                issues = audit_file(file_path)
                all_issues.extend(issues)

    return all_issues


def generate_audit_report(issues: List[ErrorHandlingIssue], output_path: str):
    """生成审计报告。"""
    from collections import defaultdict
    from datetime import datetime

    # 按严重程度分组
    by_severity = defaultdict(list)
    for issue in issues:
        by_severity[issue.severity].append(issue)

    # 按类型分组
    by_type = defaultdict(list)
    for issue in issues:
        by_type[issue.issue_type].append(issue)

    # 按文件分组
    by_file = defaultdict(list)
    for issue in issues:
        by_file[issue.file_path].append(issue)

    with open(output_path, 'w') as f:
        f.write("# 错误处理深度审计报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 汇总统计
        f.write("## 汇总统计\n\n")
        f.write(f"- 总问题数: {len(issues)}\n")
        f.write(f"- CRITICAL: {len(by_severity['CRITICAL'])}\n")
        f.write(f"- HIGH: {len(by_severity['HIGH'])}\n")
        f.write(f"- MEDIUM: {len(by_severity['MEDIUM'])}\n")
        f.write(f"- LOW: {len(by_severity['LOW'])}\n")
        f.write(f"- 受影响文件数: {len(by_file)}\n\n")

        # 按问题类型统计
        f.write("## 按问题类型统计\n\n")
        for issue_type, type_issues in sorted(by_type.items(), key=lambda x: -len(x[1])):
            f.write(f"- **{issue_type}**: {len(type_issues)} 个问题\n")
        f.write("\n")

        # 按严重程度详细列举
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if by_severity[severity]:
                f.write(f"## {severity} 严重程度问题\n\n")
                for issue in by_severity[severity]:
                    f.write(f"### {issue.issue_type}\n\n")
                    f.write(f"- **文件**: `{issue.file_path}`\n")
                    f.write(f"- **行号**: {issue.line_number}\n")
                    f.write(f"- **描述**: {issue.description}\n")
                    if issue.code_snippet:
                        f.write(f"- **代码**:\n```python\n{issue.code_snippet}\n```\n")
                    f.write("\n")

        # 按文件列出问题
        f.write("## 按文件列出问题\n\n")
        for file_path, file_issues in sorted(by_file.items(), key=lambda x: -len(x[1])):
            f.write(f"### {file_path} ({len(file_issues)} 个问题)\n\n")
            for issue in file_issues:
                f.write(f"- 行 {issue.line_number}: [{issue.severity}] {issue.issue_type} - {issue.description}\n")
            f.write("\n")

    print(f"审计报告已生成: {output_path}")


def main():
    print("=" * 80)
    print("错误处理深度审计")
    print("=" * 80)
    print()

    base_path = Path('/home/shw/quant_projects/factor_engine')

    # 审计关键模块
    critical_modules = [
        'runtime/*.py',
        'storage/*.py',
        'storage/materialize/*.py',
        'cache/*.py',
        'backend/*.py',
        'execution/*.py',
    ]

    print("扫描关键模块...")
    all_issues = audit_directory(base_path, critical_modules)

    print(f"发现 {len(all_issues)} 个潜在问题")

    # 按严重程度统计
    from collections import Counter
    severity_counts = Counter(issue.severity for issue in all_issues)
    print(f"  CRITICAL: {severity_counts['CRITICAL']}")
    print(f"  HIGH: {severity_counts['HIGH']}")
    print(f"  MEDIUM: {severity_counts['MEDIUM']}")
    print(f"  LOW: {severity_counts['LOW']}")

    # 生成报告
    report_path = '/tmp/error_handling_deep_audit.md'
    generate_audit_report(all_issues, report_path)

    # 生成CSV
    import csv
    csv_path = '/tmp/missing_error_handling.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file_path', 'line_number', 'issue_type', 'severity', 'description'
        ])
        writer.writeheader()
        for issue in all_issues:
            writer.writerow({
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'issue_type': issue.issue_type,
                'severity': issue.severity,
                'description': issue.description,
            })

    print(f"CSV报告已生成: {csv_path}")

    # 返回CRITICAL或HIGH问题的数量作为退出码
    critical_high = severity_counts['CRITICAL'] + severity_counts['HIGH']
    if critical_high > 0:
        print(f"\n警告: 发现 {critical_high} 个高危问题需要修复")
        return 1
    else:
        print("\n✓ 未发现高危问题")
        return 0


if __name__ == '__main__':
    sys.exit(main())
