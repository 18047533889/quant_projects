"""Generate a complete source-backed interface index without importing backends."""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ROOT.name
OUTPUT = ROOT / "docs" / "API_REFERENCE.md"
B = chr(96)


def cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def lead(node):
    return (ast.get_docstring(node) or "").split("\n\n", 1)[0].replace("\n", " ").strip()


def definitions(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            yield node.name, node
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        not member.name.startswith("_") or member.name in {"__init__", "__call__"}
                    ):
                        yield node.name + "." + member.name, member


def render():
    modules = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        raw = path.read_text()
        tree = ast.parse(raw)
        modules.append((path, raw, tree, list(definitions(tree))))
    count = sum(len(m[3]) for m in modules)
    lines = [f"# {ROOT.name} 完整模块与接口索引", "",
        "先读 [功能与算法手册](FUNCTIONAL_GUIDE.md)，再查本页的具体入口、参数和实现位置。",
        f"扫描实际包目录：**{len(modules)} 个 Python 模块、{count} 个公开函数/类/方法定义**。",
        "收录非下划线开头的顶层定义及类的公开方法，不把所有内部模块都承诺为稳定API；私有辅助算法见功能手册。",
        "参数、类型、默认值直接取自源码语法树，不导入或启动可选后端。类型注解不代表生产可用性。",
        "未写独立说明的入口会明确标记，不凭名称编造功能；算法讲解、约束、完整流程与例子见功能手册。", "",
        "## 模块目录", "", "| 模块 | 定义数 | 模块说明 |", "|---|---:|---|"]
    for path, raw, tree, entries in modules:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"| [{rel}](../{rel}) | {len(entries)} | {cell(lead(tree) or '参见所属功能章节')} |")
    for path, raw, tree, entries in modules:
        rel = path.relative_to(ROOT).as_posix()
        lines += ["", f"## {rel}", "", lead(tree) or "模块角色见所属功能章节。"]
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in item.targets):
                try:
                    exports = ast.literal_eval(item.value)
                    lines += ["", "显式导出（含重导出）：" + "、".join(B + str(v) + B for v in exports) + "。"]
                except (ValueError, TypeError):
                    pass
        for name, node in entries:
            lines += ["", f"### {name}", "", f"[实际实现](../{rel}#L{node.lineno})。", "",
                lead(node) or "此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。"]
            if isinstance(node, ast.ClassDef):
                if node.bases:
                    lines += ["", "基类：" + B + cell(", ".join(ast.unparse(x) for x in node.bases)) + B + "。"]
                fields = []
                for child in node.body:
                    if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and not child.target.id.startswith("_"):
                        fields.append((child.target.id, ast.unparse(child.annotation), ast.unparse(child.value) if child.value else "必填/未声明默认"))
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                fields.append((target.id, "类常量/枚举", ast.unparse(child.value)))
                if fields:
                    lines += ["", "本类声明字段（继承字段见基类；实际限制仍需合同校验）：", "",
                              "| 字段 | 类型 | 默认值/值 |", "|---|---|---|"]
                    for field, annotation, default in fields:
                        lines.append(f"| {B}{field}{B} | {B}{cell(annotation)}{B} | {B}{cell(default)}{B} |")
            else:
                lines += ["", "参数：" + B + "(" + cell(ast.unparse(node.args)) + ")" + B + "。"]
                if node.returns:
                    lines += ["", "返回类型：" + B + cell(ast.unparse(node.returns)) + B + "。"]
    lines += ["", "## 源码一致性", "", "<details>", "<summary>展开模块指纹</summary>", "",
              "| 模块 | SHA-256 |", "|---|---|"]
    for path, raw, _, _ in modules:
        lines.append(f"| {B}{path.relative_to(ROOT).as_posix()}{B} | {B}{hashlib.sha256(raw.encode()).hexdigest()}{B} |")
    lines += ["", "</details>", "", "重新生成：" + B + "python scripts/build_api_reference.py" + B +
              "；检查：" + B + "python scripts/build_api_reference.py --check" + B + "。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            raise SystemExit("API reference is stale; regenerate and review the functional guide.")
        print(f"{ROOT.name}: reference current")
    else:
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(content)
        print(f"{ROOT.name}: {len(content.splitlines())} lines")


if __name__ == "__main__":
    main()
