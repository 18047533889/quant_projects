# -*- coding: utf-8 -*-
"""M-240 支援：模型算子 silent-coercion（静默夹紧/截断）AST 扫描。

audit doc §30 batch 2+3 列出的模型算子模块中，``max(1, int(x))`` /
``min(k, int(x))`` / 裸 ``int(param)`` 会静默地把用户参数夹紧或截断
（``lag=True``、``lag=0``、``lag=-3`` → 1；``window=20.9`` → 20），制造
"两个 AST 相同输出"的搜索空间污染。本脚本对每个目标模块做 **纯 AST** 扫描
（**只读**，不 import 任何算子模块，不执行任何算子代码），把每个
silent-coercion 站点以 ``{file, line, pattern}`` 上报，供 M-240 gate 使用。

识别四类 pattern（``int`` / ``max`` / ``min`` 均指同名内置函数）：

1. ``max(...int(...))`` —— ``max`` 的参数里出现 ``int(...)``（clamp + 截断）；
2. ``min(...int(...))`` —— ``min`` 的参数里出现 ``int(...)``；
3. ``int(param)`` —— 裸 ``int()`` 施加在**参数**（当前函数或任意外层函数/
   lambda 的参数）或参数元素（``param[i]`` / ``param.attr``）上，例如
   ``x[-int(window):]``、``int(order)``、``int(vals[i])``；
4. ``int(...max/min(...))`` —— ``int()`` 包裹 ``max(...)`` / ``min(...)``
   （同类的静默截断，如 ``int(min(n_components, n_active, ...))``）。

只报告**函数/方法/lambda 体内部**（``_calculate_series`` 与 kernel 函数体）
的站点；模块级 ``int(...)`` 不报。作用于**计算值**的 ``int()``
（``int(len(x))``、``int(np.argmax(v))``、局部变量重复 ``int``）不报 ——
它们是 no-op 或显式取整，不属于用户参数被静默截断。

输出：
    docs/evidence/r35/R35_MODEL_SILENT_CLAMP_AUDIT.json
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

#: audit doc §30 batch 2+3 —— 模型算子模块（相对仓库根）。
MODULES: list[str] = [
    "cleaned_operators/cross_section/panel_model.py",
    "cleaned_operators/cross_section/pca_state.py",
    "cleaned_operators/ts_model/ar_meanrev.py",
    "cleaned_operators/regression_models.py",
    "cleaned_operators/ts_model/dynamic_regression.py",
    "cleaned_operators/ts_model/_rolling_core.py",
    "cleaned_operators/ts_model/state_space.py",
    "cleaned_operators/ts_model/volatility.py",
    "cleaned_operators/dmd.py",
    "cleaned_operators/hankel.py",
    "cleaned_operators/ts_model/sequence_anomaly.py",
    "cleaned_operators/candle_state_space.py",
    "cleaned_operators/dynamic_knn.py",
    "cleaned_operators/cross_section_local.py",
    "cleaned_operators/markov_dynamics.py",
    "cleaned_operators/state_geometry.py",
    "cleaned_operators/first_passage.py",
    "cleaned_operators/local_lyapunov.py",
    "cleaned_operators/recurrence_analysis.py",
    "cleaned_operators/advanced_information.py",
    "cleaned_operators/glr_change.py",
    "cleaned_operators/nonlinear_dependence.py",
    "cleaned_operators/ts_model/path_signature.py",
    "cleaned_operators/research_transform.py",
]

OUT = Path("docs/evidence/r35/R35_MODEL_SILENT_CLAMP_AUDIT.json")


def _call_func_name(node: ast.AST) -> str | None:
    """``int`` / ``max`` / ``min`` 的内置函数名；非 Call / 其它函数返回 None。"""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    return node.func.id


def _is_int_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_func_name(node) == "int"


def _root_name(node: ast.AST) -> str | None:
    """取 Name/Attribute/Subscript 链的根名字；其它节点返回 None。

    ``window`` -> 'window'；``vals[i]`` -> 'vals'；``x.shape[0]`` -> 'x'；
    ``np.ceil(k)``（Call）-> None。
    """
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _func_params(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    args = node.args
    params = {a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}
    if args.vararg is not None:
        params.add(args.vararg.arg)
    if args.kwarg is not None:
        params.add(args.kwarg.arg)
    return params


class _SilentCoercionFinder(ast.NodeVisitor):
    """在函数体内部定位 silent-coercion 站点。

    参数感知：裸名字 ``int(x)`` 只有当 ``x`` 是**某个外层函数/lambda 的参数**
    时才上报（局部计算变量如 ``int(cells)`` 是 no-op，不报）。
    """

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.findings: list[dict] = []
        self._fstack: list[str] = []
        self._pstack: list[set[str]] = []

    # -- 函数上下文跟踪 ------------------------------------------------
    def _ctx(self) -> str:
        return ".".join(self._fstack) or "<module>"

    def _params(self) -> set[str]:
        union: set[str] = set()
        for ps in self._pstack:
            union |= ps
        return union

    def _enter(self, name: str, params: set[str]) -> None:
        self._fstack.append(name)
        self._pstack.append(params)

    def _exit(self) -> None:
        self._fstack.pop()
        self._pstack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node.name, _func_params(node))
        self.generic_visit(node)
        self._exit()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node.name, _func_params(node))
        self.generic_visit(node)
        self._exit()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._enter("<lambda>", _func_params(node))
        self.generic_visit(node)
        self._exit()

    # -- 站点检测 ------------------------------------------------------
    def _record(self, line: int, pattern: str, source: str) -> None:
        # 仅报告函数/kernel 体内的站点（_fstack 非空）。
        if not self._fstack:
            return
        self.findings.append({
            "file": self.relpath,
            "line": line,
            "function": self._ctx(),
            "pattern": pattern,
            "source": source,
        })

    def _int_arg_coerces_param(self, int_call: ast.Call) -> bool:
        """``int(...)`` 的实参是否属于"用户参数被截断"（参数/参数元素/包裹 max-min）。"""
        if not int_call.args:
            return False
        arg = int_call.args[0]
        if _call_func_name(arg) in ("max", "min"):
            return True  # int(...max/min(...))
        root = _root_name(arg)
        if root is not None and root in self._params():
            return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        fname = _call_func_name(node)
        if fname in ("max", "min"):
            # max(...int(...)) / min(...int(...))：clamp 包裹 int 截断。
            for a in node.args:
                if _is_int_call(a) and self._int_arg_coerces_param(a):
                    self._record(
                        node.lineno,
                        f"{fname}(...int(...))",
                        ast.unparse(node)[:160],
                    )
                    break
        elif fname == "int" and self._int_arg_coerces_param(node):
            arg = node.args[0]
            if _call_func_name(arg) in ("max", "min"):
                self._record(node.lineno, "int(...max/min(...))", ast.unparse(node)[:160])
            else:
                self._record(node.lineno, "int(param)", ast.unparse(node)[:160])
        self.generic_visit(node)


def scan_file(path: Path) -> tuple[list[dict], str | None]:
    """对一个模块做 AST 扫描。返回 (findings, error)。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"cannot read: {exc}"
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [], f"syntax error: {exc}"
    finder = _SilentCoercionFinder(str(path).replace("\\", "/"))
    finder.visit(tree)
    return finder.findings, None


def main() -> int:
    root = Path(".").resolve()
    scanned: list[str] = []
    errors: dict[str, str] = {}
    findings: list[dict] = []

    for rel in MODULES:
        path = root / rel
        scanned.append(rel)
        if not path.is_file():
            errors[rel] = "missing file"
            continue
        hits, err = scan_file(path)
        if err is not None:
            errors[rel] = err
            continue
        findings.extend(hits)

    findings.sort(key=lambda f: (f["file"], f["line"]))
    by_pattern: dict[str, int] = {}
    for f in findings:
        by_pattern[f["pattern"]] = by_pattern.get(f["pattern"], 0) + 1
    per_file: dict[str, int] = {}
    for f in findings:
        per_file[f["file"]] = per_file.get(f["file"], 0) + 1

    payload = {
        "generated_by": "scripts/audit_model_silent_clamp.py",
        "purpose": "M-240 gate support: locate silent max/min clamp + bare int() "
                   "coercion of user parameters in model-operator kernel bodies "
                   "(pure AST, read-only).",
        "scanned_modules": scanned,
        "module_errors": errors,
        "pattern_counts": by_pattern,
        "total_findings": len(findings),
        "per_file_counts": dict(sorted(per_file.items())),
        "findings": findings,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- stdout 摘要（CSV 风格，便于 reconciler 直接消费）----
    print("[audit-model-silent-clamp] output:", OUT)
    print("[audit-model-silent-clamp] modules scanned:", len(scanned),
          "| errors:", len(errors), "| findings:", len(findings))
    for pat, n in sorted(by_pattern.items()):
        print(f"[audit-model-silent-clamp]   {pat}: {n}")
    print("[audit-model-silent-clamp] file,line,function,pattern,source")
    for f in findings:
        print(f"{f['file']},{f['line']},{f['function']},{f['pattern']},{f['source']}")
    if errors:
        print("[audit-model-silent-clamp] ERRORS", json.dumps(errors, ensure_ascii=False))
    return 0  # 报告工具：即使有 findings 也返回 0，由 reconciler/gate 消费 JSON。


if __name__ == "__main__":
    raise SystemExit(main())
