"""Generate the exhaustive, source-backed metric calculation reference."""
from __future__ import annotations
import argparse
import ast
import functools
import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quant_evaluator.registry.metrics import list_metrics, get_metric, CANONICAL_METRIC_ALIASES

OUTPUT = ROOT / "quant_evaluator/docs/METRIC_REFERENCE.md"


def portable_math(markdown: str) -> str:
    """Use a conservative TeX subset, including in inline expressions.

    Raw asterisks may be consumed by Markdown before math rendering.
    Some GitHub math paths reject operatorname even when KaTeX accepts it.
    """
    def replace(match):
        delimiter = "$$" if match.group(1) is not None else "$"
        body = match.group(1) if match.group(1) is not None else match.group(2)
        if delimiter == "$" and body.startswith("`") and body.endswith("`"):
            body = body[1:-1]
        body = body.replace(r"\operatorname", r"\mathrm")
        body = body.replace("^*", r"^{\ast}").replace("*", r"\ast ")
        # Explicit inline delimiters avoid Markdown/CJK boundary ambiguity.
        return "$`" + body + "`$" if delimiter == "$" else "\n```math\n" + body.strip() + "\n```\n"
    return re.sub(r"\$\$([\s\S]*?)\$\$|(?<!\$)\$([^$\n]+)\$(?!\$)", replace, markdown)


def function(fn):
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn


def key(fn):
    fn = function(fn)
    return fn.__module__ + "." + fn.__qualname__


def source(fn):
    return textwrap.dedent(inspect.getsource(function(fn))).rstrip()


def dependencies(fn):
    fn = function(fn)
    if not inspect.isfunction(fn):
        return []
    found = []
    for name in fn.__code__.co_names:
        obj = fn.__globals__.get(name)
        if inspect.isfunction(obj) and obj.__module__.startswith("quant_evaluator.metrics"):
            found.append(obj)
    # Adapters often use local imports; include their formula implementations.
    for node in ast.walk(ast.parse(source(fn))):
        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module
            if node.level:
                module_name = importlib.util.resolve_name("." * node.level + node.module, fn.__module__.rsplit(".", 1)[0])
            if not module_name.startswith("quant_evaluator.metrics"):
                continue
            module = importlib.import_module(module_name)
            for alias in node.names:
                obj = getattr(module, alias.name, None)
                if inspect.isfunction(obj):
                    found.append(obj)
    return found


def render():
    ids = sorted(list_metrics())
    formulas = {}
    for path in sorted(OUTPUT.parent.glob("formulas_*.json")):
        entries = json.loads(path.read_text())
        duplicates = set(formulas) & set(entries)
        if duplicates:
            raise ValueError(f"Duplicate formula definitions: {sorted(duplicates)}")
        formulas.update(entries)
    if set(formulas) != set(ids):
        raise ValueError(f"Formula coverage mismatch: missing={sorted(set(ids)-set(formulas))}, extra={sorted(set(formulas)-set(ids))}")
    for metric_id, definition in formulas.items():
        if "$$" not in definition or "```" in definition:
            raise ValueError(f"Expected readable mathematical formula, not code: {metric_id}")
    lines = ["# QuantEvaluator 全部注册指标计算手册", "",
        "与 [统一口径与取舍](METRIC_CONVENTIONS.md) 配套。正文使用数学公式与中文解释，源码只作为核对链接。",
        "", f"注册 ID 共 **{len(ids)}** 个，别名不重复计数。下面逐项列出全部 ID，包括实验性或不能单独执行的项目。",
        "默认参数是函数层默认；公开入口额外构建策略见统一口径，尤其 IC 的每日20配对、分桶人数及观察期。",
        "公式按当前实际实现编写，非仅按指标名称套用教科书定义。输入合同与公开入口可能比低层函数施加更严格的限制。",
        "None/NaN/unsupported不代表0；状态stable也不代表生产可交易或GPU已验收。", "",
        "## 公共符号与阅读规则", "",
        "除逐项另有定义：$t$ 为时间，$i$ 为资产，$f$ 为因子，$x$ 为因子值，$y$ 为预测标签，$r$ 为单期收益，$w$ 为权重；$T,N,Q$ 分别为有效期数、资产数、桶数。", "",
        "$$\\bar z=\\frac{1}{n}\\sum_{j=1}^{n}z_j,\\qquad s(z)=\\sqrt{\\frac{\\sum_{j=1}^{n}(z_j-\\bar z)^2}{n-1}}$$", "",
        "$\\mathbf 1(\\cdot)$ 是条件成立取1、否则取0的指示函数；$\\operatorname{rank}$ 默认使用平均并列秩；$\\operatorname{Corr}$ 是相关系数。有限值集合及有效掩码按各项定义筛选；没有足够样本时为不可用，不自动补0。某些分布指标使用总体矩或其他分母，以该项公式为准。", "",
        "GitHub 渲染数学公式；若使用本地 Markdown 阅读器，请开启 LaTeX/MathJax 数学显示。", "",
        "## 完整目录", "", "| ID | 名称 | 状态 | 输出 |", "|---|---|---|---|"]
    implementations = {}
    for metric_id in ids:
        spec = get_metric(metric_id)
        lines.append(f"| [{metric_id}](#metric-{metric_id}) | {spec.display_name} | {spec.status.value} | {spec.artifact_kind} |")
    for metric_id in ids:
        spec = get_metric(metric_id)
        aliases = sorted(a for a, target in CANONICAL_METRIC_ALIASES.items() if target == metric_id)
        lines += ["", f'<a id="metric-{metric_id}"></a>', f"## {metric_id} — {spec.display_name}", "",
            spec.description, "",
            f"- 版本：`{spec.metric_version}`；状态：`{spec.status.value}`；层级：`{spec.tier.value}`。",
            f"- 输入依赖：`{', '.join(spec.requires or []) or '见函数签名'}`。",
            f"- 输出：`{spec.artifact_kind}`；单位：`{spec.units or '注册表未标注，见函数公式'}`；方向：`{spec.direction}`。",
            f"- 缺失政策：`{spec.missing_policy}`；数值政策：`{spec.numeric_policy}`；注册最低期数：`{spec.min_periods}`。",
            f"- 别名：{', '.join(' `'+a+'` ' for a in aliases) or '无'}。",
            f"- 增量模式：`{spec.update_mode}`；注册实现定位：`{spec.implementation_id or '见下方实际函数'}`。"]
        fn = spec.compute_fn
        # Keep display math on separate lines for GitHub/MathJax renderers.
        definition = re.sub(r"\$\$(.*?)\$\$", lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n",
                            formulas[metric_id], flags=re.DOTALL)
        definition = re.sub(r"(?m)^[ \t]*,[ \t]*$", "", definition)
        lines += ["", "### 数学公式与计算口径", "", definition.strip(), ""]
        if fn is None:
            lines += ["", "没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。"]
            continue
        base = function(fn)
        identity = key(base)
        implementations[identity] = base
        defaults = [(name, p.default) for name, p in inspect.signature(fn).parameters.items()
                    if p.default is not inspect.Parameter.empty]
        if defaults:
            lines += ["### 函数层默认参数", "", "| 参数 | 默认值 |", "|---|---|"]
            for name, value in defaults:
                lines.append(f"| `{name}` | `{value!r}` |")
        if isinstance(fn, functools.partial):
            lines += ["", f"绑定参数：`args={fn.args!r}, kwargs={fn.keywords!r}`。"]
        path = Path(inspect.getsourcefile(base)).relative_to(OUTPUT.parent.parent)
        line = inspect.getsourcelines(base)[1]
        lines += ["", f"实现核对：[函数定义](../{path.as_posix()}#L{line})；`{identity}`。"]
    queue = list(implementations.values())
    seen = set(implementations)
    helpers = {}
    while queue:
        fn = queue.pop()
        for dep in dependencies(fn):
            identity = key(dep)
            if identity in seen:
                continue
            seen.add(identity)
            helpers[identity] = dep
            queue.append(dep)
    lines += ["", "## 实现核对索引（可选）", "",
        "正文不要求阅读代码。下列链接仅用于核对共享计算函数及掩码细节。", "",
        "<details>", "<summary>展开共享实现链接</summary>", ""]
    for identity, fn in sorted(helpers.items()):
        path = Path(inspect.getsourcefile(fn)).relative_to(OUTPUT.parent.parent)
        line = inspect.getsourcelines(fn)[1]
        lines.append(f"- [{identity}](../{path.as_posix()}#L{line})")
    lines += ["", "</details>"]
    lines += ["", "## 定义完整性指纹", "",
              "生成器检查全部注册指标都有数学口径；实现指纹变化时仍须人工复核公式，指纹本身不证明数学说明正确。", "",
              "<details>", "<summary>展开实现指纹</summary>", "",
              "| 函数 | SHA-256（源公式） |", "|---|---|"]
    for identity, fn in sorted({**implementations, **helpers}.items()):
        lines.append(f"| `{identity}` | `{hashlib.sha256(source(fn).encode()).hexdigest()}` |")
    lines += ["", "</details>"]
    return portable_math("\n".join(line.rstrip() for line in "\n".join(lines).splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            raise SystemExit("Metric reference is stale; regenerate and review it.")
        print("Metric reference matches every registered metric and formula.")
    else:
        OUTPUT.write_text(content)
        print(f"Wrote {OUTPUT}: {len(list_metrics())} metrics, {len(content.splitlines())} lines.")


if __name__ == "__main__":
    main()
