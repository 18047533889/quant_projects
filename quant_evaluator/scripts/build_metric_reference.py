"""Generate the exhaustive, source-backed metric calculation reference."""
from __future__ import annotations
import argparse
import ast
import functools
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quant_evaluator.registry.metrics import list_metrics, get_metric, CANONICAL_METRIC_ALIASES

OUTPUT = ROOT / "quant_evaluator/docs/METRIC_REFERENCE.md"


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
    lines = ["# QuantEvaluator 全部注册指标计算手册", "",
        "与 [统一口径与取舍](METRIC_CONVENTIONS.md) 配套。由实际注册表、函数说明与源公式生成；不得手工只改数字。",
        "", f"注册 ID 共 **{len(ids)}** 个，别名不重复计数。下面逐项列出全部 ID，包括实验性或不能单独执行的项目。",
        "默认参数是函数层默认；公开入口额外构建策略见统一口径，尤其 IC 的每日20配对、分桶人数及观察期。",
        "实现源码是精确定义的一部分：保留掩码、分母、边界分支，避免将自定义指标写成名称相近的标准公式。",
        "None/NaN/unsupported不代表0；状态stable也不代表生产可交易或GPU已验收。", "",
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
        if fn is None:
            lines += ["", "没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。"]
            continue
        base = function(fn)
        identity = key(base)
        implementations[identity] = base
        lines += ["", "### 计算定义与默认参数", "", "```python", f"{identity}{inspect.signature(fn)}", "```", "",
                  inspect.getdoc(fn) or inspect.getdoc(base) or "此函数没有独立说明；精确定义见下方源公式。"]
        if isinstance(fn, functools.partial):
            lines += ["", f"绑定参数：`args={fn.args!r}, kwargs={fn.keywords!r}`。"]
        lines += ["", "### 精确计算公式（实际实现）", "", "```python", source(base), "```"]
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
    lines += ["", "## 共享公式与掩码依赖", "",
        "以下为上文适配器引用的共享计算函数，避免只展示一层转发却遗漏真实公式。外部NumPy/SciPy标准运算按其参数解释。",
        "输入合同、交易制品与GPU派发仍以相应模块及统一口径为准；本附录不复制数据或生产产物。"]
    for identity, fn in sorted(helpers.items()):
        lines += ["", f"### {identity}", "", "```python", source(fn), "```"]
    lines += ["", "## 定义完整性指纹", "",
              "每项注册定义及上列实现源公式均参与本文内容；以下源摘要便于定位函数变化。", "",
              "| 函数 | SHA-256（源公式） |", "|---|---|"]
    for identity, fn in sorted({**implementations, **helpers}.items()):
        lines.append(f"| `{identity}` | `{hashlib.sha256(source(fn).encode()).hexdigest()}` |")
    return "\n".join(lines) + "\n"


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
