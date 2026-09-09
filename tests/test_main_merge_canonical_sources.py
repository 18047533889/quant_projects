"""M22 regression: no silently shadowed definitions or module/package split."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("quant_evaluator", "factor_assets", "factor_optimizer",
            "factor_preprocess", "factor_engine", "data_access")
IGNORE = {".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules",
          "tests", "benchmarks"}


def test_main_implementations_have_unique_python_bindings():
    failures = []
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            if any(p in IGNORE or p.endswith(".egg-info") for p in path.relative_to(ROOT).parts):
                continue
            sibling = path.with_suffix("")
            if sibling.is_dir() and (sibling / "__init__.py").is_file():
                failures.append(f"module/package collision: {path.relative_to(ROOT)}")
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            seen = set()
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if any(isinstance(d, ast.Name) and d.id == "overload" or
                       isinstance(d, ast.Attribute) and d.attr == "overload"
                       for d in node.decorator_list):
                    continue
                if node.name in seen:
                    failures.append(f"shadowed definition: {path.relative_to(ROOT)}:{node.lineno}:{node.name}")
                seen.add(node.name)
    assert failures == []
