#!/usr/bin/env python3
"""GTJA-191 独立包测试：转换逻辑、翻译正确性、投递包一致性。"""
from __future__ import annotations

import ast
import glob
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_convert_module():
    path = SCRIPTS_DIR / "convert_and_build_delivery.py"
    spec = importlib.util.spec_from_file_location("gtja_convert", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_catalog() -> dict:
    return json.loads((PACKAGE_ROOT / "dsl" / "gtja191_dsl_catalog.json").read_text(encoding="utf-8"))


def _ema_arity_issues(formula: str) -> list[str]:
    tree = ast.parse(formula, mode="eval")

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.issues: list[str] = []

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in ("EMA", "SMA", "WMA"):
                if len(node.args) != 2:
                    self.issues.append(f"{node.func.id} has {len(node.args)} args")
            self.generic_visit(node)

    v = V()
    v.visit(tree)
    return v.issues


class TestConversionHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conv = _load_convert_module()

    def test_sma_span_table(self) -> None:
        self.assertEqual(self.conv._gtja_sma_span(13, 2), 12)
        self.assertEqual(self.conv._gtja_sma_span(27, 2), 26)
        self.assertEqual(self.conv._gtja_sma_span(3, 1), 5)
        self.assertEqual(self.conv._gtja_sma_span(10, 1), 19)
        self.assertEqual(self.conv._gtja_sma_span(9, 1), 17)

    def test_ema_three_arg_nested_fold(self) -> None:
        raw = "EMA(EMA(close,13,2),10,1)"
        out = self.conv._post_process_dsl(raw)
        self.assertEqual(out, "EMA(EMA(close, 12), 19)")
        self.assertEqual(_ema_arity_issues(out), [])

    def test_rolling_max_on_composite_expr(self) -> None:
        expr = "max(((high + low + close) / 3) - close, 3)"
        out = self.conv._fix_rolling_max_min(expr)
        self.assertEqual(out, "ts_max(((high + low + close) / 3) - close, 3)")

    def test_rolling_max_nested_inside_outer_elementwise(self) -> None:
        expr = (
            "max(rank(a), rank(ts_decay_linear(max(ts_corr(rank(close), rank(volume), 4), 13), 14)))"
        )
        out = self.conv._fix_rolling_max_min_calls(expr)
        self.assertIn("ts_max(ts_corr(rank(close), rank(volume), 4), 13)", out)
        self.assertTrue(out.startswith("max(rank(a),"))

    def test_preserve_elementwise_max_zero(self) -> None:
        expr = "ts_sum(max(0, high - delay(close, 1)), 26)"
        self.assertEqual(self.conv._fix_rolling_max_min(expr), expr)

    def test_preserve_elementwise_min_two_series(self) -> None:
        expr = "min(low, delay(close, 1))"
        self.assertEqual(self.conv._fix_rolling_max_min(expr), expr)

    def test_do_not_corrupt_m_argmin(self) -> None:
        expr = "((20 - (20 - m_argmin(low, 20)))/20)*100"
        out = self.conv._post_process_dsl(expr)
        self.assertIn("m_argmin", out)
        self.assertNotIn("m_argts_min", out)


class TestTranslationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conv = _load_convert_module()
        cls.catalog = _load_catalog()

    def test_alpha_007_rolling_max_min(self) -> None:
        dsl = self.catalog["gtja191_alpha_007"]["dsl_formula"]
        self.assertIn("ts_max((((high + low + close) / 3) - close), 3)", dsl)
        self.assertIn("ts_min((((high + low + close) / 3) - close), 3)", dsl)

    def test_alpha_052_no_typo_field_l(self) -> None:
        dsl = self.catalog["gtja191_alpha_052"]["dsl_formula"]
        self.assertIn("low", dsl)
        self.assertNotRegex(dsl, r"(?<![a-z_])L(?![a-z_])")

    def test_alpha_089_sma_to_ema_spans(self) -> None:
        dsl = self.catalog["gtja191_alpha_089"]["dsl_formula"]
        self.assertEqual(dsl, "2*(EMA(close, 12)-EMA(close, 26)-EMA(EMA(close, 12)-EMA(close, 26), 9))")

    def test_alpha_064_inner_rolling_max(self) -> None:
        dsl = self.catalog["gtja191_alpha_064"]["dsl_formula"]
        self.assertIn("ts_max(ts_corr(rank(close), rank(ts_mean(volume,60)), 4), 13)", dsl)

    def test_alpha_190_count_minus_one(self) -> None:
        dsl = self.catalog["gtja191_alpha_190"]["dsl_formula"]
        self.assertIn("ts_sum(if_else(close / delay(close, 1) - 1 >", dsl)
        self.assertIn(", 20) - 1)", dsl)

    def test_auto_converted_formulas_are_idempotent(self) -> None:
        """自动翻译项：重新跑 convert 应与 catalog 一致（翻译稳定）。"""
        drift: list[str] = []
        for name, item in self.catalog.items():
            if item["conversion"] != "converted":
                continue
            expected = item["dsl_formula"]
            got = self.conv.convert_gtja_to_dsl(item["source_formula"])
            if got != expected:
                drift.append(name)
        self.assertEqual(drift, [], f"conversion drift: {drift[:5]}")


class TestCatalogIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = _load_catalog()

    def test_catalog_has_191_entries(self) -> None:
        self.assertEqual(len(self.catalog), 191)

    def test_all_catalog_entries_valid(self) -> None:
        bad = [k for k, v in self.catalog.items() if not v["valid"]]
        self.assertEqual(bad, [])

    def test_all_dsl_parse_and_balance_parens(self) -> None:
        from lib.dsl_validate import validate_formula as check_formula

        for name, item in self.catalog.items():
            dsl = item["dsl_formula"]
            with self.subTest(name=name):
                self.assertEqual(dsl.count("("), dsl.count(")"), "paren mismatch")
                ok, msg = check_formula(dsl)
                self.assertTrue(ok, msg)

    def test_no_three_arg_ema_in_catalog(self) -> None:
        bad: list[str] = []
        for name, item in self.catalog.items():
            issues = _ema_arity_issues(item["dsl_formula"])
            if issues:
                bad.append(f"{name}: {issues}")
        self.assertEqual(bad, [])

    def test_no_corrupted_operator_names(self) -> None:
        for name, item in self.catalog.items():
            dsl = item["dsl_formula"]
            with self.subTest(name=name):
                self.assertNotIn("m_argts_min", dsl)
                self.assertNotIn("m_argts_max", dsl)
                self.assertNotRegex(dsl, r"EMA\([^)]+,\s*\d+\s*,\s*\d+")

    def test_formula_files_match_catalog(self) -> None:
        for name, item in sorted(self.catalog.items()):
            num = int(name.rsplit("_", 1)[-1])
            path = PACKAGE_ROOT / "formulas" / f"gtja191_alpha_{num:03d}.dsl"
            with self.subTest(name=name):
                self.assertTrue(path.exists(), f"missing {path.name}")
                text = path.read_text(encoding="utf-8")
                last_line = [ln for ln in text.splitlines() if ln and not ln.startswith("#")][-1]
                self.assertEqual(last_line.strip(), item["dsl_formula"])


class TestDeliveryPackage(unittest.TestCase):
    CAMPAIGN = "manual_ashare_pv_202607041600"

    @classmethod
    def setUpClass(cls) -> None:
        cls.conv = _load_convert_module()
        cls.catalog = _load_catalog()
        cls.campaign_dir = PACKAGE_ROOT / "candidate_pool" / cls.CAMPAIGN

    def test_deliverable_manifest_count(self) -> None:
        manifests = sorted(glob.glob(str(self.campaign_dir / "manual_*/manifest.json")))
        deliverable = sum(
            1
            for v in self.catalog.values()
            if v["valid"] and not v.get("delivery_excluded")
        )
        self.assertEqual(len(manifests), deliverable)
        self.assertEqual(deliverable, 185)

    def test_excluded_factors_not_in_delivery(self) -> None:
        excluded = self.conv.DELIVERY_EXCLUDED
        manifests = glob.glob(str(self.campaign_dir / "manual_*/manifest.json"))
        delivered_nums = set()
        for path in manifests:
            m = json.loads(Path(path).read_text(encoding="utf-8"))
            desc = m.get("description", "")
            match = re.search(r"第 (\d+) 号", desc)
            if match:
                delivered_nums.add(int(match.group(1)))
        for name in excluded:
            num = int(name.rsplit("_", 1)[-1])
            self.assertNotIn(num, delivered_nums, f"excluded {name} was delivered")

    def test_all_manifest_formulas_match_catalog(self) -> None:
        for path in glob.glob(str(self.campaign_dir / "manual_*/manifest.json")):
            m = json.loads(Path(path).read_text(encoding="utf-8"))
            desc = m.get("description", "")
            match = re.search(r"第 (\d+) 号", desc)
            self.assertIsNotNone(match, path)
            num = int(match.group(1))  # type: ignore[union-attr]
            key = f"gtja191_alpha_{num:03d}"
            self.assertEqual(m["formula"], self.catalog[key]["dsl_formula"])
            self.assertEqual(m["expression_type"], "dsl")

    def test_all_manifests_validate(self) -> None:
        from lib.dsl_validate import validate_formula as check_formula

        for path in glob.glob(str(self.campaign_dir / "manual_*/manifest.json")):
            m = json.loads(Path(path).read_text(encoding="utf-8"))
            ok, msg = check_formula(m["formula"])
            self.assertTrue(ok, f"{path}: {msg}")


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
