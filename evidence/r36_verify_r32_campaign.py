"""Read-only independent verification of the frozen R32 paired ledger."""
import importlib.util
import json
from pathlib import Path

root = Path("evidence/r32_operator_campaign")
spec = importlib.util.spec_from_file_location("r32_review_campaign", root / "campaign_fixture_retry.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
recipes = json.loads((root / "recipes_final_current_source.json").read_text())["recipes"]
rows = [json.loads(line) for line in (root / "ledger_final_current_source_attempt_2.jsonl").read_text().splitlines()]
names = {r["canonical"] for r in recipes}
assert len(names) == len(recipes) == 60
backend = {(r["canonical"], r["backend"]): r for r in rows
           if r.get("backend") in ("pandas_numpy", "polars") and r["status"] != "RUNNING"}
parity = {r["canonical"]: r for r in rows if r["status"].startswith("CANONICAL_PARITY")}
assert len(backend) == 120 and len(parity) == 60
assert set(parity) == names
campaign.load_all()
inputs = campaign.fixture_inputs(96)
for recipe in recipes:
    name = recipe["canonical"]
    pair = parity[name]
    assert pair["status"] == "CANONICAL_PARITY" and pair["pandas_polars_equal"] is True
    for kind in ("pandas_numpy", "polars"):
        result = backend[name, kind]
        assert result["status"] == "EXECUTED_FINITE" and result["finite"] > 0, (name, kind)
        assert result["future_prefix_invariant_all_inputs"] is True
        op = campaign.OperatorRegistry.get(name, kind, mode="any")
        current = campaign.execution_fingerprint(name, kind, recipe, inputs, op)
        assert result["fingerprint"] == pair["backend_fingerprints"][kind] == current, (name, kind)
print(json.dumps({"reviewed_current_canonicals": len(names), "finite_backends": len(backend),
                  "parity_pairs": len(parity), "current_fingerprints_match": True,
                  "canonicals": sorted(names)}, sort_keys=True))
