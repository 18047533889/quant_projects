"""Generate bounded, explicitly non-certifying main-tree inventory artifacts."""
import json
from pathlib import Path
from quant_evaluator.metrics.coverage_compiler import canonical_entrypoint_inventory
from check_team_environment import runtime_identity, isolated_import_errors


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "evidence/r2"
    inventory = {
        "scope": "M22 static canonical entrypoint inventory only",
        "date": "2026-09-09", "working_tree_modified": True,
        "execution_certification": False,
        "rows": canonical_entrypoint_inventory(),
    }
    (folder / "main_merge_entrypoints_20260909.json").write_text(
        json.dumps(inventory, indent=2) + "\n")
    runtime = runtime_identity()
    runtime["import_errors"] = isolated_import_errors(root)
    (folder / "main_merge_runtime_identity_20260909.json").write_text(
        json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"inventory_rows": len(inventory["rows"]),
                      "import_errors": runtime["import_errors"]}))
    return bool(runtime["import_errors"])


if __name__ == "__main__":
    raise SystemExit(main())
