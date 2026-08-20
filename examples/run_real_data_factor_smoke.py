from __future__ import annotations

from pathlib import Path

from runtime.real_data_factor_smoke import run_all_dataset_smokes
from workspace_paths import resolve_path
import os

MASSIVE_ROOT = resolve_path(
    os.environ.get("MASSIVE_PARQUET_ROOT", "~/quant_projects/data/us_stock/cleaned_massive_data")
)


if __name__ == '__main__':
    reports = run_all_dataset_smokes(MASSIVE_ROOT, max_files=2)
    ok_count = 0
    total = 0

    for report in reports:
        print(f"\n== {report['dataset']} ==")
        for item in report['results']:
            total += 1
            if item['ok']:
                ok_count += 1
                print(
                    f"  [OK] {item['factor']}: rows={item['rows']}, "
                    f"non_nan_ratio={item['non_nan_ratio']:.4f}, lookback={item['lookback']}"
                )
            else:
                print(f"  [FAIL] {item['factor']}: {item['error']}")

    print(f"\nsummary: {ok_count}/{total} factors succeeded")
