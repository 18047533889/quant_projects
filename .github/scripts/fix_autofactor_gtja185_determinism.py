from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_generator() -> None:
    path = PROJECT / "scripts" / "generate_gtja185_pack.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("from datetime import datetime, timezone\n", "")
    old = '            "generated_at": datetime.now(timezone.utc).isoformat(),\n'
    new = '            "source_catalog_hash": source_catalog_hash,\n'
    if old not in text and new not in text:
        raise RuntimeError("generator metadata anchor not found")
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_evaluator() -> None:
    path = PROJECT / "evaluation" / "gtja185_batch.py"
    replace_exact(
        path,
        '''    purified = values.groupby(level="datetime", group_keys=False).apply(transform)
    purified.index = purified.index.droplevel(0) if purified.index.nlevels == 3 else purified.index
    purified.index = purified.index.set_names(["datetime", "asset"])
    return purified.sort_index().rename(series.name)
''',
        '''    purified = values.groupby(level="datetime", group_keys=False).transform(transform)
    purified.index = purified.index.set_names(["datetime", "asset"])
    return purified.sort_index().rename(series.name)
''',
        "stable cross-sectional transform",
    )
    replace_exact(
        path,
        '''        top = set(valid.index.get_level_values("asset")[top_mask])
        bottom = set(valid.index.get_level_values("asset")[bottom_mask])
''',
        '''        top = set(valid.index.get_level_values("asset")[np.asarray(top_mask, dtype=bool)])
        bottom = set(valid.index.get_level_values("asset")[np.asarray(bottom_mask, dtype=bool)])
''',
        "stable quantile membership",
    )
    replace_exact(
        path,
        '''    train_ic = _daily_ic(
        purified.loc[_split_mask(purified.index, "train", bounds)],
        forward_returns[min(config.horizons)].loc[_split_mask(forward_returns[min(config.horizons)].index, "train", bounds)],
        min_assets=config.min_assets,
    )
''',
        '''    primary_forward = forward_returns[min(config.horizons)].reindex(purified.index)
    train_positions = np.flatnonzero(_split_mask(purified.index, "train", bounds))
    train_ic = _daily_ic(
        purified.iloc[train_positions],
        primary_forward.iloc[train_positions],
        min_assets=config.min_assets,
    )
''',
        "aligned training direction sample",
    )
    replace_exact(
        path,
        '''    finite = np.isfinite(purified.to_numpy(dtype=float, na_value=np.nan))
''',
        '''    finite = np.isfinite(pd.to_numeric(purified, errors="coerce").to_numpy(dtype=float))
''',
        "portable finite conversion",
    )


def regenerate_pack() -> None:
    namespace = runpy.run_path(str(PROJECT / "scripts" / "generate_gtja185_pack.py"))
    result = namespace["main"]()
    if result not in (None, 0):
        raise RuntimeError(f"pack generation returned {result}")


def main() -> None:
    patch_generator()
    patch_evaluator()
    regenerate_pack()
    print("GTJA185 deterministic pack and evaluator fixes applied")


if __name__ == "__main__":
    main()
