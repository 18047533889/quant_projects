#!/usr/bin/env python
"""报告口径：每个因子一律以「全窗 RankIC 非负」的方向入库（RankIC 转正，负则加负号）。

新挖链路的评估方向只用 2016-01-04..2018-06-30（EVAL_END）训练，因此一个因子在
全窗上仍可能算出负的 Mean RankIC。本脚本找出这些因子，用**相反的固定方向**重新
跑一次 QuantEvaluator（IC / 十分层净值 / 多空序列都由 QE 重算，不做手工改符号），
然后把逐因子 manifest 原地重写。评估产物带新的 sha256，页面照常校验。

用法：
    .venv/bin/python jobs/normalize_direction.py            # 只报告，不写盘
    .venv/bin/python jobs/normalize_direction.py --apply    # 真正重跑并重写 manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

import new_mining_intake as N  # noqa: E402  (needs jobs/ on sys.path)

# Fields that the landing stage attaches to the record; the candidate file may
# not carry them, so they are carried over from the entry being replaced.
PRESERVE = ("landing_backend", "landing_backend_fallbacks", "campaign", "source_metadata",
            "factor_name", "source_formula", "formula_source", "formula_source_sha256",
            "lqtp_formula", "dsl_repair")


def manifest_dirs(root: Path):
    yield root / "factors"
    shards = root / "shards"
    if shards.is_dir():
        for shard in sorted(shards.glob("*/factors")):
            if shard.is_dir():
                yield shard


def collect(root: Path):
    """(manifest_path, name, entry, rank_ic) for every negative-RankIC entry.

    A factor can exist in several manifest locations (the root run and each
    shard).  ``_current_manifests`` resolves that by taking the first hit in the
    same precedence order used here, so only that copy is flipped: rewriting a
    stale shadow copy would create a second, contradicting source of truth.
    """
    out, seen, shadows = [], set(), []
    for base in manifest_dirs(root):
        for manifest_path in sorted(base.glob("*/report_manifest.json")):
            try:
                document = json.loads(manifest_path.read_text())
            except Exception:
                continue
            for name, entry in (document.get("factors") or {}).items():
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status") or "") == "unavailable":
                    continue
                rank_ic = (entry.get("metrics") or {}).get("rank_ic")
                if rank_ic is None or float(rank_ic) >= 0:
                    continue
                if name in seen:
                    shadows.append((name, manifest_path))
                    continue
                seen.add(name)
                out.append((manifest_path, name, entry, float(rank_ic)))
    return out, shadows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "work" / "newmining_20260919"))
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    import incremental_factor_intake as intake

    root = Path(args.root)
    candidates_path = Path(getattr(N, "CANDIDATES", root / "candidates.json"))
    payload = json.loads(candidates_path.read_text())
    candidates = {c["page_name"]: c for c in payload["candidates"]}

    targets, shadows = collect(root)
    print(f"negatives found: {len(targets)} (authoritative copies)")
    for name, shadow in shadows:
        print(f"  shadow copy ignored: {name} @ {shadow}")
    if not targets:
        return 0

    failures = 0
    for manifest_path, name, entry, rank_ic in targets:
        direction = int(entry.get("direction") or 1)
        flip = -direction
        record = candidates.get(name)
        if record is None:
            print(f"  SKIP {name}: no candidate record in {candidates_path.name}")
            failures += 1
            continue
        record, _note = N.apply_dsl_repair(record)
        resolved_matrix = intake.matrix_path(name)
        if not resolved_matrix:
            print(f"  SKIP {name}: engine resolves no owned matrix; rewriting would "
                  "mark a landed factor as never-landed")
            failures += 1
            continue
        if not args.apply:
            print(f"  WOULD FLIP {name}: rank_ic {rank_ic:+.5f} direction {direction} -> {flip}")
            continue
        try:
            batch = intake.evaluate_factor_batch([record], batch_size=1, backend=args.backend,
                                                 direction_map={name: flip})
        except Exception as error:
            print(f"  FAIL {name}: evaluation raised {type(error).__name__}: {error}")
            failures += 1
            continue
        evaluated = (batch.get("factors") or {}).get(name)
        new_ic = getattr(evaluated, "mean_rank_ic", None)
        if evaluated is None or new_ic is None or float(new_ic) < 0:
            print(f"  FAIL {name}: flip did not produce a non-negative RankIC "
                  f"({new_ic!r}); manifest left untouched")
            failures += 1
            continue
        # Write into the real directory so the artifact lands next to its
        # manifest, then carry over everything the candidate record does not hold.
        original_document = json.loads(manifest_path.read_text())
        with tempfile.TemporaryDirectory(prefix="normalize_") as staging:
            staged = Path(staging) / "report_manifest.json"
            intake.write_report_manifest([record], batch, target=staged)
            document = json.loads(staged.read_text())
            fresh = (document.get("factors") or {}).get(name)
            old = (original_document.get("factors") or {}).get(name) or {}
            for key in PRESERVE:
                if key in old and (key not in fresh or fresh.get(key) in (None, "", [])):
                    fresh[key] = old[key]
            # move the freshly evaluated artifact next to the real manifest.
            # write_report_manifest stores ``artifact`` relative to the manifest
            # directory, but a per-factor manifest is also copied into a staging
            # directory during page rendering (new_mining_publish.render_pages),
            # where a relative ref would not resolve.  Per-factor manifests in
            # this tree hold absolute refs, so store the absolute one.
            relative_artifact = str(fresh["artifact"])
            source_artifact = staged.parent / relative_artifact
            target_artifact = manifest_path.parent / relative_artifact
            if not source_artifact.exists():
                raise FileNotFoundError(f"staged artifact missing: {source_artifact}")
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            target_artifact.write_bytes(source_artifact.read_bytes())
            fresh["artifact"] = str(target_artifact.resolve())
            fresh["artifact_sha256"] = hashlib.sha256(target_artifact.read_bytes()).hexdigest()
            document["factors"] = {name: fresh}
            manifest_path.write_text(json.dumps(document, ensure_ascii=False, indent=1))
        check = json.loads(manifest_path.read_text())
        written = (check.get("factors") or {}).get(name) or {}
        print(f"  FLIPPED {name}: rank_ic {rank_ic:+.5f} -> {float(new_ic):+.5f} "
              f"direction {direction} -> {written.get('direction')} "
              f"is_flipped={written.get('is_flipped')} "
              f"artifact={Path(str(written.get('artifact'))).name}")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
