"""Read-only content/topology inventory of pre-rebuild GitHub branches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import hashlib


def git(*args):
    return subprocess.check_output(["git", *args])


def current_path_candidates(path):
    """Explicit migration candidates, not a claim of semantic equivalence."""
    stripped = path.removeprefix("AutoFactorEvaluation-RECONSTRUCT/")
    if stripped.startswith("dataaccess/"):
        stripped = "data_access/" + stripped[len("dataaccess/"):]
    return tuple(dict.fromkeys((path, stripped)))


def reconcile_inventory(inventory, root, retained_refs):
    """Bounded read-only current-blob comparison, preserving unresolved code."""
    result = []
    for branch in inventory["branches"]:
        if branch["ref"] not in retained_refs:
            continue
        rows = []
        for change in branch["changed"]:
            path = change["path"]
            candidates = current_path_candidates(path)
            current = next((root / p for p in candidates if (root / p).is_file()), None)
            row = dict(change, current_path=str(current.relative_to(root)) if current else None,
                       current_blob=None, execution_status="NOT_RUN")
            if current is not None:
                data = current.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                row["current_blob"] = blob
                row["current_sha256"] = hashlib.sha256(data).hexdigest()
                if change["status"] != "D" and blob == change["blob"]:
                    row.update(classification="CURRENT_BYTES_IDENTICAL",
                        reason="Current mapped file equals historical blob; public reachability still needs consumer evidence.")
                else:
                    row.update(classification="NEEDS_SEMANTIC_REVIEW",
                        reason="Current mapped implementation differs; retain historical ref and compare contracts/tests before migrating.")
            elif path.endswith((".md", ".bak", ".txt", ".b85")) or path.startswith(".github/"):
                row.update(classification="HISTORICAL_SUPPORTING_ARTIFACT",
                    reason="Not a current runtime implementation; preserved in immutable historical ref, not copied into live code.")
            elif path.startswith("AutoFactorEvaluation-RECONSTRUCT/"):
                row.update(classification="REPLACEMENT_CONTRACT_REVIEW",
                    current_authorities=["quant_evaluator/runtime/evaluator.py", "quant_platform/app/orchestrator.py",
                                         "factor_assets/selection/decision.py"],
                    reason="Historical standalone bundled architecture conflicts with unique-main-tree contract; numerical/consumer behavior must be adjudicated separately, not restored as another embedded platform.")
            else:
                row.update(classification="NEEDS_SEMANTIC_REVIEW",
                    reason="No verified current path mapping. Keep ref; absence alone neither authorizes deletion nor proves the old behavior is still required.")
            rows.append(row)
        result.append({"ref": branch["ref"], "tip": branch["tip"], "changed": rows,
                       "counts": dict(Counter(r["classification"] for r in rows)),
                       "semantic_closure": "NOT_VERIFIED",
                       "reason": "Anchor-only refs preserve inherited history; zero delta is not full coverage." if not rows else "Per-path comparison preserves unresolved semantic differences."})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reconcile-existing", type=Path,
                        help="Read the existing historical inventory and compare current mapped bytes; no checkout or fetch")
    args = parser.parse_args()
    if args.reconcile_existing:
        inventory = json.loads(args.reconcile_existing.read_text())
        retained = {line.split()[1] for line in git("ls-remote", "origin", "refs/heads/*").decode().splitlines()
                    if not line.endswith("refs/heads/main")}
        payload = {"schema": "legacy-current-disposition-v2",
                   "main": git("rev-parse", "HEAD").decode().strip(),
                   "source_inventory_sha256": hashlib.sha256(args.reconcile_existing.read_bytes()).hexdigest(),
                   "status": "NEEDS_SEMANTIC_REVIEW", "historical_refs_deleted": False,
                   "branches": reconcile_inventory(inventory, Path.cwd(), retained)}
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"branches": len(payload["branches"]),
                          "counts": dict(Counter(r["classification"] for b in payload["branches"] for r in b["changed"]))}))
        return
    anchor = "f56de261192ccf41086dbd8c0535d5e3d2740de7"
    current = git("rev-parse", "HEAD").decode().strip()
    refs = [line.split() for line in git("ls-remote", "origin", "refs/heads/*").decode().splitlines()
            if not line.endswith("refs/heads/main")]
    main_blobs = {line.split(b" ", 1)[0].decode()
                  for line in git("rev-list", "--objects", current).splitlines()}
    main_tree = {}
    for entry in git("ls-tree", "-rz", current).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        main_tree[path.decode()] = meta.split()[2].decode()
    branches = []
    for tip, ref in refs:
        base = git("merge-base", anchor, tip).decode().strip()
        entries = git("diff", "--raw", "--no-abbrev", "--no-renames", "-z", base, tip).split(b"\0")
        changed = []
        for index in range(0, len(entries)-1, 2):
            if not entries[index]:
                continue
            raw, path = entries[index].decode(), entries[index+1].decode()
            parts = raw.split()
            old_blob, new_blob, status = parts[2:5]
            if status == "D":
                disposition = "ALREADY_ABSENT_MAIN" if path not in main_tree else "REVIEW_DELETION"
            elif new_blob in main_blobs:
                disposition = "BLOB_PRESENT_IN_MAIN_HISTORY"
            else:
                disposition = "REVIEW_DELTA"
            changed.append(dict(path=path, status=status, blob=new_blob,
                                old_blob=old_blob, disposition=disposition))
        branches.append(dict(
            ref=ref, tip=tip,
            date=git("show", "-s", "--format=%cI", tip).decode().strip(),
            subject=git("show", "-s", "--format=%s", tip).decode().strip(),
            common_base_with_old_anchor=base,
            commits_beyond_old_anchor=int(git("rev-list", "--count", tip, "--not", anchor)),
            changed=changed, counts=dict(Counter(x["disposition"] for x in changed)),
        ))
    output = dict(schema="legacy-branch-content-inventory-v1", main=current,
                  old_anchor=anchor, note="No deletion approval inferred from age or blob matches alone.",
                  branches=branches)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    for branch in branches:
        print(branch["ref"], branch["commits_beyond_old_anchor"], branch["counts"])


if __name__ == "__main__":
    main()
