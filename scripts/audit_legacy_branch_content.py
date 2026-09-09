"""Read-only content/topology inventory of pre-rebuild GitHub branches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
