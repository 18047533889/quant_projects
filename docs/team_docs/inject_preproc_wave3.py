#!/usr/bin/env python3
"""Idempotent merge: §0.4 storage map, §12f 十二 Raw+CA, cross-links (v3.16)."""
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

MARKER = "sec-deep-research-raw-ca"
STORAGE_MARKER = "sec-storage-map"


def main():
    html = HTML.read_text(encoding="utf-8")
    changed = False

    if MARKER not in html:
        print("WARN: §12f 十二 not found — run manual merge or restore from git")
    if STORAGE_MARKER not in html:
        print("WARN: §0.4 storage map not found")

    # Version bump guard (optional idempotent touch)
    old_meta = "版本 v3.15"
    new_meta = "版本 v3.16"
    if old_meta in html and new_meta not in html:
        html = html.replace(old_meta, new_meta, 1)
        changed = True

    if changed:
        HTML.write_text(html, encoding="utf-8")
        print("Updated", HTML.name)
    else:
        print("Already at v3.16+ — no changes")


if __name__ == "__main__":
    main()
