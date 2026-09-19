#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COS landing helper for the quant factor platform (server-c).

Storage policy this tool implements
----------------------------------
* **factor values and other owned artefacts** -> Tencent COS bucket
  ``quant-factors-1425188104``, prefix ``data/``.
* **raw market data** -> local SSD ``~/cos_data`` (already mirrored there);
  never uploaded, it is the platform's read-only source of truth.

Access model
------------
The bucket is only reachable through the managed sudo gateway
``/usr/local/libexec/quantsociety-cos/factor-admin-cos`` (member of
``quant-admin``, which this account is in).  The gateway enforces:

* the local path must live under an approved root (``/home/<user>`` or
  ``/srv/quant/...``) -- so a staging file must be written under ``$HOME``;
* ``ls`` needs a COS URI as its first argument;
* ``delete`` is **not** granted to this account (write + read only).

Because delete is unavailable, the tool is deliberately append-only: it never
tries to remove a remote object.  Re-landing the same key overwrites it, which
is what a re-run wants anyway.

Usage
-----
    python3 tools/cos_land.py ls   data/factors/daily
    python3 tools/cos_land.py put  ~/stage/f.parquet data/factors/daily/f.parquet
    python3 tools/cos_land.py put  ~/stage/batch1   data/factors/daily/batch1
    python3 tools/cos_land.py stage-upload /tmp/not-allowed/x.parquet data/x.parquet

``stage-upload`` copies a file that lives outside the approved roots into a
staging directory under ``$HOME`` first, then uploads it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

GATEWAY = "/usr/local/libexec/quantsociety-cos/factor-admin-cos"
BUCKET = "quant-factors-1425188104"
COS_ROOT = f"cos://{BUCKET}/"

# The gateway only accepts local paths under one of these roots.
APPROVED_ROOTS = (Path("/home"), Path("/srv/quant"))
STAGING_DIR = Path.home() / "cos_stage"


class CosLandingError(RuntimeError):
    pass


def _run(argv: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sudo", "-n", GATEWAY, *argv],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def _cos_uri(key: str) -> str:
    key = key.strip().lstrip("/")
    if key.startswith("cos://"):
        return key
    if not key:
        raise CosLandingError("empty COS key")
    return COS_ROOT + key


def _is_approved(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any(
        resolved == root or root in resolved.parents for root in APPROVED_ROOTS
    )


def stage(path: str | os.PathLike[str]) -> Path:
    """Return an approved-root copy of ``path`` (no-op when already approved)."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise CosLandingError(f"local path does not exist: {src}")
    if _is_approved(src):
        return src
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    dest = STAGING_DIR / src.name
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    return dest


def put(local: str | os.PathLike[str], key: str, *, recursive: bool | None = None) -> str:
    """Upload ``local`` to ``key``. Returns the COS URI."""
    src = Path(local).expanduser().resolve()
    if not src.exists():
        raise CosLandingError(f"local path does not exist: {src}")
    if not _is_approved(src):
        raise CosLandingError(
            f"local path is outside approved roots {APPROVED_ROOTS}: {src}. "
            "Use stage()/stage-upload to copy it under $HOME first."
        )
    uri = _cos_uri(key)
    args = ["cp"]
    if recursive is True or (recursive is None and src.is_dir()):
        args.append("-r")
    args += [str(src), uri]
    proc = _run(args)
    if proc.returncode != 0 or "OK num: 0" in proc.stdout:
        raise CosLandingError(
            f"upload failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:500]}"
        )
    return uri


def ls(key: str) -> str:
    proc = _run(["ls", _cos_uri(key)])
    if proc.returncode != 0:
        raise CosLandingError(
            f"list failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:500]}"
        )
    return proc.stdout


def _parse_listing_row(listing: str, key: str) -> dict | None:
    """Pull one row out of the coscli table: {'key','type','modified','etag','size'}.

    ``coscli`` prints human-readable sizes ("8.77 MB"), not raw bytes, so a
    byte-for-byte comparison against the local file is not possible from the
    listing alone.  We parse the row and compare a locally formatted size.
    """
    key = key.strip().lstrip("/")
    if key.startswith("cos://"):
        key = key.split("/", 3)[-1]
    for line in listing.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6:
            continue
        if cells[0] == key:
            return {
                "key": cells[0], "type": cells[1], "modified": cells[2],
                "etag": cells[3].strip('"'), "size": cells[4],
            }
    return None


def _human_bytes(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def verify(local: str | os.PathLike[str], key: str) -> dict:
    """Confirm the remote object exists; compare sizes when the row is parseable."""
    src = Path(local).expanduser().resolve()
    listing = ls(key)
    report: dict = {
        "local": str(src),
        "cos_uri": _cos_uri(key),
        "listed": bool(listing.strip()),
        "size_match": None,
        "remote_listing": listing[:1500],
    }
    if src.is_file():
        report["local_bytes"] = src.stat().st_size
        report["local_size_human"] = _human_bytes(src.stat().st_size)
        row = _parse_listing_row(listing, key)
        if row is not None:
            report["remote"] = row
            report["size_match"] = row["size"] == report["local_size_human"]
    return report


def stage_upload(local: str | os.PathLike[str], key: str) -> dict:
    staged = stage(local)
    uri = put(staged, key)
    rep = verify(staged, key)
    rep["staged_from"] = str(Path(local).expanduser().resolve())
    return rep


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list a COS prefix")
    p_ls.add_argument("key")

    p_put = sub.add_parser("put", help="upload a file/dir (must be under $HOME or /srv/quant)")
    p_put.add_argument("local")
    p_put.add_argument("key")

    p_st = sub.add_parser("stage-upload", help="stage into $HOME then upload")
    p_st.add_argument("local")
    p_st.add_argument("key")

    p_vf = sub.add_parser("verify", help="verify a remote object against a local file")
    p_vf.add_argument("local")
    p_vf.add_argument("key")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "ls":
            print(ls(args.key))
        elif args.cmd == "put":
            print("uploaded:", put(args.local, args.key))
        elif args.cmd == "stage-upload":
            import json
            print(json.dumps(stage_upload(args.local, args.key), indent=1, ensure_ascii=False))
        elif args.cmd == "verify":
            import json
            print(json.dumps(verify(args.local, args.key), indent=1, ensure_ascii=False))
    except CosLandingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
