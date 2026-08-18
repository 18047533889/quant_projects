from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_writer(root: Path, ready: Path, release: Path) -> subprocess.Popen[str]:
    code = r'''
import sys, time
from pathlib import Path
from quant_evaluator.runtime.cache_v2 import DiskCacheLayer, create_compressor
root, ready, release = map(Path, sys.argv[1:])
cache = DiskCacheLayer(root, create_compressor("none"))
original = Path.replace

def paused_replace(self, target):
    result = original(self, target)
    if Path(target).name.endswith(".cache"):
        ready.write_text("cache-replaced", encoding="utf-8")
        while not release.exists():
            time.sleep(0.002)
    return result
Path.replace = paused_replace
assert cache.put("key", {"value": "new"}, ttl_seconds=60)
'''
    return subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(ready), str(release)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qe_torn_publish_") as raw:
        root = Path(raw) / "cache"
        ready = Path(raw) / "ready"
        release = Path(raw) / "release"
        from quant_evaluator.runtime.cache_v2 import DiskCacheLayer, create_compressor

        cache = DiskCacheLayer(root, create_compressor("none"))
        assert cache.put("key", {"value": "old"}, ttl_seconds=60)
        writer = run_writer(root, ready, release)
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.002)
        if not ready.exists():
            raise RuntimeError(writer.stderr.read() if writer.stderr else "writer did not pause")
        reader = DiskCacheLayer(root, create_compressor("none"))
        observed: list[object] = []
        reader_done = threading.Event()

        def read_entry() -> None:
            observed.append(reader.get("key"))
            reader_done.set()

        reader_thread = threading.Thread(target=read_entry)
        reader_thread.start()
        time.sleep(0.1)
        blocked = not reader_done.is_set()
        release.write_text("go", encoding="utf-8")
        reader_thread.join(timeout=10)
        if reader_thread.is_alive():
            raise RuntimeError("reader remained blocked after writer release")
        stdout, stderr = writer.communicate(timeout=10)
        if writer.returncode != 0:
            raise RuntimeError(f"writer failed: {stdout}\n{stderr}")
        value = observed[0] if observed else None
        observed_value = None if value is None else value[0]
        print(json.dumps({
            "reader_blocked": blocked,
            "observed_after_publish": observed_value,
            "torn": observed_value is None,
        }))


if __name__ == "__main__":
    main()
