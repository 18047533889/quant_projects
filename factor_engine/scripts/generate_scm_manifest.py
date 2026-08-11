#!/usr/bin/env python3
"""R39 #75 —— 构建期生成 evidence/scm_manifest.json（绑定源码树到提交）。

在**打包/构建时**（有 live .git）运行：:

    python3 scripts/generate_scm_manifest.py [--out evidence/scm_manifest.json]

生成的文件随 wheel/容器发布；运行期证据校验在没有 .git 的部署里读它，不再直接
调 git。没有 .git（本地编辑环境）→ 显式报错，不生成半成品 manifest（fail-closed）。

也提供 ``--no-write`` 只打印内容（CI 诊断用）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence.scm_manifest import (  # noqa: E402
    SCM_MANIFEST_PATH,
    generate_scm_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=SCM_MANIFEST_PATH,
        help="manifest 输出路径（默认 evidence/scm_manifest.json）",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="只打印 JSON，不写文件",
    )
    args = parser.parse_args()
    try:
        data = generate_scm_manifest(out_path=args.out)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.no_write:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
