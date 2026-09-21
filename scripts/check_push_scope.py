#!/usr/bin/env python3
"""Git pre-push guard for server-c; checks object trees, not remote nicknames."""
import os
import sys
from pathlib import Path

from push_both import ROOT, SyncError, git, sanitize, validate_publication


def main():
    try:
        if len(sys.argv) != 3:
            raise SyncError("expected Git pre-push remote name and URL")
        repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        # Hooks may inherit Git variables belonging to a mirror, not ROOT.
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            os.environ.pop(key, None)
        source = git(ROOT, "rev-parse", "refs/heads/main^{commit}")
        for line in sys.stdin:
            fields = line.split()
            if len(fields) != 4:
                raise SyncError("malformed pre-push update")
            _, commit, ref, _ = fields
            validate_publication(repo, ROOT, source, sys.argv[2], commit, ref)
        return 0
    except (SyncError, OSError) as exc:
        print("PUSH BLOCKED: " + sanitize(str(exc)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
