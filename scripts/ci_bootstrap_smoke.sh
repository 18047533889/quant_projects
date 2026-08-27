#!/usr/bin/env bash
# P0-01 CI_BOOTSTRAP smoke — hard-fail environment/checkout validation.
#
# Runs from a generic shell with NO assumptions about the repo being checked
# out (it must be).  Every step is fail-closed: a missing-binary, broken
# checkout, or unpreparable interpreter fails the job (exit != 0) — never a
# silent skip.  This is the FIRST gate of the tiered pipeline.
#
# Exit 0 only if ALL of the following are true:
#   checkout present, .github/workflows/ci.yml present,
#   python3 interpreter exists, prints a version,
#   cwd resolves to the checkout (contains a .git dir),
#   HEAD resolves and is printed.
# The final line MUST be `CI_BOOTSTRAP_PASS` on stdout so the workflow's
# CI_BOOTSTRAP job can grep it and fail hard otherwise.
set -euo pipefail

echo "==> P0-01 CI_BOOTSTRAP smoke (fail-closed; no silent skips)"

# 1. CWD must be the checkout root.
if [ ! -d .git ]; then
    echo "FATAL: CWD '$(pwd)' has no .git — not a checkout root" >&2
    exit 1
fi
echo "==> checkout root: $(pwd)"

# 2. The workflow file that defines this gate must be present (self-check).
if [ ! -f .github/workflows/ci.yml ]; then
    echo "FATAL: .github/workflows/ci.yml missing from checkout" >&2
    exit 1
fi
echo "==> ci.yml present"

# 3. python3 interpreter must exist and be runnable.
if ! command -v python3 >/dev/null 2>&1; then
    echo "FATAL: python3 not on PATH" >&2
    exit 1
fi
if ! python3 --version >/dev/null 2>&1; then
    echo "FATAL: python3 exists but does not run" >&2
    exit 1
fi
echo "==> python3: $(python3 --version)"

# 4. HEAD must resolve.
if ! head_sha=$(git rev-parse HEAD 2>/dev/null); then
    echo "FATAL: git rev-parse HEAD failed — broken checkout" >&2
    exit 1
fi
echo "==> HEAD: ${head_sha}"

# 5. Workspace must be inspectable.
echo "==> workspace top-level entries:"
ls -1 | head -40

# 6. Workflow triggers must be sane (push on main + PR + manual).
if grep -q "push:" .github/workflows/ci.yml \
   && grep -q "pull_request:" .github/workflows/ci.yml \
   && grep -q "workflow_dispatch:" .github/workflows/ci.yml; then
    echo "==> triggers present (push/pull_request/workflow_dispatch)"
else
    echo "FATAL: expected push/pull_request/workflow_dispatch triggers" >&2
    exit 1
fi

echo "CI_BOOTSTRAP_PASS"