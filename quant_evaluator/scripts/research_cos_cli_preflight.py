"""Credential-free, no-execution check for the configured research COS CLI."""
from __future__ import annotations

import shutil


def preflight_research_cos_cli() -> dict[str, object]:
    """Report whether DataAccess's exact configured CLI entry is executable.

    This only resolves the configured name/path. It does not invoke the CLI,
    inspect credentials, access COS, or disclose the configured value.
    """
    from data_access.cos.mirror import _cli_binary

    configured = _cli_binary()
    present = isinstance(configured, str) and bool(configured.strip())
    resolved = shutil.which(configured) if present else None
    ok = resolved is not None
    return {
        "configured_cli_present": present,
        "configured_cli_resolvable": ok,
        "reason_type": "resolved" if ok else "not_found_or_not_executable",
    }
