# -*- coding: utf-8 -*-
"""R28 §八: PUBLIC_RANDOM_FACTOR_TERMINALS == 0 and MINING_RANDOM_OPERATORS == 0.

Random *generation* primitives (rand_*, shuffle, sample, np.random draws) must
never be reachable factor terminals.  Fixed-seed surrogates inside research
statistics are allowed but must not be production-certified terminals.
"""
from __future__ import annotations

import io
import re
import tokenize

import pytest

from cleaned_operators import load_all


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


RANDOM_TERMINAL_PREFIXES = (
    "rand_", "shuffle", "sample", "random_", "monte_carlo",
)
RNG_CALL = re.compile(
    r"np\.random\.(rand|randn|randint|uniform|normal|choice|shuffle|sample|permutation)\s*\("
)


def _code_lines(path):
    try:
        with path.open("rb") as fh:
            toks = list(tokenize.tokenize(fh.readline))
    except Exception:
        return []
    lines = []
    buf, cur = [], 0
    for t in toks:
        if t.type in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT, tokenize.STRING):
            continue
        if t.type == tokenize.NEWLINE:
            line = " ".join(buf).strip()
            if line:
                lines.append((cur, line))
            buf = []
            continue
        if t.type in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        if buf and t.start[0] != cur:
            line = " ".join(buf).strip()
            if line:
                lines.append((cur, line))
            buf = []
        cur = t.start[0]
        buf.append(t.string)
    if buf:
        line = " ".join(buf).strip()
        if line:
            lines.append((cur, line))
    return lines


def test_no_random_named_terminals():
    from cleaned_operators.registry import OperatorRegistry
    import cleaned_operators.operator_surface as surf

    names = set(OperatorRegistry.list_canonical())
    bad = sorted(n for n in names if n.startswith(RANDOM_TERMINAL_PREFIXES))
    assert bad == [], f"random-named canonicals registered: {bad}"
    # also no random terminal reachable via authoring surfaces
    for surf_set_name, surf_set in (
        ("DAILY", surf.DAILY_CANONICALS),
        ("EXTENDED", surf.EXTENDED_ONLY_CANONICALS),
    ):
        bad_surf = sorted(n for n in surf_set if n.startswith(RANDOM_TERMINAL_PREFIXES))
        assert bad_surf == [], f"random-named in {surf_set_name}: {bad_surf}"


def _impl_files(reg):
    import sys
    from pathlib import Path

    files = set()
    for name in reg.list_canonical():
        for backend, op in (reg._operators.get(name, {}) or {}).items():
            mod = type(op).__module__
            src = getattr(sys.modules.get(mod), "__file__", None)
            if src:
                files.add(Path(src))
    return files


def test_no_unseeded_rng_in_operator_implementation():
    """Any np.random draw inside an operator implementation file must be a
    fixed-seed surrogate (deterministic, reproducible), never an unseeded draw
    reachable as a production factor."""
    import sys
    from pathlib import Path

    from cleaned_operators.operator_surface import production_certification
    from cleaned_operators.registry import OperatorRegistry

    reg = OperatorRegistry
    files = _impl_files(reg)

    # 1) no CERTIFIED operator's implementation file may contain an RNG draw.
    cert_bad = []
    for f in sorted(files):
        for lineno, code in _code_lines(f):
            if not RNG_CALL.search(code):
                continue
            # which canonicals live in this file?
            for name in sorted(reg.list_canonical()):
                try:
                    if production_certification(name).name == "CERTIFIED":
                        cert_bad.append((name, str(f), lineno, code[:90]))
                except Exception:
                    pass
    assert cert_bad == [], f"CERTIFIED operator file with RNG draw: {cert_bad}"

    # 2) any RNG draw present in an operator impl file must use default_rng(seed)
    #    (fixed-seed surrogate), not the legacy unseeded global generators.
    unseeded = []
    for f in sorted(files):
        for lineno, code in _code_lines(f):
            if RNG_CALL.search(code):
                if "default_rng" not in code:
                    unseeded.append((str(f), lineno, code[:90]))
    # default_rng() is on a separate line in many files (rng = np.random.default_rng(seed)
    # then rng.draw(...)); only flag draws that never reference an rng object.
    assert unseeded == [], f"unseeded RNG draws in operator impl files: {unseeded}"
