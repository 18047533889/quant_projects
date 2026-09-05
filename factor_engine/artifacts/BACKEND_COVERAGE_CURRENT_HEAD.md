# Backend coverage — CURRENT HEAD (regenerated 2026-09-05 11:53)

> MACHINE-GENERATED from a live registry load at current HEAD (R61 refresh of
> `artifacts/preflight/rebuild_inventory.py` outputs).  Do not edit by hand.
> Previous plane (HEAD `7628674b…`, production_admitted=0 fail-closed) is
> superseded: the two root causes recorded there (certification_source suffix
> mismatch, R47 hash validation) were fixed in R59/R60, and this file is the
> new authoritative current-HEAD plane.

## Load plane (authoritative)

- **HEAD:** `593507c1e25288e78ba95a5d7acc6bfef57daf1f`
- **Working tree:** DIRTY (R61 working state; see `git status` for the live list)
- **Load mode:** `load_all(include_research=False)`
- **canonical_count:** **1689**

## Counts (current HEAD)

| metric | count |
|---|---:|
| total canonicals | **1689** |
| lifecycle: production | 137 |
| lifecycle: experimental | 1548 |
| lifecycle: deprecated | 1 |
| lifecycle: research | 3 |
| surface: daily | 1242 |
| surface: extended | 431 |
| surface: research | 5 |
| surface: internal | 3 |
| surface: unsafe | 7 |
| surface: legacy | 1 |
| `production_certified is True` (catalog six-gate) | 86 |
| **production_admitted** | **72** |
| **directly_usable** | **72** |
| mining_visible | 1626 |
| composition_usable | 1625 |
| terminal_usable | 1461 |

## r64 causality (current HEAD)

| causal | 1582 |
| leak_detected | 16 |
| unknown | 91 |

## Direct-use ladder (R18 authority)

| metric | count |
|---|---:|
| **production_admitted** | **72** |
| **directly_usable** | **72** |
| context_admitted | 0 |

**production_admitted = 72** at current HEAD.  The R59/R60 root-cause fixes
(certification_source suffix matching + R47 64-hex strict validation fallback +
spec_completion back-fill) restored the certification fast path; the admitted
set now flows through the full five-gate chain (registry valid / causal / field
contract / backend parity / lookback contract).

## Evidence binding

- `evidence/CURRENT.json`: all artifacts CURRENT against the live tree
  (R22-CURRENT-TRUTH protocol, producer-executed binds).
- This file regenerates from `artifacts/preflight/rebuild_inventory.py` output
  (inventory_manifest.json + direct_use_matrix.json + operator_matrix.csv).

