# R2 Master Queue (compact)

**Updated:** 2026-08-19 · **HEAD:** `bce4c0a9`  
**Workflow:** `DISCOVERED → REPRODUCED → FIXING → LOCAL_TESTED → INDEPENDENT_REVIEW → REGRESSION_TESTED → CLOSED_VERIFIED`  
**History / long narratives:** do not reopen; use evidence YAMLs under `factor_engine/evidence/r2/` and `dataaccess/*MANIFEST.yaml`.

## Active / open (pointers)

| ID | Status | Pointer |
|---|---|---|
| Q2 residency / PlanNode / verifier | LOCAL focused | `Q2-P0-009`, `Q2-P0-019`, `Q2-P0-020*`, `Q2-P0-021-022*`, `Q2-P0-002-verifier*` |
| Q `ts_beta` | quarantined | `Q2-P0-008.yaml` |
| POL2 moment/kurt / rolling | LOCAL focused | `POL2-P0-002`…`004` YAMLs |
| DA catalog identity / PIT | LOCAL focused | `DA2_P0_002_MANIFEST.yaml`, `R2-P0-039_MANIFEST.yaml` |
| FE multi-region physical | LOCAL focused | `R2-P0-059-062-063_manifest.yaml` |
| FA2 adapter / assembly | LOCAL focused | `FA2-P0-003.yaml`, `FA2-P0-004.yaml` |
| QE cache / streaming / mean_ic | LOCAL focused | `quant_evaluator/tests/test_*` |
| CI2 / live COS / real Redis / broad QE | `NOT_RUN` | — |

## Closed-enough locally (not production)

Older R2-P0-032/033/036/037 and OPT2-P0-001 remain regression-tested historically; broad DA / live COS still `NOT_RUN`. See archive if needed.

## Truth

No surface is `CLOSED_VERIFIED` for production. Q remains fail-closed and not production-certified.
