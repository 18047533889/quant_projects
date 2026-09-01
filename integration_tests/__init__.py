"""Task #107 — cross-package contract integration tests.

The per-package contract work (A/C/D/E/F/G) lives in each package's own test
suite; this package is the *integration harness* that round-trips the shared
contracts THROUGH the public namespaces of the packages so CI catches contract
drift (a rename / a frozen field unfrozen / a Ref turned mutable / a digest
regression) at the cross-package seam.

The harness imports PUBLIC package namespaces only — never implementation
internals of another package.  The A股 PIT/unit contract is owned FP-side (task
D); this package's ``test_a_share_pit_unit_contract.py`` runs the ROUND-TRIP of
that contract and is written to keep working AFTER D's identity split lands.
"""
