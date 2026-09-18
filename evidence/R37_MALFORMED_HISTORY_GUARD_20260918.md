# R37 invalid history-row guard

Reproduction: negative, fractional (0.5 / 2.5), and boolean own-history rows
were silently truncated/clamped into stateless or finite-window incremental
modes. Four regression cases failed before the repair.

Both classifier and resolved incremental contract now require a non-boolean
nonnegative Integral value. Invalid history falls back to full replay, never
a reduced finite read window. NumPy integer values retain Integral support.

Evidence:
- evidence/r37-malformed-history-before.log: 4 failed, 3 passed.
- evidence/r37-malformed-history-fixed.log: 41 passed (R44 incremental plus
  financial revision histories), including negative/fractional/bool/NaN/Inf/
  malformed string declarations.
No factor formula changes, eligibility promotion, or production publication.
