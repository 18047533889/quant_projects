# R43 KM equilibrium root-selection diagnosis

The pre-repair Polars implementation selected the first downward drift crossing.
The canonical contract selects the stable root nearest the current state center
and also recognizes bracketed exact-zero plateaus.

On the deterministic deepening fixture (seed 2, 200 rows, two columns):

- finite output pairs: 365
- NaN-mask mismatches: 0
- materially different values: 22
- maximum absolute difference: 1.9100906279313756
- maximum relative difference: 1.8731658203475061
- all 22 mismatches had exactly two downward crossings
- crossing denominator absolute min/median/max: 0.4181865540 / 1.4150681061 / 2.0131847140
- near-degenerate denominators (`<= 1e-4`): 0

This was a semantic root-selection defect, not floating-point tolerance or poor
conditioning. Historical `ts_km_equilibrium_distance` values produced by the
old Polars path are affected and must be recomputed.

Post-repair verification:

- focused independent contract suite: 4 passed
- complete deepening Polars parity: 1 passed
