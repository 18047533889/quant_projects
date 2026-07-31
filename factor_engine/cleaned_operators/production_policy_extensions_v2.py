# -*- coding: utf-8 -*-
"""Production-policy extensions for v2 factor operators."""
from __future__ import annotations

import cleaned_operators.production_hardening as ph
from cleaned_operators import operator_surface as surface

_STATEFUL={
    "KAMA":("last_value","last_timestamp"),
    "Supertrend":("final_upper","final_lower","direction","last_close","last_timestamp"),
    "SupertrendDirection":("final_upper","final_lower","direction","last_close","last_timestamp"),
    "PSAR":("sar","direction","extreme_point","acceleration_factor","prev_high","prev_low","last_timestamp"),
}
ph.FULL_HISTORY_REPLAY_CANONICALS=frozenset(set(ph.FULL_HISTORY_REPLAY_CANONICALS)|set(_STATEFUL));ph.STATEFUL_CHECKPOINTS.update(_STATEFUL)

for _name in surface._FUNDAMENTAL_V2_CANONICALS:ph._SCOPE_OVERRIDES[_name]="fundamental_period"
for _name in surface._STRUCTURE_V2_CANONICALS:ph._SCOPE_OVERRIDES[_name]="ts"
for _name in surface._LIQUIDITY_V2_CANONICALS|surface._TECHNICAL_V2_CANONICALS:ph._SCOPE_OVERRIDES[_name]="ts"
ph._FUNDAMENTAL_PERIOD_CANONICALS=frozenset(set(ph._FUNDAMENTAL_PERIOD_CANONICALS)|set(surface._FUNDAMENTAL_V2_CANONICALS))

# production_tiers may have been imported by an earlier bootstrap module. Refresh
# the frozen surface-derived sets so every newly reviewed Extended canonical is
# subject to the same signature/evidence/backend rules.
import cleaned_operators.production_tiers as pt
_all_extended=set(surface.EXTENDED_ONLY_CANONICALS)
pt.PANDAS_FIRST_PRODUCTION_CANONICALS=frozenset(set(pt.PANDAS_FIRST_PRODUCTION_CANONICALS)|_all_extended)
pt.PANDAS_FIRST_SIGNATURE_GATED=frozenset(set(pt.PANDAS_FIRST_SIGNATURE_GATED)|_all_extended)
pt.PANDAS_FIRST_BACKEND_PINNED=frozenset(set(pt.PANDAS_FIRST_BACKEND_PINNED)|_all_extended)
