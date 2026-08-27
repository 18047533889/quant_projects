"""Post-fix tag-governance verification (FIX-1..12). Reads the live registry after load_all().
    Mirrors /tmp/cross_check.py category logic, adapted to current governance semantics.
"""
import json, re, sys
from collections import defaultdict

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import load_all

load_all()

cat = OperatorRegistry.catalog()
canon = OperatorRegistry.list_canonical()
anom = defaultdict(list)

def add(cls, c, detail):
    anom[cls].append((c, detail))

for c in canon:
    e = cat.get(c, {})
    tags = [str(t) for t in (e.get('tags') or ())]
    backends = list(OperatorRegistry.backends_for(c))
    status = str(e.get('status') or '').lower()
    ig = (e.get('input_grain') or '').lower()
    og = (e.get('output_grain') or '').lower()
    scope = str(e.get('scope') or '').lower()
    pit_safe = e.get('pit_safe')
    st = e.get('stateful')
    det_ver = e.get('determinism_verified')
    ou = (e.get('output_unit') or '').lower()

    # FIX-1 backend:* per registered backend
    if not any(str(t).startswith('backend:') for t in tags):
        add('missing_backend_tag', c, f"backends={backends}")
    # every registered backend must have a tag
    for b in backends:
        if f"backend:{b}" not in tags:
            add('backend_tag_missing_for_registered', c, f"backend={b}")

    # FIX-3 cost band clamp 0..8
    for t in tags:
        m = re.fullmatch(r'cost:(\d+)', str(t))
        if m and (int(m.group(1)) < 0 or int(m.group(1)) > 8):
            add('cost_out_of_range', c, f"tag={t}")

    # FIX-4 grain_minute_to_daily where declared minute->daily
    if ig == 'minute' and og == 'daily' and 'grain_minute_to_daily' not in tags:
        add('grain_minute_to_daily_missing', c, f"{ig}->{og}")
    # implied surface: if grain tag present, must have minute + daily_agg
    if 'grain_minute_to_daily' in tags:
        if 'minute' not in tags or 'daily_agg' not in tags:
            add('grain_without_minute_dailyagg', c, f"tags={tags}")

    # FIX-5/7 causal + pit_safe on causal-by-construction scope
    if scope in {'elementwise', 'cs', 'group'}:
        if 'causal' not in tags:
            add('causal_scope_no_tag', c, f"scope={scope}")
        if 'pit_safe' not in tags:
            add('pitsafe_scope_no_tag', c, f"scope={scope}")

    # FIX-9 pit_safe where field True
    if pit_safe is True and 'pit_safe' not in tags:
        add('pitsafe_field_no_tag', c, f"pit_safe={pit_safe}")

    # FIX-6 deterministic where determinism_verified True (only add direction)
    if det_ver is True and 'deterministic' not in tags:
        add('deterministic_verified_no_tag', c, f"determinism_verified={det_ver}")

    # FIX-8 stateful where field True
    if st is True and 'stateful' not in tags:
        add('stateful_field_no_tag', c, f"stateful={st}")

    # FIX-10 unit:ratio where output_unit ratio
    if ou and 'ratio' in ou and 'unit:ratio' not in tags:
        add('unit_ratio_missing', c, f"output_unit={ou}")

    # FIX-9 source_blocked tag on production ops
    if 'source_blocked' in tags and status == 'production':
        add('source_blocked_production', c, f"status={status}")

    # FIX-11 daily conflict on the two intraday ops
    if c in {'intraday_activity_duration_curvature', 'intraday_impact_decay_rate'}:
        if 'daily' in tags:
            add('intraday_daily_conflict', c, f"daily in tags")

    # empty tags FIX-12 (production ops must not be empty)
    if status == 'production' and not tags:
        add('empty_tags_production', c, f"backends={backends}")

print("=== POST-FIX ANOMALY COUNTS ===")
for cls in sorted(anom, key=lambda k: -len(anom[k])):
    print(f"{cls}: {len(anom[cls])}")
if not anom:
    print("ALL ZERO — 12 tag fixes verified against live catalog")
json.dump({k: v for k, v in anom.items()}, open('/tmp/anomalies_after_fix.json', 'w'), indent=1)