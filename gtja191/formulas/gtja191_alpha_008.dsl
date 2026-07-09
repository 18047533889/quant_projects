# GTJA-191 Alpha 008
# source: RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)

rank(ts_delta((high + low) / 2 * 0.2 + (high + low + close) / 3 * 0.8, 4) * -1)
