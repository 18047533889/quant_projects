# GTJA-191 Alpha 038
# source: (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)

where(ts_sum(high, 20) / 20 < high, -1 * ts_delta(high, 2), 0)
