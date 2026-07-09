# GTJA-191 Alpha 098
# source: ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))

where(or_(ts_delta(ts_sum(close, 100) / 100, 100) / (ts_delay(close, 100) + 1e-08) < 0.05, ts_delta(ts_sum(close, 100) / 100, 100) / (ts_delay(close, 100) + 1e-08) == 0.05), -1 * (close - ts_min(close, 100)), -1 * ts_delta(close, 3))
