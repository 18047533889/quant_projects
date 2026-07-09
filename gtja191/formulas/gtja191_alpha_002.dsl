# GTJA-191 Alpha 002
# source: (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

-1 * ts_delta((close - low - (high - close)) / (high - low), 1)
