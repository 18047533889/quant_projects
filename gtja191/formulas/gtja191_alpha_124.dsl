# GTJA-191 Alpha 124
# source: (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)

(close - col('vwap')) / ts_decay_linear(rank(ts_max(close, 30)), 2)
