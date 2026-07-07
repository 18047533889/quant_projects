# GTJA-191 Alpha 156
# source: (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)

(max(rank(ts_decay_linear(ts_delta(((high + low + close) / 3), 5), 3)), rank(ts_decay_linear(((ts_delta(((open * 0.15) + (low *0.85)), 2) / ((open * 0.15) + (low * 0.85))) * -1), 3))) * -1)
