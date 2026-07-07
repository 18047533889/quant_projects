# GTJA-191 Alpha 120
# source: (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))

(rank((((high + low + close) / 3) - close)) / rank((((high + low + close) / 3) + close)))
