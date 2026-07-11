# GTJA-191 Alpha 120
# source: (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))

rank(col('vwap') - close) / rank(col('vwap') + close)
