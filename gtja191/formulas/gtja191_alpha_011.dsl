# GTJA-191 Alpha 011
# source: SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)

ts_sum((close - low - (high - close)) / (high - low) * volume, 6)
