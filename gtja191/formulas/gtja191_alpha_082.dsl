# GTJA-191 Alpha 082
# source: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)

ts_ema((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 39)
