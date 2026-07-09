# GTJA-191 Alpha 057
# source: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)

ts_ema((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 5)
