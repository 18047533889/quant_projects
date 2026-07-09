# GTJA-191 Alpha 027
# source: WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)

WMA((close - ts_delay(close, 3)) / (ts_delay(close, 3) + 1e-08) * 100 + (close - ts_delay(close, 6)) / (ts_delay(close, 6) + 1e-08) * 100, 12)
