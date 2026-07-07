# GTJA-191 Alpha 027
# source: WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)

WMA((close - delay(close, 3)) / (delay(close, 3) + 1e-8) * 100 + (close - delay(close, 6)) / (delay(close, 6) + 1e-8) * 100, 12)
