# GTJA-191 Alpha 110
# source: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100

ts_sum(max(0,high-delay(close,1)),20)/ts_sum(max(0,delay(close,1)-low),20)*100
