# GTJA-191 Alpha 052
# source: SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)* 100

ts_sum(flex_max(0, high - ts_delay((high + low + close) / 3, 1)), 26) / (ts_sum(flex_max(0, ts_delay((high + low + close) / 3, 1) - low), 26) + 1e-08) * 100
