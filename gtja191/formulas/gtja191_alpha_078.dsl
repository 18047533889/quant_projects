# GTJA-191 Alpha 078
# source: ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))

((high + low + close) / 3 - ts_mean((high + low + close) / 3, 12)) / (0.015 * ts_mean(abs(close - ts_mean((high + low + close) / 3, 12)), 12))
