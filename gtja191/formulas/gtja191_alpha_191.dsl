# GTJA-191 Alpha 191
# source: ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)

((ts_corr(ts_mean(volume,20), low, 5) + ((high + low) / 2)) - close)
