ema((high - low) / (close - (high + low) / 2) * volume / ts_mean(volume, 20), 5)
