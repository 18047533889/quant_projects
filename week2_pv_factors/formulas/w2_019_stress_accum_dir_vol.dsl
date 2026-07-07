ema(max(0, (high - low - ts_mean(high - low, 40)) / ts_std(high - low, 40)) * min(2, volume / ts_mean(volume, 20)) * clip((high - low) / (close - open), -3, 3), 15)
