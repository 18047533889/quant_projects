ts_mean((high - low) / (2 * close - high - low) * ts_mean(close, 5) / ts_mean(high - low, 5), 3)
