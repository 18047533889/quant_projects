"""Fix SQL backend coverage for EWM/Wilder family (planned changes).

Two coordinated edits:
1. sql_tiers.py — remove the EWMA/Wilder family from the subtraction block so
   RSI_WILDER/ATR_WILDER/MACD_*/DEMA/TEMA/PPO_*/PVO_*/TSI_*/Keltner*/ADL/
   ChaikinOscillator/CMF/ForceIndex become SQL-IMPLEMENTED (candidates in
   research/validation, marker registered centrally).
2. sql_pushdown/emitter.py — implement the exact pandas adjust=False ewm
   recursion as a DuckDB recursive-CTE helper and wire it into _ewm_adjust_false_sql,
   the ts_ema/ewm_mean branch, and the ewm_std/ewm_var/ewm_cov/ewm_corr branches;
   remove the family from _SQL_FALLBACK_CANONICALS.

The current emitter uses a finite-window self-join that does NOT match pandas on
NaN-gap inputs (verified: [1,2,nan,4,...] → emitter 0.75 vs pandas 1.5 at the NaN
and 2.375 vs 3.1667 after). The recursive-CTE form matches exactly.

Exact recursion (verified against pandas 2.3.3 cython):
  weighted = x0 (first non-nan); old_wt = 1
  valid step:   old_wt *= (1-a); weighted = (old_wt*weighted + a*cur)/(old_wt + a); old_wt = 1
  NaN step:     old_wt *= (1-a); weighted unchanged (carried)
  min_periods: rows with fewer than minp non-nan obs → NULL.
"""
print("plan written")
