# Gap-Fade Strategy — Research Findings

**Period of research:** April 2025  
**Account:** Alpaca paper trading, $100,000 starting equity  
**Universe:** S&P 500 constituents (current list — survivorship bias present, noted below)

---

## What We Tested

A **gap-fade** strategy: when an S&P 500 stock gaps up or down ≥ N% at the open, trade *against* the gap on the assumption the move is an overreaction and the stock partially reverts during the session.

**Entry filters applied at open:**
- Gap threshold (varied: 2%–6%)
- Volume ratio ≥ 1.5× the 20-day prior average (yesterday's volume, no lookahead)
- RSI not overbought/oversold (< 80 long, > 20 short)

**Exit configurations tested:**
- Fixed stop-loss + fixed take-profit (multiple stop/target combinations)
- `prev_close` as the take-profit target (100% gap fill)
- EOD close (OHLC simulation, no intraday exit)

**Slippage model:** 10 bps per side (entry and exit), flat.

---

## Profit Factor Trajectory

Profit factor was the primary metric throughout — it is more robust than raw return because it captures both win rate and win/loss ratio and is not distorted by position sizing.

| Period | Config | Profit Factor | Notes |
|---|---|---|---|
| 2022–2023 (earlier runs) | Fade, gap ≥ 4%, stop 4%, target 2% | ~1.26 | Appeared to show edge |
| 2023 in-sample (optimizer) | Best fade config found | 1.31 (peak) | Optimised, not live |
| 2024 H1 out-of-sample | Same best configs | 1.02–1.16 | Sharp decay vs in-sample |
| 2024 full year | Fade, gap ≥ 4%, stop 4%, target 2% | 1.00 | Edge fully consumed by slippage |

**The trajectory is degrading over time.** A profit factor that declines from ~1.26 to 1.00 across successive out-of-sample periods is not a stable edge — it is a signal that the apparent edge is either eroding or was never large enough to survive real-world frictions.

---

## Critical Finding: Gap Fill Rate

The core hypothesis of the gap-fade strategy is that stocks revert toward their previous close after a large gap. We tested this directly by setting `prev_close` as the take-profit target.

**Result (2024 full year, gap ≥ 4%, fade):**

| Exit type | Count | % of trades |
|---|---|---|
| Stop hit (gap continued) | 251 | 57.8% |
| Target hit (gap filled) | 37 | **8.5%** |
| EOD close (neither) | 146 | 33.6% |

**The gap filled back to `prev_close` in fewer than 1-in-10 trades.** Stocks that gap ≥ 4% on the S&P 500 continue in the gap direction (or stay near the open) far more often than they retrace. The fade hypothesis — that these moves are overreactions — is not supported by 2024 data at this timeframe.

For context: the 57.8% stop rate on shorts (longs hit their upside stop) means the gap *continued* more than it faded in the majority of trades. This is a momentum signal, not a mean-reversion signal.

---

## Slippage as Edge Killer

Even before the `prev_close` test, the fixed-target fade showed:

- **Pre-slippage P&L:** approximately +$44,000 (gross alpha exists)
- **Slippage cost:** approximately −$43,400 across 434 trades
- **Net result:** +$970 (+0.97%), profit factor 1.00

10 bps per side on ~434 trades consumed the *entire* edge. Any real implementation would also face:
- Market impact (larger than 10 bps on illiquid opens)
- Bid-ask spread at the open (often wider than during the session)
- Partial fills on bracket orders

The gross signal exists but is too thin to survive execution costs at S&P 500 scale.

---

## Optimizer Results (In-Sample 2023, OOS 2024 H1)

A 200-combination grid search was run (see `optimizer_results.json`). Parameters swept:

- Gap threshold: 2%, 3%, 4%, 5%, 6%
- Stop: 1%, 2%, 3%, 4%
- Target: 1%, 2%, 3%, 4%, prev_close
- Direction: momentum or fade

**Top 10 in-sample configs and their OOS profit factors:**

| Rank | Gap | Stop | Target | Fade | IS PF | OOS PF |
|---|---|---|---|---|---|---|
| 1 | 5% | 4% | 1% | Y | 1.311 | 0.951 |
| 2 | 6% | 4% | 1% | Y | 1.294 | 1.011 |
| 3 | 4% | 4% | 1% | Y | 1.283 | 1.022 |
| 4 | **6%** | **3%** | **1%** | **Y** | **1.266** | **1.123** |
| 5 | 5% | 3% | 1% | Y | 1.265 | 1.013 |
| 6 | **6%** | **2%** | **1%** | **Y** | **1.261** | **1.161** |
| 7 | 4% | 3% | 1% | Y | 1.250 | 1.070 |
| 8 | 6% | 1% | 1% | Y | 1.238 | 1.026 |
| 9 | 5% | 2% | 1% | Y | 1.215 | 1.071 |
| 10 | 5% | 4% | 4% | Y | 1.208 | 0.770 |

**Key observations:**
- Every top-10 in-sample config is fade. Momentum produced no top-10 entries.
- The target is 1% in 9 of the top 10 — tight, asymmetric exits.
- In-sample PF consistently overstates OOS PF. The drop ranges from modest (rank 6: 1.261 → 1.161) to severe (rank 10: 1.208 → 0.770).
- The two most stable configs OOS are rank 4 (gap 6%, stop 3%) and rank 6 (gap 6%, stop 2%). Both use 6% gap threshold, suggesting wider gaps have a marginally more reliable fade signal — but PF 1.12–1.16 after slippage leaves almost no real margin.

**No configuration survived walk-forward validation with a profit factor meaningfully above 1.0.**

---

## Methodology Notes

### Walk-Forward Validation

All optimizer runs used a strict train/test split:
- **In-sample (train):** 2023 calendar year
- **Out-of-sample (test):** 2024 H1 (Jan–Jun)

Parameters were selected solely on in-sample performance; OOS was run once per config, never used to re-select. This is the minimum credible validation for a strategy optimised on historical data.

A single OOS period is still limited. Proper validation would use rolling walk-forward windows (e.g. optimise on year N, test on year N+1, repeat across 5+ years). The current data suggests the IS→OOS decay is real and not period-specific.

### Known Limitations

1. **Survivorship bias:** The universe is the *current* S&P 500 list. Stocks that were delisted or demoted between 2022–2024 are absent. This overstates historical performance — stocks that gapped badly and were later removed from the index are missing from the loss column.

2. **Open price approximation:** Entry uses the bar's `open` price + 10 bps slippage. In practice, the open auction on a gapping stock is volatile; fills are often materially worse than the printed open.

3. **Single market regime:** 2023–2024 was a strongly trending bull market. Gap-fade strategies tend to work better in range-bound, high-volatility regimes. Results may not generalise to other conditions.

4. **No transaction costs beyond slippage:** Brokerage commissions and SEC fees are not modelled (minor at current rates but non-zero).

---

## Conclusion

The gap-fade strategy on S&P 500 stocks does not produce a reliable edge at the parameters and timeframe tested:

- The gross signal is marginal (~1.26 PF at peak) and declines over time
- Slippage at realistic rates consumes the entire net edge
- The core assumption (gaps revert) is not supported — gaps fill to `prev_close` in only ~8.5% of cases
- Walk-forward validation shows consistent IS→OOS profit factor decay

**This is not a failure — it is a valid research result.** Ruling out a strategy variant saves real capital that would otherwise be deployed into a losing system. The infrastructure built (backtesting engine, parameter optimizer, walk-forward framework) is reusable for any future strategy hypothesis.

### What to Test Next (if returning to this codebase)

- **Different universe:** Small/mid-cap stocks or ETFs, where gaps are more likely to be noise-driven overreactions
- **Multi-day hold:** Allowing positions to revert over 2–5 days rather than forcing same-day closure
- **Volatility regime filter:** Only trade the fade when VIX > 20 (high-volatility environments where mean-reversion is stronger)
- **Earnings gap isolation:** Restrict to non-earnings gaps (earnings gaps have specific dynamics that may distort the signal)

---

*Backtest engine: `backtest_engine.py` | Optimizer: `optimizer.py` | Raw results: `optimizer_results.json` | Full trade log: `../backtest_results.json` (4 MB, git-ignored if large)*
