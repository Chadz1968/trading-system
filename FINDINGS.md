# Gap-Fade Strategy — Research Findings

**Research period:** April 2025  
**Universe:** S&P 500 current constituents  
**Account:** Alpaca paper, $100,000 starting equity  
**Detailed archive:** `gap-fade-archive/` (engine, optimizer, full results)

---

## Verdict

The gap-fade edge is real in earlier data and vanishes by 2024. Slippage consumes
what survives. Not tradeable at S&P 500 scale with a flat 10 bps cost model.

---

## Profit Factor Trajectory

Profit factor (gross wins ÷ gross losses) was the primary metric — more stable
than raw return and insensitive to position sizing choices.

| Period | Configuration | Profit Factor |
|---|---|---|
| 2022 (earlier data) | Fade ≥ 4%, stop 4%, target 2% | ~1.26 |
| 2023 in-sample (optimizer best) | Fade ≥ 5–6%, stop 2–3%, target 1% | 1.26–1.31 |
| 2024 H1 out-of-sample | Same configs, unseen data | 1.02–1.16 |
| 2024 full year | Fade ≥ 4%, stop 4%, target 2% | **1.00** |

The trajectory is unambiguous: an edge that started at ~1.26 decays to 1.00 over
three successive years. Any profit factor below ~1.20 before costs is not
deployable — there is no margin for slippage, spread, or market impact.

---

## The Gap Fill Rate Discovery

The core hypothesis: stocks that gap ≥ 4% at the open are overreacting and will
partially retrace during the session. We tested this directly by setting the
take-profit target at `prev_close` — the full gap fill level.

**Result: 2024 full year, 434 trades, gap ≥ 4%, fade direction**

| Exit | Count | Rate |
|---|---|---|
| Stop hit (gap continued) | 251 | 57.8% |
| Target hit (gap fully filled) | 37 | **~8.5%** |
| EOD close (neither) | 146 | 33.6% |

Fewer than 1-in-10 trades reached `prev_close`. The gap *continued* in the
original direction in the majority of trades — the opposite of what a fade
strategy requires. This falsifies the thesis at the S&P 500 / same-day timeframe.

Even the fixed 2% target (half the gap on average) showed the same pattern:
gross alpha of ~$44,000 across 434 trades, consumed almost entirely by ~$43,400
in slippage at 10 bps per side. Net result: +$970, profit factor 1.00.

---

## Optimizer Results (200 Combinations)

Grid swept: gap threshold × stop % × target % × direction × prev_close flag.
In-sample: 2023. Out-of-sample: 2024 H1. Data fetched once; no re-use of OOS.

Top in-sample configs and their OOS validation:

| Rank | Gap | Stop | Target | IS PF | OOS PF |
|---|---|---|---|---|---|
| 1 | 5% | 4% | 1% | 1.311 | 0.951 |
| 2 | 6% | 4% | 1% | 1.294 | 1.011 |
| 3 | 4% | 4% | 1% | 1.283 | 1.022 |
| 4 | 6% | 3% | 1% | 1.266 | 1.123 |
| 5 | 5% | 3% | 1% | 1.265 | 1.013 |
| **6** | **6%** | **2%** | **1%** | **1.261** | **1.161** |
| 7 | 4% | 3% | 1% | 1.250 | 1.070 |
| 8 | 6% | 1% | 1% | 1.238 | 1.026 |
| 9 | 5% | 2% | 1% | 1.215 | 1.071 |
| 10 | 5% | 4% | 4% | 1.208 | 0.770 |

All top-10 in-sample configs were fade. Momentum produced no top-10 entries.
The best OOS performer (rank 6, PF 1.161) still leaves almost no margin after
real-world execution costs. IS→OOS decay is consistent across all configs.

Full results: `gap-fade-archive/optimizer_results.json`

---

## What We Learned About Walk-Forward Validation

Running backtests on the same data you used to select parameters inflates
apparent performance — the parameters have been fitted to noise in that period.
Walk-forward validation catches this:

1. **Optimise on in-sample data** — find the best parameter set
2. **Run once on unseen out-of-sample data** — measure true generalisation
3. **Never loop back** — if OOS disappoints, the answer is to stop, not to
   re-optimise and try again on the same OOS window

In this research, every config that looked strong in-sample (PF 1.24–1.31) showed
meaningful decay out-of-sample (PF 1.00–1.16). The consistent IS→OOS drop is
evidence that the in-sample signal is partly noise, not that the OOS period was
unlucky. A genuinely robust edge would show stable PF across both windows.

The infrastructure is reusable for any future strategy: `optimizer.py` runs a
full walk-forward sweep in one command once data is loaded.

---

## Known Limitations of This Research

- **Survivorship bias:** Universe is current S&P 500. Removed stocks (typically
  underperformers) are absent — this overstates historical performance.
- **Open price model:** Entry uses the printed open + 10 bps. Gap opens are
  volatile; real fills are often materially worse.
- **Single regime:** 2022–2024 was a strong bull market. Gap-fade may behave
  differently in range-bound or high-volatility regimes (VIX > 25).
- **No multi-leg costs:** SEC fees and commissions not modelled.

---

## If Returning to This Codebase

Strategies worth testing with the existing engine and optimizer:

- **Different universe:** Small/mid-cap, where gap moves are more likely noise
- **Earnings filter:** Exclude earnings-gap days — these have different mechanics
- **VIX regime filter:** Only fade when VIX > 20 (mean-reversion stronger in high vol)
- **Multi-day hold:** Allow the position to revert over 2–5 sessions rather than
  forcing intraday closure

The live trading system (bracket orders, reflector, scheduler) is production-ready
for paper testing any new strategy that produces the same `trade dict` interface.
