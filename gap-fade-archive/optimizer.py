"""
Grid-search optimizer for the gap trading strategy.

Fetches OHLCV data once, then sweeps all parameter combinations over an
in-sample period. The top performers are validated on a held-out
out-of-sample period to guard against overfitting.

Objective metric: profit factor (gross wins / gross losses) — more stable
than raw return and penalises both low win-rate and poor win/loss ratio.

Usage:
    py -3.14 optimizer.py
    py -3.14 optimizer.py --in-sample-start 2023-01-01 --in-sample-end 2023-12-31 \
                          --oos-start 2024-01-01 --oos-end 2024-12-31
    py -3.14 optimizer.py --top-n 15 --min-trades 50
"""

import argparse
import itertools
import json
from datetime import date, timedelta

from config import API_KEY, SECRET_KEY, MAX_TRADES_PER_DAY, RISK_PER_TRADE
from finder_agent import get_sp500_symbols
from backtest_engine import (
    _fetch_all_bars,
    _build_date_index,
    _simulate_day,
    STOP_PCT,
)

# ── Parameter grid ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "gap_threshold":     [0.02, 0.03, 0.04, 0.05, 0.06],
    "stop_pct":          [0.01, 0.02, 0.03, 0.04],
    "target_pct":        [0.01, 0.02, 0.03, 0.04],
    "fade":              [False, True],
    "prev_close_target": [False, True],
}

DEFAULT_MIN_TRADES = 30   # discard configs with too few trades
DEFAULT_TOP_N      = 10   # how many in-sample winners to validate OOS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enum_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _run_period(
    symbol_bars: dict,
    symbol_index: dict,
    days: list[date],
    params: dict,
    initial_equity: float = 100_000.0,
) -> list[dict]:
    """Simulate every day in `days` with `params`. Returns flat trade list."""
    equity, trades = initial_equity, []
    for test_day in days:
        day_trades = _simulate_day(
            test_day, symbol_bars, symbol_index,
            equity,
            params["gap_threshold"],
            MAX_TRADES_PER_DAY,
            RISK_PER_TRADE,
            fade=params["fade"],
            stop_pct=params["stop_pct"],
            target_pct=params["target_pct"],
            prev_close_target=params["prev_close_target"],
        )
        for t in day_trades:
            equity += t["pnl"]
        trades.extend(day_trades)
    return trades


def _metrics(trades: list[dict], initial_equity: float, min_trades: int) -> dict | None:
    if len(trades) < min_trades:
        return None
    pnls    = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers  = [p for p in pnls if p <= 0]
    if not losers:
        return None
    stops   = sum(1 for t in trades if t["exit_reason"] == "stop")
    targets = sum(1 for t in trades if t["exit_reason"] == "target")
    eod     = sum(1 for t in trades if t["exit_reason"] == "eod")
    return {
        "total_trades":     len(trades),
        "win_rate":         round(len(winners) / len(pnls) * 100, 1),
        "profit_factor":    round(sum(winners) / abs(sum(losers)), 3),
        "total_return_pct": round(sum(pnls) / initial_equity * 100, 2),
        "avg_win":          round(sum(winners) / len(winners), 2) if winners else 0,
        "avg_loss":         round(sum(losers)  / len(losers),  2) if losers  else 0,
        "stop_rate":        round(stops   / len(trades) * 100, 1),
        "target_rate":      round(targets / len(trades) * 100, 1),
        "eod_rate":         round(eod     / len(trades) * 100, 1),
    }


def _build_combos() -> list[dict]:
    """
    Enumerate unique parameter combinations.
    When prev_close_target=True, target_pct is overridden — so only one
    target_pct value is needed for that branch (saves ~75 redundant runs).
    """
    combos = []
    for gap, stop, target, fade, prev_close in itertools.product(
        PARAM_GRID["gap_threshold"],
        PARAM_GRID["stop_pct"],
        PARAM_GRID["target_pct"],
        PARAM_GRID["fade"],
        PARAM_GRID["prev_close_target"],
    ):
        if prev_close and target != PARAM_GRID["target_pct"][0]:
            continue  # deduplicate: target_pct irrelevant when prev_close overrides
        combos.append({
            "gap_threshold":     gap,
            "stop_pct":          stop,
            "target_pct":        target,
            "fade":              fade,
            "prev_close_target": prev_close,
        })
    return combos


def _fmt_target(p: dict) -> str:
    return "prev_cl" if p["prev_close_target"] else f"{p['target_pct']*100:.0f}%"


def _print_table_header_is():
    print(f"\n{'#':>3}  {'Gap':>5}  {'Stop':>5}  {'Target':>7}  {'Fade':>4}  "
          f"{'PF':>6}  {'Win%':>5}  {'Ret%':>7}  {'Stop%':>6}  {'Tgt%':>5}  {'Trades':>6}")
    print("─" * 82)


def _print_row_is(rank: int, p: dict, m: dict):
    print(
        f"{rank:>3}  {p['gap_threshold']*100:>4.0f}%  {p['stop_pct']*100:>4.0f}%  "
        f"{_fmt_target(p):>7}  {'Y' if p['fade'] else 'N':>4}  "
        f"{m['profit_factor']:>6.3f}  {m['win_rate']:>5.1f}  "
        f"{m['total_return_pct']:>+7.1f}  {m['stop_rate']:>6.1f}  "
        f"{m['target_rate']:>5.1f}  {m['total_trades']:>6}"
    )


def _print_table_header_oos():
    print(f"\n{'#':>3}  {'Gap':>5}  {'Stop':>5}  {'Target':>7}  {'Fade':>4}  "
          f"{'IS-PF':>6}  {'OOS-PF':>7}  {'OOS-Ret%':>9}  {'OOS-Trades':>10}")
    print("─" * 76)


def _print_row_oos(rank: int, p: dict, is_pf: float, oos_m: dict | None):
    if oos_m is None:
        print(f"{rank:>3}  {p['gap_threshold']*100:>4.0f}%  {p['stop_pct']*100:>4.0f}%  "
              f"{_fmt_target(p):>7}  {'Y' if p['fade'] else 'N':>4}  "
              f"{is_pf:>6.3f}  (too few OOS trades)")
        return
    print(
        f"{rank:>3}  {p['gap_threshold']*100:>4.0f}%  {p['stop_pct']*100:>4.0f}%  "
        f"{_fmt_target(p):>7}  {'Y' if p['fade'] else 'N':>4}  "
        f"{is_pf:>6.3f}  {oos_m['profit_factor']:>7.3f}  "
        f"{oos_m['total_return_pct']:>+9.1f}  {oos_m['total_trades']:>10}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gap strategy parameter optimizer")
    parser.add_argument("--in-sample-start",  default="2023-01-01", help="In-sample start YYYY-MM-DD")
    parser.add_argument("--in-sample-end",    default="2023-12-31", help="In-sample end   YYYY-MM-DD")
    parser.add_argument("--oos-start",        default="2024-01-01", help="OOS start       YYYY-MM-DD")
    parser.add_argument("--oos-end",          default="2024-06-30", help="OOS end         YYYY-MM-DD")
    parser.add_argument("--top-n",    type=int, default=DEFAULT_TOP_N,      help="Top N configs to validate OOS")
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES, help="Min trades to include a config")
    args = parser.parse_args()

    in_start  = date.fromisoformat(args.in_sample_start)
    in_end    = date.fromisoformat(args.in_sample_end)
    oos_start = date.fromisoformat(args.oos_start)
    oos_end   = date.fromisoformat(args.oos_end)

    print("\n" + "=" * 60)
    print("  GAP STRATEGY — PARAMETER OPTIMIZER")
    print("=" * 60)
    print(f"  In-sample:     {in_start} → {in_end}")
    print(f"  Out-of-sample: {oos_start} → {oos_end}")
    print(f"  Min trades:    {args.min_trades}  |  Top-N: {args.top_n}")

    # ── Fetch data once for entire range ──────────────────────────────────────
    buffer_start = in_start - timedelta(days=60)
    fetch_end    = max(in_end, oos_end)
    symbols = get_sp500_symbols()
    print(f"\nLoaded {len(symbols)} S&P 500 symbols.")
    print(f"Fetching bars from {buffer_start} to {fetch_end} (fetched once for all runs)...")

    symbol_bars  = _fetch_all_bars(symbols, buffer_start, fetch_end)
    symbol_index = {sym: _build_date_index(bars) for sym, bars in symbol_bars.items()}
    print(f"Data loaded for {len(symbol_bars)} symbols.\n")

    in_days  = _enum_days(in_start, in_end)
    oos_days = _enum_days(oos_start, oos_end)
    combos   = _build_combos()
    total    = len(combos)

    # ── In-sample grid search ─────────────────────────────────────────────────
    print(f"Sweeping {total} parameter combinations on in-sample period...")
    valid_results = []

    for i, params in enumerate(combos, 1):
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} combinations tested...", end="\r")
        trades  = _run_period(symbol_bars, symbol_index, in_days, params)
        metrics = _metrics(trades, 100_000.0, args.min_trades)
        if metrics:
            valid_results.append({"params": params, "in_sample": metrics})

    valid_results.sort(key=lambda r: r["in_sample"]["profit_factor"], reverse=True)
    top = valid_results[:args.top_n]

    print(f"\n\nIn-sample results: {len(valid_results)}/{total} configs had ≥{args.min_trades} trades.")
    print(f"\nTop {args.top_n} by profit factor (in-sample {in_start} → {in_end}):")
    _print_table_header_is()
    for rank, r in enumerate(top, 1):
        _print_row_is(rank, r["params"], r["in_sample"])

    # ── Out-of-sample validation ───────────────────────────────────────────────
    print(f"\n\nOut-of-sample validation ({oos_start} → {oos_end}):")
    _print_table_header_oos()

    oos_output = []
    for rank, r in enumerate(top, 1):
        params     = r["params"]
        oos_trades = _run_period(symbol_bars, symbol_index, oos_days, params)
        oos_m      = _metrics(oos_trades, 100_000.0, args.min_trades)
        _print_row_oos(rank, params, r["in_sample"]["profit_factor"], oos_m)
        oos_output.append({
            "rank":       rank,
            "params":     params,
            "in_sample":  r["in_sample"],
            "oos":        oos_m,
        })

    # ── Highlight the best OOS performer ──────────────────────────────────────
    oos_valid = [r for r in oos_output if r["oos"] is not None]
    if oos_valid:
        best = max(oos_valid, key=lambda r: r["oos"]["profit_factor"])
        print(f"\n★  Best OOS config (PF {best['oos']['profit_factor']:.3f}):")
        p = best["params"]
        print(f"   --gap {p['gap_threshold']} --stop-pct {p['stop_pct']} "
              f"--target-pct {p['target_pct']} "
              f"{'--fade ' if p['fade'] else ''}"
              f"{'--prev-close-target' if p['prev_close_target'] else ''}")
        print(f"   Return: {best['oos']['total_return_pct']:+.1f}%  "
              f"Win rate: {best['oos']['win_rate']:.1f}%  "
              f"Trades: {best['oos']['total_trades']}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "in_sample_period":  f"{in_start} to {in_end}",
        "oos_period":        f"{oos_start} to {oos_end}",
        "total_combos":      total,
        "valid_combos":      len(valid_results),
        "min_trades_filter": args.min_trades,
        "results":           oos_output,
    }
    with open("optimizer_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to optimizer_results.json")


if __name__ == "__main__":
    main()
