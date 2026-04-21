"""
Backtesting engine for the Gap & Momentum day trading strategy.

Replays the Finder → Filter → Risk pipeline on historical OHLCV data without
placing any real orders.  Exits are simulated from daily OHLC bars:
  - Stop-loss at entry ± STOP_PCT (2%) — conservative: stop wins ties with target
  - Take-profit at 2:1 risk-reward (4% from entry)
  - EOD close if neither level is touched

Known limitations:
  - Survivorship bias: uses today's S&P 500 list, not historical constituents
  - Holiday calendar: weekday filter; no-trade days on US holidays are harmless zeros
  - Slippage model: flat 10bps per side (see SLIPPAGE constant)

Usage:
    python backtest_engine.py --start 2024-01-01 --end 2024-06-30
    python backtest_engine.py --start 2025-01-01 --end 2025-12-31 --equity 50000
    python backtest_engine.py --start 2025-01-01 --end 2025-12-31 --ignore-drawdown-stop
"""

import json
import math
import statistics
import zoneinfo
from datetime import date, datetime, time, timedelta

_ET = zoneinfo.ZoneInfo("America/New_York")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import API_KEY, SECRET_KEY, GAP_THRESHOLD, MAX_TRADES_PER_DAY, RISK_PER_TRADE, MAX_DRAWDOWN
from finder_agent import get_sp500_symbols

STOP_PCT   = 0.02    # mirrors risk_agent.STOP_PCT
TARGET_MULT = 2.0    # 2:1 reward-to-risk ratio
SLIPPAGE   = 0.001   # 10 bps per side (entry and exit)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_one(client: StockHistoricalDataClient, symbol: str, start: date, end: date) -> list | None:
    """Fetch bars for a single symbol; return None and log on failure."""
    try:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
        bars = client.get_stock_bars(req)
        data = bars.data.get(symbol)
        return sorted(data, key=lambda b: b.timestamp) if data else None
    except Exception as exc:
        print(f"    Skip {symbol}: {exc}")
        return None


def _fetch_all_bars(symbols: list[str], start: date, end: date) -> dict[str, list]:
    """
    Fetch daily bars for all symbols over [start, end] in batches of 100.
    If a batch fails (e.g. one invalid symbol), retries each symbol individually
    so one bad ticker doesn't drop 99 good ones.
    Returns {symbol: [Bar, ...]} sorted ascending by timestamp.
    """
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    all_bars: dict[str, list] = {}
    batch_size = 100
    total_batches = math.ceil(len(symbols) / batch_size)

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} symbols)...")
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(req)
            for sym, sym_bars in bars.data.items():
                all_bars[sym] = sorted(sym_bars, key=lambda b: b.timestamp)
        except Exception as exc:
            print(f"  Batch {batch_num} failed ({exc}), retrying individually...")
            skipped = 0
            for sym in batch:
                result = _fetch_one(client, sym, start, end)
                if result:
                    all_bars[sym] = result
                else:
                    skipped += 1
            if skipped:
                print(f"    Skipped {skipped} invalid symbol(s) in batch {batch_num}.")

    return all_bars


def _build_date_index(bars: list) -> dict[date, int]:
    """Map each bar's date → its index in the list for O(1) lookup."""
    return {b.timestamp.date(): idx for idx, b in enumerate(bars)}


def _fetch_minute_bars(
    client: StockHistoricalDataClient, symbol: str, trade_date: date, exit_time: time
) -> list:
    """Fetch 1-min bars for symbol from 9:30 ET to exit_time ET on trade_date."""
    open_dt  = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=_ET)
    close_dt = datetime(trade_date.year, trade_date.month, trade_date.day,
                        exit_time.hour, exit_time.minute, tzinfo=_ET)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=open_dt,
            end=close_dt,
        )
        bars = client.get_stock_bars(req)
        data = bars.data.get(symbol)
        return sorted(data, key=lambda b: b.timestamp) if data else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Indicator helpers (mirror filter_agent logic exactly)
# ---------------------------------------------------------------------------

def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent   = deltas[-period:]
    avg_gain = sum(d for d in recent if d > 0) / period
    avg_loss = sum(-d for d in recent if d < 0) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)


def _volume_ratio(prior_volumes: list[float], compare_vol: float) -> float:
    """
    Ratio of compare_vol to the 20-day average of prior_volumes[-21:-1].
    prior_volumes should end with the most-recent known bar (yesterday).
    compare_vol is the volume being tested (also yesterday in backtest mode).
    """
    hist = prior_volumes[-21:-1] if len(prior_volumes) >= 21 else prior_volumes[:-1]
    avg  = sum(hist) / len(hist) if hist else 0
    return round(compare_vol / avg, 2) if avg > 0 else 0.0


# ---------------------------------------------------------------------------
# Exit helpers
# ---------------------------------------------------------------------------

def _ohlc_exit(
    high: float, low: float, close: float,
    is_long: bool, raw_stop: float, raw_target: float,
) -> tuple[float, str]:
    """Simulate exit from daily OHLC. Stop wins ties with target."""
    if is_long:
        if low  <= raw_stop:   return raw_stop,   "stop"
        if high >= raw_target: return raw_target, "target"
    else:
        if high >= raw_stop:   return raw_stop,   "stop"
        if low  <= raw_target: return raw_target, "target"
    return close, "eod"


def _intraday_exit(
    minute_bars: list, is_long: bool, raw_stop: float, raw_target: float,
) -> tuple[float, str]:
    """Walk minute bars bar-by-bar; return (raw_exit_price, reason).
    'time' means neither level was hit before the cutoff."""
    for bar in minute_bars:
        if is_long:
            if bar.low  <= raw_stop:   return raw_stop,   "stop"
            if bar.high >= raw_target: return raw_target, "target"
        else:
            if bar.high >= raw_stop:   return raw_stop,   "stop"
            if bar.low  <= raw_target: return raw_target, "target"
    return (minute_bars[-1].close if minute_bars else 0.0), "time"


def _trailing_stop_exit(
    minute_bars: list, is_long: bool, entry: float, initial_stop: float, trail_pct: float,
) -> tuple[float, str]:
    """Walk minute bars with a trailing stop.
    The stop starts at initial_stop and ratchets up (long) or down (short) as
    price moves in our favour. Returns (raw_exit_price, reason):
      'stop'  — initial hard stop hit (price never moved in our favour)
      'trail' — trailing stop fired after a favourable move
      'eod'   — held to end of bar window without a stop fire
    """
    trail_stop = initial_stop

    for bar in minute_bars:
        if is_long:
            # Ratchet trail up behind the bar's high
            trail_stop = max(trail_stop, bar.high * (1 - trail_pct))
            if bar.low <= trail_stop:
                reason = "trail" if trail_stop > initial_stop else "stop"
                return trail_stop, reason
        else:
            # Ratchet trail down behind the bar's low
            trail_stop = min(trail_stop, bar.low * (1 + trail_pct))
            if bar.high >= trail_stop:
                reason = "trail" if trail_stop < initial_stop else "stop"
                return trail_stop, reason

    return (minute_bars[-1].close if minute_bars else 0.0), "eod"


# ---------------------------------------------------------------------------
# Per-day simulation
# ---------------------------------------------------------------------------

def _simulate_day(
    test_day: date,
    symbol_bars: dict[str, list],
    symbol_index: dict[str, dict[date, int]],
    day_equity: float,          # equity at start of day — used for ALL sizing
    gap_threshold: float,
    max_trades: int,
    risk_per_trade: float,
    fade: bool = False,         # True = gap-fade (trade against the gap)
    stop_pct: float = STOP_PCT,
    target_pct: float = STOP_PCT * TARGET_MULT,
    exit_time: time | None = None,
    minute_client: StockHistoricalDataClient | None = None,
    trail_pct: float | None = None,
    prev_close_target: bool = False,
) -> list[dict]:
    """
    Run one trading day through the pipeline.  Returns a list of trade dicts.
    All positions are sized from day_equity (pre-open), matching live behaviour
    where all orders are submitted simultaneously at the open.

    fade=True inverts direction: short up-gappers, long down-gappers.
    Adjust stop_pct / target_pct to match the fade setup (typically stop wider
    than target since fade targets mean-reversion, not continuation).
    """
    candidates = []

    for sym, bars in symbol_bars.items():
        date_idx  = symbol_index[sym]
        today_pos = date_idx.get(test_day)
        if today_pos is None or today_pos < 2:
            continue  # need at least two prior bars

        prior_bars  = bars[:today_pos]   # everything strictly before today
        today_bar   = bars[today_pos]

        prev_close  = prior_bars[-1].close
        today_open  = today_bar.open
        today_high  = today_bar.high
        today_low   = today_bar.low
        today_close = today_bar.close

        gap_pct = (today_open - prev_close) / prev_close
        if abs(gap_pct) < gap_threshold:
            continue

        # Momentum: follow the gap. Fade: trade against it.
        is_long = (gap_pct < 0) if fade else (gap_pct > 0)

        # --- Filter: volume ratio (FIX: use yesterday's volume, known at open) ---
        # Compares yesterday's volume against its own 20-day prior average.
        # Avoids lookahead — today's full-day volume is not known at 9:30 AM.
        hist_vols    = [b.volume for b in prior_bars]
        if len(hist_vols) < 5:
            continue
        yesterday_vol = prior_bars[-1].volume
        vol_ratio = _volume_ratio(hist_vols, yesterday_vol)
        if vol_ratio < 1.5:
            continue

        # --- Filter: RSI (uses prior closes only — no lookahead) ---
        closes = [b.close for b in prior_bars]
        rsi = _compute_rsi(closes)
        if is_long and rsi > 80:
            continue
        if not is_long and rsi < 20:
            continue

        candidates.append({
            "symbol":    sym,
            "gap_pct":   gap_pct,
            "is_long":   is_long,
            "prev_close": prev_close,
            "open":      today_open,
            "high":      today_high,
            "low":       today_low,
            "close":     today_close,
            "vol_ratio": vol_ratio,
            "rsi":       rsi,
        })

    # Sort by absolute gap size, cap at daily max
    candidates.sort(key=lambda c: abs(c["gap_pct"]), reverse=True)
    candidates = candidates[:max_trades]

    trades = []
    for c in candidates:
        raw_open = c["open"]
        is_long  = c["is_long"]

        # --- Slippage on entry (FIX: realistic fill vs printed open) ---
        entry = raw_open * (1 + SLIPPAGE) if is_long else raw_open * (1 - SLIPPAGE)

        # --- Risk: position sizing from pre-open equity (no intraday compounding) ---
        dollar_risk   = day_equity * risk_per_trade
        stop_distance = entry * stop_pct
        shares = math.floor(dollar_risk / stop_distance)
        if shares < 1:
            continue

        raw_stop   = entry * (1 - stop_pct) if is_long else entry * (1 + stop_pct)
        raw_target = c["prev_close"] if prev_close_target else (
            entry * (1 + target_pct) if is_long else entry * (1 - target_pct)
        )

        # --- Exit simulation ---
        if minute_client is not None:
            m_bars = _fetch_minute_bars(minute_client, c["symbol"], test_day, exit_time)
            if m_bars:
                if trail_pct is not None:
                    raw_exit, exit_reason = _trailing_stop_exit(m_bars, is_long, entry, raw_stop, trail_pct)
                else:
                    raw_exit, exit_reason = _intraday_exit(m_bars, is_long, raw_stop, raw_target)
                if not raw_exit:
                    continue
            else:
                raw_exit, exit_reason = _ohlc_exit(c["high"], c["low"], c["close"], is_long, raw_stop, raw_target)
        else:
            raw_exit, exit_reason = _ohlc_exit(c["high"], c["low"], c["close"], is_long, raw_stop, raw_target)

        exit_price = raw_exit * (1 - SLIPPAGE) if is_long else raw_exit * (1 + SLIPPAGE)

        pnl = (exit_price - entry) * shares * (1 if is_long else -1)

        trades.append({
            "date":        str(test_day),
            "symbol":      c["symbol"],
            "direction":   "long" if is_long else "short",
            "entry":       round(entry, 4),
            "exit":        round(exit_price, 4),
            "stop":        round(raw_stop, 4),
            "target":      round(raw_target, 4),
            "shares":      shares,
            "pnl":         round(pnl, 2),
            "exit_reason": exit_reason,
            "gap_pct":     round(c["gap_pct"] * 100, 2),
            "vol_ratio":   c["vol_ratio"],
            "rsi":         c["rsi"],
        })

    return trades


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

_RESULTS_FILE = "backtest_results.json"
_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def _append_results(result: dict) -> None:
    """Append result to backtest_results.json. Clears the file if it exceeds 5 MB."""
    import os
    from datetime import timezone

    result["run_timestamp"] = datetime.now(timezone.utc).isoformat()

    history: list[dict] = []
    if os.path.exists(_RESULTS_FILE):
        if os.path.getsize(_RESULTS_FILE) > _MAX_FILE_BYTES:
            print(f"[Results] File exceeds 5 MB — clearing history.")
        else:
            try:
                with open(_RESULTS_FILE) as fh:
                    existing = json.load(fh)
                # Handle old single-object format
                history = existing if isinstance(existing, list) else [existing]
            except (json.JSONDecodeError, OSError):
                history = []

    history.append(result)

    with open(_RESULTS_FILE, "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"[Results] Appended run #{len(history)} to {_RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    start: date,
    end: date,
    initial_equity: float = 100_000.0,
    gap_threshold: float = GAP_THRESHOLD,
    max_trades_per_day: int = MAX_TRADES_PER_DAY,
    risk_per_trade: float = RISK_PER_TRADE,
    max_drawdown: float = MAX_DRAWDOWN,
    enforce_drawdown_stop: bool = True,
    fade: bool = False,
    stop_pct: float = STOP_PCT,
    target_pct: float = STOP_PCT * TARGET_MULT,
    exit_time: time | None = None,
    trail_pct: float | None = None,
    prev_close_target: bool = False,
) -> dict:
    """
    Run the full gap strategy backtest over [start, end].

    Returns a results dict with aggregate metrics and a per-trade log.
    Also writes backtest_results.json.

    enforce_drawdown_stop=False lets the backtest run the full window so you
    can see the full distribution of returns; set to True to match live behaviour.

    fade=True inverts direction (short up-gappers, long down-gappers).
    For fade mode the typical setup is stop_pct=0.04, target_pct=0.02
    (wide stop, tight target) since you're betting on mean-reversion.
    """
    strategy_label = "GAP-FADE" if fade else "GAP-MOMENTUM"
    print(f"\n=== {strategy_label} Backtest: {start} → {end} ===")
    print(f"    Initial equity:       ${initial_equity:,.0f}")
    print(f"    Gap threshold:        {gap_threshold*100:.1f}%  |  Max trades/day: {max_trades_per_day}")
    print(f"    Risk/trade:           {risk_per_trade*100:.1f}%  |  Max drawdown:   {max_drawdown*100:.1f}%")
    print(f"    Stop:                 {stop_pct*100:.1f}%         |  Target:          {target_pct*100:.1f}%")
    print(f"    Slippage:             {SLIPPAGE*10000:.0f}bps per side")
    print(f"    Exit time:            {'EOD (OHLC sim)' if exit_time is None else exit_time.strftime('%H:%M') + ' ET (minute bars)'}")
    if trail_pct is not None:
        print(f"    Trailing stop:        {trail_pct*100:.1f}% (replaces fixed target)")
    if prev_close_target:
        print(f"    Target:               prev_close (mean-reversion to yesterday's close)")
    print(f"    Drawdown stop active: {enforce_drawdown_stop}\n")

    symbols = get_sp500_symbols()
    print(f"Loaded {len(symbols)} S&P 500 symbols.")

    # Fetch all data once — 60-day buffer before start for indicator warmup
    buffer_start = start - timedelta(days=60)
    print(f"Fetching OHLCV bars from {buffer_start} to {end}...")
    symbol_bars = _fetch_all_bars(symbols, buffer_start, end)
    print(f"Data loaded for {len(symbol_bars)} symbols.\n")

    # Trailing stop requires minute bars for the full day if no exit_time given
    if trail_pct is not None and exit_time is None:
        exit_time = time(16, 0)
    minute_client = StockHistoricalDataClient(API_KEY, SECRET_KEY) if (exit_time is not None or trail_pct is not None) else None

    # Pre-build date → index maps for O(1) day lookup
    symbol_index = {sym: _build_date_index(bars) for sym, bars in symbol_bars.items()}

    # Enumerate weekdays in [start, end] as proxy trading-day calendar
    all_days: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            all_days.append(d)
        d += timedelta(days=1)

    equity      = initial_equity
    peak_equity = initial_equity
    all_trades: list[dict] = []

    # Sharpe fix: one return per calendar day (0 on no-trade days)
    daily_return_series: list[float] = []

    for test_day in all_days:
        day_start_equity = equity

        # Hard stop check (FIX: gated by enforce_drawdown_stop flag)
        drawdown = (peak_equity - equity) / peak_equity
        if enforce_drawdown_stop and drawdown >= max_drawdown:
            print(f"{test_day}: HARD STOP — drawdown {drawdown:.1%} >= {max_drawdown:.1%}. Halting.")
            # Fill remaining days with 0 so Sharpe denominator is correct
            remaining = len(all_days) - all_days.index(test_day)
            daily_return_series.extend([0.0] * remaining)
            break

        day_trades = _simulate_day(
            test_day, symbol_bars, symbol_index,
            equity, gap_threshold, max_trades_per_day, risk_per_trade,
            fade=fade, stop_pct=stop_pct, target_pct=target_pct,
            exit_time=exit_time, minute_client=minute_client, trail_pct=trail_pct,
            prev_close_target=prev_close_target,
        )

        day_pnl = 0.0
        for t in day_trades:
            equity      += t["pnl"]
            peak_equity  = max(peak_equity, equity)
            day_pnl     += t["pnl"]
            t["equity_after"] = round(equity, 2)

        all_trades.extend(day_trades)

        # FIX: daily return uses start-of-day equity as denominator
        day_ret = (equity - day_start_equity) / day_start_equity if day_start_equity > 0 else 0.0
        daily_return_series.append(day_ret)

        if day_trades:
            print(
                f"{test_day}: {len(day_trades)} trade(s) | "
                f"day P&L={day_pnl:+.2f} | equity={equity:,.2f}"
            )

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    if not all_trades:
        metrics: dict = {
            "total_trades": 0,
            "note": "No trades passed all filters in the date range.",
        }
    else:
        pnls    = [t["pnl"] for t in all_trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]

        # Max drawdown from equity curve
        peak   = initial_equity
        max_dd = 0.0
        for t in all_trades:
            eq = t["equity_after"]
            peak   = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)

        # Sharpe ratio (annualised) — FIX: includes all days (zeros for flat days)
        if len(daily_return_series) > 1:
            mean_r = sum(daily_return_series) / len(daily_return_series)
            std_r  = statistics.stdev(daily_return_series)
            sharpe = round(mean_r / std_r * math.sqrt(252), 2) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        daily_active = sum(1 for r in daily_return_series if r != 0.0)
        exit_counts  = {"stop": 0, "target": 0, "trail": 0, "eod": 0, "time": 0}
        for t in all_trades:
            exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

        metrics = {
            "period":              f"{start} to {end}",
            "strategy":            strategy_label,
            "stop_pct":            f"{stop_pct*100:.1f}%",
            "target_pct":          f"{target_pct*100:.1f}%",
            "initial_equity":      initial_equity,
            "final_equity":        round(equity, 2),
            "total_pnl":           round(equity - initial_equity, 2),
            "total_return_pct":    round((equity - initial_equity) / initial_equity * 100, 2),
            "total_trades":        len(all_trades),
            "trading_days_active": daily_active,
            "trading_days_total":  len(daily_return_series),
            "win_rate_pct":        round(len(winners) / len(pnls) * 100, 2),
            "avg_win":             round(sum(winners) / len(winners), 2) if winners else 0,
            "avg_loss":            round(sum(losers)  / len(losers),  2) if losers  else 0,
            "profit_factor":       round(sum(winners) / abs(sum(losers)), 2) if losers else float("inf"),
            "max_drawdown_pct":    round(max_dd * 100, 2),
            "sharpe_ratio":        sharpe,
            "slippage_per_side":   f"{SLIPPAGE*10000:.0f}bps",
            "trailing_stop_pct":   f"{trail_pct*100:.1f}%" if trail_pct is not None else "off",
            "prev_close_target":   prev_close_target,
            "exit_time":           "EOD" if exit_time is None else exit_time.strftime("%H:%M ET"),
            "exit_breakdown":      exit_counts,
            "notes": [
                "Volume filter uses yesterday's volume vs prior 20-day avg (no lookahead)",
                "Survivorship bias present: uses current S&P 500 constituents",
            ],
        }

    results = {"metrics": metrics, "trades": all_trades}
    _append_results(results)

    print("\n=== Backtest Complete ===")
    print(json.dumps(metrics, indent=2))
    print("\nFull trade log saved to backtest_results.json")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gap & Momentum backtest engine")
    parser.add_argument("--start",  default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-06-30", help="End date   YYYY-MM-DD")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Starting equity (default: 100000)")
    parser.add_argument("--gap",    type=float, default=GAP_THRESHOLD, help="Gap threshold 0-1 (default: 0.02)")
    parser.add_argument("--max-trades", type=int, default=MAX_TRADES_PER_DAY, help="Max trades/day (default: 5)")
    parser.add_argument(
        "--ignore-drawdown-stop",
        action="store_true",
        help="Disable the 5%% drawdown hard stop so the full window is tested (research mode)",
    )
    parser.add_argument(
        "--fade",
        action="store_true",
        help="Gap-fade mode: short up-gappers, long down-gappers (mean-reversion)",
    )
    parser.add_argument("--stop-pct",   type=float, default=None, help="Stop distance 0-1 (default: 0.02 momentum / 0.04 fade)")
    parser.add_argument("--target-pct", type=float, default=None, help="Target distance 0-1 (default: 0.04 momentum / 0.02 fade)")
    parser.add_argument(
        "--exit-time",
        default=None,
        help="Intraday exit time HH:MM ET (e.g. 10:30). Enables minute-bar simulation. Omit for EOD OHLC simulation.",
    )
    parser.add_argument(
        "--trail-pct",
        type=float,
        default=None,
        help="Trailing stop distance 0-1 (e.g. 0.015 = 1.5%%). Replaces fixed target. Enables full-day minute bars.",
    )
    parser.add_argument(
        "--prev-close-target",
        action="store_true",
        help="Use yesterday's close as the take-profit target (mean-reversion exit). Overrides --target-pct.",
    )
    args = parser.parse_args()

    # Sensible defaults differ by mode
    stop_pct   = args.stop_pct   if args.stop_pct   is not None else (0.04 if args.fade else STOP_PCT)
    target_pct = args.target_pct if args.target_pct is not None else (0.02 if args.fade else STOP_PCT * TARGET_MULT)

    exit_time = None
    if args.exit_time:
        hh, mm = args.exit_time.split(":")
        exit_time = time(int(hh), int(mm))

    run_backtest(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        initial_equity=args.equity,
        gap_threshold=args.gap,
        max_trades_per_day=args.max_trades,
        enforce_drawdown_stop=not args.ignore_drawdown_stop,
        fade=args.fade,
        stop_pct=stop_pct,
        target_pct=target_pct,
        exit_time=exit_time,
        trail_pct=args.trail_pct,
        prev_close_target=args.prev_close_target,
    )
