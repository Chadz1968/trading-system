"""
Filter Agent: Validates technical quality of gap candidates and
enforces a daily trade-count cap to prevent over-trading.

Checks:
  - Volume is at least 1.5x the 20-day average (confirms genuine interest)
  - RSI(14) is not in extreme overbought/oversold territory (40-80 for longs, 20-60 for shorts)
  - Daily trade count has not hit MAX_TRADES_PER_DAY
"""

from datetime import date, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import API_KEY, SECRET_KEY, MAX_TRADES_PER_DAY


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0  # neutral if insufficient data
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _get_technicals(symbol: str, client: StockHistoricalDataClient) -> dict:
    """Fetch 30 days of daily bars and compute volume ratio and RSI."""
    today = date.today()
    start = today - timedelta(days=45)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=today,
    )
    bars = client.get_stock_bars(request)[symbol]
    if not bars or len(bars) < 5:
        return {"volume_ratio": 0, "rsi": 50}

    volumes = [b.volume for b in bars]
    closes = [b.close for b in bars]

    avg_volume_20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes[:-1]) / max(len(volumes) - 1, 1)
    today_volume = volumes[-1]
    volume_ratio = round(today_volume / avg_volume_20, 2) if avg_volume_20 > 0 else 0

    rsi = _compute_rsi(closes)
    return {"volume_ratio": volume_ratio, "rsi": rsi}


def _passes_technicals(stock: dict, technicals: dict) -> tuple[bool, str]:
    """Return (passes, reason) based on volume and RSI rules."""
    vr = technicals["volume_ratio"]
    rsi = technicals["rsi"]
    gap_pct = stock["gap_pct"]
    is_long = gap_pct > 0

    if vr < 1.5:
        return False, f"volume ratio {vr:.2f} < 1.5 (weak confirmation)"

    if is_long and rsi > 80:
        return False, f"RSI {rsi} overbought for long setup"
    if not is_long and rsi < 20:
        return False, f"RSI {rsi} oversold for short setup"

    return True, f"volume_ratio={vr:.2f}, RSI={rsi}"


def run(candidates: list[dict], trades_today: int = 0) -> list[dict]:
    """
    Filter gap candidates by technical quality and trade-count cap.

    Args:
        candidates: Output list from finder_agent.run()
        trades_today: Number of trades already taken today

    Returns:
        List of approved candidates, each enriched with technical details.
    """
    print("[Filter] Validating technical signals...")

    if trades_today >= MAX_TRADES_PER_DAY:
        print(f"[Filter] Daily trade cap ({MAX_TRADES_PER_DAY}) reached. No new trades.")
        return []

    slots_remaining = MAX_TRADES_PER_DAY - trades_today
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    approved = []

    for stock in candidates:
        if len(approved) >= slots_remaining:
            break
        symbol = stock["symbol"]
        try:
            tech = _get_technicals(symbol, client)
            passes, reason = _passes_technicals(stock, tech)
            stock.update(tech)
            if passes:
                approved.append(stock)
                print(f"[Filter]   PASS {symbol:6s} | {reason}")
            else:
                print(f"[Filter]   FAIL {symbol:6s} | {reason}")
        except Exception as e:
            print(f"[Filter]   SKIP {symbol:6s} | error: {e}")

    print(f"[Filter] {len(approved)} candidates passed.")
    return approved
