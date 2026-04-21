"""
Finder Agent: Scans S&P 500 for stocks with >2% gap at open,
then uses an LLM to identify likely news catalysts.
"""

import json
from datetime import date, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from openai import OpenAI

from config import API_KEY, SECRET_KEY, OPENAI_KEY, GAP_THRESHOLD


def get_sp500_symbols() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    symbols = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    return symbols


def _fetch_bars_in_batches(client: StockHistoricalDataClient, symbols: list[str], start, end, batch_size=100):
    """Fetch daily bars for symbols in batches to avoid API limits."""
    all_bars = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        all_bars.update(bars.data)
    return all_bars


def find_gaps(symbols: list[str], gap_threshold: float = GAP_THRESHOLD) -> list[dict]:
    """Return stocks whose today-open vs prev-close gap exceeds threshold."""
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    today = date.today()
    start = today - timedelta(days=7)   # wide window to ensure we get 2 trading days

    bars_data = _fetch_bars_in_batches(client, symbols, start=start, end=today)

    candidates = []
    for symbol, bars in bars_data.items():
        if len(bars) < 2:
            continue
        prev_close = bars[-2].close
        today_open = bars[-1].open
        gap_pct = (today_open - prev_close) / prev_close
        if abs(gap_pct) >= gap_threshold:
            candidates.append(
                {
                    "symbol": symbol,
                    "prev_close": round(prev_close, 2),
                    "today_open": round(today_open, 2),
                    "gap_pct": round(gap_pct, 4),
                    "volume": bars[-1].volume,
                }
            )

    return sorted(candidates, key=lambda x: abs(x["gap_pct"]), reverse=True)


def analyze_catalyst(symbol: str, gap_pct: float, openai_client: OpenAI) -> str:
    """Ask the LLM to hypothesize the catalyst and rate the setup."""
    direction = "up" if gap_pct > 0 else "down"
    prompt = (
        f"{symbol} is gapping {direction} {abs(gap_pct) * 100:.1f}% today. "
        "What are the most likely news catalysts? "
        "Rate the gap-and-momentum day-trade setup quality 1-10 and briefly explain why. "
        "Keep the response under 3 sentences."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def run(gap_threshold: float = GAP_THRESHOLD, top_n: int = 10) -> list[dict]:
    """
    Main entry point. Returns up to top_n gap candidates enriched with
    LLM catalyst analysis.
    """
    print("[Finder] Scanning S&P 500 for gap opportunities...")
    symbols = get_sp500_symbols()
    print(f"[Finder] Loaded {len(symbols)} symbols.")

    gaps = find_gaps(symbols, gap_threshold)
    print(f"[Finder] {len(gaps)} stocks gapped >{gap_threshold * 100:.0f}%.")

    if not gaps:
        return []

    openai_client = OpenAI(api_key=OPENAI_KEY)
    results = []
    for stock in gaps[:top_n]:
        catalyst = analyze_catalyst(stock["symbol"], stock["gap_pct"], openai_client)
        stock["catalyst"] = catalyst
        results.append(stock)
        direction = "+" if stock["gap_pct"] > 0 else ""
        print(f"[Finder]   {stock['symbol']:6s} {direction}{stock['gap_pct']*100:.1f}% | {catalyst[:80]}...")

    return results


if __name__ == "__main__":
    results = run()
    print(json.dumps(results, indent=2, default=str))
