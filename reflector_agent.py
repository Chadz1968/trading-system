"""
Reflector Agent: Logs every trade with the agent's reasoning at entry,
then runs an end-of-day LLM post-mortem to extract lessons for tomorrow.

Trade log is stored in trade_log.json (one entry per trade).
Daily summaries are appended to daily_summaries.json.
"""

import json
import os
from datetime import date, datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from openai import OpenAI

from config import API_KEY, SECRET_KEY, OPENAI_KEY

TRADE_LOG = "trade_log.json"
SUMMARY_LOG = "daily_summaries.json"


def _load_json(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_json(path: str, data: list) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def log_trade(trade: dict, order: dict) -> None:
    """
    Persist a trade entry combining the risk agent's trade dict and
    the Alpaca order confirmation. Called immediately after order submission.
    order must contain 'id' (parent bracket order ID) and 'target_price'.
    """
    entry = {
        "date": str(date.today()),
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": trade["symbol"],
        "side": trade["side"],
        "shares": trade["shares"],
        "entry_price": trade["entry_price"],
        "stop_price": trade["stop_price"],
        "target_price": order.get("target_price"),
        "gap_pct": trade["gap_pct"],
        "volume_ratio": trade.get("volume_ratio"),
        "rsi": trade.get("rsi"),
        "catalyst": trade.get("catalyst", ""),
        "dollar_risk": trade["dollar_risk"],
        "order_id": order.get("id"),   # parent bracket order ID
        "exit_price": None,
        "exit_reason": None,
        "pnl": None,
        "outcome": "open",
    }
    log = _load_json(TRADE_LOG)
    log.append(entry)
    _save_json(TRADE_LOG, log)
    print(f"[Reflector] Logged trade: {entry['symbol']} {entry['side']} {entry['shares']} shares")


def update_exit(order_id: str, exit_price: float, exit_reason: str) -> None:
    """Update a trade record with exit price, reason, and P&L after close."""
    log = _load_json(TRADE_LOG)
    for entry in log:
        if entry.get("order_id") == order_id and entry["outcome"] == "open":
            entry["exit_price"] = exit_price
            entry["exit_reason"] = exit_reason
            multiplier = 1 if entry["side"] == "buy" else -1
            entry["pnl"] = round((exit_price - entry["entry_price"]) * entry["shares"] * multiplier, 2)
            entry["outcome"] = "win" if entry["pnl"] > 0 else "loss"
            break
    _save_json(TRADE_LOG, log)


def _collect_todays_trades() -> list[dict]:
    today = str(date.today())
    return [t for t in _load_json(TRADE_LOG) if t.get("date") == today]


def _fetch_exit_fills(client: TradingClient, open_trades: list[dict]) -> dict[str, tuple[float, str]]:
    """
    For each open trade, fetch its parent bracket order from Alpaca and look for
    a filled child leg (stop or take-profit).  Falls back to scanning all filled
    orders for a same-symbol market order in the opposite direction (flatten case).

    Returns {parent_order_id: (exit_price, reason)} where reason is one of
    'stop', 'target', or 'flatten'.
    """
    results: dict[str, tuple[float, str]] = {}

    # Build a symbol → list of all filled orders map for the flatten fallback.
    try:
        all_orders = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200)
        )
    except Exception as e:
        print(f"[Reflector] Could not fetch order list: {e}")
        all_orders = []

    fills_by_symbol: dict[str, list] = {}
    for o in all_orders:
        if str(o.status) == "filled" and o.filled_avg_price:
            fills_by_symbol.setdefault(o.symbol, []).append(o)

    for trade in open_trades:
        parent_id = trade.get("order_id")
        if not parent_id:
            continue

        # --- Primary: check bracket child legs ---
        try:
            parent = client.get_order_by_id(parent_id)
            for leg in (parent.legs or []):
                if str(leg.status) == "filled" and leg.filled_avg_price:
                    order_type = str(leg.order_type).lower()
                    reason = "stop" if "stop" in order_type else "target"
                    results[parent_id] = (float(leg.filled_avg_price), reason)
                    break
            if parent_id in results:
                continue
        except Exception as e:
            print(f"[Reflector] Could not fetch parent order {parent_id}: {e}")

        # --- Fallback: flatten — look for opposite-side market fill ---
        opposite_side = "sell" if trade["side"] == "buy" else "buy"
        for o in fills_by_symbol.get(trade["symbol"], []):
            if str(o.side).lower() == opposite_side and str(o.order_type).lower() == "market":
                results[parent_id] = (float(o.filled_avg_price), "flatten")
                break

    return results


def close_day() -> dict:
    """
    Called at end of trading day:
      1. Reconciles open trade records with Alpaca fill prices
      2. Runs LLM post-mortem
      3. Appends summary to daily_summaries.json
    Returns the summary dict.
    """
    print("[Reflector] Running end-of-day reconciliation...")

    client = TradingClient(API_KEY, SECRET_KEY, paper=True)

    log = _load_json(TRADE_LOG)
    open_trades = [t for t in log if t["outcome"] == "open"]
    exits = _fetch_exit_fills(client, open_trades)

    for entry in open_trades:
        parent_id = entry.get("order_id")
        if parent_id and parent_id in exits:
            exit_price, exit_reason = exits[parent_id]
            update_exit(parent_id, exit_price, exit_reason)

    trades_today = _collect_todays_trades()
    summary = _build_summary(trades_today)
    summary["insights"] = _run_postmortem(trades_today)

    summaries = _load_json(SUMMARY_LOG)
    summaries.append(summary)
    _save_json(SUMMARY_LOG, summaries)

    print(f"[Reflector] Day closed. Trades={summary['total_trades']} | P&L=${summary['total_pnl']}")
    print(f"[Reflector] Insights: {summary['insights'][:120]}...")
    return summary


def _build_summary(trades: list[dict]) -> dict:
    closed = [t for t in trades if t["outcome"] != "open"]
    total_pnl = round(sum(t["pnl"] for t in closed if t["pnl"] is not None), 2)
    wins = sum(1 for t in closed if t["outcome"] == "win")
    losses = sum(1 for t in closed if t["outcome"] == "loss")
    return {
        "date": str(date.today()),
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(closed), 2) if closed else 0,
        "total_pnl": total_pnl,
    }


def _run_postmortem(trades: list[dict]) -> str:
    """Ask the LLM to analyse today's trades and suggest improvements."""
    if not trades:
        return "No trades today — nothing to analyse."

    trade_lines = "\n".join(
        f"- {t['symbol']} {t['side'].upper()} gap={t['gap_pct']*100:+.1f}% "
        f"RSI={t.get('rsi','?')} vol_ratio={t.get('volume_ratio','?')} "
        f"outcome={t['outcome']} pnl=${t.get('pnl','?')} "
        f"catalyst: {t.get('catalyst','')[:60]}"
        for t in trades
    )

    prompt = (
        "You are a trading coach reviewing today's gap-and-momentum trades.\n\n"
        f"Today's trades:\n{trade_lines}\n\n"
        "In 3-5 bullet points: what worked, what didn't, and one specific rule "
        "to add or adjust for tomorrow's session."
    )

    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
