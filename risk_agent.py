"""
Risk Agent: Enforces position sizing (1% account risk per trade) and
the 5% total account drawdown hard stop.

For each approved candidate it:
  1. Pulls current account equity from Alpaca
  2. Checks whether cumulative drawdown from peak equity would exceed MAX_DRAWDOWN
  3. Calculates share quantity using a fixed-stop model:
       stop_distance = today_open * STOP_PCT  (default 2% below open for longs)
       shares = floor((equity * RISK_PER_TRADE) / stop_distance)
  4. Returns a trade order dict ready for execution, or rejects if risk rules fail.
"""

import math
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce

from config import API_KEY, SECRET_KEY, BASE_URL, RISK_PER_TRADE, MAX_DRAWDOWN

STOP_PCT    = 0.02  # stop placed 2% from entry
TARGET_MULT = 2.0   # take-profit at 2:1 reward-to-risk (4% from entry)


def _get_account(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "last_equity": float(account.last_equity),
        "portfolio_value": float(account.portfolio_value),
    }


def _drawdown_ok(account: dict) -> tuple[bool, float]:
    """Return (ok, current_drawdown_pct) vs last recorded equity high-water mark."""
    equity = account["equity"]
    # Use last_equity as a proxy for the previous session's equity (Alpaca field)
    peak = max(equity, account["last_equity"])
    drawdown = (peak - equity) / peak if peak > 0 else 0
    return drawdown < MAX_DRAWDOWN, round(drawdown, 4)


def _size_position(equity: float, entry_price: float, is_long: bool) -> tuple[int, float]:
    """
    Calculate share quantity and stop price.
    Dollar risk = equity * RISK_PER_TRADE
    Stop distance = entry_price * STOP_PCT
    Shares = floor(dollar_risk / stop_distance)
    """
    dollar_risk = equity * RISK_PER_TRADE
    stop_distance = entry_price * STOP_PCT
    shares = math.floor(dollar_risk / stop_distance)
    stop_price = round(entry_price * (1 - STOP_PCT) if is_long else entry_price * (1 + STOP_PCT), 2)
    return max(shares, 1), stop_price


def evaluate(candidates: list[dict]) -> list[dict]:
    """
    Evaluate each candidate against risk rules.

    Returns a list of approved trade dicts with order parameters attached.
    Stops ALL trading if drawdown limit is hit.
    """
    print("[Risk] Evaluating position sizes and drawdown...")

    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = _get_account(client)
    equity = account["equity"]

    dd_ok, drawdown = _drawdown_ok(account)
    print(f"[Risk] Equity=${equity:,.2f} | Drawdown={drawdown*100:.2f}% (limit {MAX_DRAWDOWN*100:.0f}%)")

    if not dd_ok:
        print(f"[Risk] HARD STOP — drawdown {drawdown*100:.2f}% exceeds {MAX_DRAWDOWN*100:.0f}%. No trades today.")
        return []

    approved = []
    for stock in candidates:
        symbol = stock["symbol"]
        entry = stock["today_open"]
        is_long = stock["gap_pct"] > 0
        shares, stop_price = _size_position(equity, entry, is_long)

        trade = {
            **stock,
            "side": "buy" if is_long else "sell",
            "shares": shares,
            "entry_price": entry,
            "stop_price": stop_price,
            "dollar_risk": round(shares * abs(entry - stop_price), 2),
            "account_equity": equity,
        }
        approved.append(trade)
        print(
            f"[Risk]   {symbol:6s} {'LONG' if is_long else 'SHORT'} "
            f"{shares} shares @ ${entry} | stop ${stop_price} | risk ${trade['dollar_risk']}"
        )

    return approved


def place_order(trade: dict) -> dict:
    """
    Submit a bracket order to Alpaca paper trading.
    A bracket order submits the entry market order plus two exit legs in one
    request: a stop-loss leg and a take-profit limit leg.  Alpaca monitors both
    legs; whichever fills first cancels the other.

    Returns a minimal dict with order ID and calculated target price so the
    caller can persist them in the trade log.
    """
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    is_long = trade["side"] == "buy"
    side    = OrderSide.BUY if is_long else OrderSide.SELL
    entry   = trade["entry_price"]
    stop    = trade["stop_price"]   # pre-calculated by _size_position
    target  = round(
        entry * (1 + STOP_PCT * TARGET_MULT) if is_long else entry * (1 - STOP_PCT * TARGET_MULT),
        2,
    )

    order_request = MarketOrderRequest(
        symbol=trade["symbol"],
        qty=trade["shares"],
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        stop_loss=StopLossRequest(stop_price=stop),
        take_profit=TakeProfitRequest(limit_price=target),
    )
    order = client.submit_order(order_request)
    direction = "LONG" if is_long else "SHORT"
    print(
        f"[Risk] Bracket order {order.id} — {direction} {trade['shares']} {trade['symbol']} "
        f"| stop=${stop}  target=${target}"
    )
    return {"id": str(order.id), "target_price": target}
