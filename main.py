"""
Main orchestrator — runs the full agent pipeline:
  Finder → Filter → Risk → (place orders) → Reflector log

Usage:
  python main.py           # run morning scan + place trades
  python main.py --eod     # run end-of-day reconciliation + post-mortem
"""

import argparse
import sys

import finder_agent
import filter_agent
import risk_agent
import reflector_agent


def morning_run():
    print("=" * 60)
    print("  TRADING SYSTEM — MORNING RUN")
    print("=" * 60)

    # 1. Find gap candidates
    candidates = finder_agent.run()
    if not candidates:
        print("\n[Main] No gap candidates found. Done.")
        return

    # 2. Filter by technical quality
    filtered = filter_agent.run(candidates)
    if not filtered:
        print("\n[Main] No candidates survived filtering. Done.")
        return

    # 3. Risk-size and drawdown check
    approved = risk_agent.evaluate(filtered)
    if not approved:
        print("\n[Main] No trades approved by risk agent. Done.")
        return

    # 4. Place orders and log each trade
    print(f"\n[Main] Placing {len(approved)} order(s)...")
    for trade in approved:
        try:
            order = risk_agent.place_order(trade)
            reflector_agent.log_trade(trade, order)
        except Exception as e:
            print(f"[Main] Failed to place order for {trade['symbol']}: {e}")

    print("\n[Main] Morning run complete.")


def eod_run():
    print("=" * 60)
    print("  TRADING SYSTEM — END OF DAY")
    print("=" * 60)
    summary = reflector_agent.close_day()
    print("\n--- Daily Summary ---")
    for k, v in summary.items():
        if k != "insights":
            print(f"  {k}: {v}")
    print("\n--- Insights ---")
    print(summary.get("insights", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Trading System")
    parser.add_argument("--eod", action="store_true", help="Run end-of-day reflector instead of morning scan")
    args = parser.parse_args()

    if args.eod:
        eod_run()
    else:
        morning_run()
