"""
Scheduler: Automatically fires the trading pipeline on US market weekdays.

Keep this process running during the trading day. It checks the time every
30 seconds and fires each pipeline stage once per day at the right moment.

All internal logic uses US Eastern Time (ET) so it self-corrects for DST on
both sides of the Atlantic — no manual time adjustments needed when clocks change.

Schedule (ET → approximate UK time shown at startup):
  09:31 ET  — morning scan + order placement
  15:55 ET  — flatten all open positions before close
  16:05 ET  — end-of-day reflector + post-mortem

Run with:
    python scheduler.py

Stop with Ctrl+C.  Safe to restart at any point — already-fired stages for the
current day are tracked in memory and won't re-fire.
"""

import time
import zoneinfo
from datetime import datetime

import main as trading_main

_ET = zoneinfo.ZoneInfo("America/New_York")
_UK = zoneinfo.ZoneInfo("Europe/London")

# (ET hour, ET minute, window_minutes, label, function)
_SCHEDULE: list[tuple[int, int, int, str, callable]] = [
    (9,  31, 5, "MORNING SCAN",        trading_main.morning_run),
    (15, 55, 4, "FLATTEN POSITIONS",   trading_main.flatten_run),
    (16,  5, 10, "END OF DAY",         trading_main.eod_run),
]


def _now_et() -> datetime:
    return datetime.now(_ET)


def _now_uk() -> datetime:
    return datetime.now(_UK)


def _fmt_time() -> str:
    et = _now_et()
    uk = _now_uk()
    return f"{et.strftime('%H:%M ET')}  /  {uk.strftime('%H:%M %Z')}"


def _is_weekday() -> bool:
    return _now_et().weekday() < 5  # Mon–Fri; US holidays not filtered


def main() -> None:
    et = _now_et()
    uk = _now_uk()
    print("=" * 60)
    print("  TRADING SYSTEM SCHEDULER")
    print("=" * 60)
    print(f"  Started:     {et.strftime('%A %d %b %Y')}")
    print(f"  Current ET:  {et.strftime('%H:%M')}")
    print(f"  Current UK:  {uk.strftime('%H:%M %Z')}")
    print()
    print("  Daily schedule (ET → UK approximate):")
    for h, m, _, label, _ in _SCHEDULE:
        trigger_et = et.replace(hour=h, minute=m, second=0, microsecond=0)
        trigger_uk = trigger_et.astimezone(_UK)
        print(f"    {h:02d}:{m:02d} ET  ({trigger_uk.strftime('%H:%M %Z')})  —  {label}")
    print()
    print("  Press Ctrl+C to stop.\n")

    fired: set[str] = set()

    while True:
        now = _now_et()
        today = now.strftime("%Y-%m-%d")

        if _is_weekday():
            hhmm = now.hour * 100 + now.minute

            for h, m, window, label, fn in _SCHEDULE:
                trigger_hhmm = h * 100 + m
                end_hhmm     = trigger_hhmm + window
                key          = f"{today}-{label}"

                if trigger_hhmm <= hhmm < end_hhmm and key not in fired:
                    fired.add(key)
                    print(f"\n{'='*60}")
                    print(f"  {_fmt_time()}  —  {label}")
                    print(f"{'='*60}")
                    try:
                        fn()
                    except Exception as exc:
                        print(f"[Scheduler] ERROR in {label}: {exc}")

        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[Scheduler] Stopped at {_fmt_time()}.")
