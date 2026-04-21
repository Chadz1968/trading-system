# Multi-Agent S&P 500 Day Trading System

## Project Goal
Build a self-improving, modular multi-agent system for S&P 500 day trading using a Gap & Momentum strategy, running on Alpaca Paper Trading.

---

## Strategy
- **Market:** S&P 500 Equities
- **Edge:** Gap & Momentum — trade stocks with significant overnight gaps backed by news catalysts and volume confirmation
- **Risk Parameters:**
  - Max account drawdown: 5% total (hard stop — no trades if breached)
  - Risk per trade: 1% of account equity
  - Max trades per day: 5

---

## Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.14 | Core runtime |
| alpaca-py | Paper trading API + market data |
| OpenAI API (gpt-4o-mini) | LLM catalyst analysis + post-mortems |
| pandas | Data handling |
| python-dotenv | Secret management |
| VS Code | IDE |

---

## Project Structure
```
trading-system/
├── .env                    ← your secrets (never committed)
├── .env.example            ← template
├── .gitignore
├── config.py               ← loads env vars + global constants
├── requirements.txt
├── finder_agent.py         ← scans S&P 500 for gap candidates
├── filter_agent.py         ← validates technicals, enforces trade cap
├── risk_agent.py           ← sizes positions, enforces drawdown stop
├── reflector_agent.py      ← logs trades, runs LLM post-mortems
├── main.py                 ← orchestrates the full pipeline
├── trade_log.json          ← created at runtime
└── daily_summaries.json    ← created at runtime
```

---

## Agent Pipeline

```
Finder → Filter → Risk → Place Orders → Reflector Log
                                              ↓
                                     End-of-Day Post-Mortem (LLM)
```

### Finder Agent (`finder_agent.py`)
- Fetches live S&P 500 ticker list from Wikipedia
- Pulls daily bars from Alpaca in batches of 100
- Flags stocks with `abs(gap%) >= 2%` vs previous close
- Calls `gpt-4o-mini` to hypothesize news catalysts and rate each setup

### Filter Agent (`filter_agent.py`)
- Rejects if today's volume < 1.5x the 20-day average
- Rejects longs with RSI > 80 or shorts with RSI < 20
- Stops passing new candidates once `MAX_TRADES_PER_DAY` is reached

### Risk Agent (`risk_agent.py`)
- Checks account equity and triggers a **hard stop** if drawdown ≥ 5%
- Sizes each position: `shares = floor((equity × 1%) / (entry × 2%))`
- Sets stop price 2% from entry
- Submits market orders via Alpaca paper trading

### Reflector Agent (`reflector_agent.py`)
- Logs every trade immediately after submission with full agent reasoning
- At end of day, reconciles fills from Alpaca and calculates P&L
- Runs an LLM post-mortem and appends insights to `daily_summaries.json`

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure secrets
```bash
cp .env.example .env
# Edit .env and add your keys
```

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
OPENAI_API_KEY=...
```

### 3. Run
```bash
# Morning scan — find, filter, size, and place trades
python main.py

# End of day — reconcile fills and run post-mortem
python main.py --eod
```

---

## Project Rules
1. **Security first** — all secrets via `.env` through `config.py`. Never hardcode keys.
2. **Modularity** — each agent is a standalone script; debug them independently.
3. **Feedback loop** — every trade is logged with the agent's reasoning at entry time.
4. **No live trading** — Alpaca paper trading only until the system proves consistent.

---

## Completed Milestones
- [x] Project structure, `.env` security, `.gitignore`
- [x] `config.py` with all global constants
- [x] Finder Agent — gap scan + LLM catalyst analysis
- [x] Filter Agent — volume ratio, RSI validation, trade cap
- [x] Risk Agent — position sizing, drawdown hard stop, order submission
- [x] Reflector Agent — trade logging, P&L reconciliation, LLM post-mortem
- [x] `main.py` orchestrator with `--eod` flag

## Next Steps
- [ ] Backtest gap-and-momentum signals on historical data
- [ ] Add bracket orders (take-profit target at 2:1 R/R)
- [ ] Feed prior day's Reflector insights into the Finder prompt
- [ ] Add a dashboard / daily email summary
