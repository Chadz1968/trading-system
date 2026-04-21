import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# "paper" only — live mode is not supported yet. Change here when ready.
TRADING_MODE = "paper"

GAP_THRESHOLD = 0.02        # 2% minimum gap to qualify
MAX_TRADES_PER_DAY = 5      # filter agent cap
RISK_PER_TRADE = 0.01       # 1% of account per trade
MAX_DRAWDOWN = 0.05         # 5% total account drawdown stop


def get_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(API_KEY, SECRET_KEY, paper=(TRADING_MODE == "paper"))
