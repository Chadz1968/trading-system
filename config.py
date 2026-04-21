import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

GAP_THRESHOLD = 0.02        # 2% minimum gap to qualify
MAX_TRADES_PER_DAY = 5      # filter agent cap
RISK_PER_TRADE = 0.01       # 1% of account per trade
MAX_DRAWDOWN = 0.05         # 5% total account drawdown stop
