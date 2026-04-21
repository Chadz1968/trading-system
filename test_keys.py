import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("Testing credentials...")

alpaca_key = os.getenv("ALPACA_API_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print("\n[1] Env vars:")
print(f"    ALPACA_API_KEY:    {'OK' if alpaca_key else 'MISSING'}")
print(f"    ALPACA_SECRET_KEY: {'OK' if alpaca_secret else 'MISSING'}")
print(f"    OPENAI_API_KEY:    {'OK' if openai_key else 'MISSING'}")

if not all([alpaca_key, alpaca_secret, openai_key]):
    print("\nFix missing keys in .env before continuing.")
    exit(1)

print("\n[2] Alpaca trading...")
try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(alpaca_key, alpaca_secret, paper=True)
    account = client.get_account()
    print(f"    OK — status={account.status}  equity=${float(account.equity):,.2f}  buying_power=${float(account.buying_power):,.2f}")
except Exception as e:
    print(f"    FAIL: {e}")

print("\n[3] Alpaca market data...")
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
    quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="AAPL"))
    print(f"    OK — AAPL bid=${quote['AAPL'].bid_price}  ask=${quote['AAPL'].ask_price}")
except Exception as e:
    print(f"    FAIL: {e}")

print("\n[4] OpenAI...")
try:
    from openai import OpenAI
    response = OpenAI(api_key=openai_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
    )
    print(f"    OK — '{response.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"    FAIL: {e}")

print("\n" + "=" * 50)
