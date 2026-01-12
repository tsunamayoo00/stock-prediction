import yfinance as yf
import pandas as pd

def check_ticker_info(ticker_symbol):
    print(f"--- Checking {ticker_symbol} ---")
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Info (Fundamentals)
    try:
        info = ticker.info
        print("keys in info:", list(info.keys())[:10]) # Show first 10 keys
        print("Sector:", info.get('sector'))
        print("Industry:", info.get('industry'))
        print("PE Ratio:", info.get('trailingPE'))
        print("Market Cap:", info.get('marketCap'))
    except Exception as e:
        print("Info fetch failed:", e)

    # 2. Financials (Income Statement)
    try:
        fin = ticker.financials
        if not fin.empty:
            print("\nFinancials (Head):\n", fin.head(3))
        else:
            print("\nFinancials is empty")
    except Exception as e:
        print("Financials fetch failed:", e)

    # 3. News
    try:
        news = ticker.news
        if news:
            print(f"\nNews count: {len(news)}")
            print("Latest Headline:", news[0].get('title'))
        else:
            print("\nNo news found")
    except Exception as e:
        print("News fetch failed:", e)

if __name__ == "__main__":
    check_ticker_info("7203.T") # Toyota
    print("\n" + "="*30 + "\n")
    check_ticker_info("AAPL")   # Apple
