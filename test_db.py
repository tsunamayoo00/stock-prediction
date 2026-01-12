from data_manager import DataManager
import pandas as pd

def test_dm():
    print("Initializing DataManager...")
    try:
        db = DataManager()
        print("DB Initialized.")
    except Exception as e:
        print(f"Failed init: {e}")
        return

    print("Checking connection...")
    try:
        with db.get_connection() as conn:
            print("Connection successful.")
    except Exception as e:
        print(f"Failed conn: {e}")
        return

    print("Checking get_all_tickers...")
    try:
        df = db.get_all_tickers()
        print(f"Tickers found: {len(df)}")
    except Exception as e:
        print(f"Failed get_tickers: {e}")

    print("Test Complete.")

if __name__ == "__main__":
    test_dm()
