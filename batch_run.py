import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_pipeline import StockPredictionPipeline
from sector_map import get_universe_tickers, get_ticker_info
from data_manager import DataManager

# 並列数 (Stooqへの負荷を考慮し4程度に)
MAX_WORKERS = 4

def process_ticker(ticker, db_info=None):
    """
    1銘柄の処理を行うワーカ関数
    """
    db = DataManager()
    
    # 完了済みかチェック (今日の日付で予測が存在するか)
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    if db.is_predicted(ticker, today_str):
        return f"SKIP: {ticker} (Already done)"

    # DBから属性情報取得
    if db_info is not None:
        name_jp = db_info.get('name_jp') or ticker
        sector = db_info.get('sector') or "Unknown"
    else:
        # DB情報がない場合のフォールバック
        info = get_ticker_info(ticker)
        name_jp = info["Name_JP"]
        sector = info["Sector"]

    try:
        # パイプライン実行
        pipeline = StockPredictionPipeline()
        pipeline.ticker = ticker
        
        results = pipeline.run()
        
        prediction = results["prediction"]
        # In classification mode, 'rmse' is meaningless (or logloss). We can set it to 0 or use confidence.
        rmse = results.get("rmse", 0)
        
        # DB保存 (predicted_price column will store Probability 0.0~1.0)
        db.save_prediction(
            date=prediction["date"],
            ticker=ticker,
            name_jp=name_jp,
            sector=sector,
            current=prediction["current"],
            predicted=prediction["next"], # This is Up Probability
            rmse=rmse
        )
        return f"DONE: {name_jp} (Up Prob: {prediction['next']:.1%})"
        
    except Exception as e:
        return f"ERROR: {ticker} ({str(e)})"


from ai_tickers import AI_SECTOR_TICKERS

# ... (MAX_WORKERS etc)

def run_batch():
    print("Fetching universe from database...")
    db = DataManager()
    df_tickers = db.get_all_tickers()
    
    if df_tickers.empty:
        print("No tickers found in DB. Please run 'fetch_jpx_tickers.py' first.")
        return

    # [AI検証モード] ユーザー要望によりAI関連銘柄のみにフィルタリング
    print(f"🔍 AI Verification Mode: Filtering for {len(AI_SECTOR_TICKERS)} AI-related stocks.")
    df_tickers = df_tickers[df_tickers['ticker'].isin(AI_SECTOR_TICKERS)]
    
    if df_tickers.empty:
        # DBにAI銘柄がない場合は強制的にリストを使う（fetch_jpxで取れてない場合など）
        # ただし属性情報が取れないので、最低限のDFを作成
        print("Warning: AI tickers not found in DB list. Using raw list.")
        df_tickers = pd.DataFrame({'ticker': AI_SECTOR_TICKERS, 'name_jp': ['AI-Stock']*len(AI_SECTOR_TICKERS), 'sector': ['AI']*len(AI_SECTOR_TICKERS)})

    # 属性情報を辞書化してワーカに渡す
    tickers = df_tickers['ticker'].tolist()
    db_info_map = df_tickers.set_index('ticker').to_dict(orient='index')
    
    total = len(tickers)
    print(f"Starting batch prediction for {total} tickers with {MAX_WORKERS} workers...")
    print("Press Ctrl+C to stop. Progress is saved automatically.")
    
    # 並列実行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Future オブジェクトのリスト作成
        future_to_ticker = {
            executor.submit(process_ticker, ticker, db_info_map.get(ticker)): ticker 
            for ticker in tickers
        }
        
        count = 0
        for future in as_completed(future_to_ticker):
            count += 1
            result = future.result()
            print(f"[{count}/{total}] {result}")
            
    print("Batch processing complete!")

if __name__ == "__main__":
    run_batch()
