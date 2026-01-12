import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_pipeline import StockPredictionPipeline
from ai_tickers import AI_SECTOR_TICKERS
from config import Config
import time

def calculate_metrics(y_true, y_pred, price_current):
    """
    RMSE, MAE, Directional Accuracyを計算
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 方向一致率 (Directional Accuracy)
    # 実際の変化
    actual_diff = y_true - price_current
    # 予測の変化
    pred_diff = y_pred - price_current
    
    # 符号が同じなら正解 (0の場合は除外あるいは正解扱い)
    # ここでは変化なし(0)は不一致とする厳しめの判定
    hits = np.sign(actual_diff) == np.sign(pred_diff)
    accuracy = np.mean(hits)
    
    return rmse, mae, accuracy

def run_simple_validation(tickers):
    """AI銘柄全件での単純時系列分割検証"""
    print("\n=== Standard Temporal Split Validation (All AI Tickers) ===")
    results = []
    
    for ticker in tickers:
        print(f"Testing {ticker}...", end=" ")
        pipeline = StockPredictionPipeline()
        pipeline.ticker = ticker
        
        try:
            # 1. データ準備
            df = pipeline.fetch_data()
            if df.empty:
                print("Skipped (No Data)")
                continue
                
            df = pipeline.create_features(df, dropna=True)
            
            # 手動スプリット (pipeline.runと同じロジックだが検証用にデータを保持)
            split_idx = int(len(df) * (1 - Config.TEST_SIZE))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            if len(test_df) < 10:
                print("Skipped (Not enough test data)")
                continue

            # 特徴量抽出
            features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            X_train, y_train = train_df[features], train_df['Target']
            X_test, y_test = test_df[features], test_df['Target']
            
            # 2. 学習
            # verbose=-1 to silence lightgbm
            pipeline.model = pipeline.train(X_train, y_train)
            
            # 3. 予測
            preds = pipeline.model.predict(X_test)
            
            # 4. 評価
            # 基準価格（その日のClose）を取得して方向的中率を計算
            # y_test[t] は Close[t+1] なので、基準は Close[t] になるはずだが
            # create_featuresで、各行 t には Close[t] が入っており、Target[t] = Close[t+1]
            # したがって、test_df['Close'] が「今日の価格」
            current_prices = test_df['Close'].values
            
            rmse, mae, acc = calculate_metrics(y_test.values, preds, current_prices)
            
            print(f"RMSE: {rmse:.2f}, Acc: {acc:.2%}")
            results.append({
                'Ticker': ticker,
                'RMSE': rmse,
                'MAE': mae,
                'Accuracy': acc,
                'Samples': len(test_df)
            })
            
        except Exception as e:
            print(f"Error: {e}")

    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Summary Results ---")
        print(res_df.describe())
        print(f"\nAverage Directional Accuracy: {res_df['Accuracy'].mean():.2%}")
        return res_df
    return pd.DataFrame()

def run_walk_forward_validation(ticker):
    """
    1銘柄に対する厳密なバックテスト (Walk-Forward Validation)
    過去データのみを使って再学習を繰り返しながら予測する
    """
    print(f"\n=== Walk-Forward Validation (Backtest) for {ticker} ===")
    print("Simulating trading year by year... (This may take a minute)")
    
    pipeline = StockPredictionPipeline()
    pipeline.ticker = ticker
    df = pipeline.fetch_data()
    df = pipeline.create_features(df, dropna=True)
    
    # データ期間の確認
    dates = df.index
    print(f"Data Period: {dates[0].date()} to {dates[-1].date()}")
    
    # 最初の学習期間 (例: 最初の2年)
    start_train_size = int(len(df) * 0.5) 
    # ステップサイズ (例: 1ヶ月ごとに再学習、あるいはもっと粗く3ヶ月)
    # 高速化のため、今回は「1年単位」でスライディングするか、
    # シンプルに「時系列クロスバリデーション (5分割)」を行う
    
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4) # 4分割 = 進行度 20%, 40%, 60%, 80% で区切る
    
    fold = 0
    metrics = []
    
    plt.figure(figsize=(15, 8))
    
    # 全体をプロット準備
    plt.plot(df.index, df['Close'], label='Actual Price', color='gray', alpha=0.3)
    
    features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    for train_index, test_index in tscv.split(df):
        fold += 1
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        X_train, y_train = train_df[features], train_df['Target']
        X_test, y_test = test_df[features], test_df['Target']
        
        # 学習
        model = pipeline.train(X_train, y_train)
        
        # 予測
        preds = model.predict(X_test)
        
        # 評価
        rmse, mae, acc = calculate_metrics(y_test.values, preds, test_df['Close'].values)
        metrics.append({'Fold': fold, 'RMSE': rmse, 'Accuracy': acc})
        
        # プロット
        # test期間の予測線を引く
        plt.plot(test_df.index, preds, label=f'Fold {fold} Pred (Acc: {acc:.1%})')
        
        print(f"Fold {fold}: Test Period {test_df.index[0].date()}~{test_df.index[-1].date()} | RMSE: {rmse:.2f} | Acc: {acc:.2%}")

    plt.title(f"Walk-Forward Validation: {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_result.png')
    print("Backtest plot saved to 'backtest_result.png'")
    
    # スコア平均
    avg_acc = np.mean([m['Accuracy'] for m in metrics])
    print(f"\nAverage Backtest Accuracy: {avg_acc:.2%}")

def main():
    # 1. AI銘柄全体での統計検証
    run_simple_validation(AI_SECTOR_TICKERS)
    
    # 2. 代表銘柄 (ソフトバンクG) での可視化検証
    run_walk_forward_validation("9984.JP")

if __name__ == "__main__":
    main()
