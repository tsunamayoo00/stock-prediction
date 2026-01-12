import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from model_pipeline import StockPredictionPipeline
from ensemble_model import EnsemblePredictor
from config import Config

def calculate_metrics(y_true, y_pred, price_current):
    """
    RMSE, MAE, Directional Accuracy
    Note: Target is 5-day future price.
    Direction is (Target - Current) vs (Pred - Current).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 実際のリターン (5日後 - 現在)
    actual_diff = y_true - price_current
    # 予測リターン (5日後予測 - 現在)
    pred_diff = y_pred - price_current
    
    # 方向一致判定
    hits = np.sign(actual_diff) == np.sign(pred_diff)
    accuracy = np.mean(hits)
    
    return rmse, mae, accuracy

def run_verification(ticker):
    print(f"\n=== v11 Validation (Ensemble + 5-Day) for {ticker} ===")
    print(f"Horizon: {getattr(Config, 'FORECAST_HORIZON', 1)} days")
    
    pipeline = StockPredictionPipeline()
    pipeline.ticker = ticker
    
    # Data Fetch
    try:
        df = pipeline.fetch_data()
        df = pipeline.create_features(df, dropna=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Check Data
    print(f"Data: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} rows)")
    print("Columns:", [c for c in df.columns if 'Nikkei' in c or 'VIX' in c])
    
    # Time Series Split (5 Fold)
    tscv = TimeSeriesSplit(n_splits=5)
    features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'name_jp', 'sector', 'Target_Date']]
    
    accuracies = []
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual', color='gray', alpha=0.3)
    
    fold = 0
    for train_idx, test_idx in tscv.split(df):
        fold += 1
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        X_train, y_train = train_df[features], train_df['Target']
        X_test, y_test = test_df[features], test_df['Target']
        
        # Ensemble Learning
        model = EnsemblePredictor()
        # verbose suppression handled inside class hopefully
        model.fit(X_train, y_train, X_valid=X_test, y_valid=y_test)
        
        # Predict
        preds = model.predict(X_test)
        
        # Eval
        # For evaluation, we compare Prediction vs "Current Price at prediction time"
        # X_test corresponds to time t. y_test is Close[t+5].
        # Current price is Close[t].
        current_prices = df['Close'].iloc[test_idx].values
        
        rmse, mae, acc = calculate_metrics(y_test.values, preds, current_prices)
        accuracies.append(acc)
        
        print(f"Fold {fold}: {test_df.index[0].date()}~ | Acc: {acc:.2%} (RMSE: {rmse:.1f})")
        
        # Plot (Shift prediction to align with Target date if desired, but plotting on 't' is standard for forecasts)
        # Or better: plot preds at t+5. 
        # But simple plot on t is easier to see "Signal".
        plt.plot(test_df.index, preds, label=f'Pred Fold{fold} ({acc:.1%})', linewidth=1)

    avg_acc = np.mean(accuracies)
    print(f"\nAverage Directional Accuracy (5-day): {avg_acc:.2%}")
    
    plt.title(f"v11 Ensemble Backtest (5-day Horizon): {ticker} Acc={avg_acc:.1%}")
    plt.legend()
    plt.savefig(f'verify_v11_{ticker}.png')
    print(f"Saved plot: verify_v11_{ticker}.png")
    
    return avg_acc

if __name__ == "__main__":
    # Test on SoftBank (High Volatility)
    acc_sb = run_verification("9984.JP")
    
    # Test on Toyota (Stable)
    acc_ty = run_verification("7203.JP")
    
    print("\n=== Final Verdict ===")
    print(f"SoftBank (Volatile): {acc_sb:.2%}")
    print(f"Toyota (Stable):   {acc_ty:.2%}")
