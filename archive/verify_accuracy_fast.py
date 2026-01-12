import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from model_pipeline import StockPredictionPipeline

def calculate_metrics(y_true, y_pred, price_current):
    """RMSE, MAE, Directional Accuracy"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Directional Accuracy (Strict)
    actual_diff = y_true - price_current
    pred_diff = y_pred - price_current
    hits = np.sign(actual_diff) == np.sign(pred_diff)
    accuracy = np.mean(hits)
    
    return rmse, mae, accuracy

def run_walk_forward_validation(ticker):
    print(f"\n=== Walk-Forward Validation (Backtest) for {ticker} ===")
    
    pipeline = StockPredictionPipeline()
    pipeline.ticker = ticker
    # Fetch data (might skip if recent enough)
    try:
        df = pipeline.fetch_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data.")
        return

    df = pipeline.create_features(df, dropna=True)
    print(f"Data Period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} rows)")
    
    # Time Series Split (5 folds)
    # This respects time order: Train on past, Test on future segment
    tscv = TimeSeriesSplit(n_splits=5)
    
    features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Price', color='gray', alpha=0.5)
    
    fold = 0
    accuracies = []
    
    for train_idx, test_idx in tscv.split(df):
        fold += 1
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        X_train, y_train = train_df[features], train_df['Target']
        X_test, y_test = test_df[features], test_df['Target']
        
        # Train
        model = pipeline.train(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Evaluate
        # Use test_df['Close'] as "Current Price" (at time t) to compare with y_test (Target at t+1)
        rmse, mae, acc = calculate_metrics(y_test.values, preds, test_df['Close'].values)
        accuracies.append(acc)
        
        # Log to file
        with open("validation_results.txt", "a") as f:
            f.write(f"Fold {fold}: {test_df.index[0].date()}~{test_df.index[-1].date()} | Acc: {acc:.2%} (RMSE: {rmse:.1f})\n")
        
        # Plot segment
        plt.plot(test_df.index, preds, label=f'Pred (Fold {fold})')

    avg_acc = np.mean(accuracies)
    print(f"\nAverage Directional Accuracy: {avg_acc:.2%}")
    with open("validation_results.txt", "a") as f:
        f.write(f"\nAverage Directional Accuracy: {avg_acc:.2%}\n")
    
    plt.title(f"Walk-Forward Backtest: {ticker} (Avg Acc: {avg_acc:.1%})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'backtest_{ticker}.png')
    print(f"Plot saved: backtest_{ticker}.png")

def main():
    # Clear log
    with open("validation_results.txt", "w") as f:
        f.write("Validation Results\n")
        
    # Run for SoftBank (Representative AI)
    run_walk_forward_validation("9984.JP")

if __name__ == "__main__":
    main()
