from ensemble_model import EnsemblePredictor
from model_pipeline import StockPredictionPipeline
import pandas as pd
import numpy as np

def test_ensemble():
    print("Testing Ensemble Model...")
    pipeline = StockPredictionPipeline()
    pipeline.ticker = "9984.JP" # SoftBank
    
    # Data
    try:
        df = pipeline.fetch_data()
        if df.empty:
            print("No data found.")
            return
            
        df = pipeline.create_features(df, dropna=True)
        
        # Split (Simple temporal split)
        test_size = 0.2
        split_idx = int(len(df) * (1 - test_size))
        
        features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'name_jp', 'sector']]
        X = df[features]
        y = df['Target']
        
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        # Initialize & Fit Ensemble
        ensemble = EnsemblePredictor()
        ensemble.fit(X_train, y_train, X_valid=X_test, y_valid=y_test)
        
        # Predict
        preds = ensemble.predict(X_test)
        
        print("\nPredictions (First 5):", preds[:5])
        print("Actuals (First 5):", y_test.values[:5])
        
        # Simple Eval
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Ensemble RMSE: {rmse:.2f}")
        
        # Compare with single models if possible (internal access)
        print("Model Weights:", ensemble.weights)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ensemble()
