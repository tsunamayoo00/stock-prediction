import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from model_pipeline import StockPredictionPipeline
from ensemble_model import EnsemblePredictor
from config import Config

# Plot settings
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Meiryo']

def run_verification(ticker="9984.JP"):
    print(f"=== v13 Classification Verification for {ticker} ===")
    
    # 1. Fetch Data
    pipeline = StockPredictionPipeline()
    pipeline.ticker = ticker
    df = pipeline.fetch_data()
    df = pipeline.create_features(df, dropna=True)
    
    # Check Target Distribution
    print(f"Target Distribution (1=Up, 0=Down):")
    print(df['Target'].value_counts(normalize=True))
    
    features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'name_jp', 'sector']]
    X = df[features]
    y = df['Target']
    
    # 2. TimeSeriesSplit (Walk-Forward)
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    precisions = []
    aucs = []
    fold = 0
    
    results = []

    for train_index, valid_index in tscv.split(X):
        fold += 1
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # Train Ensemble (Classifier)
        model = EnsemblePredictor()
        model.fit(X_train, y_train, X_valid, y_valid)
        
        # Predict (Probability)
        y_pred_proba = model.predict(X_valid)
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_valid, y_pred_class)
        prec = precision_score(y_valid, y_pred_class, zero_division=0)
        auc = roc_auc_score(y_valid, y_pred_proba)
        
        accuracies.append(acc)
        precisions.append(prec)
        aucs.append(auc)
        
        period = f"{df.index[valid_index[0]].date()}~"
        print(f"Fold {fold}: {period} | Acc: {acc:.2%} | Precision: {prec:.2%} | AUC: {auc:.4f}")
        
        # Store for plotting
        valid_dates = df.index[valid_index]
        for date, actual, prob in zip(valid_dates, y_valid, y_pred_proba):
            results.append({
                'Date': date,
                'Actual': actual,
                'Probability': prob,
                'Prediction': 1 if prob > 0.5 else 0
            })
            
    print("\n=== Summary ===")
    print(f"Average Accuracy:  {np.mean(accuracies):.2%}")
    print(f"Average Precision: {np.mean(precisions):.2%}")
    print(f"Average AUC:       {np.mean(aucs):.4f}")
    
    # 3. Visualization
    res_df = pd.DataFrame(results).set_index('Date')
    
    # Plot Probability vs Actual
    plt.figure(figsize=(15, 6))
    plt.plot(res_df.index, res_df['Probability'], label='Up Probability', alpha=0.7, color='cyan')
    # Actual Up (1) as green dots, Down (0) as red dots
    up_dates = res_df[res_df['Actual'] == 1].index
    down_dates = res_df[res_df['Actual'] == 0].index
    plt.scatter(up_dates, [1.02] * len(up_dates), color='green', marker='^', s=10, label='Actual Up')
    plt.scatter(down_dates, [-0.02] * len(down_dates), color='red', marker='v', s=10, label='Actual Down')
    
    plt.axhline(0.5, color='white', linestyle='--', alpha=0.5)
    plt.title(f"Classification Prediction ({ticker}) - Up Prob vs Actual")
    plt.legend()
    plt.savefig(f"verify_class_{ticker}.png")
    print(f"Saved plot: verify_class_{ticker}.png")

if __name__ == "__main__":
    # Test on SoftBank (Volatile) and Toyota (Stable)
    run_verification("9984.JP")
    run_verification("7203.JP")
