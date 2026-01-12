import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from model_pipeline import StockPredictionPipeline
from config import Config

from sklearn.metrics import roc_auc_score

def objective(trial, X, y):
    # ハイパーパラメータの探索範囲定義
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        # Class weight handling (optional, but good for unbalanced)
        # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), 
    }

    # Time Series Cross Validation (3-fold for speed)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 学習
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )
        
        # Predict (Probability)
        preds = model.predict(X_test)
        try:
            auc = roc_auc_score(y_test, preds)
        except ValueError:
            # Handle cases where y_test has only one class
            auc = 0.5
            
        scores.append(auc)
    
    return np.mean(scores)

def main():
    print("Preparing Data for Optimization (Once)...")
    pipeline = StockPredictionPipeline()
    pipeline.ticker = "9984.JP" # SoftBank
    
    # データ取得・生成 (1回のみ実行)
    df = pipeline.fetch_data()
    # 特徴量生成
    df = pipeline.create_features(df, dropna=True)
    
    # 特徴量指定
    features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'name_jp', 'sector', 'Target_Date']]
    X = df[features]
    y = df['Target']
    
    print(f"Data Prepared. Shape: {X.shape}")

    print("Starting Optuna Optimization (Maximize AUC)...")
    study = optuna.create_study(direction='maximize')
    
    # データ(X, y)を渡す
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20) 
    
    print("\nBest Params:")
    print(study.best_params)
    
    # 結果をJSON保存またはConfig更新用に出力
    import json
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Saved to best_params.json")

if __name__ == "__main__":
    main()
