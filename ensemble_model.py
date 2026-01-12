import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import Config

class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        # 重み（単純平均）
        self.weights = {
            'lgbm': 0.34,
            'xgb': 0.33,
            'cat': 0.33
        }
        
    def _preprocess(self, X):
        """カテゴリ変数を数値化 (XGBoost対応)"""
        X = X.copy()
        for col in X.columns:
            if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes
        return X
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        3つのモデル(LightGBM, XGBoost, CatBoost)を学習させる (Classification)
        """
        print("Training Ensemble Models (Classification)...")
        
        # Preprocess (Label Encoding)
        X_train_enc = self._preprocess(X_train)
        X_valid_enc = self._preprocess(X_valid) if X_valid is not None else None
        
        # 1. LightGBM
        print("  - Training LightGBM...")
        lgb_params = Config.LGBM_PARAMS.copy()
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_valid is not None and y_valid is not None:
            valid_sets.append(lgb.Dataset(X_valid, label=y_valid))
            
        self.models['lgbm'] = lgb.train(
            lgb_params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # 2. XGBoost
        print("  - Training XGBoost...")
        model_xgb = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            early_stopping_rounds=50,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        model_xgb.fit(
            X_train_enc, y_train,
            eval_set=[(X_valid_enc, y_valid)] if X_valid is not None else None,
            verbose=False
        )
        self.models['xgb'] = model_xgb
        
        # 3. CatBoost
        print("  - Training CatBoost...")
        cat_features = [c for c in X_train.columns if c in Config.CATEGORICAL_FEATURES]
        if not cat_features:
            cat_features = None
            
        model_cat = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            verbose=0,
            allow_writing_files=False,
            cat_features=cat_features
        )
        model_cat.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid) if X_valid is not None else None,
            early_stopping_rounds=50
        )
        self.models['cat'] = model_cat
        
        print("Ensemble Training Complete.")
        
    def predict(self, X):
        """
        重み付き平均で予測 (確率 0 ~ 1)
        """
        # 1. LightGBM (predict returns prob for positive class in binary objective)
        preds_lgbm = self.models['lgbm'].predict(X)
        
        # 2. XGBoost (predict_proba returns [prob0, prob1])
        X_enc = self._preprocess(X)
        preds_xgb = self.models['xgb'].predict_proba(X_enc)[:, 1]
        
        # 3. CatBoost (predict_proba returns [prob0, prob1])
        preds_cat = self.models['cat'].predict_proba(X)[:, 1]
        
        final_pred = (
            preds_lgbm * self.weights['lgbm'] +
            preds_xgb * self.weights['xgb'] +
            preds_cat * self.weights['cat']
        )
        return final_pred
