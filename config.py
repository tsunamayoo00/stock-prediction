class Config:
    # データ取得設定
    TICKER = "7203.JP"
    
    # 関連銘柄 (競合他社)
    RELATED_TICKERS = ["7267.JP", "7201.JP"]

    # マクロ経済指標 (Stooqコード)
    # USDJPY(為替), 10USY(米10年債), CL.F(原油), GC.F(金), ^NKX(日経平均), ^SPX(S&P500), ^VIX(恐怖指数)
    # Stooqではコードが少し特殊な場合がある: 
    # USDJPY -> "USDJPY"
    # 米10年債 -> "10USY.B"
    # 原油 -> "CL.F" 
    # 日経平均 -> "^NKX"
    MACRO_TICKERS = {
        "USDJPY": "USDJPY",      # ドル円
        "US10Y": "10USY.B",      # 米国10年債利回り
        "Oil": "CL.F",           # WTI原油
        "Gold": "GC.F",          # 金先物
        "Nikkei225": "NK.F",     # 日経平均先物 (CME)
        "SP500": "^SPX",         # S&P500
        "VIX": "^VIX"            # VIX (恐怖指数)
    }

    PERIOD = "5y"
    INTERVAL = "1d"
    
    # 予測期間 (Target Horizon)
    # 1 = 翌日, 5 = 5日後(スイング)
    FORECAST_HORIZON = 5

    # 特徴量生成設定: テクニカル指標用パラメータ
    TechnicalParams = {
        "SMA": [5, 20, 60, 200],
        "RSI": 14,
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "Bollinger": {"window": 20, "window_dev": 2},
        "Stoch": {"window": 14, "smooth": 3},
        "ADX": 14,
        "ATR": 14,
        "ICHIMOKU": {"conv": 9, "base": 26, "span2": 52}
    }
    
    # ラグ特徴量
    LOG_RETURN_LAGS = [1, 2, 3, 5, 10, 20] 

    # カテゴリ特徴量
    CATEGORICAL_FEATURES = ['Sector', 'Industry']

    # モデル学習設定
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
