
import pandas as pd
import numpy as np
import lightgbm as lgb
import ta # for direct access to ta.trend, ta.momentum, etc.
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# バッチ処理などでGUIエラーが出るのを防ぐ
plt.switch_backend('Agg')
from config import Config

class StockPredictionPipeline:
    def __init__(self):
        self.ticker = Config.TICKER
        self.period = Config.PERIOD
        self.interval = Config.INTERVAL
        self.model = None
        self.db = None

    def fetch_data(self):
        """
        DataManager経由でデータを取得・結合する
        """
        from data_manager import DataManager
        self.db = DataManager()
        
        # 1. 必要なデータをすべて更新(ダウンロード)する & 結合リスト作成
        # Target
        self.db.update_data(self.ticker, period=self.period)
        
        # Related
        for relative in Config.RELATED_TICKERS:
            self.db.update_data(relative, period=self.period)

        # Macro
        for name, code in Config.MACRO_TICKERS.items():
            self.db.update_data(code, period=self.period)
            
        # 2. DBからデータをロードして結合する
        print(f"Loading target data from DB: {self.ticker}...")
        main_df = self.db.get_data(self.ticker)
        
        if main_df.empty:
            raise ValueError("No data available for this ticker")

        # 関連銘柄の結合
        for relative in Config.RELATED_TICKERS:
            rel_df = self.db.get_data(relative)
            if not rel_df.empty:
                rel_close = rel_df[['Close']].rename(columns={'Close': f'Close_{relative}'})
                rel_df['Log_Return'] = np.log(rel_df['Close'] / rel_df['Close'].shift(1)).fillna(0)
                rel_feature = rel_df[['Log_Return']].rename(columns={'Log_Return': f'Log_Return_{relative}'})
                main_df = main_df.join(rel_close, how='left')
                main_df = main_df.join(rel_feature, how='left')

        # マクロ指標の結合
        for name, code in Config.MACRO_TICKERS.items():
            macro_df = self.db.get_data(code)
            if not macro_df.empty:
                # 終値 = その指標の値 (例: 1ドル140円)
                macro_val = macro_df[['Close']].rename(columns={'Close': name})
                
                # 変化率も重要 (為替が「動いた」ことが影響するため)
                macro_df['Pct_Change'] = macro_df['Close'].pct_change().fillna(0)
                macro_change = macro_df[['Pct_Change']].rename(columns={'Pct_Change': f'{name}_Change'})
                
                main_df = main_df.join(macro_val, how='left')
                main_df = main_df.join(macro_change, how='left')

        # ニュース情報の結合 [NEW]
        # 1. 市場全体 (Market Sentiment)
        market_news = self.db.get_news_data(ticker='MARKET')
        if not market_news.empty:
            market_news = market_news[['sentiment_score', 'news_count', 'keyword_count']].rename(columns={
                'sentiment_score': 'Market_Sentiment', 
                'news_count': 'Market_News_Count',
                'keyword_count': 'Market_Keyword_Count'
            })
            main_df = main_df.join(market_news, how='left')
        
        # 2. 個別銘柄 (Ticker Sentiment)
        ticker_news = self.db.get_news_data(ticker=self.ticker)
        if not ticker_news.empty:
            ticker_news = ticker_news[['sentiment_score', 'news_count', 'keyword_count']].rename(columns={
                'sentiment_score': 'Ticker_Sentiment', 
                'news_count': 'Ticker_News_Count',
                'keyword_count': 'Ticker_Keyword_Count'
            })
            main_df = main_df.join(ticker_news, how='left')

        # ニュースがない日は 0 (中立) で埋める
        news_cols = ['Market_Sentiment', 'Market_News_Count', 'Market_Keyword_Count', 
                     'Ticker_Sentiment', 'Ticker_News_Count', 'Ticker_Keyword_Count']
        for col in news_cols:
            if col in main_df.columns:
                main_df[col] = main_df[col].fillna(0)
            else:
                # データが全くない場合もカラムを作っておく（モデルの一貫性のため）
                main_df[col] = 0.0

        # 欠損値処理 (各国の祝日が違うため、前方埋めが必須)
        main_df = main_df.fillna(method='ffill')
        return main_df.dropna()
    

    
    def generate_synthetic_data(self):
        # (既存のコードと同じため省略せず、もし必要ならここも記述するが、
        #  今回はfetch_dataが大きく変わるので再度記述しておく方が安全)
        # ...今回は既存メソッドがそのまま使える構造ではないため、簡易的なダミーを返すよう修正
        dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
        df = pd.DataFrame(index=dates)
        df['Close'] = 100 + np.random.randn(len(dates)).cumsum()
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 1000
        return df

    def create_features(self, df, dropna=True):
        """特徴量生成"""
        print("Creating features...")
        df = df.copy()

        # ターゲット変数作成: Config.FORECAST_HORIZON 日後の終値が現在より高いか (1: Up, 0: Down)
        horizon = getattr(Config, 'FORECAST_HORIZON', 1)
        # Shift(-h) is future price.
        # Future > Current => 1, else 0
        future_close = df['Close'].shift(-horizon)
        df['Target'] = (future_close > df['Close']).astype(int)
        
        # Nan handling for the last 'horizon' rows (where future is unknown)
        # They will be 0 by default with comparison, but strictly they are invalid for training.
        # We should set them to NaN to be dropped later, OR handle them carefully.
        # Comparison with NaN returns False (0), which creates false negatives.
        # Must force NaN where future_close is NaN.
        df.loc[future_close.isna(), 'Target'] = float('nan')

        # 1. テクニカル指標 (対象銘柄のみ計算)
        # Config params
        tp = Config.TechnicalParams

        # SMA & EMA
        for window in tp["SMA"]:
            df[f'SMA_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
            df[f'EMA_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
            
        # 移動平均乖離率
        df['SMA_20_Diff'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=tp["RSI"])
        # RSIの期間違いもあったほうが良い
        df['RSI_Short'] = ta.momentum.rsi(df['Close'], window=7)

        # ROC (Rate of Change)
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Williams %R
        df['WilliamsR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

        # MACD
        macd = ta.trend.MACD(df['Close'], 
                             window_slow=tp["MACD"]["slow"], 
                             window_fast=tp["MACD"]["fast"], 
                             window_sign=tp["MACD"]["signal"])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], 
                                          window=tp["Bollinger"]["window"], 
                                          window_dev=tp["Bollinger"]["window_dev"])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Width'] = bb.bollinger_wband()
        # バンド内位置 (0=Low, 1=High)
        df['BB_Pos'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                 window=tp["Stoch"]["window"], 
                                                 smooth_window=tp["Stoch"]["smooth"])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=tp["ADX"])
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()

        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=tp["ATR"])
        
        # OBV (On-Balance Volume)
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # MFI (Money Flow Index)
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

        # Aroon
        aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=25)
        df['Aroon_Up'] = aroon.aroon_up()
        df['Aroon_Down'] = aroon.aroon_down()
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        # 上昇/下降で値が入るカラムが分かれるため結合（簡易）
        psar_up = ta.trend.psar_up(df['High'], df['Low'], df['Close'])
        df['PSAR'] = df['PSAR'].fillna(0) + psar_up.fillna(0) # 簡易合成

        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'], 
                                              window1=tp["ICHIMOKU"]["conv"], 
                                              window2=tp["ICHIMOKU"]["base"], 
                                              window3=tp["ICHIMOKU"]["span2"])
        df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
        
        # --- プライスアクション（ローソク足特徴量） ---
        # 実体長
        df['Body_Len'] = (df['Close'] - df['Open']).abs()
        # ヒゲ長
        df['Shadow_Upper'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Shadow_Lower'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        # ギャップ (今日の始値 - 前日の終値)
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        
        # 簡易センチメント (Volume Change & Volatility Change)
        df['Vol_Change'] = df['Volume'].pct_change()
        # ボラティリティ急増 (Daily Rangeの移動平均乖離)
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Range_MA'] = df['Daily_Range'].rolling(20).mean()
        df['Volatility_Spike'] = df['Daily_Range'] / df['Range_MA']

        # 2. 対数収益率 (対象銘柄)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for lag in Config.LOG_RETURN_LAGS:
            df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

        # NOTE: 関連銘柄の Log_Return_{Ticker} は既に fetch_data で結合済み
        # 必要であれば、それらのラグ特徴量もここで作ることは可能だが、
        # まずは「前日の動き」が既に入っているのでそのままとする。

        # 3. 日付情報
        df['Month'] = df.index.month
        df['Weekday'] = df.index.dayofweek

        # 4. セクター情報 (カテゴリ変数を数値化: Label Encoding)
        # 本来はLightGBMのcategory機能を使うが、ここでは簡易に数値化しておく
        from sector_map import get_ticker_info
        info = get_ticker_info(self.ticker)
        
        # 全行に同じ値が入る（静的特徴量）
        # 文字列のままだとエラーになるため、ハッシュ値などで数値化するか、カテゴリとして扱う
        # 今回は予測対象が1銘柄ごとの時系列モデルなので、これらは「定数」となり
        # 決定木の分岐には使われない（無視される）可能性が高い。
        # ★重要: 「複数銘柄を混在させて1つのモデルで学習する」場合のみ意味がある。
        # 今回は「単一銘柄モデル」なので、実はあまり意味がないが、
        # 将来的に Global Model (全銘柄共通モデル) にした時に効いてくる。
        
        df['Sector'] = info['Sector']
        df['Industry'] = info['Industry']
        
        # LightGBM用にcategory型に変換
        for col in ['Sector', 'Industry']:
            df[col] = df[col].astype('category')

        if dropna:
            return df.dropna()
        else:
            return df
            
    def predict_next_day(self, model, df_all_features):
        """
        最新データ(TargetがNaNの行)を使って翌日の株価を予測する
        df_all_features: dropnaしていない特徴量付きDataFrame
        """
        # 最後の行（最新日）
        last_row = df_all_features.iloc[[-1]] 
        
        # 学習時と同じ特徴量カラムを抽出
        features = [c for c in df_all_features.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        X_new = last_row[features]
        
        # 予測
        pred_price = model.predict(X_new)[0]
        current_price = last_row['Close'].values[0]
        date = last_row.index[-1]
        
        return date, current_price, pred_price

    def split_data(self, df):
        """時系列を維持してTrain/Test分割"""
        # Dateカラムはindexにある前提
        features = [c for c in df.columns if c not in ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        # dropna済みのdfが渡ってくる前提
        target = 'Target'


        split_idx = int(len(df) * (1 - Config.TEST_SIZE))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        return X_train, y_train, X_test, y_test, features

    def train(self, X_train, y_train):
        """LightGBMモデルの学習"""
        print("Training model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        
        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_data,
            num_boost_round=1000
        )
        return self.model

    def evaluate(self, model, X_test, y_test):
        """評価と可視化"""
        print("Evaluating model...")
        # Classification (Binary): Probability of class 1
        predictions = model.predict(X_test)
        
        # Calculate Metric
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            score = roc_auc_score(y_test, predictions)
            metric_name = "AUC"
        except:
            # Fallback (e.g. only one class in test set)
            score = 0
            metric_name = "AUC(N/A)"

        print(f"{metric_name}: {score:.4f}")

        # 可視化
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, predictions, label='Up Probability', alpha=0.7, color='cyan')
        # Actual Up (1) as Green dots
        up_idx = y_test[y_test==1].index
        plt.scatter(up_idx, [1.02]*len(up_idx), color='green', marker='^', label='Actual Up', s=20)
        # Actual Down (0) as Red dots
        down_idx = y_test[y_test==0].index
        plt.scatter(down_idx, [-0.02]*len(down_idx), color='red', marker='v', label='Actual Down', s=20)
        
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title(f"Stock Trend Prediction ({Config.TICKER}) - {metric_name}: {score:.4f}")
        plt.xlabel("Date")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        
        output_path = "prediction_result.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        
        # Return generic 'rmse' dict key for compatibility, but value is score
        return score, predictions



    def run(self):
        """パイプライン実行 (GUI対応版)"""
        raw_df = self.fetch_data()
        
        # 特徴量生成 (全データ)
        df_full = self.create_features(raw_df, dropna=False)
        
        # 学習用データ (NaNを含む行=最新行 を削除)
        df_train_valid = df_full.dropna()
        
        X_train, y_train, X_test, y_test, features = self.split_data(df_train_valid)
        
        model = self.train(X_train, y_train)
        rmse, preds = self.evaluate(model, X_test, y_test)
        
        # 特徴量重要度
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(importance.head(10))
        
        # 翌日予測
        latest_date, current_price, next_price = self.predict_next_day(model, df_full)
        
        # ニュース感情スコアの取得 (最新行)
        last_row = df_full.iloc[-1]
        market_sent = last_row.get('Market_Sentiment', 0)
        ticker_sent = last_row.get('Ticker_Sentiment', 0)
        news_count = float(last_row.get('Ticker_News_Count', 0))

        return {
            "model": model,
            "rmse": rmse,
            "importance": importance,
            "test_data": {"y_test": y_test, "preds": preds},
            "prediction": {
                "date": latest_date,
                "current": current_price,
                "next": next_price,
                "market_sentiment": market_sent,
                "ticker_sentiment": ticker_sent,
                "news_count": news_count
            }
        }
        

