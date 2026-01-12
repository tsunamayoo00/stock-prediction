# 株価予測AIツール システム仕様書 (Version 5.0)

## 1. プロジェクト概要
本システムは、機械学習（LightGBM）を用いて「特定の株式銘柄の翌日の終値」を予測するAIツールです。
単なる過去の価格データだけでなく、競合他社の動向、マクロ経済指標、高度なテクニカル分析、業種属性などを「全方位的」に学習し、高精度な予測を行うことを目的としています。

## 2. 技術スタック
*   **言語**: Python 3.8+
*   **コアライブラリ**:
    *   `lightgbm`: 予測モデル (Gradient Boosting Decision Tree)
    *   `pandas`, `numpy`: データ操作・数値計算
    *   `pandas-datareader`: データ取得 (Stooq API利用)
    *   `ta`: テクニカル分析ライブラリ
    *   `sqlite3`: データキャッシング
    *   `matplotlib`: 結果可視化

## 3. ディレクトリ構成
```text
stock-prediction/
├── main.py             # エントリーポイント (実行スクリプト)
├── config.py           # 設定ファイル (銘柄、期間、ハイパーパラメータ定義)
├── data_manager.py     # データ取得・SQLiteキャッシュ管理
├── model_pipeline.py   # 学習・推論・特徴量生成エンジン
├── sector_map.py       # 銘柄属性（セクター・業種）定義ファイル
├── stock_data.db       # ローカルキャッシュ用データベース (SQLite)
├── requirements.txt    # 依存ライブラリ一覧
└── prediction_result_*.png # 実行結果の可視化画像
```

## 4. 機能・ロジック詳細

### 4.1. データ取得戦略 (Data Strategy)
API制限回避と高速化のため、ハイブリッドなデータ取得・管理を採用しています。

*   **データソース**: **Stooq** (pandas-datareader経由)
    *   Yahoo Finance (yfinance) のAPI制限問題を解決するため実装。
    *   JPXコード (`7203.JP`) および米国株コード (`TSLA.US`) に対応。
*   **キャッシング**: `data_manager.py`
    *   取得したデータはすべてローカルの `stock_data.db` (SQLite) に保存。
    *   **差分更新機能**: 次回実行時は「最終更新日の翌日」以降のデータのみをフェッチし、通信量を最小限に抑えます。

### 4.2. 特徴量エンジニアリング (Feature Engineering)
`model_pipeline.py` にて、以下の4カテゴリ・計30種類以上の特徴量を生成します。

#### A. テクニカル指標 (Technical)
*   **トレンド**: SMA (5, 20, 60, 200日), MACD, 一目均衡表 (基準線, 転換線, 先行スパン), ADX (トレンド強度)
*   **オシレーター**: RSI (14日), ストキャスティクス (K, D)
*   **ボラティリティ**: Bollinger Bands (±2σ, Bandwidth), ATR (Average True Range)

#### B. マクロ経済・市場環境 (Macro/Market)
*   **為替**: USD/JPY (ドル円レート) 及其の変化率
*   **金利**: 米国10年国債利回り (US10Y)
*   **商品**: WTI原油先物 (Oil), 金先物 (Gold)
*   **指数**: 日経平均 (^NKX), S&P500 (^SPX)
*   ※ 各データはターゲット銘柄のデータフレームに日付ベースで結合されます。

#### C. マルチティッカー・競合分析 (Multi-Ticker)
*   対象銘柄だけでなく、`config.py` で定義された「関連銘柄（競合他社）」の株価データを同時学習。
*   特に「対数収益率 (Log Return)」を特徴量として用いることで、セクター全体の連動性をモデルに取り込みます。

#### D. センチメント・属性・アノマリー (Others)
*   **センチメント代替**: 出来高急増 (Volume Spike)、価格変動率の異常値 (Volatility Spike)
*   **カレンダー**: 月初(Month_Start)、月末(Month_End)、曜日、月
*   **属性情報**: `sector_map.py` に基づくセクター・業種カテゴリ (Category Features)

### 4.3. 予測モデル (Model)
*   **アルゴリズム**: LightGBM (Regressor)
*   **学習設定**:
    *   目的変数: 翌日の終値 (`Close` shifted by -1)
    *   検証期間: 過去5年分 (開発効率化のため制限中。設定で変更可能)
    *   Train/Test分割: 時系列ホールドアウト法 (直近20%をテストデータに使用)
*   **評価指標**: RMSE (Root Mean Squared Error)

## 5. 設定マニュアル (`config.py`)
ユーザーはこのファイルを編集するだけで分析対象を変更できます。

| パラメータ名 | 説明 | 設定例 |
| :--- | :--- | :--- |
| `TICKER` | 予測対象の銘柄コード | `"7203.JP"` (トヨタ) |
| `RELATED_TICKERS` | 予測の参考にする競合銘柄リスト | `["7267.JP", "7201.JP"]` |
| `MACRO_TICKERS` | マクロ指標のコード定義 | (デフォルト推奨) |
| `PERIOD` | データ取得期間 | `"5y"` (5年), `"10y"` など |
| `LGBM_PARAMS` | LightGBMのハイパーパラメータ | 辞書形式で定義 |

## 6. 今後の拡張性 (Roadmap)
*   **パラメータチューニング**: Optuna等による自動最適化の実装。
*   **Webアプリ化**: Streamlit 等を用いたGUIフロントエンドの実装。
*   **リアルタイム予測**: J-Quants API等を用いたザラ場中の予測対応。
