# AI Stock Prediction System

プロフェッショナルな株価予測・分析システムです。
アンサンブル学習（LightGBM, XGBoost, CatBoost）とマクロ経済指標、ニュース感情分析を駆使し、**5日後の株価方向（上昇/下落）**を予測します。

## 主な機能
*   **AI予測**: 複数の機械学習モデルによる高精度なトレンド予測。
*   **ニュース分析**: RSSフィードから最新ニュースを取得し、感情スコアを算出。
*   **マクロ分析**: 日経先物、VIX指数などの市場全体の動きを加味。
*   **Web UI**: Streamlitによる直感的なダッシュボード。

## ファイル構成
### Core System
*   `app.py`: Webアプリケーション (Streamlit) のエントリポイント
*   `batch_run.py`: 全銘柄のバッチ予測実行スクリプト（日次運用はこれを実行）
*   `model_pipeline.py`: データ取得〜特徴量生成〜学習・予測のパイプライン
*   `ensemble_model.py`: アンサンブルモデル（分類器）の実装
*   `config.py`: 設定ファイル（パラメータ、対象銘柄、マクロ指標）
*   `data_manager.py`: データベース (SQLite) 操作クラス

### Utilities
*   `fetch_jpx_tickers.py`: 東証銘柄リストの更新
*   `fetch_news_rss.py`: ニュース収集・分析
*   `optimize_params.py`: ハイパーパラメータ最適化 (Optuna)
*   `ai_tickers.py`: AI関連銘柄リスト定義
*   `sector_map.py`: セクター情報のマッピング

### Folders
*   `archive/`: 過去の検証スクリプトや画像

## 使い方
### 1. 予測の実行 (日次)
```powershell
python batch_run.py
```
実行結果は `stock_data.db` に保存されます。

### 2. UIの起動 (確認)
```powershell
streamlit run app.py
```
ブラウザで予測結果やニュース分析を確認できます。

### 3. パラメータ調整 (メンテナンス)
```powershell
python optimize_params.py
```
モデルの精度を再チューニングする場合に使用します。
