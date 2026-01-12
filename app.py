import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_manager import DataManager
from config import Config
from model_pipeline import StockPredictionPipeline

st.set_page_config(page_title="AI株価予測ダッシュボード", layout="wide")

st.title("📈 AI株価予測ダッシュボード")
st.markdown("全上場銘柄のAI予測結果を一元管理・比較するためのダッシュボードです。")

# データのロード
db = DataManager()
try:
    df_pred = db.get_latest_predictions()
except Exception as e:
    st.error(f"データベース読み込みエラー: {e}")
    df_pred = pd.DataFrame()

if df_pred.empty:
    st.warning("予測データが見つかりません。先に `batch_run.py` を実行してください。")
    st.stop()

# --- サイドバー (フィルタ) ---
st.sidebar.header("フィルタ設定")

# 業種フィルタ
# sectorカラムに日本語が入るようになるため、そのまま使用
if 'sector' in df_pred.columns:
    sectors = ["すべて"] + sorted([s for s in df_pred['sector'].dropna().unique() if s != "Index"])
    sector_filter = st.sidebar.selectbox("業種を選択", sectors)

    # フィルタリング
    if sector_filter != "すべて":
        df_view = df_pred[df_pred['sector'] == sector_filter]
    else:
        df_view = df_pred
else:
    df_view = df_pred

# 表示項目の整理 (カラム名を日本語化)
# DBカラム: date, ticker, name_jp, sector, predicted_price, current_price, diff_pct, signal, rmse
display_map = {
    'ticker': '銘柄コード',
    'name_jp': '銘柄名',
    'sector': '業種',
    'current_price': '現在値',
    'predicted_price': '予測値',
    'diff_pct': '騰落率',
    'signal': 'シグナル',
    'rmse': '予測誤差(RMSE)'
}

# カラムが存在するか確認してからリネーム
cols_to_show = [c for c in display_map.keys() if c in df_view.columns]
df_view = df_view[cols_to_show].rename(columns=display_map)

# ソート (騰落率の降順)
if '騰落率' in df_view.columns:
    df_view = df_view.sort_values(by='騰落率', ascending=False)

# 数値フォーマットの適用
# pandasのStylerを使うとリッチだが、ここでは単純に文字列変換で対応
if '騰落率' in df_view.columns:
    df_view['騰落率'] = df_view['騰落率'].apply(lambda x: f"{x:+.2f}%")
if '現在値' in df_view.columns:
    df_view['現在値'] = df_view['現在値'].apply(lambda x: f"{x:,.1f}")
if '予測値' in df_view.columns:
    df_view['予測値'] = df_view['予測値'].apply(lambda x: f"{x:,.1f}")
if '予測誤差(RMSE)' in df_view.columns:
    df_view['予測誤差(RMSE)'] = df_view['予測誤差(RMSE)'].apply(lambda x: f"{x:,.1f}")

# シグナルの日本語化
if 'シグナル' in df_view.columns:
    df_view['シグナル'] = df_view['シグナル'].map({'BUY': '買い', 'SELL': '売り', 'HOLD': '中立'})

# メインテーブル表示
st.header("📈 銘柄ランキング")
st.dataframe(df_view, height=500, use_container_width=True)

# --- 詳細分析 ---
st.markdown("---")
st.header("🔍 個別銘柄分析")

# 選択ボックス (コードと名前を結合して検索しやすくする)
# df_pred (元データ) を使う
options = df_pred.apply(lambda row: f"{row['ticker']} : {row.get('name_jp', 'Unknown')}", axis=1).tolist()
selected_option = st.selectbox("銘柄を選択または検索", options)

if st.button("詳細分析を実行"):
    selected_ticker = selected_option.split(" : ")[0]
    
    with st.spinner(f"[{selected_ticker}] を分析中..."):
        pipeline = StockPredictionPipeline()
        pipeline.ticker = selected_ticker
        
        # 実行
        try:
            results = pipeline.run()
            
            # 結果表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("株価予測チャート")
                st.image("prediction_result.png", caption="予測結果 (テストデータ)", use_column_width=True)
                
            with col2:
                st.subheader("AIが重視した特徴量")
                importance = results["importance"]
                # 日本語化対応が必要ならここでするが、Feature名は英語のままのほうがわかりやすい場合も
                st.dataframe(importance.head(10), height=300)
                st.bar_chart(importance.set_index("Feature")["Importance"].head(10))
            
            # --- ニュース分析セクション [NEW] ---
            st.markdown("---")
            st.subheader("📰 ニュースセンチメント分析 (本日の速報)")
            
            pred_data = results.get("prediction", {})
            m_sent = pred_data.get("market_sentiment", 0)
            t_sent = pred_data.get("ticker_sentiment", 0)
            n_count = int(pred_data.get("news_count", 0))
            
            ncol1, ncol2, ncol3 = st.columns(3)
            with ncol1:
                st.metric("市場全体のセンチメント", f"{m_sent:.2f}", delta=f"{m_sent:.2f}")
            with ncol2:
                # ニュースがない場合は0だが、deltaの色で雰囲気がわかる
                st.metric(f"個別銘柄センチメント ({selected_ticker})", f"{t_sent:.2f}", delta=f"{t_sent:.2f}")
            with ncol3:
                st.metric("関連ニュース件数", f"{n_count}件")
            
            if n_count > 0:
                if t_sent > 0.1:
                    st.success("ポジティブなニュースが検出されました (上方修正、増益など)")
                elif t_sent < -0.1:
                    st.error("ネガティブなニュースが検出されました (下方修正、減益など)")
                else:
                    st.info("ニュースはありますが、感情スコアは中立です")
            else:
                st.caption("※ 本日のこの銘柄に関する直接的なニュースは検出されませんでした。")                
        except Exception as e:
            st.error(f"分析エラー: {e}")

    st.success("分析完了")
