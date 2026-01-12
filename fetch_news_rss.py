import feedparser
from asari.api import Sonar
import pandas as pd
import datetime
import re
from data_manager import DataManager

# RSS Feeds (Free sources)
RSS_URLS = {
    "Nikkei": "https://assets.wor.jp/rss/rdf/nikkei/news.rdf", 
    "Yahoo_Economy": "https://news.yahoo.co.jp/rss/categories/business.xml",
    "Reuters_Biz": "http://feeds.reuters.com/reuters/JPBusinessNews"
}

# 重要キーワード (簡易版)
KEYWORDS = {
    "Upward": ["上方修正", "増益", "最高益", "黒字化", "増配", "自社株買い"],
    "Downward": ["下方修正", "減益", "赤字", "不祥事", "減配"]
}

def clean_text(text):
    """HTMLタグなどを除去"""
    text = re.sub(r'<[^>]+>', '', text)
    return text

def main():
    print("Initializing News Fetcher...")
    
    # NLP Analyzer (asari)
    sonar = Sonar()
    db = DataManager()
    
    # 1. 銘柄辞書の作成 (Entity Linking用水)
    print("Loading ticker Map for Entity Linking...")
    df_tickers = db.get_all_tickers()
    
    ticker_map = {}
    if not df_tickers.empty:
        for _, row in df_tickers.iterrows():
            name = row.get('name_jp')
            code = row.get('ticker')
            if name and len(name) > 2:
                ticker_map[name] = code

    print(f"Loaded {len(ticker_map)} entities.")

    # 集計用変数
    daily_scores = []
    daily_keywords = 0
    
    # 銘柄ごとのスコア (ticker -> [scores])
    ticker_scores = {}
    ticker_keywords = {}

    # 今日の日付
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    # 2. RSS取得ループ
    for source, url in RSS_URLS.items():
        print(f"Fetching {source}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                
                # 感情分析 (asari)
                res = sonar.ping(text=title)
                # res['classes'] = [{'class_name': 'negative', 'confidence': 0.53}, ...]
                
                pos_conf = 0.0
                neg_conf = 0.0
                for c in res['classes']:
                    if c['class_name'] == 'positive':
                        pos_conf = c['confidence']
                    elif c['class_name'] == 'negative':
                        neg_conf = c['confidence']
                
                # スコア計算 (-1.0 ~ 1.0)
                score = pos_conf - neg_conf
                
                # キーワード判定
                keyword_hit = 0
                for k in KEYWORDS["Upward"]:
                    if k in title: keyword_hit += 1
                for k in KEYWORDS["Downward"]:
                    if k in title: keyword_hit += 1 
                
                # 市場全体集計
                daily_scores.append(score)
                daily_keywords += keyword_hit
                
                # Entity Linking (銘柄特定)
                for name, code in ticker_map.items():
                    if name in title:
                        if code not in ticker_scores:
                            ticker_scores[code] = []
                            ticker_keywords[code] = 0
                        ticker_scores[code].append(score)
                        ticker_keywords[code] += keyword_hit
                        print(f"  -> Matched: {name} ({code}) : {score:.2f}")

        except Exception as e:
            print(f"Error fetching {source}: {e}")

    # 3. 保存 (Market Sentiment)
    if daily_scores:
        avg_score = sum(daily_scores) / len(daily_scores)
        count = len(daily_scores)
        print(f"Market Sentiment: Score={avg_score:.2f}, Count={count}, Keywords={daily_keywords}")
        
        db.save_news_sentiment(today_str, avg_score, count, daily_keywords, ticker="MARKET")
    else:
        print("No news found.")

    # 4. 保存 (Ticker Sentiment)
    for ticker, scores in ticker_scores.items():
        avg_s = sum(scores) / len(scores)
        cnt = len(scores)
        k_cnt = ticker_keywords[ticker]
        
        db.save_news_sentiment(today_str, avg_s, cnt, k_cnt, ticker=ticker)
        print(f"Saved {ticker}: Score={avg_s:.2f}")

if __name__ == "__main__":
    main()
