import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import os
import time
from sqlalchemy import create_engine, text, inspect
from config import Config

class DataManager:
    # Default to SQLite local file if DATABASE_URL not set
    # Supabase/Postgres URL: postgresql://user:pass@host:port/dbname
    DB_URL = os.environ.get("DATABASE_URL", "sqlite:///stock_data.db")
    
    def __init__(self):
        self.engine = create_engine(self.DB_URL)
        self._init_db()
    
    def _init_db(self):
        """データベースとテーブルの初期化"""
        with self.engine.connect() as conn:
            # テーブル定義 (SQLAlchemy Textで実行)
            # NOTE: TEXT, REAL, INTEGER are generally compatible.
            # PRIMARY KEY constraint is standard.
            
            # 1. prices table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prices (
                    date DATE,
                    ticker VARCHAR(20),
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            # 2. tickers table (Use VARCHAR for text)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker VARCHAR(20) PRIMARY KEY,
                    name_jp TEXT,
                    sector TEXT,
                    industry TEXT,
                    last_updated DATE
                )
            """))

            # 3. predictions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    date DATE,
                    ticker VARCHAR(20),
                    name_jp TEXT,
                    sector TEXT,
                    predicted_price REAL,
                    current_price REAL,
                    diff_pct REAL,
                    signal VARCHAR(10),
                    rmse REAL,
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            # 4. news_sentiment table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    date DATE,
                    ticker VARCHAR(20),
                    sentiment_score REAL,
                    news_count INTEGER,
                    keyword_count INTEGER,
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            conn.commit()

    def get_connection(self):
        """SQLAlchemy Connectionを返す"""
        return self.engine.connect()

    def _execute_upsert(self, table, keys, data_rows, columns):
        """
        DBに応じたUPSERT処理を実行
        keys: UNIQUE constraint columns (list)
        data_rows: list of tuples values
        columns: list of column names
        """
        if not data_rows:
            return

        with self.engine.connect() as conn:
            dialect = self.engine.dialect.name
            
            # Construct VALUES placeholder
            placeholders = ", ".join([":" + c for c in columns])
            
            if dialect == 'sqlite':
                # SQLite: INSERT OR REPLACE
                stmt = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                conn.execute(text(stmt), [dict(zip(columns, row)) for row in data_rows])
                
            elif dialect == 'postgresql':
                # PostgreSQL: INSERT ... ON CONFLICT DO UPDATE
                # EXCLUDED table holds the proposed values
                update_cols = [f"{c} = EXCLUDED.{c}" for c in columns if c not in keys]
                update_stmt = ", ".join(update_cols)
                on_conflict = f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {update_stmt}"
                
                stmt = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) {on_conflict}"
                conn.execute(text(stmt), [dict(zip(columns, row)) for row in data_rows])
            
            else:
                # Fallback: Plain insert (might fail)
                stmt = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                conn.execute(text(stmt), [dict(zip(columns, row)) for row in data_rows])
                
            conn.commit()

    def update_data(self, ticker, period="5y"):
        """Stooqからデータを取得してDBを更新する"""
        last_date = self._get_last_updated_date(ticker)
        
        if last_date:
            start_date = (pd.to_datetime(last_date) + timedelta(days=1))
            if start_date >= pd.Timestamp.now().normalize():
                print(f"[{ticker}] Data is up to date.")
                return
        else:
            try:
                years = int(period.replace("y", ""))
            except:
                years = 5
            start_date = pd.Timestamp.now() - pd.DateOffset(years=years)

        print(f"[{ticker}] Fetching from Stooq (start={start_date.date()})...")
        
        try:
            time.sleep(1)
            df = web.DataReader(ticker, 'stooq', start=start_date)
            df = df.sort_index()
            df.columns = [c.capitalize() for c in df.columns]
        except Exception as e:
            print(f"[{ticker}] Download failed: {e}")
            return

        if df.empty:
            print(f"[{ticker}] No new data found.")
            # Still update ticker info to record check attempt if needed, 
            # but currently we only update on success.
            return

        self._save_to_db(ticker, df)
        
        new_last_date = df.index.max().strftime('%Y-%m-%d')
        self._update_ticker_info(ticker, new_last_date)
        print(f"[{ticker}] Database updated. Last date: {new_last_date}")
        
    def _save_to_db(self, ticker, df):
        """pricesテーブルへのUPSERT"""
        rows = []
        for dt, row in df.iterrows():
            date_str = dt.strftime('%Y-%m-%d')
            rows.append((
                date_str, ticker,
                row.get('Open', 0), row.get('High', 0), row.get('Low', 0),
                row.get('Close', 0), row.get('Volume', 0)
            ))
        
        columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        self._execute_upsert('prices', ['date', 'ticker'], rows, columns)

    def get_data(self, ticker):
        """DBからデータを取得"""
        with self.engine.connect() as conn:
            # text query with bind params
            query = text("SELECT date, open, high, low, close, volume FROM prices WHERE ticker = :ticker ORDER BY date")
            df = pd.read_sql(query, conn, params={"ticker": ticker}, index_col='date', parse_dates=['date'])
            df.columns = [c.title() for c in df.columns]
            return df

    def _get_last_updated_date(self, ticker):
        with self.engine.connect() as conn:
            query = text("SELECT last_updated FROM tickers WHERE ticker = :ticker")
            result = conn.execute(query, {"ticker": ticker}).fetchone()
            return result[0] if result else None

    def register_ticker(self, ticker, name_jp, sector, industry):
        """登録のみ（更新日は変更しない）"""
        # First check existing to preserve last_updated
        last_updated = self._get_last_updated_date(ticker)
        
        rows = [(ticker, name_jp, sector, industry, last_updated)]
        columns = ['ticker', 'name_jp', 'sector', 'industry', 'last_updated']
        self._execute_upsert('tickers', ['ticker'], rows, columns)

    def get_all_tickers(self):
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM tickers"), conn)
            
    def _update_ticker_info(self, ticker, last_updated):
        """最終更新日を更新 (属性維持)"""
        with self.engine.connect() as conn:
            query = text("SELECT name_jp, sector, industry FROM tickers WHERE ticker = :ticker")
            row = conn.execute(query, {"ticker": ticker}).fetchone()
            name_jp, sector, industry = row if row else (None, None, None)
        
        rows = [(ticker, name_jp, sector, industry, last_updated)]
        columns = ['ticker', 'name_jp', 'sector', 'industry', 'last_updated']
        self._execute_upsert('tickers', ['ticker'], rows, columns)

    def save_prediction(self, date, ticker, name_jp, sector, current, predicted, rmse):
        """予測結果保存 (UPSERT)"""
        # date cleanup
        if isinstance(date, pd.Timestamp) or hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
            
        diff_pct = ((predicted - current) / current) * 100
        # Simple signal
        signal = "HOLD"
        # Since we are essentially binary prediction now, predicted is Prob (0-1).
        # We should store "next" in predicted. Wait, predicted was Probability in batch_run.
        # But `predicted` column is REAL.
        # Signal logic: if predicted > 0.5 => BUY.
        if predicted > 0.6: # High confidence
            signal = "BUY"
        elif predicted < 0.4:
            signal = "SELL"
            
        rows = [(date_str, ticker, name_jp, sector, predicted, current, diff_pct, signal, rmse)]
        columns = ['date', 'ticker', 'name_jp', 'sector', 'predicted_price', 'current_price', 'diff_pct', 'signal', 'rmse']
        self._execute_upsert('predictions', ['date', 'ticker'], rows, columns)

    def get_latest_predictions(self):
        with self.engine.connect() as conn:
            # Find max date
            res = conn.execute(text("SELECT MAX(date) FROM predictions")).fetchone()
            latest_date = res[0]
            
            if not latest_date:
                return pd.DataFrame()
                
            query = text("SELECT * FROM predictions WHERE date = :date")
            # date comparison needs care between string and date type. 
            # In SQL, usually direct comparison works if formats match.
            df = pd.read_sql(query, conn, params={"date": latest_date})
            return df

    def is_predicted(self, ticker, date_str=None):
        if date_str is None:
            date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        with self.engine.connect() as conn:
            query = text("SELECT 1 FROM predictions WHERE ticker = :ticker AND date >= :date")
            result = conn.execute(query, {"ticker": ticker, "date": date_str}).fetchone()
            return result is not None

    def save_news_sentiment(self, date_str, score, count, keyword_count, ticker="MARKET"):
        rows = [(date_str, ticker, score, count, keyword_count)]
        columns = ['date', 'ticker', 'sentiment_score', 'news_count', 'keyword_count']
        self._execute_upsert('news_sentiment', ['date', 'ticker'], rows, columns)

    def get_news_data(self, ticker="MARKET"):
        with self.engine.connect() as conn:
            query = text("SELECT * FROM news_sentiment WHERE ticker = :ticker")
            df = pd.read_sql(query, conn, params={"ticker": ticker})
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            return df
