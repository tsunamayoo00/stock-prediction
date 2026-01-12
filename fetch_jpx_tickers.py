import pandas as pd
import requests
import io
from data_manager import DataManager

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

def main():
    print(f"Downloading ticker list from JPX: {JPX_URL}")
    
    try:
        resp = requests.get(JPX_URL)
        resp.raise_for_status()
        
        # pandasでExcelを読み込む
        # JPXのファイル形式に合わせて調整
        df = pd.read_excel(io.BytesIO(resp.content))
        
        print(f"Downloaded {len(df)} rows.")
        
        # カラム名の確認と整理
        # 実際にダウンロードしてみないとカラム名が正確かわからないが、
        # 通常: 'コード', '銘柄名', '33業種区分', '17業種区分' などが含まれる
        print("Columns:", df.columns.tolist())
        
        db = DataManager()
        
        count = 0
        for _, row in df.iterrows():
            code = row.get('コード')
            name = row.get('銘柄名')
            sector33 = row.get('33業種区分')
            sector17 = row.get('17業種区分')
            
            if pd.isna(code):
                continue
                
            # コードは 7203 or 72030 (統合コード) の場合がある
            # Stooq用に '.JP' を付与する
            # また、末尾が0以外の5桁コードなどの扱いには注意が必要だが
            # 一般的な4桁コード + JP を基本とする
            
            code_str = str(code)
            if len(code_str) > 4:
                # 5桁の場合、最後の一桁が予備コード(通常0)
                # Stooqは4桁.JPで通るものが多いが、ETFなどは違うかも
                # ひとまず4桁に短縮して保存
                ticker = f"{code_str[:4]}.JP"
            else:
                ticker = f"{code_str}.JP"
                
            # 登録
            # sectorは33業種を使う (例: 輸送用機器)
            db.register_ticker(
                ticker=ticker,
                name_jp=name,
                sector=str(sector33),
                industry=str(sector17) # industryとして17業種を入れておく（逆でもよい）
            )
            count += 1
            
        print(f"Registered {count} tickers to database.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
