# 銘柄・セクター・企業名マッピング定義
# Stooqコードをキーとして、属性情報を管理する

SECTOR_MAP = {
    # --- 自動車 (Automobile) ---
    "7203.JP": {
        "Name_JP": "トヨタ自動車", "Name_EN": "Toyota Motor",
        "Sector": "Automobile", "Industry": "Transport Equipment"
    },
    "7267.JP": {
        "Name_JP": "本田技研工業", "Name_EN": "Honda Motor",
        "Sector": "Automobile", "Industry": "Transport Equipment"
    },
    "7201.JP": {
        "Name_JP": "日産自動車", "Name_EN": "Nissan Motor",
        "Sector": "Automobile", "Industry": "Transport Equipment"
    },
    "TSLA.US": {
        "Name_JP": "テスラ", "Name_EN": "Tesla Inc",
        "Sector": "Automobile", "Industry": "EV / Tech"
    },

    # --- 電機・ハイテク (Technology / Electronics) ---
    "6758.JP": {
        "Name_JP": "ソニーグループ", "Name_EN": "Sony Group",
        "Sector": "Technology", "Industry": "Consumer Electronics"
    },
    "6501.JP": {
        "Name_JP": "日立製作所", "Name_EN": "Hitachi",
        "Sector": "Technology", "Industry": "Electric Appliances"
    },
    "6702.JP": {
        "Name_JP": "富士通", "Name_EN": "Fujitsu",
        "Sector": "Technology", "Industry": "IT Services"
    },
    "AAPL.US": {
        "Name_JP": "アップル", "Name_EN": "Apple Inc",
        "Sector": "Technology", "Industry": "Consumer Electronics"
    },
    "MSFT.US": {
        "Name_JP": "マイクロソフト", "Name_EN": "Microsoft",
        "Sector": "Technology", "Industry": "Software"
    },

    # --- 半導体 (Semiconductor) ---
    "8035.JP": {
        "Name_JP": "東京エレクトロン", "Name_EN": "Tokyo Electron",
        "Sector": "Technology", "Industry": "Semiconductor Equip"
    },
    "6857.JP": {
        "Name_JP": "アドバンテスト", "Name_EN": "Advantest",
        "Sector": "Technology", "Industry": "Semiconductor Equip"
    },
    "NVDA.US": {
        "Name_JP": "エヌビディア", "Name_EN": "NVIDIA",
        "Sector": "Technology", "Industry": "Semiconductors"
    },

    # --- 通信 (Telecommunication) ---
    "9432.JP": {
        "Name_JP": "日本電信電話", "Name_EN": "NTT",
        "Sector": "Communication", "Industry": "Telecom Services"
    },
    "9984.JP": {
        "Name_JP": "ソフトバンクG", "Name_EN": "Softbank Group",
        "Sector": "Communication", "Industry": "Telecom / Investment"
    },

    # --- 商社 (Trading) ---
    "8001.JP": {
        "Name_JP": "伊藤忠商事", "Name_EN": "Itochu",
        "Sector": "Commercial", "Industry": "Trading Companies"
    },
    "8031.JP": {
        "Name_JP": "三井物産", "Name_EN": "Mitsui & Co",
        "Sector": "Commercial", "Industry": "Trading Companies"
    },

    # --- 金融 (Finance) ---
    "8306.JP": {
        "Name_JP": "三菱UFJ", "Name_EN": "MUFG",
        "Sector": "Finance", "Industry": "Banks"
    },
    "8316.JP": {
        "Name_JP": "三井住友FG", "Name_EN": "SMFG",
        "Sector": "Finance", "Industry": "Banks"
    },
    
    # --- 小売 (Retail) ---
    "9983.JP": {
        "Name_JP": "ファーストリテイリング", "Name_EN": "Fast Retailing",
        "Sector": "Retail", "Industry": "Apparel"
    },
    "7974.JP": {
        "Name_JP": "任天堂", "Name_EN": "Nintendo",
        "Sector": "Any", "Industry": "Entertainment"
    },

    # --- 指数 (Indices) ---
    "^N225":   {"Name_JP": "日経平均", "Name_EN": "Nikkei 225", "Sector": "Index", "Industry": "Market Index"},
}

def get_ticker_info(ticker):
    """
    ティッカーごとの属性情報を返す
    """
    return SECTOR_MAP.get(ticker, {
        "Name_JP": "不明", 
        "Name_EN": "Unknown",
        "Sector": "Unknown",
        "Industry": "Unknown"
    })

def get_universe_tickers():
    """
    指数(^N225など)を除く、予測対象の全銘柄コードリストを返す
    """
    return [k for k, v in SECTOR_MAP.items() if v["Sector"] != "Index"]
