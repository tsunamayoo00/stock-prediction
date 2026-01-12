from model_pipeline import StockPredictionPipeline
from config import Config
import pandas as pd

def test():
    print("Testing Feature Generation in Pipeline...")
    pipeline = StockPredictionPipeline()
    pipeline.ticker = "7203.JP" # Major ticker
    
    try:
        df = pipeline.fetch_data()
        df = pipeline.create_features(df, dropna=True)
        print(f"Generated {len(df.columns)} features.")
        print("Columns:", df.columns.tolist())
        
        # Check specific new features
        new_features = ['CCI', 'OBV', 'Body_Len', 'Shadow_Upper', 'Gap', 'EMA_20']
        missing = [f for f in new_features if f not in df.columns]
        
        if not missing:
            print("SUCCESS: All new features found.")
        else:
            print(f"FAILURE: Missing features: {missing}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
