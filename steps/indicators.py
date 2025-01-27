import pandas as pd
import os

def add_indicators(data):
    data = data.copy()

    # Ensure 'Close' is numeric, converting errors to NaN
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # Drop rows where 'Close' is NaN after conversion
    data = data.dropna(subset=['Close'])
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data.dropna()

def process_stock_data(file_path):
    data = pd.read_csv(file_path)
    if 'Close' not in data.columns:
        print("Error: 'Close' column is missing from the data.")
        return
    data_with_indicators = add_indicators(data)
    output_path = os.path.splitext(file_path)[0] + '_with_indicators.csv'
    data_with_indicators.to_csv(output_path, index=False)
    print(f"Data with indicators saved to {output_path}")

file_path = 'TCS.NS.CSV'  # Replace with your actual file path
process_stock_data(file_path)
