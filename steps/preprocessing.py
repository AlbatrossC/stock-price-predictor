import pandas as pd
from sklearn.preprocessing import StandardScaler
from steps.indicators import add_indicators

def preprocess_data(data):
    data = data.copy()
    
    # Define features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']
    
    # Create target variable (1 if price goes up, 0 if down)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Separate features and target
    X = data[features]
    y = data['Target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features, index=X.index), y, data

def process_stock_data(file_path):
    data = pd.read_csv(file_path)
    
    if 'Close' not in data.columns:
        print("Error: 'Close' column is missing from the data.")
        return
    
    data_with_indicators = add_indicators(data)
    
    X_scaled, y, data_with_target = preprocess_data(data_with_indicators)
    
    output_path = file_path.replace('.csv', '_processed.csv')
    data_with_target.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Predefined file path
file_path = 'TCS.NS_with_indicators.csv'  # Replace this with the actual file path
process_stock_data(file_path)
