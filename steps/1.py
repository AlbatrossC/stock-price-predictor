import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_stock_data(stock_symbol):
    data = yf.download(stock_symbol, period="5y", interval="1d")
    return data

def add_indicators(data):
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

def preprocess_data(data):
    data = data.copy()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI']
    data = data[features]
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    X = data[features]
    y = data['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=features, index=X.index), y, data

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_next_day(model, X, raw_data):
    # Get the most recent data
    last_row = X.tail(1)
    prediction = model.predict(last_row)
    prediction_proba = model.predict_proba(last_row)
    
    # Determine confidence
    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    direction = "UP" if prediction[0] == 1 else "DOWN"
    
    # Calculate the prediction date (assumes the next trading day is tomorrow)
    last_date = raw_data.index[-1]
    next_date = last_date + pd.Timedelta(days=1)  # Add one day to the most recent date
    
    # Print prediction results
    print("\n=== Prediction for the Next Day ===")
    print(f"Date: {next_date.strftime('%Y-%m-%d')} (Tomorrow)")
    print(f"Prediction: The stock price will go {direction}.")
    print(f"Confidence: {confidence:.2%}")
    return direction, confidence

def main(stock_symbol):
    print(f"Fetching data for {stock_symbol}...")
    data = get_stock_data(stock_symbol)
    print("Adding indicators...")
    data = add_indicators(data)
    print("Preprocessing data...")
    X, y, raw_data = preprocess_data(data)
    print("Training and evaluating model...")
    model = train_and_evaluate_model(X, y)
    predict_next_day(model, X, raw_data)
    print("Analysis complete!")

if __name__ == "__main__":
    stock_symbol = "AXISBANK.NS"  # Example stock symbol
    main(stock_symbol)
