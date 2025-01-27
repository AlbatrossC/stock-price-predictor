from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

def get_stock_data(stock_symbol):
    try:
        # Fetch 1 year of daily data
        data = yf.download(stock_symbol, period="1y", interval="1d")
        if data.empty:
            raise ValueError(f"No data found for symbol {stock_symbol}")
        return data
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def add_indicators(data):
    data = data.copy()

    # Moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data.dropna()

def preprocess_data(data):
    data = data.copy()

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']

    # Create target variable (1 if price goes up, 0 if down)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    data = data.dropna()

    X = data[features]
    y = data['Target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=features, index=X.index), y, data

def train_model(X, y):
    """Train the Random Forest model"""
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_accuracy = 0
    best_report = None

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_report = report

    return best_model, best_accuracy, best_report

def predict_next_day(model, X, raw_data):
    last_data = X.tail(1)
    prediction = model.predict(last_data)
    probability = model.predict_proba(last_data)

    direction = "UP" if prediction[0] == 1 else "DOWN"
    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]

    # Calculate tomorrow's date
    last_date = raw_data.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    return direction, float(confidence), next_date.strftime('%Y-%m-%d')

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get stock symbol from form
        stock_symbol = request.form['stock_symbol'].upper()

        # Get stock data
        data = get_stock_data(stock_symbol)

        # Add technical indicators
        data_with_indicators = add_indicators(data)

        # Preprocess data
        X, y, raw_data = preprocess_data(data_with_indicators)

        # Train model
        model, accuracy, report = train_model(X, y)

        # Get prediction
        direction, confidence, prediction_date = predict_next_day(model, X, raw_data)

        # Prepare historical data (last 30 days)
        historical_data = raw_data.tail(30).copy()
        dates = historical_data.index.strftime('%Y-%m-%d').tolist()

        response_data = {
            'success': True,
            'prediction': direction,
            'confidence': confidence,
            'prediction_date': prediction_date,
            'accuracy': float(accuracy),
            'report': report,
            'historical_data': {
                'dates': dates,
                'prices': historical_data['Close'].values.tolist(),
                'sma20': historical_data['SMA_20'].values.tolist(),
                'ema12': historical_data['EMA_12'].values.tolist(),
                'ema26': historical_data['EMA_26'].values.tolist(),
                'rsi': historical_data['RSI'].values.tolist(),
                'macd': historical_data['MACD'].values.tolist(),
                'macd_signal': historical_data['MACD_Signal'].values.tolist()
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
