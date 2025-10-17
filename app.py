from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

app = Flask(__name__)

# ---------------------- LOGGING CONFIG ----------------------
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.LIGHTBLACK_EX}[%(asctime)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)

def log_info(msg):
    logging.info(Fore.CYAN + msg + Style.RESET_ALL)

def log_success(msg):
    logging.info(Fore.GREEN + msg + Style.RESET_ALL)

def log_warning(msg):
    logging.warning(Fore.YELLOW + msg + Style.RESET_ALL)

def log_error(msg):
    logging.error(Fore.RED + msg + Style.RESET_ALL)

# -------------------------------------------------------------


def get_stock_data(stock_symbol):
    try:
        log_info(f"Fetching data for stock: {Fore.LIGHTYELLOW_EX}{stock_symbol}")
        data = yf.download(stock_symbol, period="1y", interval="1d", progress=False)
        if data.empty:
            raise ValueError(f"No data found for symbol {stock_symbol}")
        log_success(f"Data fetched successfully for {stock_symbol} ({len(data)} records)")
        return data
    except Exception as e:
        log_error(f"Error fetching data for {stock_symbol}: {e}")
        raise Exception(f"Error fetching data: {str(e)}")


def add_indicators(data):
    data = data.copy()
    log_info("Adding technical indicators...")

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

    data = data.dropna()
    log_success("Indicators added successfully.")
    return data


def preprocess_data(data):
    log_info("Preprocessing data for model training...")
    data = data.copy()

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    X = data[features]
    y = data['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log_success(f"Data preprocessed: {len(X)} samples ready.")
    return pd.DataFrame(X_scaled, columns=features, index=X.index), y, data


def train_model(X, y):
    log_info("Training Random Forest model...")
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_accuracy = 0
    best_report = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], y_pred)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_report = classification_report(y.iloc[test_idx], y_pred, output_dict=False)

        log_info(f"Fold {fold} accuracy: {Fore.LIGHTYELLOW_EX}{acc:.4f}")

    log_success(f"Model trained successfully. Best accuracy: {Fore.LIGHTGREEN_EX}{best_accuracy:.4f}")
    return best_model, best_accuracy, best_report


def predict_next_day(model, X, raw_data):
    last_data = X.tail(1)
    prediction = model.predict(last_data)
    probability = model.predict_proba(last_data)

    direction = "UP" if prediction[0] == 1 else "DOWN"
    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]

    last_date = raw_data.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    log_success(f"Prediction: {Fore.LIGHTCYAN_EX}{direction} "
                f"{Fore.WHITE}(Confidence: {confidence:.2f}) for {next_date.strftime('%Y-%m-%d')}")
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
        stock_symbol = request.form['stock_symbol'].upper().strip()
        log_info(f"--- New Prediction Request for {Fore.LIGHTMAGENTA_EX}{stock_symbol} ---")

        data = get_stock_data(stock_symbol)
        data_with_indicators = add_indicators(data)
        X, y, raw_data = preprocess_data(data_with_indicators)
        model, accuracy, report = train_model(X, y)
        direction, confidence, prediction_date = predict_next_day(model, X, raw_data)

        historical_data = raw_data.tail(30).copy()
        dates = historical_data.index.strftime('%Y-%m-%d').tolist()

        log_success(f"Prediction completed for {stock_symbol} ✅")

        return jsonify({
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
        })

    except Exception as e:
        log_error(f"Error processing stock: {e}")
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    log_info(f"{Fore.LIGHTGREEN_EX}Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
