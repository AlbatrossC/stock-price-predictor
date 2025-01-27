import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta

def train_model(X, y):
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Initialize RandomForest model with specified hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train the model with the training data
    model.fit(X_train, y_train)
    
    # Predict using the test data
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report

def load_and_train_model(file_path):
    # Read the data from a CSV file
    data = pd.read_csv(file_path)
    
    # Convert the date column to datetime (assuming there's a date column)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    
    # Ensure 'Target' column exists in the dataset
    if 'Target' not in data.columns:
        print("Error: 'Target' column is missing from the data.")
        return None, None
    
    # Define the features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']
    X = data[features]
    y = data['Target']
    
    # Train the model and get evaluation results
    model, accuracy, report = train_model(X, y)
    
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    return model, data

def predict_next_day(model, X, raw_data):
    # Use the latest data point for prediction
    last_data = X.tail(1)
    prediction = model.predict(last_data)
    probability = model.predict_proba(last_data)
    
    # Determine direction and confidence based on the prediction
    direction = "UP" if prediction[0] == 1 else "DOWN"
    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]
    
    # Get tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    prediction_date = tomorrow.strftime('%Y-%m-%d')
    
    return direction, float(confidence), prediction_date

# Load the model and data, then predict the next day's movement
file_path = 'TCS.NS_with_indicators_processed.csv'
model, raw_data = load_and_train_model(file_path)

if model is not None and raw_data is not None:
    # Define the feature columns used for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']
    X = raw_data[features]
    
    # Predict the next day's movement and print the result
    direction, confidence, next_date = predict_next_day(model, X, raw_data)
    print(f"Prediction for {next_date}: Direction = {direction}, Confidence = {confidence:.2f}")