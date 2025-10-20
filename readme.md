# 📈 Stock Price Predictor

A machine learning-powered web application that predicts stock price movements using technical indicators and historical data. Built with Flask and powered by Random Forest classification, this tool provides next-day price direction predictions with confidence scores and interactive visualizations.

## 🌐 Live Demo

**Deployment Link:** [https://stock-price-predictor-cfq7.onrender.com/](https://stock-price-predictor-cfq7.onrender.com/)

---

## 🛠️ Tech Stack

### **Backend**
- **Framework:** Flask (Python)
- **Machine Learning:** scikit-learn (Random Forest Classifier)
- **Data Processing:** pandas, numpy
- **Model Training:** TimeSeriesSplit for cross-validation

### **Frontend**
- **Languages:** HTML5, CSS3, JavaScript (ES6+)
- **Styling:** Tailwind CSS
- **Charts:** Chart.js
- **Design:** Responsive, minimalist UI

### **Data Source**
- **API:** Yahoo Finance (yfinance)
- **Data Type:** Real-time stock market data
- **Indicators:** SMA, EMA, RSI, MACD

### **Additional Tools**
- **Logging:** colorama (colored terminal output)
- **Deployment:** Render

---

## 🚀 Running Locally

Follow these steps to run the Stock Price Predictor on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/AlbatrossC/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Access the Application

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 📊 Features

- Real-time stock price predictions
- Technical indicator analysis (RSI, MACD, Moving Averages)
- Interactive charts and visualizations
- Model confidence and accuracy metrics
- Responsive design for mobile and desktop
- Company symbol search with autocomplete