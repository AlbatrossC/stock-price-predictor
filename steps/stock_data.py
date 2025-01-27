import yfinance as yf

def stock_data(stock_symbol):
    data = yf.download(stock_symbol, period="1y", interval="1d")
    
    if data.empty:
        print(f"No data found for symbol {stock_symbol}")
        return None
    
    # Save to CSV 
    filename = f"{stock_symbol}.csv"
    data.to_csv(filename)
    print("Data Saved")
    return data

stock_symbol = input("Enter a Stock Symbol: ")
data = stock_data(stock_symbol)
