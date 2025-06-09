# update_nifty.py

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

FILE_NAME = "NIFTY_50.xlsx"

def fetch_nifty_data():
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download("^NSEI", start=today, end=today)
    if data.empty:
        print("No data available for today.")
        return None
    data.reset_index(inplace=True)
    return data

def update_excel():
    new_data = fetch_nifty_data()
    if new_data is None:
        return

    if os.path.exists(FILE_NAME):
        df_existing = pd.read_excel(FILE_NAME)
        combined = pd.concat([df_existing, new_data], ignore_index=True)
        combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    else:
        combined = new_data

    combined.to_excel(FILE_NAME, index=False)
    print("Excel updated successfully.")

if __name__ == "__main__":
    update_excel()
