# update_nifty_nsepy.py

pip install nsepy pandas openpyxl

from nsepy import get_history
from datetime import date, datetime
import pandas as pd
import os

FILE_NAME = "NIFTY_50.xlsx"

def fetch_nifty_data():
    today = date.today()
    
    # NSE only publishes data till last completed trading day
    # So we fetch for yesterday or earlier and get the last available row
    start_date = today.replace(day=1)  # start of the month
    end_date = today

    data = get_history(symbol="NIFTY", 
                       index=True,
                       start=start_date,
                       end=end_date)

    if data.empty:
        print("No data available for the given range.")
        return None

    data.reset_index(inplace=True)
    return data.tail(1)  # Get the most recent available day

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
    print("✅ Excel updated with latest available NIFTY data from NSE.")

if __name__ == "__main__":
    update_excel()
